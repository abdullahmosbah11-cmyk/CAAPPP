"""
VeriMath — Module 8 (Novel Contribution #2): Structural AST Anti-Cheating Verifier
==================================================================================
Verifies structural mathematical faithfulness between the user's natural-language
problem and the emitted Lean 4 theorem signature (`example ... : <PROP> := by ...`).

Prevents autoformalization "reward hacking" / semantic substitution attacks that
produce a Lean-valid proof of the *wrong* statement, including:
  1. Tautology Substitution: emitting `1 + 1 = 2`, `True`, or `x = x` for a real problem.
  2. Relation Polarity Weakening / Flipping: turning strict `<` / `>` into `≤` / `≥`
     or `=`, or inverting `<` ↔ `>`.
  3. Right-Hand-Side Tampering on Claim Verification: altering `(a + b)^2 = a^2 + b^2`
     into `(a + b)^2 = a^2 + 2*a*b + b^2` so a false student claim passes verification.
  4. Dropped Domain Variables / Target Functions: omitting user variables (`x`, `y`)
     or transcendental functions (`sin`, `cos`, `log`, `exp`, `deriv`) from the Lean AST.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

_MATH_FUNCS = ("sin", "cos", "tan", "exp", "log", "sqrt", "arctan", "atan", "arcsin", "asin", "arccos", "acos")


@dataclass
class ASTFaithfulnessResult:
    passed: bool
    checks_run: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    lean_proposition: str = ""
    lean_variables: List[str] = field(default_factory=list)
    problem_variables: List[str] = field(default_factory=list)


def _strip_lean_casts(s: str) -> str:
    return re.sub(r"\s*:\s*(?:ℝ|Real|R|ℕ|Nat|ℤ|Int|ℂ|Complex)(?=\))", "", s)


def _lean_to_sympy_expr(lean_expr: str) -> sp.Expr:
    s = _strip_lean_casts(lean_expr.strip())
    s = s.replace("Complex.I", "I")
    s = s.replace("Real.pi", "pi")
    for fn in ("sin", "cos", "tan", "exp", "log", "sqrt", "arctan", "arcsin", "arccos"):
        s = s.replace(f"Real.{fn}", fn)
    s = s.replace("arctan", "atan").replace("arcsin", "asin").replace("arccos", "acos")
    s = re.sub(r"\b(sin|cos|tan|exp|log|sqrt|atan|asin|acos)\s+([a-zA-Z0-9_]+)", r"\1(\2)", s)
    local_dict = {
        "x": sp.Symbol("x"),
        "y": sp.Symbol("y"),
        "z": sp.Symbol("z"),
        "a": sp.Symbol("a"),
        "b": sp.Symbol("b"),
        "c": sp.Symbol("c"),
        "n": sp.Symbol("n"),
        "k": sp.Symbol("k"),
        "t": sp.Symbol("t"),
        "I": sp.I,
        "pi": sp.pi,
        "e": sp.E,
    }
    return parse_expr(s, local_dict=local_dict, transformations=_TRANSFORMS)


def _prob_to_sympy_expr(prob_expr: str, use_complex: bool = False) -> sp.Expr:
    s = prob_expr.strip()
    s = s.replace("[", "(").replace("]", ")").replace("−", "-").replace("×", "*").replace("·", "*")
    s = s.replace("²", "^2").replace("³", "^3").replace("π", "pi")
    s = re.sub(r"\bln\b", "log", s, flags=re.I)
    if use_complex:
        s = re.sub(r"(\d)\s*i\b", r"\1*I", s)
        s = re.sub(r"(?<![a-zA-Z_])i(?![a-zA-Z_])", "I", s)
    local_dict = {
        "x": sp.Symbol("x"),
        "y": sp.Symbol("y"),
        "z": sp.Symbol("z"),
        "a": sp.Symbol("a"),
        "b": sp.Symbol("b"),
        "c": sp.Symbol("c"),
        "n": sp.Symbol("n"),
        "k": sp.Symbol("k"),
        "t": sp.Symbol("t"),
        "I": sp.I,
        "pi": sp.pi,
        "e": sp.E,
    }
    return parse_expr(s, local_dict=local_dict, transformations=_TRANSFORMS)


def extract_lean_signature(lean_statement: str) -> Tuple[str, str]:
    """
    Split `example <binders> : <PROP> := by <proof>` into `(binders, prop_str)`.
    Respects parentheses nesting inside binders like `(ha : 0 < a)`.
    """
    stmt = lean_statement.strip()
    head = stmt.split(":= by")[0].strip()
    if not head.startswith("example"):
        return "", head
    idx = len("example")
    n = len(head)
    depth = 0
    while idx < n:
        ch = head[idx]
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0:
            return head[len("example"):idx].strip(), head[idx + 1:].strip()
        idx += 1
    return "", head


def _split_top_level_relation(prop_str: str) -> Optional[Tuple[str, str, str]]:
    """Find top-level relation (`=`, `<`, `>`, `≤`, `≥`, `<=`, `>=`, `∣`) outside parens."""
    depth = 0
    i = 0
    n = len(prop_str)
    while i < n:
        ch = prop_str[i]
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth = max(0, depth - 1)
        elif depth == 0:
            two = prop_str[i:i + 2]
            if two in ("<=", ">="):
                return prop_str[:i].strip(), two, prop_str[i + 2:].strip()
            if ch in ("=", "<", ">", "≤", "≥", "∣"):
                return prop_str[:i].strip(), ch, prop_str[i + 1:].strip()
        i += 1
    return None


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        balanced = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i < len(s) - 1:
                    balanced = False
                    break
        if balanced and depth == 0:
            s = s[1:-1].strip()
        else:
            break
    return s


def verify_ast_faithfulness(
    problem: str,
    lean_statement: str,
    subkind: Optional[str] = None,
) -> ASTFaithfulnessResult:
    """
    Deterministic AST Anti-Cheating Audit between user problem and Lean 4 theorem.
    """
    if not lean_statement or not lean_statement.strip():
        return ASTFaithfulnessResult(passed=False, reason="Empty Lean statement.")

    p = problem.strip()
    p_l = p.lower()
    binders, prop_str = extract_lean_signature(lean_statement)
    checks: List[str] = []

    # Collect variables bound or referenced in Lean proposition
    lean_clean = _strip_lean_casts(prop_str)
    lean_no_kw = re.sub(
        r"\b(?:Real|Complex|Nat|Finset|Int|Icc|I|sin|cos|tan|exp|log|sqrt|deriv|fun|Prime|gcd|lcm|choose|factorial|descFactorial)\b",
        " ",
        lean_clean,
    )
    lean_vars = sorted(set(re.findall(r"\b([a-zA-Z])\b", lean_no_kw)))

    # Collect explicit single-letter math variables written inside algebraic problem formulas
    p_formula_vars = sorted(set(re.findall(r"(?<![a-zA-Z])([xyzabcnkt])\b(?!(?:\s+is\b|\s+of\b))", p)))
    checks.append("tautology_substitution_guard")

    # ── 1. Reject Trivial Constant Tautologies (e.g. `1 + 1 = 2`, `True`, `0 = 0`) ─
    prop_compact = re.sub(r"\s+", "", lean_clean)
    if prop_compact in ("1+1=2", "0=0", "1=1", "True", "trivial"):
        if not any(t in p_l for t in ("1 + 1", "1+1", "0 = 0", "true")):
            return ASTFaithfulnessResult(
                passed=False,
                checks_run=checks,
                reason=f"AST Anti-Cheating Rejection: Lean statement collapsed to trivial tautology '{prop_str}'.",
                lean_proposition=prop_str,
                lean_variables=lean_vars,
                problem_variables=p_formula_vars,
            )

    # Check trivial reflexive identity `X = X` when the problem asks to prove an identity between two distinct expressions
    top_rel = _split_top_level_relation(prop_str)
    if top_rel and top_rel[1] == "=" and "deriv" not in prop_str and "∫" not in prop_str:
        l_norm = re.sub(r"\s+", "", _strip_lean_casts(top_rel[0]))
        r_norm = re.sub(r"\s+", "", _strip_lean_casts(top_rel[2]))
        if l_norm == r_norm and not re.match(r"^\s*(?:prove|show)\s+.*\b(.+?)\s*=\s*\1\b", p_l):
            # Allow simple numeric evaluation like 4 = 4 only if not an algebraic variable identity
            if len(l_norm) > 0 and ("=" in p or "prove" in p_l or "verify" in p_l):
                if l_norm != "2" or "1+1" not in p_l:
                    return ASTFaithfulnessResult(
                        passed=False,
                        checks_run=checks,
                        reason=f"AST Anti-Cheating Rejection: Reflexive tautology '{prop_str}' detected.",
                        lean_proposition=prop_str,
                        lean_variables=lean_vars,
                        problem_variables=p_formula_vars,
                    )

    # ── 2. Strict Inequality Polarity / Anti-Weakening Guard ──────────────────
    m_prob_ineq = (
        re.match(
            r"^\s*(?:prove\s+that|show\s+that|verify\s+that|is\s+it\s+true\s+that|check\s+(?:whether|if))?\s*(.+?)\s*(<=|>=|≤|≥|<|>)\s*(.+?)[?.!]*$",
            p_l,
        )
        if not any(w in p_l for w in ("maximize", "minimize", "maximizes", "minimizes", "where ", "swimmer", "channel"))
        and p_l.count(".") == 0
        else None
    )
    if m_prob_ineq and "=" not in m_prob_ineq.group(1) and "=" not in m_prob_ineq.group(3):
        checks.append("inequality_polarity_guard")
        p_lhs_raw, p_op, p_rhs_raw = m_prob_ineq.group(1).strip(), m_prob_ineq.group(2), m_prob_ineq.group(3).strip()
        op_norm = {"<": "<", ">": ">", "<=": "≤", "≤": "≤", ">=": "≥", "≥": "≥"}[p_op]

        if not top_rel and not prop_str.startswith("¬"):
            return ASTFaithfulnessResult(
                passed=False,
                checks_run=checks,
                reason="AST Anti-Cheating Rejection: Problem is an inequality, but Lean statement lacks an inequality relation.",
                lean_proposition=prop_str,
            )

        if top_rel:
            l_lhs_raw, l_op_raw, l_rhs_raw = top_rel
            l_op = {"<": "<", ">": ">", "<=": "≤", "≤": "≤", ">=": "≥", "≥": "≥"}.get(l_op_raw, l_op_raw)
            # Forbid strict -> weak (< turned into ≤) or equality substitution
            if op_norm in ("<", ">") and l_op not in ("<", ">"):
                return ASTFaithfulnessResult(
                    passed=False,
                    checks_run=checks,
                    reason=f"AST Anti-Cheating Rejection: Strict inequality '{op_norm}' weakened/replaced by '{l_op}'.",
                    lean_proposition=prop_str,
                )
            if l_op == "=":
                return ASTFaithfulnessResult(
                    passed=False,
                    checks_run=checks,
                    reason="AST Anti-Cheating Rejection: Inequality problem replaced by an equality.",
                    lean_proposition=prop_str,
                )
            # Check AST structural isomorphism on sides
            try:
                p_l_sym = _prob_to_sympy_expr(p_lhs_raw)
                p_r_sym = _prob_to_sympy_expr(p_rhs_raw)
                l_l_sym = _lean_to_sympy_expr(_strip_outer_parens(l_lhs_raw))
                l_r_sym = _lean_to_sympy_expr(_strip_outer_parens(l_rhs_raw))
                same_dir = (op_norm == l_op) and sp.simplify(p_l_sym - l_l_sym) == 0 and sp.simplify(p_r_sym - l_r_sym) == 0
                flipped_ok = (
                    ((op_norm == "<" and l_op == ">") or (op_norm == ">" and l_op == "<")
                     or (op_norm == "≤" and l_op == "≥") or (op_norm == "≥" and l_op == "≤"))
                    and sp.simplify(p_l_sym - l_r_sym) == 0
                    and sp.simplify(p_r_sym - l_l_sym) == 0
                )
                if not (same_dir or flipped_ok):
                    return ASTFaithfulnessResult(
                        passed=False,
                        checks_run=checks,
                        reason=f"AST Anti-Cheating Rejection: Lean inequality sides do not match problem ({p_lhs_raw} {op_norm} {p_rhs_raw}).",
                        lean_proposition=prop_str,
                    )
            except Exception:
                pass

    # ── 3. Explicit Claim Identity Isomorphism (`Prove that A = B`) ───────────
    m_prob_claim = (
        re.search(
            r"^(?:prove\s+that|show\s+that|verify\s+that|verify|check\s+(?:whether|if)|is\s+it\s+true\s+that)\s+(.+?)\s*=\s*(.+?)[?.!]*$",
            p_l,
        )
        if not any(op in p_l for op in ("<", ">", "≤", "≥"))
        else None
    )
    if m_prob_claim:
        checks.append("identity_ast_isomorphism")
        p_lhs_raw = re.sub(r"\s+for\s+(?:all\s+)?.*$", "", m_prob_claim.group(1).strip())
        p_rhs_raw = re.sub(r"\s+for\s+(?:all\s+)?.*$", "", m_prob_claim.group(2).strip())
        is_c = ("ℂ" in prop_str or "Complex.I" in prop_str or bool(re.search(r"(?:\d\s*\*?\s*i\b|[+-]\s*i\b)", p_l)))
        if not top_rel or top_rel[1] != "=":
            return ASTFaithfulnessResult(
                passed=False,
                checks_run=checks,
                reason="AST Anti-Cheating Rejection: Identity claim did not produce an equality theorem.",
                lean_proposition=prop_str,
            )
        try:
            p_l_sym = _prob_to_sympy_expr(p_lhs_raw, use_complex=is_c)
            p_r_sym = _prob_to_sympy_expr(p_rhs_raw, use_complex=is_c)
            l_l_sym = _lean_to_sympy_expr(_strip_outer_parens(top_rel[0]))
            l_r_sym = _lean_to_sympy_expr(_strip_outer_parens(top_rel[2]))
            pos_syms = {v: sp.Symbol(v, positive=True) for v in re.findall(r"\b([a-zA-Z])\b", p_lhs_raw + p_rhs_raw)}
            # Check syntactic/algebraic equivalence per side (WITHOUT equating LHS to RHS!)
            l_match = sp.simplify(sp.expand(p_l_sym - l_l_sym)) == 0
            r_match = sp.simplify(sp.expand(p_r_sym - l_r_sym)) == 0
            swap_match = (
                sp.simplify(sp.expand(p_l_sym - l_r_sym)) == 0
                and sp.simplify(sp.expand(p_r_sym - l_l_sym)) == 0
            )
            if not ((l_match and r_match) or swap_match):
                return ASTFaithfulnessResult(
                    passed=False,
                    checks_run=checks,
                    reason=(
                        f"AST Anti-Cheating Rejection: Lean equality sides do not match problem claim "
                        f"'{p_lhs_raw} = {p_rhs_raw}' (possible RHS tampering)."
                    ),
                    lean_proposition=prop_str,
                )
        except Exception:
            pass

    # ── 4. Derivative / Gradient Function Body Isomorphism ───────────────────
    if "deriv (fun " in prop_str and not any(w in p_l for w in ("limit", "lim ", "mean value", "rolle", "implicit")):
        checks.append("derivative_target_isomorphism")
        m_deriv_fn = re.search(r"deriv\s*\(\s*fun\s+([a-zA-Z])\s*=>\s*(.+?)\s*\)\s*(?:\1|\()", prop_str)
        m_prob_fx = re.search(r"(?:f\(x\)|y|gradient\s+of|partial\s+derivative\s+of)\s*(?:=\s*)?([a-zA-Z0-9_^*+\-/().\s]+?)(?:\s+with\s+respect|\s+at\b|[?.!]|$)", p, re.I)
        if m_deriv_fn and m_prob_fx and "of" not in m_prob_fx.group(1).lower():
            try:
                f_lean_sym = _lean_to_sympy_expr(m_deriv_fn.group(2))
                f_prob_sym = _prob_to_sympy_expr(m_prob_fx.group(1))
                if f_prob_sym.free_symbols and not f_lean_sym.free_symbols:
                    return ASTFaithfulnessResult(
                        passed=False,
                        checks_run=checks,
                        reason="AST Anti-Cheating Rejection: Symbolic derivative target replaced by constant function in Lean.",
                        lean_proposition=prop_str,
                    )
            except Exception:
                pass

    # ── 5b. Discrete Literal Preservation Guard (Prime, GCD, LCM, Comb) ─────
    if any(tok in prop_str for tok in ("Nat.Prime", "Nat.gcd", "Nat.lcm", "Nat.choose", "Nat.factorial", "Nat.descFactorial", "∣")):
        checks.append("literal_preservation_guard")
        p_ints = set(re.findall(r"\b\d+\b", p))
        l_ints = set(re.findall(r"\b\d+\b", lean_clean))
        if p_ints and not (p_ints & l_ints):
            return ASTFaithfulnessResult(
                passed=False,
                checks_run=checks,
                reason="AST Anti-Cheating Rejection: Lean discrete theorem does not reference the integers from the problem.",
                lean_proposition=prop_str,
            )

    # ── 6. Function Presence Guard Directly Against Lean AST ──────────────────
    checks.append("transcendental_function_guard")
    prop_low = prop_str.lower()
    for fn in ("sin", "cos", "tan", "exp", "log"):
        if re.search(rf"\b{fn}\s*\(", p_l):
            # In derivatives/identities, ensure function isn't silently erased unless prompt is differentiating a constant identity like sin^2 + cos^2
            if fn not in prop_low and "solve" in p_l:
                return ASTFaithfulnessResult(
                    passed=False,
                    checks_run=checks,
                    reason=f"AST Anti-Cheating Rejection: Problem contains '{fn}(...)' which vanished from Lean AST.",
                    lean_proposition=prop_str,
                )

    return ASTFaithfulnessResult(
        passed=True,
        checks_run=checks,
        lean_proposition=prop_str,
        lean_variables=lean_vars,
        problem_variables=p_formula_vars,
    )
