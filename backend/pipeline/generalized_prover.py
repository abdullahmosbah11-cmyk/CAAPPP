"""
VeriMath — Module 6 (Novel Contribution): Generalized Out-of-Domain Autoformalizer & Prover
=============================================================================================
Invoked IF AND ONLY IF a problem does NOT match any of VeriMath's 244 assigned
templates (`classify_problem(problem) is None`) and is not in `demo_cache.db`.

Architecture (Zero-Trust Propose → Formalize → Judge → Lean Verify / Reject):
  1. Domain Gate: Rejects non-mathematical prompts cleanly.
  2. Proposal Generation:
       a) When a trained/local GGUF LLM is loaded, prompts it to propose a
          candidate answer AND a standalone Lean 4 theorem (`example ... := by ...`)
          without `sorry`/`admit`/`axiom`.
       b) Deterministic Generalized Symbolic Autoformalizer handles broad unseen
          mathematical structures outside the fixed templates:
            - Linear systems (2x2 and 3x3)
            - 2x2 and 3x3 matrix determinants
            - Vector dot products, 3D cross products & Euclidean norms
            - Multivariable partial derivatives & gradients (∇f)
            - Divisibility (a ∣ b), modulo (a % m), GCD, LCM, factorials, C(n, k), finite sums
            - Numeric & symbolic inequalities (nlinarith / positivity / norm_num)
            - Algebraic, polynomial & trigonometric identities
  3. Faithfulness & Zero-Trust Lean 4 Kernel Gate:
       - Rejects any `sorry`/`admit`/`axiom`.
       - Verifies with `verify_async()` (Lean 4.36.0 + Mathlib4).
       - If the proposed claim is false or unprovable by Lean, REJECTS entirely
         (`status = "REJECTED"`) — never leaks unverified answers to students.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from backend.pipeline.solver import _get_llm, _extract_json

logger = logging.getLogger(__name__)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


@dataclass
class GeneralizedProposal:
    subkind: str                 # e.g. "generalized_identity", "generalized_inequality", ...
    candidate_answer: str        # Human-readable proposed answer
    lean_statement: str          # Standalone Lean 4 `example ... := by ...`
    back_translation: str        # Plain-English restatement for faithfulness check
    reasoning: str               # Conceptual reasoning summary
    is_math_problem: bool = True


_GENERALIZED_SYSTEM_PROMPT = """\
You are VeriMath Generalized Autoformalizer. The user's math problem is OUTSIDE the predefined template set.
Propose a concise candidate answer AND a standalone, valid Lean 4 theorem (using Mathlib tactics like norm_num, ring, omega, linarith, nlinarith, positivity, decide, or simp) that formally proves the answer.

STRICT RULES:
1. NEVER emit unproven placeholders or custom axioms; every theorem must be completely proved.
2. Output ONLY valid JSON with keys:
   {"is_math": true, "answer": "...", "lean_statement": "example ... := by ...", "back_translation": "...", "reasoning": "..."}
3. If the prompt is NOT a mathematical problem, output: {"is_math": false}.
"""


def _looks_like_math(problem: str) -> bool:
    p_l = problem.lower()
    if re.search(r"[0-9=<>+\-*/^∫∂√π∇!]|<=|>=|≠|≤|≥|∣", problem):
        return True
    if re.search(r"\bis\s+(?:\d|prime|divisible|equal|greater|less)", p_l):
        return True
    math_words = (
        "prove", "verify", "show that", "does ", "divide", "divisible",
        "mod", "modulo", "remainder", "gcd", "lcm", "factorial", "choose",
        "binomial", "sum of", "partial derivative", "gradient", "determinant", "matrix", "matrices", "permutation",
        "dot product", "cross product", "magnitude", "norm of", "system",
        "complex", "imaginary", "logarithm", "exponential",
        "inequality", "identity", "polynomial", "factor",
        "simplify", "expand", "evaluate", "compute", "calculate", "derivative",
        "integral", "solve", "prime", "square", "cube", "root",
    )
    return any(w in p_l for w in math_words)


def _to_lean_expr(expr_str: str) -> str:
    s = expr_str.strip().replace("**", "^")
    s = re.sub(r"\bsin\(", "Real.sin (", s)
    s = re.sub(r"\bcos\(", "Real.cos (", s)
    s = re.sub(r"\btan\(", "Real.tan (", s)
    s = re.sub(r"\bexp\(", "Real.exp (", s)
    s = re.sub(r"\blog\(", "Real.log (", s)
    s = re.sub(r"\bln\(", "Real.log (", s)
    s = re.sub(r"\bsqrt\(", "Real.sqrt (", s)
    s = re.sub(r"\bpi\b", "Real.pi", s)
    return s


def _safe_sympy(expr_str: str) -> sp.Expr:
    cleaned = expr_str.strip()
    cleaned = cleaned.replace("[", "(").replace("]", ")").replace("−", "-").replace("×", "*").replace("·", "*")
    cleaned = cleaned.replace("²", "^2").replace("³", "^3").replace("π", "pi")
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
        "e": sp.E,
        "pi": sp.pi,
        "ln": sp.log,
    }
    return parse_expr(cleaned, local_dict=local_dict, transformations=_TRANSFORMS)


def _fmt_rational(val: sp.Expr) -> str:
    if getattr(val, "is_Integer", False) or val == int(val):
        return str(int(val))
    return str(val)


def _propose_symbolic(problem: str) -> Optional[GeneralizedProposal]:
    p = problem.strip()
    p_l = p.lower()

    # 1. Multivariable Gradient (∇f)
    if "gradient" in p_l or "∇" in p:
        m_grad = re.search(r"(?:gradient\s+(?:of\s+)?|∇\s*)(?:f\([^)]+\)\s*=\s*)?(.+?)[?.!]*$", p, re.I)
        if m_grad:
            raw_expr = m_grad.group(1).strip()
            try:
                sym_expr = _safe_sympy(raw_expr)
                vars_list = sorted({str(s) for s in sym_expr.free_symbols}, key=lambda v: ("xyz".find(v) if v in "xyz" else 99, v))
                if not vars_list:
                    vars_list = ["x", "y"]
                partials = [sp.simplify(sp.diff(sym_expr, sp.Symbol(v))) for v in vars_list]
                binders = " ".join(vars_list)
                lean_expr = _to_lean_expr(str(sym_expr))
                conjuncts = []
                for v, d_v in zip(vars_list, partials):
                    conjuncts.append(f"deriv (fun {v} => {lean_expr}) {v} = {_to_lean_expr(str(d_v))}")
                simp_tactic = (
                    "simp (disch := fun_prop) [deriv_comp, deriv_mul, deriv_add, deriv_sub, deriv_div, "
                    "deriv_pow, Real.deriv_sin, Real.deriv_cos, Real.deriv_exp, Real.deriv_log]; try ring"
                )
                n_c = len(conjuncts)
                if n_c == 1:
                    lean_stmt = f"example ({binders} : ℝ) : {conjuncts[0]} := by\n  {simp_tactic}"
                else:
                    holes = ", ".join(["?_"] * n_c)
                    lean_stmt = f"example ({binders} : ℝ) : {' ∧ '.join(conjuncts)} := by\n  refine ⟨{holes}⟩ <;> ({simp_tactic})"
                grad_disp = "(" + ", ".join(str(d).replace("**", "^") for d in partials) + ")"
                return GeneralizedProposal(
                    subkind="generalized_gradient",
                    candidate_answer=grad_disp,
                    lean_statement=lean_stmt,
                    back_translation=f"Find the gradient of {raw_expr} with respect to ({', '.join(vars_list)}). The gradient vector is {grad_disp}.",
                    reasoning=f"Computing each partial derivative ({', '.join(f'∂/∂{v}' for v in vars_list)}) gives {grad_disp}.",
                )
            except Exception:
                pass

    # 2. Partial Derivatives: ∂/∂x or "partial derivative of ... with respect to x/y/z"
    if "partial" in p_l or "∂" in p:
        var_m = re.search(r"(?:with\s+respect\s+to|w\.?r\.?t\.?|∂\s*/\s*∂)\s*([xyz])", p_l)
        wrt = var_m.group(1) if var_m else "x"
        expr_m = re.search(
            r"(?:partial\s+derivative\s+(?:of\s+)?|∂/∂[xyz]\s*(?:of\s+)?)(.+?)(?:\s+(?:with\s+respect\s+to|w\.?r\.?t\.?)\s+[xyz].*)?$",
            p,
            re.I,
        )
        if expr_m:
            raw_expr = re.sub(r"^[fgh]\([^)]+\)\s*=\s*", "", expr_m.group(1).strip()).rstrip("?.!")
            try:
                sym_expr = _safe_sympy(raw_expr)
                v = sp.Symbol(wrt)
                ans_expr = sp.simplify(sp.diff(sym_expr, v))
                free_vars = sorted({str(s) for s in (sym_expr.free_symbols | ans_expr.free_symbols | {v})})
                binders = " ".join(free_vars)
                lean_expr = _to_lean_expr(str(sym_expr))
                lean_ans = _to_lean_expr(str(ans_expr))
                ans_disp = str(ans_expr).replace("**", "^").replace("log(", "ln(")
                lean_stmt = (
                    f"example ({binders} : ℝ) : deriv (fun {wrt} => {lean_expr}) {wrt} = {lean_ans} := by\n"
                    f"  simp (disch := fun_prop) [deriv_comp, deriv_mul, deriv_add, deriv_sub, deriv_div, deriv_pow, Real.deriv_sin, Real.deriv_cos, Real.deriv_exp, Real.deriv_log]\n"
                    f"  try ring"
                )
                return GeneralizedProposal(
                    subkind="generalized_partial_derivative",
                    candidate_answer=ans_disp,
                    lean_statement=lean_stmt,
                    back_translation=f"Find the partial derivative of {raw_expr} with respect to {wrt}. The result is {ans_disp}.",
                    reasoning=f"Holding all variables other than {wrt} constant and differentiating {raw_expr} with respect to {wrt} yields {ans_disp}.",
                )
            except Exception:
                pass

    # 3. Linear Systems (2x2 or 3x3): "Solve the system: 2x + 3y = 13, x - y = -1"
    if "system" in p_l or (p.count("=") >= 2 and any(sep in p for sep in (",", ";", " and "))):
        body = re.sub(r"^.*?(?:system(?:\s+of\s+equations)?\s*:?)\s*", "", p, flags=re.I)
        raw_eqs = [eq.strip().rstrip("?.!") for eq in re.split(r"[;,]|(?:\band\b)", body) if "=" in eq]
        if 2 <= len(raw_eqs) <= 3:
            try:
                sym_eqs = []
                all_syms = set()
                parsed_halves = []
                for eq in raw_eqs:
                    lhs_s, rhs_s = eq.split("=", 1)
                    l_e, r_e = _safe_sympy(lhs_s), _safe_sympy(rhs_s)
                    sym_eqs.append(sp.Eq(l_e, r_e))
                    all_syms |= l_e.free_symbols | r_e.free_symbols
                    parsed_halves.append((l_e, r_e))
                vars_sorted = sorted(all_syms, key=lambda s: ("xyz".find(str(s)) if str(s) in "xyz" else 99, str(s)))
                if len(vars_sorted) == len(raw_eqs):
                    sol = sp.solve(sym_eqs, vars_sorted, dict=True)
                    if sol and len(sol) == 1:
                        sol_map = sol[0]
                        ans_parts = [f"{v} = {_fmt_rational(sol_map[v])}" for v in vars_sorted]
                        ans_disp = ", ".join(ans_parts)
                        # Build Lean conjunction substituting verified roots back into every equation
                        checks = []
                        for idx_eq, (l_e, r_e) in enumerate(parsed_halves):
                            l_sub, r_sub = l_e, r_e
                            for v in vars_sorted:
                                val_s = f"({_fmt_rational(sol_map[v])})"
                                l_sub = l_sub.subs(v, sp.Symbol(val_s))
                                r_sub = r_sub.subs(v, sp.Symbol(val_s))
                            l_str = _to_lean_expr(str(l_e))
                            r_str = _to_lean_expr(str(r_e))
                            for v in vars_sorted:
                                val_lean = f"({_fmt_rational(sol_map[v])} : ℝ)"
                                l_str = re.sub(rf"\b{re.escape(str(v))}\b", val_lean, l_str)
                                r_str = re.sub(rf"\b{re.escape(str(v))}\b", val_lean, r_str)
                            checks.append(f"({l_str} : ℝ) = ({r_str} : ℝ)")
                        lean_stmt = f"example : {' ∧ '.join(checks)} := by norm_num"
                        return GeneralizedProposal(
                            subkind="generalized_linear_system",
                            candidate_answer=ans_disp,
                            lean_statement=lean_stmt,
                            back_translation=f"Solve the linear system {'; '.join(raw_eqs)}. The solution is {ans_disp}.",
                            reasoning=f"Eliminating variables across the {len(raw_eqs)} linear equations gives {ans_disp}.",
                        )
            except Exception:
                pass

    # 4. 3x3 Matrix Determinant: [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]
    num_pat = r"(-?\d+(?:\.\d+)?)"
    row3_pat = rf"\[\s*{num_pat}\s*,\s*{num_pat}\s*,\s*{num_pat}\s*\]"
    m_det3 = re.search(rf"determinant.*?\[\s*{row3_pat}\s*,\s*{row3_pat}\s*,\s*{row3_pat}\s*\]", p_l)
    if m_det3:
        vals = [sp.Rational(x) for x in m_det3.groups()]
        a1, a2, a3, b1, b2, b3, c1, c2, c3 = vals
        det_val = a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)
        det_s = _fmt_rational(det_val)
        lean_stmt = (
            f"example : ({a1} : ℝ) * (({b2}) * ({c3}) - ({b3}) * ({c2})) "
            f"- ({a2}) * (({b1}) * ({c3}) - ({b3}) * ({c1})) "
            f"+ ({a3}) * (({b1}) * ({c2}) - ({b2}) * ({c1})) = {det_s} := by norm_num"
        )
        return GeneralizedProposal(
            subkind="generalized_linear_algebra",
            candidate_answer=det_s,
            lean_statement=lean_stmt,
            back_translation=f"Find the determinant of the 3x3 matrix [[{a1}, {a2}, {a3}], [{b1}, {b2}, {b3}], [{c1}, {c2}, {c3}]]. The determinant is {det_s}.",
            reasoning=f"Expanding along the first row via cofactors yields determinant = {det_s}.",
        )

    # 5. 2x2 Matrix Determinant: [[a, b], [c, d]]
    m_det = re.search(
        r"determinant.*?\[\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*,\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*\]",
        p_l,
    )
    if m_det:
        a, b, c, d = m_det.groups()
        det_val = sp.Rational(a) * sp.Rational(d) - sp.Rational(b) * sp.Rational(c)
        det_s = _fmt_rational(det_val)
        lean_stmt = f"example : ({a} : ℝ) * ({d}) - ({b}) * ({c}) = {det_s} := by norm_num"
        return GeneralizedProposal(
            subkind="generalized_linear_algebra",
            candidate_answer=det_s,
            lean_statement=lean_stmt,
            back_translation=f"Find the determinant of the 2x2 matrix [[{a}, {b}], [{c}, {d}]]. The determinant is {det_s}.",
            reasoning=f"For a 2x2 matrix [[a, b], [c, d]], det = a*d - b*c = ({a})*({d}) - ({b})*({c}) = {det_s}.",
        )

    # 6. 3D Vector Cross Product: cross product of (a1, a2, a3) and (b1, b2, b3)
    m_cross = re.search(
        r"cross\s+product.*?\(\s*([^\)]+)\s*\)\s*(?:and|with|×|x)\s*\(\s*([^\)]+)\s*\)",
        p_l,
    )
    if m_cross:
        v1 = [x.strip() for x in m_cross.group(1).split(",")]
        v2 = [x.strip() for x in m_cross.group(2).split(",")]
        if len(v1) == 3 and len(v2) == 3:
            try:
                u1, u2, u3 = (sp.Rational(x) for x in v1)
                w1, w2, w3 = (sp.Rational(x) for x in v2)
                c1 = u2 * w3 - u3 * w2
                c2 = u3 * w1 - u1 * w3
                c3 = u1 * w2 - u2 * w1
                c1s, c2s, c3s = _fmt_rational(c1), _fmt_rational(c2), _fmt_rational(c3)
                ans_disp = f"({c1s}, {c2s}, {c3s})"
                lean_stmt = (
                    f"example : (({u2} : ℝ) * ({w3}) - ({u3}) * ({w2}) = ({c1s})) ∧ "
                    f"(({u3} : ℝ) * ({w1}) - ({u1}) * ({w3}) = ({c2s})) ∧ "
                    f"(({u1} : ℝ) * ({w2}) - ({u2}) * ({w1}) = ({c3s})) := by norm_num"
                )
                return GeneralizedProposal(
                    subkind="generalized_linear_algebra",
                    candidate_answer=ans_disp,
                    lean_statement=lean_stmt,
                    back_translation=f"Find the cross product of ({u1}, {u2}, {u3}) and ({w1}, {w2}, {w3}). The result is {ans_disp}.",
                    reasoning=f"Computing u × w = (u₂w₃ − u₃w₂, u₃w₁ − u₁w₃, u₁w₂ − u₂w₁) gives {ans_disp}.",
                )
            except Exception:
                pass

    # 7. Vector Dot Product: dot product of (a, b, ...) and (c, d, ...)
    m_dot = re.search(
        r"dot\s+product.*?\(\s*([^\)]+)\s*\)\s*(?:and|with|·|\*)\s*\(\s*([^\)]+)\s*\)",
        p_l,
    )
    if m_dot:
        v1 = [x.strip() for x in m_dot.group(1).split(",")]
        v2 = [x.strip() for x in m_dot.group(2).split(",")]
        if len(v1) == len(v2) and v1:
            try:
                terms = [f"({a} : ℝ) * ({b})" for a, b in zip(v1, v2)]
                total = sum(sp.Rational(a) * sp.Rational(b) for a, b in zip(v1, v2))
                tot_s = _fmt_rational(total)
                lean_stmt = f"example : {' + '.join(terms)} = {tot_s} := by norm_num"
                return GeneralizedProposal(
                    subkind="generalized_linear_algebra",
                    candidate_answer=tot_s,
                    lean_statement=lean_stmt,
                    back_translation=f"Find the dot product of ({', '.join(v1)}) and ({', '.join(v2)}). The answer is {tot_s}.",
                    reasoning=f"Multiply matching components and sum: {' + '.join(terms)} = {tot_s}.",
                )
            except Exception:
                pass

    # 8. Vector Magnitude / Euclidean Norm: norm/magnitude of vector (3, 4, 12)
    m_norm = re.search(r"(?:magnitude|norm|length)\s+(?:of\s+(?:the\s+)?(?:vector\s+)?)?\(\s*([^\)]+)\s*\)", p_l)
    if m_norm:
        comps = [x.strip() for x in m_norm.group(1).split(",")]
        if comps:
            try:
                sq_sum = sum(sp.Rational(c) ** 2 for c in comps)
                norm_val = sp.sqrt(sq_sum)
                ans_disp = str(norm_val).replace("**", "^")
                sq_terms = " + ".join(f"({c} : ℝ)^2" for c in comps)
                if getattr(norm_val, "is_Rational", False):
                    nv_s = _fmt_rational(norm_val)
                    lean_stmt = f"example : ({sq_terms}) = ({nv_s} : ℝ)^2 := by norm_num"
                else:
                    lean_stmt = f"example : ({sq_terms}) = ({_fmt_rational(sq_sum)} : ℝ) := by norm_num"
                return GeneralizedProposal(
                    subkind="generalized_linear_algebra",
                    candidate_answer=ans_disp,
                    lean_statement=lean_stmt,
                    back_translation=f"Find the magnitude of vector ({', '.join(comps)}). The norm is {ans_disp} (squared norm {_fmt_rational(sq_sum)}).",
                    reasoning=f"Summing squared components gives {_fmt_rational(sq_sum)}, so the Euclidean norm is {ans_disp}.",
                )
            except Exception:
                pass

    # 9. Divisibility: "Does a divide b?" / "Is b divisible by a?"
    m_div1 = re.search(r"does\s+(\d+)\s+divide\s+(\d+)", p_l)
    m_div2 = re.search(r"is\s+(\d+)\s+divisible\s+by\s+(\d+)", p_l)
    if m_div1 or m_div2:
        a, b = (m_div1.group(1), m_div1.group(2)) if m_div1 else (m_div2.group(2), m_div2.group(1))
        a_i, b_i = int(a), int(b)
        is_true = (a_i != 0 and b_i % a_i == 0)
        if "prove" in p_l or "show" in p_l or "verify" in p_l or is_true:
            lean_stmt = f"example : ({a_i} : ℕ) ∣ {b_i} := by decide"
            return GeneralizedProposal(
                subkind="generalized_number_theory",
                candidate_answer=f"Yes, {a_i} divides {b_i}" if is_true else f"{a_i} divides {b_i}",
                lean_statement=lean_stmt,
                back_translation=f"Does {a_i} divide {b_i}? Claim: {a_i} divides {b_i}.",
                reasoning=f"Check divisibility {b_i} mod {a_i} = {b_i % a_i if a_i else 'undefined'}.",
            )
        else:
            lean_stmt = f"example : ¬ (({a_i} : ℕ) ∣ {b_i}) := by decide"
            return GeneralizedProposal(
                subkind="generalized_number_theory",
                candidate_answer=f"No, {a_i} does not divide {b_i}",
                lean_statement=lean_stmt,
                back_translation=f"Does {a_i} divide {b_i}? No, {a_i} does not divide {b_i}.",
                reasoning=f"{b_i} divided by {a_i} leaves remainder {b_i % a_i} ≠ 0, so {a_i} does not divide {b_i}.",
            )

    def _extract_claimed_nat(prompt_txt: str) -> Optional[int]:
        m_cl = re.search(r"=\s*(\d+)\s*[?.!]*$", prompt_txt)
        return int(m_cl.group(1)) if m_cl else None

    # 10. Modular arithmetic / GCD / LCM / Factorial / Binomial / Finite Sum
    m_mod = re.search(r"(\d+)\s*(?:mod|modulo|%)\s*(\d+)", p_l)
    if m_mod:
        a_i, m_i = int(m_mod.group(1)), int(m_mod.group(2))
        if m_i > 0:
            claimed = _extract_claimed_nat(p_l)
            target = claimed if claimed is not None else (a_i % m_i)
            return GeneralizedProposal(
                subkind="generalized_number_theory",
                candidate_answer=str(target),
                lean_statement=f"example : ({a_i} : ℕ) % {m_i} = {target} := by norm_num",
                back_translation=f"Compute {a_i} mod {m_i}. Claimed remainder is {target}.",
                reasoning=f"Dividing {a_i} by {m_i} leaves remainder {a_i % m_i}.",
            )

    m_gcd = re.search(r"\bgcd\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", p_l)
    if m_gcd:
        a_i, b_i = int(m_gcd.group(1)), int(m_gcd.group(2))
        g = math.gcd(a_i, b_i)
        claimed = _extract_claimed_nat(p_l)
        target = claimed if claimed is not None else g
        return GeneralizedProposal(
            subkind="generalized_number_theory",
            candidate_answer=str(target),
            lean_statement=f"example : Nat.gcd {a_i} {b_i} = {target} := by decide",
            back_translation=f"Find gcd({a_i}, {b_i}). Claimed answer is {target}.",
            reasoning=f"Using Euclidean division, the greatest common divisor of {a_i} and {b_i} is {g}.",
        )

    m_lcm = re.search(r"\blcm\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", p_l)
    if m_lcm:
        a_i, b_i = int(m_lcm.group(1)), int(m_lcm.group(2))
        l_val = math.lcm(a_i, b_i)
        claimed = _extract_claimed_nat(p_l)
        target = claimed if claimed is not None else l_val
        return GeneralizedProposal(
            subkind="generalized_number_theory",
            candidate_answer=str(target),
            lean_statement=f"example : Nat.lcm {a_i} {b_i} = {target} := by decide",
            back_translation=f"Find lcm({a_i}, {b_i}). Claimed answer is {target}.",
            reasoning=f"Using lcm(a, b) = (a * b) / gcd(a, b), lcm({a_i}, {b_i}) = {l_val}.",
        )

    m_fact = re.search(r"(?:\bfactorial\s+(?:of\s+)?(\d+)|(\d+)\s*!)", p_l)
    if m_fact:
        n_i = int(m_fact.group(1) or m_fact.group(2))
        if 0 <= n_i <= 12:
            val = math.factorial(n_i)
            claimed = _extract_claimed_nat(p_l)
            target = claimed if claimed is not None else val
            return GeneralizedProposal(
                subkind="generalized_series_combinatorics",
                candidate_answer=str(target),
                lean_statement=f"example : Nat.factorial {n_i} = {target} := by decide",
                back_translation=f"Compute {n_i}!. Claimed factorial is {target}.",
                reasoning=f"Multiplying integers from 1 up to {n_i} gives {val}.",
            )

    m_choose = re.search(r"(?:choose|C)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", p, re.I)
    if m_choose:
        n_i, k_i = int(m_choose.group(1)), int(m_choose.group(2))
        if n_i <= 20 and k_i <= n_i:
            val = math.comb(n_i, k_i)
            claimed = _extract_claimed_nat(p_l)
            target = claimed if claimed is not None else val
            return GeneralizedProposal(
                subkind="generalized_series_combinatorics",
                candidate_answer=str(target),
                lean_statement=f"example : Nat.choose {n_i} {k_i} = {target} := by decide",
                back_translation=f"Compute choose({n_i}, {k_i}). Claimed value is {target}.",
                reasoning=f"C({n_i}, {k_i}) = {n_i}! / ({k_i}! * ({n_i}-{k_i})!) = {val}.",
            )

    m_perm = re.search(r"(?:\bpermutations?\s*|\bP)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", p)
    if m_perm:
        n_i, k_i = int(m_perm.group(1)), int(m_perm.group(2))
        if n_i <= 20 and k_i <= n_i:
            val = math.perm(n_i, k_i)
            claimed = _extract_claimed_nat(p_l)
            target = claimed if claimed is not None else val
            return GeneralizedProposal(
                subkind="generalized_series_combinatorics",
                candidate_answer=str(target),
                lean_statement=f"example : Nat.descFactorial {n_i} {k_i} = {target} := by decide",
                back_translation=f"Find permutations P({n_i}, {k_i}). Claimed value is {target}.",
                reasoning=f"P({n_i}, {k_i}) = {n_i}! / ({n_i} - {k_i})! = {val}.",
            )

    m_sum_sq = re.search(r"sum\s+of\s+(?:the\s+)?squares\s+of\s+(?:the\s+)?first\s+(\d+)\s+positive\s+integers", p_l)
    if m_sum_sq:
        n_i = int(m_sum_sq.group(1))
        if 1 <= n_i <= 100:
            total = n_i * (n_i + 1) * (2 * n_i + 1) // 6
            return GeneralizedProposal(
                subkind="generalized_series_combinatorics",
                candidate_answer=str(total),
                lean_statement=f"example : (∑ k ∈ Finset.Icc 1 {n_i}, k ^ 2) = {total} := by decide",
                back_translation=f"Find the sum of squares of the first {n_i} positive integers. The sum is {total}.",
                reasoning=f"Using n(n+1)(2n+1)/6 for n = {n_i} gives {total}.",
            )

    m_sum = re.search(r"sum\s+of\s+(?:the\s+)?first\s+(\d+)\s+positive\s+integers", p_l)
    if m_sum:
        n_i = int(m_sum.group(1))
        if 1 <= n_i <= 200:
            total = n_i * (n_i + 1) // 2
            return GeneralizedProposal(
                subkind="generalized_series_combinatorics",
                candidate_answer=str(total),
                lean_statement=f"example : (∑ k ∈ Finset.Icc 1 {n_i}, k) = {total} := by decide",
                back_translation=f"Find the sum of the first {n_i} positive integers. The sum is {total}.",
                reasoning=f"Summing 1 + 2 + ... + {n_i} using Gauss pairing gives {n_i}*({n_i}+1)/2 = {total}.",
            )

    # 10b. Complex Number Arithmetic: (a + bi) * (c + di), (a + bi) + (c + di), or identity claims
    if (
        "complex" in p_l or re.search(r"(?:\d\s*\*?\s*i\b|[+-]\s*i\b|\(\s*i\b)", p_l)
    ) and not any(w in p_l for w in ("derivative", "integral", "matrix", "determinant")):
        def _to_lean_c(expr_txt: str) -> str:
            s = re.sub(r"\)\s*\(", ") * (", expr_txt.strip())
            s = re.sub(r"(\d)\s*\(", r"\1 * (", s)
            s = re.sub(r"(\d)\s*\*?\s*i\b", r"\1 * Complex.I", s)
            s = re.sub(r"(?<![a-zA-Z_])i(?![a-zA-Z_])", "Complex.I", s)
            return s

        def _parse_c(expr_txt: str) -> sp.Expr:
            s = re.sub(r"(\d)\s*i\b", r"\1*I", expr_txt.strip())
            s = re.sub(r"(?<![a-zA-Z_])i(?![a-zA-Z_])", "I", s)
            s = s.replace("−", "-").replace("×", "*").replace("·", "*")
            return parse_expr(s, local_dict={"I": sp.I}, transformations=_TRANSFORMS)

        def _fmt_complex(re_p: sp.Expr, im_p: sp.Expr) -> tuple[str, str]:
            r_s = _fmt_rational(re_p)
            if im_p == 0:
                return f"({r_s} : ℂ)", r_s
            im_abs = abs(im_p)
            sign_op = "+" if im_p > 0 else "-"
            im_lean = "Complex.I" if im_abs == 1 else f"{_fmt_rational(im_abs)} * Complex.I"
            im_disp = "i" if im_abs == 1 else f"{_fmt_rational(im_abs)}i"
            if re_p == 0:
                lean_v = im_lean if im_p > 0 else f"-{im_lean}"
                disp_v = im_disp if im_p > 0 else f"-{im_disp}"
            else:
                lean_v = f"({r_s} : ℂ) {sign_op} {im_lean}"
                disp_v = f"{r_s} {sign_op} {im_disp}"
            return lean_v, disp_v

        body_c = re.sub(
            r"^\s*(?:prove\s+that|show\s+that|verify\s+that|check\s+(?:whether|if)|is\s+it\s+true\s+that|compute|simplify|evaluate|calculate|multiply|add|subtract|find(?:\s+the\s+value\s+of)?)\s+",
            "",
            p,
            flags=re.I,
        ).rstrip("?.! ").strip()
        try:
            if "=" in body_c:
                c_lhs_raw, c_rhs_raw = [x.strip() for x in body_c.split("=", 1)]
                _ = _parse_c(c_lhs_raw)
                _ = _parse_c(c_rhs_raw)
                lean_lhs_c = _to_lean_c(c_lhs_raw)
                lean_rhs_c = _to_lean_c(c_rhs_raw)
                lean_stmt = (
                    f"example : ({lean_lhs_c} : ℂ) = ({lean_rhs_c} : ℂ) := by\n"
                    f"  apply Complex.ext <;> simp <;> norm_num"
                )
                return GeneralizedProposal(
                    subkind="generalized_complex",
                    candidate_answer=f"{c_lhs_raw} = {c_rhs_raw}",
                    lean_statement=lean_stmt,
                    back_translation=f"Verify the complex identity {c_lhs_raw} = {c_rhs_raw}.",
                    reasoning=f"Expanding complex terms with i^2 = -1 and comparing real and imaginary components.",
                )
            else:
                c_expr = _parse_c(body_c)
                c_exp = sp.expand(c_expr)
                re_p, im_p = c_exp.as_real_imag()
                if getattr(re_p, "is_Rational", False) and getattr(im_p, "is_Rational", False):
                    lean_rhs_c, disp_c = _fmt_complex(re_p, im_p)
                    lean_lhs_c = _to_lean_c(body_c)
                    lean_stmt = (
                        f"example : ({lean_lhs_c} : ℂ) = ({lean_rhs_c}) := by\n"
                        f"  apply Complex.ext <;> simp <;> norm_num"
                    )
                    return GeneralizedProposal(
                        subkind="generalized_complex",
                        candidate_answer=disp_c,
                        lean_statement=lean_stmt,
                        back_translation=f"Evaluate the complex expression {body_c}. Result: {disp_c}.",
                        reasoning=f"Distributing terms and substituting i^2 = -1 groups real part {_fmt_rational(re_p)} and imaginary part {_fmt_rational(im_p)} into {disp_c}.",
                    )
        except Exception:
            pass

    # 10c. Logarithmic Product / Quotient Identities (e.g. log(a*b) = log(a) + log(b))
    if ("=" in p) and ("log(" in p_l or "ln(" in p_l):
        m_log_id = re.search(
            r"(?:prove\s+that|show\s+that|verify\s+that|verify|check\s+(?:whether|if)|is\s+it\s+true\s+that)?\s*(.+?)\s*=\s*(.+?)[?.!]*$",
            p_l,
        )
        if m_log_id:
            lhs_raw = re.sub(r"\s+for\s+(?:all\s+)?.*$", "", m_log_id.group(1).strip())
            rhs_raw = re.sub(r"\s+for\s+(?:all\s+)?.*$", "", m_log_id.group(2).strip())
            try:
                l_sym = _safe_sympy(lhs_raw)
                r_sym = _safe_sympy(rhs_raw)
                free_vars = sorted({str(s) for s in (l_sym.free_symbols | r_sym.free_symbols)})
                if free_vars:
                    binders = " ".join(free_vars)
                    pos_hyps = " ".join(f"(h{v} : 0 < {v})" for v in free_vars)
                    lean_lhs = _to_lean_expr(str(l_sym))
                    lean_rhs = _to_lean_expr(str(r_sym))
                    ne_args = " ".join(f"h{v}.ne'" for v in free_vars[:2])
                    if "/" in lhs_raw or "/" in rhs_raw:
                        tactic = f"exact Real.log_div {ne_args}"
                    else:
                        tactic = f"simp [Real.log_mul {ne_args}, Real.log_div {ne_args}]"
                    lean_stmt = f"example ({binders} : ℝ) {pos_hyps} : {lean_lhs} = {lean_rhs} := by\n  {tactic}"
                    return GeneralizedProposal(
                        subkind="generalized_log_identity",
                        candidate_answer=f"{lhs_raw} = {rhs_raw}",
                        lean_statement=lean_stmt,
                        back_translation=f"Verify for positive variables ({', '.join(free_vars)}) whether {lhs_raw} = {rhs_raw}.",
                        reasoning=f"Testing the logarithmic identity for positive arguments ({', '.join(free_vars)}).",
                    )
            except Exception:
                pass

    # 10d. Exponential Equations: e.g. "Solve 2^x = 32", "Solve 2^x = 1/8", "Solve 4^x = 8"
    m_exp_eq = re.search(r"\b(\d+)\s*\^\s*\(?([a-zA-Z])\)?\s*=\s*(\d+(?:\s*/\s*\d+)?)\b", p)
    if m_exp_eq and not any(op in p for op in ("<", ">", "≤", "≥")):
        base_i = int(m_exp_eq.group(1))
        var_name = m_exp_eq.group(2)
        rhs_str = m_exp_eq.group(3).replace(" ", "")
        rhs_rat = sp.Rational(rhs_str)
        if base_i > 1 and rhs_rat > 0:
            exp_exact = sp.simplify(sp.log(rhs_rat, base_i))
            rhs_lean = rhs_str if "/" not in rhs_str else f"({rhs_str} : ℝ)"
            if getattr(exp_exact, "is_Integer", False):
                k_val = int(exp_exact)
                k_lean = str(k_val) if k_val >= 0 else f"({k_val} : ℝ)"
                lean_stmt = (
                    f"example ({var_name} : ℝ) : (({base_i} : ℝ) ^ {var_name} = {rhs_lean}) ↔ ({var_name} = {k_lean}) := by\n"
                    f"  constructor\n"
                    f"  · intro h\n"
                    f"    have h1 : ({base_i} : ℝ) ^ {var_name} = ({base_i} : ℝ) ^ ({k_val} : ℝ) := by norm_num; exact h\n"
                    f"    exact (Real.rpow_right_inj (by norm_num) (by norm_num)).mp h1\n"
                    f"  · intro h; subst h; norm_num"
                )
                return GeneralizedProposal(
                    subkind="generalized_exponential_equation",
                    candidate_answer=f"{var_name} = {k_val}",
                    lean_statement=lean_stmt,
                    back_translation=f"Solve {base_i}^{var_name} = {rhs_str} for real {var_name}. The unique solution is {var_name} = {k_val}.",
                    reasoning=f"Rewriting {rhs_str} as {base_i}^{k_val} matches bases on both sides, yielding {var_name} = {k_val}.",
                )
            elif getattr(exp_exact, "is_Rational", False):
                # Common base cb^p = base_i, cb^q = rhs_rat
                cb = None
                p_exp = q_exp = 0
                for cand in range(2, base_i + 1):
                    lp = sp.simplify(sp.log(base_i, cand))
                    lq = sp.simplify(sp.log(rhs_rat, cand))
                    if getattr(lp, "is_Integer", False) and getattr(lq, "is_Integer", False):
                        cb, p_exp, q_exp = cand, int(lp), int(lq)
                        break
                if cb is not None and p_exp > 0:
                    sol_str = f"{q_exp}/{p_exp}" if sp.Rational(q_exp, p_exp).q != 1 else str(sp.Rational(q_exp, p_exp))
                    sol_lean = f"({sol_str} : ℝ)"
                    lean_stmt = (
                        f"example ({var_name} : ℝ) : (({base_i} : ℝ) ^ {var_name} = {rhs_lean}) ↔ ({var_name} = {sol_lean}) := by\n"
                        f"  constructor\n"
                        f"  · intro h\n"
                        f"    have h1 : ({cb} : ℝ) ^ (({p_exp} : ℝ) * {var_name}) = ({cb} : ℝ) ^ ({q_exp} : ℝ) := by\n"
                        f"      calc ({cb} : ℝ) ^ (({p_exp} : ℝ) * {var_name})\n"
                        f"          = (({cb} : ℝ) ^ ({p_exp} : ℝ)) ^ {var_name} := by rw [Real.rpow_mul (by norm_num)]\n"
                        f"        _ = ({base_i} : ℝ) ^ {var_name} := by norm_num\n"
                        f"        _ = {rhs_lean} := h\n"
                        f"        _ = ({cb} : ℝ) ^ ({q_exp} : ℝ) := by norm_num\n"
                        f"    have h2 : ({p_exp} : ℝ) * {var_name} = ({q_exp} : ℝ) := (Real.rpow_right_inj (by norm_num) (by norm_num)).mp h1\n"
                        f"    linarith\n"
                        f"  · intro h; subst h; norm_num"
                    )
                    return GeneralizedProposal(
                        subkind="generalized_exponential_equation",
                        candidate_answer=f"{var_name} = {sol_str}",
                        lean_statement=lean_stmt,
                        back_translation=f"Solve {base_i}^{var_name} = {rhs_str} for real {var_name}. The unique solution is {var_name} = {sol_str}.",
                        reasoning=f"Expressing both {base_i} = {cb}^{p_exp} and {rhs_str} = {cb}^{q_exp} equates {p_exp}*{var_name} = {q_exp}, giving {var_name} = {sol_str}.",
                    )

    # 10e. 2x2 Matrix Multiplication: "Multiply matrices [[a, b], [c, d]] and [[e, f], [g, h]]"
    if "matrix" in p_l or "matrices" in p_l:
        m_mm = re.search(
            r"\[\s*\[\s*([^\],]+)\s*,\s*([^\]]+)\s*\]\s*,\s*\[\s*([^\],]+)\s*,\s*([^\]]+)\s*\]\s*\]"
            r".*?"
            r"\[\s*\[\s*([^\],]+)\s*,\s*([^\]]+)\s*\]\s*,\s*\[\s*([^\],]+)\s*,\s*([^\]]+)\s*\]\s*\]",
            p,
        )
        if m_mm and ("multiply" in p_l or "product" in p_l or "*" in p or "times" in p_l):
            try:
                vals = [sp.Rational(x.strip()) for x in m_mm.groups()]
                a, b, c, d, e, f, g, h = vals
                c11, c12 = a * e + b * g, a * f + b * h
                c21, c22 = c * e + d * g, c * f + d * h
                fa, fb, fc, fd, fe, ff, fg, fh = [_fmt_rational(v) for v in vals]
                r11, r12, r21, r22 = [_fmt_rational(v) for v in (c11, c12, c21, c22)]
                lean_stmt = (
                    f"example : (({fa} : ℝ)*({fe}) + ({fb})*({fg}) = ({r11})) ∧ "
                    f"(({fa} : ℝ)*({ff}) + ({fb})*({fh}) = ({r12})) ∧ "
                    f"(({fc} : ℝ)*({fe}) + ({fd})*({fg}) = ({r21})) ∧ "
                    f"(({fc} : ℝ)*({ff}) + ({fd})*({fh}) = ({r22})) := by norm_num"
                )
                ans_mat = f"[[{r11}, {r12}], [{r21}, {r22}]]"
                return GeneralizedProposal(
                    subkind="generalized_linear_algebra",
                    candidate_answer=ans_mat,
                    lean_statement=lean_stmt,
                    back_translation=f"Multiply 2x2 matrices [[{fa}, {fb}], [{fc}, {fd}]] and [[{fe}, {ff}], [{fg}, {fh}]]. Result: {ans_mat}.",
                    reasoning=f"Computing row-by-column dot products gives entries [{r11}, {r12}] and [{r21}, {r22}].",
                )
            except Exception:
                pass

    # 10f. Primality Claims: "Prove that N is prime" (accepts primes, strictly rejects composites!)
    m_prime_claim = re.search(r"(?:prove|show|verify)\s+(?:that\s+)?(\d+)\s+is\s+(?:a\s+)?prime\b", p_l)
    if m_prime_claim:
        n_i = int(m_prime_claim.group(1))
        if 2 <= n_i <= 10000:
            return GeneralizedProposal(
                subkind="generalized_number_theory",
                candidate_answer=f"{n_i} is prime",
                lean_statement=f"example : Nat.Prime {n_i} := by decide",
                back_translation=f"Verify whether {n_i} is a prime number.",
                reasoning=f"Testing whether {n_i} has only divisors 1 and {n_i}.",
            )

    # 11. Numeric & Symbolic Inequalities (both true and false claims!)
    m_ineq = re.search(
        r"(?:prove\s+that|show\s+that|verify\s+that|is|check\s+(?:whether|if))?\s*(.+?)\s*(<=|>=|≤|≥|<|>)\s*(.+?)[?.!]*$",
        p_l,
    )
    if m_ineq and "=" not in m_ineq.group(1) and "=" not in m_ineq.group(3):
        lhs_s, op_s, rhs_s = m_ineq.group(1).strip(), m_ineq.group(2), m_ineq.group(3).strip()
        op_map = {"<=": "≤", ">=": "≥", "≤": "≤", "≥": "≥", "<": "<", ">": ">"}
        lean_op = op_map[op_s]
        has_vars = bool(re.search(r"[a-zA-Z]", lhs_s + rhs_s))
        try:
            l_sym = _safe_sympy(lhs_s)
            r_sym = _safe_sympy(rhs_s)
            if not has_vars:
                is_true = bool(
                    (lean_op == "<" and l_sym < r_sym)
                    or (lean_op == ">" and l_sym > r_sym)
                    or (lean_op == "≤" and l_sym <= r_sym)
                    or (lean_op == "≥" and l_sym >= r_sym)
                )
                if "prove" in p_l or "show" in p_l or "verify" in p_l or is_true:
                    lean_stmt = f"example : ({lhs_s} : ℝ) {lean_op} ({rhs_s} : ℝ) := by norm_num"
                    ans_txt = f"True ({lhs_s} {lean_op} {rhs_s})" if is_true else f"{lhs_s} {lean_op} {rhs_s}"
                else:
                    lean_stmt = f"example : ¬ (({lhs_s} : ℝ) {lean_op} ({rhs_s} : ℝ)) := by norm_num"
                    ans_txt = f"False (¬ ({lhs_s} {lean_op} {rhs_s}))"
                return GeneralizedProposal(
                    subkind="generalized_inequality",
                    candidate_answer=ans_txt,
                    lean_statement=lean_stmt,
                    back_translation=f"Compare {lhs_s} {op_s} {rhs_s}. Result: {ans_txt}.",
                    reasoning=f"Evaluating both sides numerically: {lhs_s} = {l_sym} and {rhs_s} = {r_sym}.",
                )
            else:
                free_vars = sorted({str(s) for s in (l_sym.free_symbols | r_sym.free_symbols)})
                binders = " ".join(free_vars)
                lean_lhs = _to_lean_expr(str(l_sym))
                lean_rhs = _to_lean_expr(str(r_sym))
                # Try nlinarith with square-nonneg hints (e.g. sq_nonneg (x - y)) + positivity fallback
                hint = ""
                if len(free_vars) == 2:
                    v0, v1 = free_vars[0], free_vars[1]
                    hint = f" [sq_nonneg ({v0} - {v1}), sq_nonneg ({v0} + {v1})]"
                elif len(free_vars) == 1:
                    hint = f" [sq_nonneg ({free_vars[0]})]"
                lean_stmt = (
                    f"example ({binders} : ℝ) : {lean_lhs} {lean_op} {lean_rhs} := by\n"
                    f"  first | positivity | nlinarith{hint}"
                )
                return GeneralizedProposal(
                    subkind="generalized_symbolic_inequality",
                    candidate_answer=f"{lhs_s} {lean_op} {rhs_s}",
                    lean_statement=lean_stmt,
                    back_translation=f"Prove the symbolic inequality {lhs_s} {op_s} {rhs_s}.",
                    reasoning=f"Testing whether {lhs_s} {lean_op} {rhs_s} holds for all real values ({', '.join(free_vars)}).",
                )
        except Exception:
            pass

    # 12. Algebraic / Polynomial / Trigonometric Identities (true & false claim stress tests!)
    m_id = re.search(
        r"(?:prove\s+that|show\s+that|verify\s+that|verify|check\s+(?:whether|if)|is\s+it\s+true\s+that)\s+(.+?)\s*=\s*(.+?)[?.!]*$",
        p_l,
    )
    if m_id:
        lhs_raw = re.sub(r"\s+for\s+all\b.*$", "", m_id.group(1).strip())
        rhs_raw = re.sub(r"\s+for\s+all\b.*$", "", m_id.group(2).strip())
        try:
            l_sym = _safe_sympy(lhs_raw)
            r_sym = _safe_sympy(rhs_raw)
            free_vars = sorted({str(s) for s in (l_sym.free_symbols | r_sym.free_symbols)})
            lean_lhs = _to_lean_expr(str(l_sym))
            lean_rhs = _to_lean_expr(str(r_sym))
            is_trig = any(fn in (lhs_raw + rhs_raw) for fn in ("sin", "cos", "tan"))
            if is_trig and free_vars:
                binders = " ".join(free_vars)
                v0 = free_vars[0]
                lean_stmt = (
                    f"example ({binders} : ℝ) : {lean_lhs} = {lean_rhs} := by\n"
                    f"  try simp [Real.sin_sq_add_cos_sq, Real.cos_two_mul, Real.sin_two_mul]\n"
                    f"  try (ring_nf; rw [← Real.sin_sq_add_cos_sq {v0}]; ring)\n"
                    f"  try ring"
                )
                subk = "generalized_trig_identity"
            elif free_vars:
                binders = " ".join(free_vars)
                lean_stmt = f"example ({binders} : ℝ) : {lean_lhs} = {lean_rhs} := by ring"
                subk = "generalized_identity"
            else:
                lean_stmt = f"example : ({lean_lhs} : ℝ) = ({lean_rhs} : ℝ) := by norm_num"
                subk = "generalized_identity"
            return GeneralizedProposal(
                subkind=subk,
                candidate_answer=f"{lhs_raw} = {rhs_raw}",
                lean_statement=lean_stmt,
                back_translation=f"Verify whether {lhs_raw} = {rhs_raw}.",
                reasoning=f"Checking symbolic equivalence between {lhs_raw} and {rhs_raw}.",
            )
        except Exception:
            pass

    return None


def propose_generalized(problem: str, feedback: Optional[str] = None) -> GeneralizedProposal:
    """
    Propose a candidate solution and standalone Lean 4 theorem for an unseen,
    out-of-domain mathematical problem (outside all 244 trained templates).
    """
    if not _looks_like_math(problem):
        return GeneralizedProposal(
            subkind="non_math",
            candidate_answer="",
            lean_statement="",
            back_translation="",
            reasoning="Prompt does not appear to be a mathematical problem.",
            is_math_problem=False,
        )

    # 1. Try local LLM synthesis if a real GGUF model is loaded
    llm = _get_llm()
    if llm.__class__.__name__ != "_MockLLM":
        fb = f"\nPrevious verification feedback (fix this): {feedback}" if feedback else ""
        prompt = (
            f"<|im_start|>system\n{_GENERALIZED_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{problem}{fb}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            resp = llm(
                prompt,
                max_tokens=380,
                temperature=0.1,
                stop=["<|im_end|>", "<|user|>"],
                echo=False,
            )
            raw = resp["choices"][0]["text"].strip()
            parsed = _extract_json(raw)
            if isinstance(parsed, dict):
                if parsed.get("is_math") is False:
                    return GeneralizedProposal(
                        subkind="non_math",
                        candidate_answer="",
                        lean_statement="",
                        back_translation="",
                        reasoning="Classified as non-mathematical prompt.",
                        is_math_problem=False,
                    )
                stmt = str(parsed.get("lean_statement", "")).strip()
                ans = str(parsed.get("answer", "")).strip()
                if stmt and ans:
                    return GeneralizedProposal(
                        subkind="generalized_llm_synthesis",
                        candidate_answer=ans,
                        lean_statement=stmt,
                        back_translation=str(parsed.get("back_translation", f"{problem} Answer: {ans}.")),
                        reasoning=str(parsed.get("reasoning", "")),
                        is_math_problem=True,
                    )
        except Exception as e:
            logger.warning(f"Generalized LLM proposal failed, trying symbolic autoformalizer: {e}")

    # 2. Generalized Symbolic Autoformalizer
    sym_prop = _propose_symbolic(problem)
    if sym_prop is not None:
        return sym_prop

    # 3. Unformalizable mathematical claim -> empty Lean statement (will be REJECTED by gate)
    return GeneralizedProposal(
        subkind="generalized_unformalized",
        candidate_answer="",
        lean_statement="",
        back_translation="",
        reasoning="Could not construct a valid formal Lean 4 statement for this out-of-domain problem.",
        is_math_problem=True,
    )
