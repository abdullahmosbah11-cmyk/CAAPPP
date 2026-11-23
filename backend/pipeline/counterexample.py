"""
VeriMath — Module 7 (Novel Contribution #1): Lean-Certified Counterexample Synthesis
=====================================================================================
Turns opaque `REJECTED` failures on false out-of-domain claims into mathematically
grounded, Lean 4 kernel-verified disproofs with minimal counterexample witnesses.

Pipeline when a generalized claim fails Lean verification:
  1. Inspect the false claim's domain (algebraic/polynomial identity, trig identity,
     logarithmic identity, complex arithmetic, numeric/symbolic inequality,
     divisibility, or primality claim).
  2. Synthesize a minimal, human-friendly witness (e.g. a = 1, b = 1 or x = 0)
     where the LHS and RHS demonstrably disagree or the relation evaluates to False.
  3. Construct a standalone Lean 4 negation / inequality theorem (`≠` or `¬ (...)`)
     at that witness WITHOUT `sorry`/`admit`/`axiom`.
  4. Verify the disproof theorem with `verify_async()` (Lean 4.36.0) so the student
     sees a formally certified counterexample ("Mathematically Disproven").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from backend.pipeline.generalized_prover import _to_lean_expr, _safe_sympy, _fmt_rational

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


@dataclass
class CounterexampleResult:
    found: bool
    witness: Dict[str, str] = field(default_factory=dict)
    witness_display: str = ""
    lhs_value: str = ""
    rhs_value: str = ""
    disproof_lean_statement: str = ""
    explanation: str = ""
    lean_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "witness": self.witness,
            "witness_display": self.witness_display,
            "lhs_value": self.lhs_value,
            "rhs_value": self.rhs_value,
            "disproof_lean_statement": self.disproof_lean_statement,
            "explanation": self.explanation,
            "lean_verified": self.lean_verified,
        }


def _fmt_lean_val(val: sp.Expr) -> str:
    """Format an exact rational or trig multiple of pi into Lean 4 real syntax."""
    if val == 0:
        return "(0 : ℝ)"
    if val == sp.pi:
        return "Real.pi"
    if val == sp.pi / 4:
        return "(Real.pi / 4)"
    if val == sp.pi / 2:
        return "(Real.pi / 2)"
    if val == sp.pi / 6:
        return "(Real.pi / 6)"
    if getattr(val, "is_Integer", False):
        return f"({int(val)} : ℝ)"
    if getattr(val, "is_Rational", False):
        return f"({val} : ℝ)"
    return f"({val} : ℝ)"


def _subst_lean_expr(expr_sym: sp.Expr, subs_map: Dict[str, sp.Expr]) -> str:
    """Substitute witness values into a symbolic expression and convert to Lean 4."""
    # Convert abstract expression to Lean first, then substitute concrete Lean values cleanly
    expr_str = str(expr_sym)
    # Replace single-letter variables deterministically by token
    lean_s = _to_lean_expr(expr_str)
    for var_name, val in subs_map.items():
        lean_val = _fmt_lean_val(val)
        lean_s = re.sub(rf"\b{re.escape(var_name)}\b", lean_val, lean_s)
    return lean_s


def synthesize_counterexample_proposal(
    problem: str,
    lean_statement: Optional[str] = None,
) -> Optional[CounterexampleResult]:
    """
    Synchronously synthesize a minimal counterexample and standalone Lean 4
    negation theorem (`example : ... ≠ ...` or `example : ¬ (...)`) for a false claim.
    """
    p = problem.strip()
    p_l = p.lower()

    # 1. False Primality Claim: e.g. "Prove that 91 is prime"
    m_prime = re.search(r"(?:prove|show|verify)\s+(?:that\s+)?(\d+)\s+is\s+(?:a\s+)?prime\b", p_l)
    if m_prime:
        n_i = int(m_prime.group(1))
        if n_i >= 2 and not sp.isprime(n_i):
            factors = sp.factorint(n_i)
            smallest = min(factors.keys())
            quotient = n_i // smallest
            return CounterexampleResult(
                found=True,
                witness={"n": str(n_i), "divisor": str(smallest)},
                witness_display=f"{n_i} = {smallest} × {quotient}",
                lhs_value=str(n_i),
                rhs_value=f"{smallest} × {quotient}",
                disproof_lean_statement=f"example : ¬ Nat.Prime {n_i} := by decide",
                explanation=(
                    f"{n_i} is not prime because it has a non-trivial factor {smallest}: "
                    f"{smallest} × {quotient} = {n_i}."
                ),
            )

    # 2. False Divisibility Claim: e.g. "Prove that 7 divides 100" or "Does 7 divide 100?"
    m_div = re.search(r"(\d+)\s+divides?\s+(\d+)", p_l) or re.search(r"is\s+(\d+)\s+divisible\s+by\s+(\d+)", p_l)
    if m_div:
        if "divisible" in p_l:
            b_i, a_i = int(m_div.group(1)), int(m_div.group(2))
        else:
            a_i, b_i = int(m_div.group(1)), int(m_div.group(2))
        if a_i != 0 and b_i % a_i != 0:
            rem = b_i % a_i
            return CounterexampleResult(
                found=True,
                witness={"divisor": str(a_i), "dividend": str(b_i), "remainder": str(rem)},
                witness_display=f"{b_i} mod {a_i} = {rem}",
                lhs_value=f"{b_i} mod {a_i}",
                rhs_value=str(rem),
                disproof_lean_statement=f"example : ¬ (({a_i} : ℕ) ∣ {b_i}) := by decide",
                explanation=(
                    f"{a_i} does not divide {b_i} evenly: dividing {b_i} by {a_i} leaves a remainder of {rem} (≠ 0)."
                ),
            )

    # 3. False Combinatorics / Number-Theory Claim (factorial, gcd, lcm, choose, perm, mod)
    m_cl_eq = re.search(r"=\s*(\d+)\s*[?.!]*$", p_l)
    if m_cl_eq:
        claimed_i = int(m_cl_eq.group(1))
        m_fact = re.search(r"(?:\bfactorial\s+(?:of\s+)?(\d+)|(\d+)\s*!)", p_l)
        if m_fact:
            n_i = int(m_fact.group(1) or m_fact.group(2))
            actual = math.factorial(n_i)
            if actual != claimed_i:
                return CounterexampleResult(
                    found=True,
                    witness={"n": str(n_i)},
                    witness_display=f"{n_i}! = {actual}",
                    lhs_value=str(actual),
                    rhs_value=str(claimed_i),
                    disproof_lean_statement=f"example : Nat.factorial {n_i} ≠ {claimed_i} := by decide",
                    explanation=f"Evaluating {n_i}! gives {actual}, not {claimed_i} ({actual} ≠ {claimed_i}).",
                )
        m_gcd = re.search(r"\bgcd\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", p_l)
        if m_gcd:
            a_i, b_i = int(m_gcd.group(1)), int(m_gcd.group(2))
            actual = math.gcd(a_i, b_i)
            if actual != claimed_i:
                return CounterexampleResult(
                    found=True,
                    witness={"a": str(a_i), "b": str(b_i)},
                    witness_display=f"gcd({a_i}, {b_i}) = {actual}",
                    lhs_value=str(actual),
                    rhs_value=str(claimed_i),
                    disproof_lean_statement=f"example : Nat.gcd {a_i} {b_i} ≠ {claimed_i} := by decide",
                    explanation=f"The greatest common divisor of {a_i} and {b_i} is {actual}, not {claimed_i} ({actual} ≠ {claimed_i}).",
                )
        m_ch = re.search(r"(?:choose|C)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", p, re.I)
        if m_ch:
            n_i, k_i = int(m_ch.group(1)), int(m_ch.group(2))
            actual = math.comb(n_i, k_i)
            if actual != claimed_i:
                return CounterexampleResult(
                    found=True,
                    witness={"n": str(n_i), "k": str(k_i)},
                    witness_display=f"C({n_i}, {k_i}) = {actual}",
                    lhs_value=str(actual),
                    rhs_value=str(claimed_i),
                    disproof_lean_statement=f"example : Nat.choose {n_i} {k_i} ≠ {claimed_i} := by decide",
                    explanation=f"Evaluating C({n_i}, {k_i}) gives {actual}, not {claimed_i} ({actual} ≠ {claimed_i}).",
                )

    # 3b. False Complex Arithmetic Identity: e.g. "Prove that (3 + 4i) * (1 - 2i) = 11 + 2i"
    if (
        "=" in p and ("complex" in p_l or re.search(r"(?:\d\s*\*?\s*i\b|[+-]\s*i\b)", p_l))
        and not any(w in p_l for w in ("derivative", "integral", "matrix", "determinant"))
    ):
        body_c = re.sub(
            r"^\s*(?:prove\s+that|show\s+that|verify\s+that|check\s+(?:whether|if)|is\s+it\s+true\s+that)\s+",
            "",
            p,
            flags=re.I,
        ).rstrip("?.! ").strip()
        if "=" in body_c:
            lhs_c_raw, rhs_c_raw = [x.strip() for x in body_c.split("=", 1)]
            try:
                def _parse_c(txt: str) -> sp.Expr:
                    s = re.sub(r"(\d)\s*i\b", r"\1*I", txt)
                    s = re.sub(r"(?<![a-zA-Z_])i(?![a-zA-Z_])", "I", s)
                    return parse_expr(s, local_dict={"I": sp.I}, transformations=_TRANSFORMS)

                def _to_lean_c(txt: str) -> str:
                    s = re.sub(r"\)\s*\(", ") * (", txt)
                    s = re.sub(r"(\d)\s*\(", r"\1 * (", s)
                    s = re.sub(r"(\d)\s*\*?\s*i\b", r"\1 * Complex.I", s)
                    s = re.sub(r"(?<![a-zA-Z_])i(?![a-zA-Z_])", "Complex.I", s)
                    return s

                l_val = sp.expand(_parse_c(lhs_c_raw))
                r_val = sp.expand(_parse_c(rhs_c_raw))
                l_re, l_im = l_val.as_real_imag()
                r_re, r_im = r_val.as_real_imag()
                if l_re != r_re or l_im != r_im:
                    # Pick the exact differing projection so Lean 4 closes without unused-tactic warnings
                    if l_im != r_im:
                        proj_block = "  have him := congr_arg Complex.im h\n  simp at him\n  norm_num at him"
                        mismatch_note = f"imaginary part is {l_im} instead of {r_im}"
                    else:
                        proj_block = "  have hre := congr_arg Complex.re h\n  simp at hre\n  norm_num at hre"
                        mismatch_note = f"real part is {l_re} instead of {r_re}"
                    disproof = (
                        f"example : (({_to_lean_c(lhs_c_raw)}) : ℂ) ≠ (({_to_lean_c(rhs_c_raw)}) : ℂ) := by\n"
                        f"  intro h\n"
                        f"{proj_block}"
                    )
                    l_disp = str(l_val).replace("*I", "i").replace("I", "i")
                    r_disp = str(r_val).replace("*I", "i").replace("I", "i")
                    return CounterexampleResult(
                        found=True,
                        witness={"expression": lhs_c_raw},
                        witness_display=lhs_c_raw,
                        lhs_value=l_disp,
                        rhs_value=r_disp,
                        disproof_lean_statement=disproof,
                        explanation=(
                            f"Expanding the left side gives {l_disp}, while the claimed right side is {r_disp} "
                            f"(the {mismatch_note})."
                        ),
                    )
            except Exception:
                pass

    # 4. False Numeric or Symbolic Inequalities: e.g. "Prove that 2^10 < 1000" or "Prove that x^2 + y^2 < 2*x*y"
    m_ineq = re.search(
        r"(?:prove\s+that|show\s+that|verify\s+that|is|check\s+(?:whether|if))?\s*(.+?)\s*(<=|>=|≤|≥|<|>)\s*(.+?)[?.!]*$",
        p_l,
    )
    if m_ineq and "=" not in m_ineq.group(1) and "=" not in m_ineq.group(3):
        lhs_s, op_s, rhs_s = m_ineq.group(1).strip(), m_ineq.group(2), m_ineq.group(3).strip()
        op_map = {"<=": "≤", ">=": "≥", "≤": "≤", "≥": "≥", "<": "<", ">": ">"}
        lean_op = op_map[op_s]
        try:
            l_sym = _safe_sympy(lhs_s)
            r_sym = _safe_sympy(rhs_s)
            free_vars = sorted({str(s) for s in (l_sym.free_symbols | r_sym.free_symbols)})

            def _holds(lv: float, rv: float) -> bool:
                if lean_op == "<":
                    return lv < rv - 1e-12
                if lean_op == ">":
                    return lv > rv + 1e-12
                if lean_op == "≤":
                    return lv <= rv + 1e-12
                return lv >= rv - 1e-12

            if not free_vars:
                lv_e = l_sym.evalf()
                rv_e = r_sym.evalf()
                if not _holds(float(lv_e), float(rv_e)):
                    lean_l = _to_lean_expr(lhs_s)
                    lean_r = _to_lean_expr(rhs_s)
                    disproof = f"example : ¬ (({lean_l} : ℝ) {lean_op} ({lean_r} : ℝ)) := by norm_num"
                    lv_disp = _fmt_rational(l_sym) if getattr(l_sym, "is_Rational", False) else str(lv_e)
                    rv_disp = _fmt_rational(r_sym) if getattr(r_sym, "is_Rational", False) else str(rv_e)
                    return CounterexampleResult(
                        found=True,
                        witness={"LHS": lv_disp, "RHS": rv_disp},
                        witness_display=f"{lhs_s} = {lv_disp}, {rhs_s} = {rv_disp}",
                        lhs_value=lv_disp,
                        rhs_value=rv_disp,
                        disproof_lean_statement=disproof,
                        explanation=f"Evaluating both sides gives {lv_disp} vs. {rv_disp}, so {lv_disp} {lean_op} {rv_disp} is false.",
                    )
            else:
                import itertools
                candidates = [sp.Integer(0), sp.Integer(1), sp.Integer(-1), sp.Integer(2), sp.Integer(-2)]
                for combo in itertools.product(candidates, repeat=len(free_vars)):
                    subs = dict(zip(free_vars, combo))
                    lv_sub = sp.simplify(l_sym.subs({sp.Symbol(k): v for k, v in subs.items()}))
                    rv_sub = sp.simplify(r_sym.subs({sp.Symbol(k): v for k, v in subs.items()}))
                    if not _holds(float(lv_sub.evalf()), float(rv_sub.evalf())):
                        lean_l = _subst_lean_expr(l_sym, subs)
                        lean_r = _subst_lean_expr(r_sym, subs)
                        disproof = f"example : ¬ (({lean_l}) {lean_op} ({lean_r})) := by norm_num"
                        w_dict = {k: str(v) for k, v in subs.items()}
                        w_disp = ", ".join(f"{k} = {v}" for k, v in w_dict.items())
                        return CounterexampleResult(
                            found=True,
                            witness=w_dict,
                            witness_display=w_disp,
                            lhs_value=str(lv_sub),
                            rhs_value=str(rv_sub),
                            disproof_lean_statement=disproof,
                            explanation=(
                                f"Counterexample at {w_disp}: the left side evaluates to {lv_sub} "
                                f"and the right side evaluates to {rv_sub}, which violates {lhs_s} {lean_op} {rhs_s}."
                            ),
                        )
        except Exception:
            pass

    # 5. False Algebraic, Trig, or Logarithmic Identities: e.g. "(a+b)^2 = a^2+b^2", "log(a+b) = log(a)+log(b)"
    m_id = re.search(
        r"(?:prove\s+that|show\s+that|verify\s+that|verify|check\s+(?:whether|if)|is\s+it\s+true\s+that)?\s*(.+?)\s*=\s*(.+?)[?.!]*$",
        p_l,
    )
    if m_id:
        lhs_raw = re.sub(r"\s+for\s+(?:all\s+)?.*$", "", m_id.group(1).strip())
        rhs_raw = re.sub(r"\s+for\s+(?:all\s+)?.*$", "", m_id.group(2).strip())
        try:
            l_sym = _safe_sympy(lhs_raw)
            r_sym = _safe_sympy(rhs_raw)
            free_vars = sorted({str(s) for s in (l_sym.free_symbols | r_sym.free_symbols)})
            has_log = ("log" in lhs_raw or "ln" in lhs_raw or "log" in rhs_raw or "ln" in rhs_raw)
            has_trig = any(fn in (lhs_raw + rhs_raw) for fn in ("sin", "cos", "tan"))

            import itertools
            if has_log:
                # Positive integers avoiding 1 first so exp_add + exp_log + norm_num closes cleanly
                candidates = [sp.Integer(2), sp.Integer(3), sp.Integer(4), sp.Integer(1)]
            elif has_trig:
                candidates = [sp.Integer(0), sp.pi / 2, sp.pi / 4, sp.pi / 6]
            else:
                candidates = [sp.Integer(1), sp.Integer(2), sp.Integer(0), sp.Integer(-1)]

            if not free_vars:
                lv_f = complex(l_sym.evalf())
                rv_f = complex(r_sym.evalf())
                if abs(lv_f - rv_f) > 1e-6:
                    lean_l = _to_lean_expr(str(l_sym))
                    lean_r = _to_lean_expr(str(r_sym))
                    disproof = f"example : ({lean_l} : ℝ) ≠ ({lean_r} : ℝ) := by norm_num"
                    return CounterexampleResult(
                        found=True,
                        witness={"expression": lhs_raw},
                        witness_display=f"{lhs_raw} vs {rhs_raw}",
                        lhs_value=str(l_sym),
                        rhs_value=str(r_sym),
                        disproof_lean_statement=disproof,
                        explanation=f"Evaluating both sides gives {l_sym} ≠ {r_sym}.",
                    )
            else:
                # Prefer all-equal / all-positive simple witnesses (like (1, 1) or (2, 3))
                for combo in itertools.product(candidates, repeat=len(free_vars)):
                    if has_log and combo == tuple(sp.Integer(1) for _ in free_vars):
                        # (1, 1) also works, but skip 1+1 if (2, 3) is cleaner; try later if needed
                        pass
                    subs_sym = {sp.Symbol(k): v for k, v in zip(free_vars, combo)}
                    lv_sub = sp.simplify(l_sym.subs(subs_sym))
                    rv_sub = sp.simplify(r_sym.subs(subs_sym))
                    try:
                        diff_num = complex((lv_sub - rv_sub).evalf())
                    except Exception:
                        continue
                    if abs(diff_num) > 1e-6:
                        subs = dict(zip(free_vars, combo))
                        lean_l = _subst_lean_expr(l_sym, subs)
                        lean_r = _subst_lean_expr(r_sym, subs)
                        if has_log:
                            disproof = (
                                f"example : ({lean_l}) ≠ ({lean_r}) := by\n"
                                f"  intro h\n"
                                f"  have he := congr_arg Real.exp h\n"
                                f"  simp (disch := norm_num) [Real.exp_add, Real.exp_log] at he\n"
                                f"  try norm_num at he"
                            )
                        elif has_trig:
                            disproof = (
                                f"example : ({lean_l}) ≠ ({lean_r}) := by\n"
                                f"  simp [Real.sin_add, Real.cos_add, Real.sin_two_mul, Real.cos_two_mul, "
                                f"Real.sin_pi, Real.cos_pi, Real.sin_pi_div_two, Real.cos_pi_div_two]\n"
                                f"  try norm_num"
                            )
                        else:
                            disproof = f"example : ({lean_l}) ≠ ({lean_r}) := by norm_num"

                        def _nice_disp(v: sp.Expr) -> str:
                            return str(v).replace("pi", "π")

                        w_dict = {k: _nice_disp(v) for k, v in subs.items()}
                        w_disp = ", ".join(f"{k} = {v}" for k, v in w_dict.items())
                        lv_disp = _nice_disp(lv_sub)
                        rv_disp = _nice_disp(rv_sub)
                        return CounterexampleResult(
                            found=True,
                            witness=w_dict,
                            witness_display=w_disp,
                            lhs_value=lv_disp,
                            rhs_value=rv_disp,
                            disproof_lean_statement=disproof,
                            explanation=(
                                f"Counterexample at {w_disp}: the left side equals {lv_disp} "
                                f"whereas the right side equals {rv_disp} ({lv_disp} ≠ {rv_disp})."
                            ),
                        )
        except Exception:
            pass

    return None
