"""
VeriMath — Student Multi-Step Derivation Grader (SymPy CAS)
=============================================================
Grades and diagnoses a student's own step-by-step mathematical derivation
rather than only solving from scratch.

Supported modes:
  - algebra:    verifies step_i == step_{i+1} symbolically
  - equation:   verifies equation solution equivalence across steps
  - derivative: verifies d/dx(step_0) == step_1 (or prompt expression) and subsequent simplifications
  - integral:   verifies d/dx(step_1) == step_0 (ignoring + C) and subsequent simplifications
  - auto:       infers mode from problem prompt
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Set, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from backend.pipeline.generalized_prover import _to_lean_expr

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


@dataclass
class StepGrade:
    step_index: int
    step_text: str
    valid: bool
    feedback: str
    error_type: Optional[str] = None  # None | "sign_error" | "constant_error" | "factor_error" | "dropped_root_error" | "extraneous_root_error" | "algebra_mismatch" | "parse_error"
    implicit_condition: Optional[str] = None


@dataclass
class DerivationGradeResult:
    overall_correct: bool
    operation: str
    first_error_step: Optional[int]
    step_results: List[StepGrade]
    summary: str
    implicit_conditions: List[str] = field(default_factory=list)
    lean_certificate: Optional[str] = None
    lean_verified: bool = False

    def to_dict(self) -> dict:
        return {
            "overall_correct": self.overall_correct,
            "operation": self.operation,
            "first_error_step": self.first_error_step,
            "step_results": [asdict(s) for s in self.step_results],
            "summary": self.summary,
            "implicit_conditions": self.implicit_conditions,
            "lean_certificate": self.lean_certificate,
            "lean_verified": self.lean_verified,
        }


def _clean_math_text(s: str) -> str:
    s = s.strip()
    s = s.replace("[", "(").replace("]", ")").replace("−", "-").replace("·", "*").replace("×", "*")
    s = s.replace("²", "^2").replace("³", "^3")
    # Strip trailing "+ C" or "+ c" in antiderivatives
    s = re.sub(r"\+\s*[cC]\s*$", "", s).strip()
    # Strip common prefixes like "f'(x) =", "dy/dx =", "y' =", "Step 1:"
    s = re.sub(r"^(?:step\s*\d+\s*:\s*)", "", s, flags=re.I)
    s = re.sub(r"^(?:f'\(x\)|dy/dx|y'|d/dx\([^)]+\))\s*=\s*", "", s, flags=re.I)
    return s.strip()


def _parse_math(s: str) -> sp.Expr:
    cleaned = _clean_math_text(s)
    local_dict = {
        "x": sp.Symbol("x"),
        "y": sp.Symbol("y"),
        "t": sp.Symbol("t"),
        "e": sp.E,
        "pi": sp.pi,
        "ln": sp.log,
    }
    return parse_expr(cleaned, local_dict=local_dict, transformations=_TRANSFORMS)


def _infer_operation(problem: str, steps: List[str], operation: str) -> str:
    if operation != "auto":
        return operation
    p_l = problem.lower()
    if any(k in p_l for k in ("integral", "integrate", "antiderivative", "∫")):
        return "integral"
    if any(k in p_l for k in ("derivative", "differentiate", "d/dx", "f'(x)", "dy/dx")):
        return "derivative"
    if any("=" in s for s in steps) or "solve" in p_l:
        return "equation"
    return "algebra"


def _diagnose_expr_mismatch(expected: sp.Expr, actual: sp.Expr, var: sp.Symbol) -> tuple[str, str]:
    try:
        if sp.simplify(expected + actual) == 0:
            return "sign_error", "Sign inversion detected: your expression has the opposite sign."
        diff_expr = sp.simplify(expected - actual)
        if diff_expr.is_number and diff_expr != 0:
            return "constant_error", f"Constant term mismatch (off by {diff_expr})."
        if actual != 0:
            ratio = sp.simplify(expected / actual)
            if ratio.is_number and ratio != 1:
                return "factor_error", f"Missing or incorrect constant coefficient factor (off by a factor of {ratio})."
    except Exception:
        pass
    return "algebra_mismatch", f"Expression is not equivalent to the expected value ({expected})."


def _find_removed_denominator_factors(prev_expr: sp.Expr, curr_expr: sp.Expr, var: sp.Symbol) -> Tuple[List[sp.Expr], List[str]]:
    """Detect non-constant denominator factors that vanished between algebraic steps (e.g. (x - 2) in (x^2 - 4)/(x - 2) -> x + 2)."""
    try:
        _, den_prev = sp.fraction(prev_expr)
        _, den_curr = sp.fraction(curr_expr)
        if den_prev == 1:
            return [], []
        zeros_prev = sp.solve(den_prev, var)
        zeros_curr = set(sp.solve(den_curr, var)) if den_curr != 1 else set()
        lost_roots = [z for z in zeros_prev if z not in zeros_curr]
        if not lost_roots:
            return [], []
        conds = [f"{var} ≠ {z}" for z in lost_roots]
        non_zero_factors = [den_prev] if den_prev.free_symbols else []
        return non_zero_factors, conds
    except Exception:
        return [], []


def synthesize_steps_lean_certificate(
    op: str,
    raw_steps: List[str],
    parsed_steps: List[Tuple[str, object]],
    denom_factors: List[sp.Expr],
) -> Optional[str]:
    """Synthesize a standalone Lean 4 calc/theorem certificate proving the student's derivation."""
    if len(parsed_steps) < 2:
        return None
    try:
        if op == "algebra":
            lean_exprs = [_to_lean_expr(str(val)) for _, val in parsed_steps]
            all_syms: Set[str] = set()
            for _, val in parsed_steps:
                all_syms.update(str(s) for s in val.free_symbols)  # type: ignore[attr-defined]
            binders = f"({' '.join(sorted(all_syms))} : ℝ) " if all_syms else ""
            hyps = ""
            hyp_names: List[str] = []
            for idx_d, df in enumerate(denom_factors, 1):
                h_name = f"hden{idx_d}"
                hyp_names.append(h_name)
                hyps += f"({h_name} : ({_to_lean_expr(str(df))}) ≠ 0) "
            tactic = f"field_simp [{', '.join(hyp_names)}]; ring" if hyp_names else "ring"
            calc_lines = [f"    ({lean_exprs[0]}) = ({lean_exprs[1]}) := by {tactic}"]
            for e_next in lean_exprs[2:]:
                calc_lines.append(f"    _ = ({e_next}) := by {tactic}")
            return (
                f"example {binders}{hyps}: ({lean_exprs[0]}) = ({lean_exprs[-1]}) := by\n"
                f"  calc\n" + "\n".join(calc_lines)
            )

        if op == "derivative":
            f0 = _to_lean_expr(str(parsed_steps[0][1]))
            fn_end = _to_lean_expr(str(parsed_steps[-1][1]))
            return (
                f"example (x : ℝ) : deriv (fun x => {f0}) x = {fn_end} := by\n"
                f"  simp (disch := fun_prop) [deriv_comp, deriv_mul, deriv_add, deriv_sub, deriv_div, "
                f"deriv_pow, Real.deriv_sin, Real.deriv_cos, Real.deriv_exp, Real.deriv_log]\n"
                f"  try ring"
            )

        if op == "integral":
            f0 = _to_lean_expr(str(parsed_steps[0][1]))
            F_end = _to_lean_expr(str(parsed_steps[-1][1]))
            return (
                f"example (x : ℝ) : deriv (fun x => {F_end}) x = {f0} := by\n"
                f"  simp (disch := fun_prop) [deriv_comp, deriv_mul, deriv_add, deriv_sub, deriv_div, "
                f"deriv_pow, Real.deriv_sin, Real.deriv_cos, Real.deriv_exp, Real.deriv_log]\n"
                f"  try ring"
            )

        if op == "equation" and "=" in raw_steps[0] and "=" in raw_steps[-1]:
            l0, r0 = _clean_math_text(raw_steps[0]).split("=", 1)
            lk, rk = _clean_math_text(raw_steps[-1]).split("=", 1)
            l0_l = _to_lean_expr(str(_parse_math(l0)))
            r0_l = _to_lean_expr(str(_parse_math(r0)))
            lk_l = _to_lean_expr(str(_parse_math(lk)))
            rk_l = _to_lean_expr(str(_parse_math(rk)))
            return (
                f"example (x : ℝ) : ({l0_l} = {r0_l}) ↔ ({lk_l} = {rk_l}) := by\n"
                f"  constructor <;> intro h <;> linarith"
            )
    except Exception:
        return None
    return None


def grade_student_steps(
    problem: str,
    student_steps: List[str],
    operation: str = "auto",
) -> DerivationGradeResult:
    if not student_steps:
        return DerivationGradeResult(
            overall_correct=False,
            operation=operation,
            first_error_step=0,
            step_results=[],
            summary="No student steps provided to grade.",
        )

    op = _infer_operation(problem, student_steps, operation)
    x = sp.Symbol("x")
    results: List[StepGrade] = []
    first_error: Optional[int] = None
    implicit_conditions: List[str] = []
    denom_factors: List[sp.Expr] = []

    # Parse all steps
    parsed_steps: List[Tuple[str, object]] = []
    for idx, raw in enumerate(student_steps):
        try:
            if op == "equation" and "=" in raw:
                lhs_s, rhs_s = _clean_math_text(raw).split("=", 1)
                parsed_steps.append(("eq", (_parse_math(lhs_s), _parse_math(rhs_s))))
            else:
                parsed_steps.append(("expr", _parse_math(raw)))
        except Exception as e:
            if first_error is None:
                first_error = idx
            results.append(
                StepGrade(
                    step_index=idx,
                    step_text=raw,
                    valid=False,
                    feedback=f"Could not parse mathematical expression: {e}",
                    error_type="parse_error",
                )
            )
            return DerivationGradeResult(
                overall_correct=False,
                operation=op,
                first_error_step=first_error,
                step_results=results,
                summary=f"Syntax/parse error at step {idx + 1}.",
            )

    # Step 0 is always accepted as the starting expression/equation
    results.append(
        StepGrade(
            step_index=0,
            step_text=student_steps[0],
            valid=True,
            feedback="Starting expression/equation parsed successfully.",
        )
    )

    for idx in range(1, len(student_steps)):
        prev_kind, prev_val = parsed_steps[idx - 1]
        curr_kind, curr_val = parsed_steps[idx]

        valid = False
        feedback = ""
        err_type: Optional[str] = None
        step_cond: Optional[str] = None

        if op == "equation" and prev_kind == "eq" and curr_kind == "eq":
            pl, pr = prev_val  # type: ignore[misc]
            cl, cr = curr_val  # type: ignore[misc]
            sol_prev = sp.solve(pl - pr, x)
            sol_curr = sp.solve(cl - cr, x)
            set_prev = set(sol_prev)
            set_curr = set(sol_curr)
            if set_prev == set_curr:
                valid = True
                feedback = "Valid equivalent equation transformation."
            elif set_curr < set_prev:
                lost = sorted(set_prev - set_curr, key=str)
                lost_str = ", ".join(str(r) for r in lost)
                err_type = "dropped_root_error"
                step_cond = " ∧ ".join(f"x ≠ {r}" for r in lost)
                implicit_conditions.append(step_cond)
                feedback = (
                    f"Implicit division-by-zero / dropped root: this step lost solution(s) x = {lost_str} "
                    f"(check whether you divided both sides by an expression that can be 0)."
                )
            elif set_prev < set_curr:
                extra = sorted(set_curr - set_prev, key=str)
                extra_str = ", ".join(str(r) for r in extra)
                err_type = "extraneous_root_error"
                step_cond = "domain / sign check required"
                implicit_conditions.append(step_cond)
                feedback = (
                    f"Extraneous root introduced: solution set gained {extra_str} "
                    f"(check irreversible operations like squaring both sides)."
                )
            else:
                err_type = "algebra_mismatch"
                feedback = f"Solution set changed from {sol_prev} to {sol_curr}."

        elif op == "derivative" and idx == 1 and prev_kind == "expr" and curr_kind == "expr":
            expected_deriv = sp.diff(prev_val, x)  # type: ignore[arg-type]
            if sp.simplify(expected_deriv - curr_val) == 0:  # type: ignore[operator]
                valid = True
                feedback = "Correct differentiation step."
            else:
                err_type, feedback = _diagnose_expr_mismatch(expected_deriv, curr_val, x)  # type: ignore[arg-type]

        elif op == "integral" and idx == 1 and prev_kind == "expr" and curr_kind == "expr":
            recovered = sp.diff(curr_val, x)  # type: ignore[arg-type]
            if sp.simplify(recovered - prev_val) == 0:  # type: ignore[operator]
                valid = True
                feedback = "Correct antiderivative step (verified via d/dx)."
            else:
                expected_int = sp.integrate(prev_val, x)  # type: ignore[arg-type]
                err_type, feedback = _diagnose_expr_mismatch(expected_int, curr_val, x)  # type: ignore[arg-type]

        elif prev_kind == "expr" and curr_kind == "expr":
            if sp.simplify(prev_val - curr_val) == 0:  # type: ignore[operator]
                valid = True
                new_denoms, cond_list = _find_removed_denominator_factors(prev_val, curr_val, x)  # type: ignore[arg-type]
                if cond_list:
                    denom_factors.extend(new_denoms)
                    step_cond = ", ".join(cond_list)
                    implicit_conditions.extend(cond_list)
                    feedback = f"Valid algebraic simplification under implicit domain condition ({step_cond})."
                else:
                    feedback = "Valid algebraic simplification."
            else:
                err_type, feedback = _diagnose_expr_mismatch(prev_val, curr_val, x)  # type: ignore[arg-type]

        if not valid and first_error is None:
            first_error = idx

        results.append(
            StepGrade(
                step_index=idx,
                step_text=student_steps[idx],
                valid=valid,
                feedback=feedback,
                error_type=err_type,
                implicit_condition=step_cond,
            )
        )

    overall = first_error is None
    dedup_conds = list(dict.fromkeys(implicit_conditions))
    lean_cert = synthesize_steps_lean_certificate(op, student_steps, parsed_steps, denom_factors) if overall else None
    summary = (
        f"All {len(student_steps)} steps verified."
        if overall
        else f"Error found at step {first_error + 1}: {results[first_error].feedback}"
    )

    return DerivationGradeResult(
        overall_correct=overall,
        operation=op,
        first_error_step=first_error,
        step_results=results,
        summary=summary,
        implicit_conditions=dedup_conds,
        lean_certificate=lean_cert,
        lean_verified=False,
    )
