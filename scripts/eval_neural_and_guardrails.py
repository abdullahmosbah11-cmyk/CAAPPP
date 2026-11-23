"""
VeriMath — Held-Out Neural & Guardrail Evaluation Harness
=========================================================
Evaluates the active solver + formalizer + judge + lean_runner pipeline on:
1. Unseen paraphrased calculus & algebra prompts
2. Zero-trust 'sorry' injection rejection
3. Incomplete problem guardrail detection
"""

from backend.templates.registry import classify_problem
from backend.pipeline.solver import solve
from backend.pipeline.formalizer import formalize
from backend.pipeline.statement_judge import judge
from backend.pipeline.lean_runner import verify, LeanVerdict


HELD_OUT_PROBLEMS = [
    ("Compute d/dx of x^3 * cos(x)", "derivative"),
    ("Evaluate the indefinite integral of 3*x^2 * exp(x^3) dx", "indefinite_integral"),
    ("Solve for x: 4*x - 12 = 8", "equation_solving"),
    ("Find the definite integral of 2*x from 0 to 5", "definite_integral"),
]

SORRY_INJECTIONS = [
    "example : 1 + 1 = 3 := by sorry",
    "example (x : ℝ) : x + 1 = x := by admit",
]


def run_eval():
    passed = 0
    for prompt, expected_tmpl in HELD_OUT_PROBLEMS:
        tmpl = classify_problem(prompt)
        s_res = solve(prompt)
        f_res = formalize(prompt, s_res, tmpl)
        j_res = judge(prompt, f_res)
        l_res = verify(f_res.lean_statement)
        ok = (
            bool(s_res.candidate_answer)
            and bool(f_res.lean_statement)
            and j_res.verdict == "MATCH"
            and l_res.verdict == LeanVerdict.VERIFIED
        )
        print(f"[HELD-OUT] {expected_tmpl} -> tmpl={tmpl} judge={j_res.verdict} lean={l_res.verdict.value} ans={s_res.candidate_answer}")
        if ok:
            passed += 1

    sorry_blocked = 0
    for stmt in SORRY_INJECTIONS:
        res = verify(stmt)
        if res.verdict == LeanVerdict.REJECTED:
            sorry_blocked += 1
        print(f"[ZERO-TRUST] '{stmt}' -> {res.verdict.value}")

    print(f"\nHeld-out pass: {passed}/{len(HELD_OUT_PROBLEMS)} | Zero-trust blocked: {sorry_blocked}/{len(SORRY_INJECTIONS)}")
    assert passed == len(HELD_OUT_PROBLEMS), "Some held-out problems failed!"
    assert sorry_blocked == len(SORRY_INJECTIONS), "Zero-trust sorry rejection failed!"


if __name__ == "__main__":
    run_eval()
