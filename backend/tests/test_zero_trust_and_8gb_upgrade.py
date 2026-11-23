"""
Unit tests for Zero-Trust Lean Kernel Hardening, Semantic Judge,
Student Step Grader, and 8 GB Pi 4 Upgrades
================================================================
"""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.pipeline.lean_runner import verify, LeanVerdict, _has_lean_errors
from backend.pipeline.formalizer import FormalizerResult
from backend.pipeline.statement_judge import judge, _check_semantic_inversions
from backend.pipeline.step_grader import grade_student_steps
from backend.pipeline.explainer import explain, build_intuitive_guide


class TestZeroTrustAnd8GBUpgrade(unittest.TestCase):
    def test_rejects_sorry_statement(self):
        res = verify("example : 1 + 1 = 3 := by sorry")
        self.assertEqual(res.verdict, LeanVerdict.REJECTED)
        self.assertIn("sorry", res.lean_output.lower())

    def test_rejects_admit_and_axiom(self):
        res_admit = verify("example : 1 = 2 := by admit")
        self.assertEqual(res_admit.verdict, LeanVerdict.REJECTED)
        res_axiom = verify("axiom fake : 1 = 2\nexample : 1 = 2 := fake")
        self.assertEqual(res_axiom.verdict, LeanVerdict.REJECTED)

    def test_has_lean_errors_catches_sorry_warning(self):
        self.assertTrue(_has_lean_errors("warning: declaration uses 'sorry'"))
        self.assertFalse(_has_lean_errors(""))

    def test_strict_lean_mode_fails_closed_when_missing(self):
        with patch.dict(os.environ, {"VERIMATH_STRICT_LEAN": "true"}):
            with patch("subprocess.run", side_effect=FileNotFoundError("lean not found")):
                res = verify("example : 1 + 1 = 2 := by norm_num")
                self.assertEqual(res.verdict, LeanVerdict.ENGINE_ERROR)

    def test_statement_judge_rejects_number_mismatch(self):
        f_res = FormalizerResult(
            lean_statement="example : 3 * 5 + 7 = 99 := by norm_num",
            template_id="equation_solving",
            slot_values={"coeff": "3", "a": "x", "b": "7", "c": "99", "answer": "5"},
            elapsed_ms=1,
            attempts=1,
        )
        j_res = judge("Solve 3x + 7 = 22", f_res)
        self.assertEqual(j_res.verdict, "MISMATCH")

    def test_semantic_inversion_catches_flipped_inequality(self):
        res = _check_semantic_inversions(
            "Find values where x < 5",
            "Find values where x > 5",
        )
        self.assertFalse(res.passed)

    def test_semantic_inversion_catches_swapped_derivative_target(self):
        res = _check_semantic_inversions(
            "Find dy/dx for x^2 + y^2 = 9",
            "Find dx/dy for x^2 + y^2 = 9",
        )
        self.assertFalse(res.passed)

    def test_semantic_inversion_catches_swapped_variable_bindings(self):
        res = _check_semantic_inversions(
            "Evaluate at x = 2, y = 5",
            "Evaluate at x = 5, y = 2",
        )
        self.assertFalse(res.passed)

    def test_semantic_inversion_catches_sign_flip_in_equation(self):
        f_res = FormalizerResult(
            lean_statement="example : (3 * x + -7 = 22) ↔ (x = 5) := by omega",
            template_id="equation_solving",
            slot_values={"coeff": "3", "a": "x", "b": "-7", "c": "22", "answer": "5"},
            elapsed_ms=1,
            attempts=1,
        )
        j_res = judge("Solve 3x + 7 = 22", f_res)
        self.assertEqual(j_res.verdict, "MISMATCH")

    def test_step_grader_valid_and_invalid_derivations(self):
        ok = grade_student_steps(
            "Find the derivative of x^3 + 2*x",
            ["x^3 + 2*x", "3*x^2 + 2"],
        )
        self.assertTrue(ok.overall_correct)

        bad_sign = grade_student_steps(
            "Find the derivative of x^2 - 3*x",
            ["x^2 - 3*x", "-2*x + 3"],
        )
        self.assertFalse(bad_sign.overall_correct)
        self.assertEqual(bad_sign.first_error_step, 1)
        self.assertEqual(bad_sign.step_results[1].error_type, "sign_error")

        bad_eq = grade_student_steps(
            "Solve 3x + 7 = 22",
            ["3*x + 7 = 22", "3*x = 29", "x = 29/3"],
        )
        self.assertFalse(bad_eq.overall_correct)
        self.assertEqual(bad_eq.first_error_step, 1)

        ok_int = grade_student_steps(
            "Integrate 2*x * exp(x^2)",
            ["2*x * exp(x^2)", "exp(x^2) + C"],
        )
        self.assertTrue(ok_int.overall_correct)

    def test_grade_steps_endpoint(self):
        with TestClient(app) as client:
            r = client.post(
                "/grade-steps",
                json={
                    "problem": "Solve 3x + 7 = 22",
                    "student_steps": ["3*x + 7 = 22", "3*x = 15", "x = 5"],
                },
            )
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json()["overall_correct"])

    def test_intuitive_explainer_layer_and_sanitization(self):
        g_prod = build_intuitive_guide(
            "Find the derivative of x^2 * sin(x)",
            "2*x * Real.sin x + x^2 * Real.cos x",
            "According to El-Moasser, verified by Lean 4.",
            template_id="derivative",
        )
        self.assertIn("First", g_prod.mental_model)
        self.assertIn("derivative of Second", g_prod.mental_model)
        for forbidden in ("Real.", "Lean", "El-Moasser", "Book of Integrals", "Senior 3"):
            self.assertNotIn(forbidden.lower(), g_prod.full_explanation.lower())

        txt = explain("Evaluate the integral of sin(x)/(sin(x)+cos(x)) from 0 to pi/2", "pi/4", template_id="kings_rule_integral")
        self.assertIn("Symmetry", txt)
        self.assertIn("pi/4", txt)

        with TestClient(app) as client:
            r = client.post("/solve", json={"problem": "Integrate: x^2 * sqrt(x^3 + 4) dx"})
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertTrue(data["explanation"])
            self.assertTrue(data["mental_model"])
            self.assertTrue(data["pitfall_tip"])

    def test_all_concept_guides_comply_with_rule_22_and_23(self):
        from backend.pipeline.explainer import _CONCEPT_GUIDES
        forbidden_tokens = (
            "el-moasser",
            "book of integrals",
            "senior 3",
            "miguel santiago",
            "real.",
            "complex.i",
            "lean 4",
            "mathlib",
        )
        for key, guide in _CONCEPT_GUIDES.items():
            blob = " ".join(guide.values()).lower()
            for tok in forbidden_tokens:
                self.assertNotIn(tok, blob, f"Rule 22/23 violation '{tok}' in guide '{key}'")

    def test_ast_faithfulness_anti_cheating_guards(self):
        from backend.pipeline.ast_faithfulness import verify_ast_faithfulness

        # 1. Tautology substitution (`1 + 1 = 2` for an algebra problem)
        res_taut = verify_ast_faithfulness("Solve 3x + 7 = 22", "example : 1 + 1 = 2 := by norm_num")
        self.assertFalse(res_taut.passed)

        # 2. Reflexive tautology (`x = x` on an identity proof)
        res_refl = verify_ast_faithfulness("Prove that (a + b)^2 = a^2 + b^2", "example (x : ℝ) : x = x := by rfl")
        self.assertFalse(res_refl.passed)

        # 3. Inequality weakening (`>` replaced with `≥`)
        res_weak = verify_ast_faithfulness("Prove that x^2 + 1 > 0", "example (x : ℝ) : x^2 + 1 ≥ 0 := by positivity")
        self.assertFalse(res_weak.passed)

        # 4. RHS tampering on a false claim (`a^2 + b^2` silently replaced by `a^2 + 2*a*b + b^2`)
        res_tamper = verify_ast_faithfulness(
            "Prove that (a + b)^2 = a^2 + b^2",
            "example (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by ring",
        )
        self.assertFalse(res_tamper.passed)

        # 5. Discrete literal substitution (`Nat.Prime 7` submitted for `Prove that 91 is prime`)
        res_swap_lit = verify_ast_faithfulness(
            "Prove that 91 is prime",
            "example : Nat.Prime 7 := by decide",
        )
        self.assertFalse(res_swap_lit.passed)

        # 6. Legitimate faithful identity passes
        res_ok = verify_ast_faithfulness(
            "Prove that (a + b)^2 = a^2 + 2*a*b + b^2",
            "example (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by ring",
        )
        self.assertTrue(res_ok.passed)

    def test_step_grader_lean_calc_and_implicit_conditions(self):
        # 1. Dropped root diagnostic (`x*(x - 2) = 0` -> `x = 0` implicitly divides by x - 2 = 0)
        dropped = grade_student_steps(
            "Solve x*(x - 2) = 0",
            ["x*(x - 2) = 0", "x = 0"],
        )
        self.assertFalse(dropped.overall_correct)
        self.assertEqual(dropped.step_results[1].error_type, "dropped_root_error")
        self.assertIn("x ≠ 2", dropped.implicit_conditions)

        # 2. Extraneous root diagnostic (`x = 2` -> `x^2 = 4`)
        extra = grade_student_steps(
            "Solve x = 2",
            ["x = 2", "x^2 = 4"],
        )
        self.assertFalse(extra.overall_correct)
        self.assertEqual(extra.step_results[1].error_type, "extraneous_root_error")

        # 3. Rational denominator pole diagnostic (`(x^2 - 4)/(x - 2)` -> `x + 2` assumes `x ≠ 2`)
        rat = grade_student_steps(
            "Simplify (x^2 - 4)/(x - 2)",
            ["(x^2 - 4)/(x - 2)", "x + 2"],
            operation="algebra",
        )
        self.assertTrue(rat.overall_correct)
        self.assertIn("x ≠ 2", rat.implicit_conditions)
        self.assertIsNotNone(rat.lean_certificate)
        self.assertIn("field_simp", rat.lean_certificate)

        # 4. Algebra calc certificate + /grade-steps lean_certify=True
        with TestClient(app) as client:
            r = client.post(
                "/grade-steps",
                json={
                    "problem": "Expand (x + 2)*(x - 3)",
                    "student_steps": ["(x + 2)*(x - 3)", "x^2 - 3*x + 2*x - 6", "x^2 - x - 6"],
                    "operation": "algebra",
                    "lean_certify": True,
                },
            )
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertTrue(data["overall_correct"])
            self.assertTrue(data["lean_verified"])
            self.assertIn("calc", data["lean_certificate"])


if __name__ == "__main__":
    unittest.main()
