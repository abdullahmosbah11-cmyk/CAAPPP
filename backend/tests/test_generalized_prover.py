"""
Unit & Integration Tests for VeriMath Generalized Out-of-Domain Autoformalizer & Prover
======================================================================================
Verifies:
1. Triggers ONLY when problem is outside the 244 trained templates (`generalized_mode=True`).
2. Proposes and formally verifies unseen out-of-domain problems (identities, inequalities,
   divisibility, GCD, partial derivatives, determinants, finite sums).
3. Strictly REJECTS false out-of-domain claims (`status == "REJECTED"`).
4. Keeps in-domain problems on the standard template pipeline (`generalized_mode=False`).
5. Rejects non-mathematical prompts as `UNSUPPORTED`.
"""

import unittest
from fastapi.testclient import TestClient

from backend.main import app


class TestGeneralizedOutOfDomainProver(unittest.TestCase):
    def test_in_domain_does_not_use_generalized_mode(self):
        with TestClient(app) as client:
            r = client.post("/solve", json={"problem": "Integrate: x^2 * sqrt(x^3 + 4) dx"})
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["status"], "VERIFIED")
            self.assertFalse(data["generalized_mode"])

    def test_generalized_true_out_of_domain_problems(self):
        cases = [
            ("Prove that (a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3", "generalized_identity"),
            ("Prove that 2^10 > 1000", "generalized_inequality"),
            ("Does 7 divide 1001?", "generalized_number_theory"),
            ("Find gcd(84, 126)", "generalized_number_theory"),
            ("Find lcm(12, 18)", "generalized_number_theory"),
            ("Find 7!", "generalized_series_combinatorics"),
            ("Find the partial derivative of x^2*y + y^3 with respect to x", "generalized_partial_derivative"),
            ("Find the gradient of x^2*y + y*z^2", "generalized_gradient"),
            ("Find the determinant of [[3, 4], [2, 5]]", "generalized_linear_algebra"),
            ("Find the determinant of [[1, 2, 3], [0, 4, 5], [1, 0, 6]]", "generalized_linear_algebra"),
            ("Find the cross product of (1, 2, 3) and (4, 5, 6)", "generalized_linear_algebra"),
            ("Find the magnitude of the vector (3, 4, 12)", "generalized_linear_algebra"),
            ("Solve the system: 2x + 3y = 13, x - y = -1", "generalized_linear_system"),
            ("Find the sum of the first 10 positive integers", "generalized_series_combinatorics"),
            ("Prove that sin(x)^2 + cos(x)^2 = 1", "generalized_trig_identity"),
            ("Prove that sin(2*x) = 2*sin(x)*cos(x)", "generalized_trig_identity"),
            ("Prove that x^2 + y^2 >= 2*x*y", "generalized_symbolic_inequality"),
            ("Prove that x^2 + 1 > 0", "generalized_symbolic_inequality"),
            ("Compute (3 + 4i) * (1 - 2i)", "generalized_complex"),
            ("Prove that (2 + 3i) * (2 - 3i) = 13", "generalized_complex"),
            ("Prove that log(a * b) = log(a) + log(b)", "generalized_log_identity"),
            ("Prove that log(a / b) = log(a) - log(b)", "generalized_log_identity"),
            ("Solve 2^x = 32", "generalized_exponential_equation"),
            ("Solve 2^x = 1/8", "generalized_exponential_equation"),
            ("Solve 4^x = 8", "generalized_exponential_equation"),
            ("Find P(6, 2)", "generalized_series_combinatorics"),
            ("Find the sum of squares of the first 5 positive integers", "generalized_series_combinatorics"),
            ("Multiply matrices [[1, 2], [3, 4]] and [[5, 6], [7, 8]]", "generalized_linear_algebra"),
            ("Prove that 97 is prime", "generalized_number_theory"),
        ]
        with TestClient(app) as client:
            for prob, expected_kind in cases:
                r = client.post("/solve", json={"problem": prob})
                self.assertEqual(r.status_code, 200, f"HTTP failure for: {prob}")
                data = r.json()
                self.assertEqual(data["status"], "VERIFIED", f"Expected VERIFIED for '{prob}', got {data}")
                self.assertTrue(data["generalized_mode"])
                self.assertTrue(data.get("ast_verified", False))
                self.assertTrue(data.get("ast_checks"))
                self.assertEqual(data["template_id"], expected_kind)
                self.assertTrue(data["explanation"])
                self.assertTrue(data["mental_model"])
                self.assertTrue(data["pitfall_tip"])

    def test_generalized_strictly_rejects_false_claims(self):
        false_claims = [
            "Prove that (a + b)^2 = a^2 + b^2",
            "Prove that 2^10 < 1000",
            "Prove that 7 divides 100",
            "Prove that sin(x)^2 + cos(x)^2 = 2",
            "Prove that sin(x + y) = sin(x) + sin(y)",
            "Prove that x^2 + y^2 < 2*x*y",
            "Prove that (3 + 4i) * (1 - 2i) = 11 + 2i",
            "Prove that log(a + b) = log(a) + log(b)",
            "Prove that 91 is prime",
            "Prove that 5! = 100",
            "Prove that gcd(12, 18) = 4",
        ]
        with TestClient(app) as client:
            for claim in false_claims:
                r = client.post("/solve", json={"problem": claim})
                self.assertEqual(r.status_code, 200)
                data = r.json()
                self.assertEqual(data["status"], "REJECTED", f"False claim '{claim}' was not REJECTED: {data}")
                self.assertTrue(data["generalized_mode"])
                self.assertIsNone(data["verified_answer"])
                ce = data.get("counterexample")
                self.assertIsNotNone(ce, f"Expected synthesized counterexample for '{claim}'")
                self.assertTrue(ce["found"], f"Counterexample not found for '{claim}'")
                self.assertTrue(ce["lean_verified"], f"Counterexample not Lean-verified for '{claim}': {ce}")
                self.assertTrue(ce["witness_display"])
                self.assertTrue(ce["disproof_lean_statement"])

    def test_non_math_prompt_returns_unsupported(self):
        with TestClient(app) as client:
            r = client.post("/solve", json={"problem": "What is the capital of France?"})
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["status"], "UNSUPPORTED")
            self.assertFalse(data["generalized_mode"])


if __name__ == "__main__":
    unittest.main()
