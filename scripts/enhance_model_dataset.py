"""
VeriMath — Model Fine-Tuning Dataset Mathematical Enhancement
==============================================================
Appends verified, `sorry`-free generalized out-of-domain autoformalization
training pairs to `verimath_finetune_dataset.jsonl` so the fine-tuned LLM
learns to synthesize standalone Lean 4 theorems for higher-level mathematics
(linear systems, 3x3 determinants, 3D cross products, vector norms, gradients,
partial derivatives, trig identities, symbolic inequalities, divisibility,
GCD/LCM/factorials, and non-math rejection).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from backend.pipeline.generalized_prover import _GENERALIZED_SYSTEM_PROMPT, propose_generalized

DATASET_PATH = Path(__file__).parent.parent / "verimath_finetune_dataset.jsonl"

GENERALIZED_PROMPTS = [
    # Linear Systems
    "Solve the system: 2x + 3y = 13, x - y = -1",
    "Solve the system of equations: 3x + 2y = 12, 5x - 2y = 4",
    "Solve the system: x + y + z = 6, 2x - y + z = 3, x + 2y - z = 2",
    "Solve the system: 4x - y = 7, 2x + 3y = 14",
    # 3x3 & 2x2 Determinants + Vectors
    "Find the determinant of [[1, 2, 3], [0, 4, 5], [1, 0, 6]]",
    "Find the determinant of [[2, -1, 0], [3, 4, 2], [1, 0, 5]]",
    "Find the determinant of [[5, 3], [2, 7]]",
    "Find the cross product of (1, 2, 3) and (4, 5, 6)",
    "Find the cross product of (2, -1, 3) and (1, 4, -2)",
    "Find the dot product of (3, 4, -1) and (2, 5, 6)",
    "Find the magnitude of the vector (3, 4, 12)",
    "Find the norm of vector (6, 8)",
    # Multivariable Gradients & Partial Derivatives
    "Find the gradient of x^2*y + y*z^2",
    "Find the gradient of 3*x*y + x^2 - 2*z",
    "Find the partial derivative of x^3*y^2 + sin(y) with respect to x",
    "Find the partial derivative of x^2*y + exp(y) with respect to y",
    # Trigonometric Identities
    "Prove that sin(x)^2 + cos(x)^2 = 1",
    "Prove that cos(2*x) = cos(x)^2 - sin(x)^2",
    "Prove that sin(2*x) = 2*sin(x)*cos(x)",
    # Symbolic & Numeric Inequalities
    "Prove that x^2 + y^2 >= 2*x*y",
    "Prove that x^2 + 1 > 0",
    "Prove that (a - b)^2 >= 0",
    "Prove that 2^10 > 1000",
    "Prove that 3/7 < 1/2",
    # Polynomial Identities
    "Prove that (a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3",
    "Prove that (x - y)*(x^2 + x*y + y^2) = x^3 - y^3",
    "Prove that (x + y)*(x - y) = x^2 - y^2",
    # Number Theory, Combinatorics & Finite Sums
    "Does 7 divide 1001?",
    "Does 11 divide 1331?",
    "Find 1234 mod 17",
    "Find gcd(84, 126)",
    "Find lcm(12, 18)",
    "Find 7!",
    "Find factorial of 6",
    "Find choose(10, 3)",
    "Find the sum of the first 10 positive integers",
    "Find the sum of the first 25 positive integers",
    # Complex Arithmetic
    "Compute (3 + 4i) * (1 - 2i)",
    "Simplify (5 - 2i) + (-1 + 7i)",
    "Prove that (2 + 3i) * (2 - 3i) = 13",
    # Logarithmic Laws & Exponential Equations
    "Prove that log(a * b) = log(a) + log(b)",
    "Prove that log(a / b) = log(a) - log(b)",
    "Solve 2^x = 32",
    "Solve 3^x = 81",
    "Solve 2^x = 1/8",
    "Solve 4^x = 8",
    # Permutations, Sum of Squares, Matrix Product & Primality
    "Find P(6, 2)",
    "Find the sum of squares of the first 5 positive integers",
    "Multiply matrices [[1, 2], [3, 4]] and [[5, 6], [7, 8]]",
    "Prove that 97 is prime",
]

NON_MATH_PROMPTS = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "Explain photosynthesis in simple terms.",
    "Recommend a good movie to watch tonight.",
]


def main():
    existing = set()
    lines = DATASET_PATH.read_text(encoding="utf-8").splitlines()
    for l in lines:
        obj = json.loads(l)
        existing.add((obj["messages"][0]["content"], obj["messages"][1]["content"]))

    added = 0
    new_rows = []

    variations = [
        "",
        "With formal verification: ",
        "Solve and prove in Lean 4: ",
    ]

    for base_p in GENERALIZED_PROMPTS:
        prop = propose_generalized(base_p)
        assert prop.is_math_problem and prop.lean_statement and "sorry" not in prop.lean_statement, f"Bad proposal for {base_p}"
        ans_json = json.dumps(
            {
                "is_math": True,
                "answer": prop.candidate_answer,
                "lean_statement": prop.lean_statement,
                "back_translation": prop.back_translation,
                "reasoning": prop.reasoning,
            },
            indent=2,
        )
        for pref in variations:
            user_p = f"{pref}{base_p}".strip()
            key = (_GENERALIZED_SYSTEM_PROMPT, user_p)
            if key not in existing:
                existing.add(key)
                new_rows.append(
                    {
                        "messages": [
                            {"role": "system", "content": _GENERALIZED_SYSTEM_PROMPT},
                            {"role": "user", "content": user_p},
                            {"role": "assistant", "content": ans_json},
                        ]
                    }
                )
                added += 1

    for nm in NON_MATH_PROMPTS:
        key = (_GENERALIZED_SYSTEM_PROMPT, nm)
        if key not in existing:
            existing.add(key)
            new_rows.append(
                {
                    "messages": [
                        {"role": "system", "content": _GENERALIZED_SYSTEM_PROMPT},
                        {"role": "user", "content": nm},
                        {"role": "assistant", "content": json.dumps({"is_math": False}, indent=2)},
                    ]
                }
            )
            added += 1

    if new_rows:
        with DATASET_PATH.open("a", encoding="utf-8") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")

    print(f"Enhanced dataset: added {added} generalized autoformalization rows (total={len(lines) + added}).")


if __name__ == "__main__":
    main()
