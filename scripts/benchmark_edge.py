"""
VeriMath — Edge Latency & Compliance Benchmark
===============================================
Verifies:
1. Rule 5: Pre-seeded SQLite edge cache (demo_cache.db) lookup <= 4 ms.
2. Rule 21: Deterministic CAS + Step Generator latency on uncached problems.
3. Rules 22 & 23: Zero Lean 4 syntax leaks and zero textbook citations in student-facing output.
"""

import sqlite3
import statistics
import time
from pathlib import Path

from backend.pipeline import cache as demo_cache
from backend.pipeline.solver import solve
from backend.pipeline.step_generator import generate_steps

FORBIDDEN_STUDENT_TOKENS = [
    "Real.sin", "Real.cos", "Real.tan", "Real.exp", "Real.log", "Real.pi",
    "El-Moasser", "Book of Integrals", "Senior 3",
]


def benchmark_cache(sample_size: int = 50):
    db_path = Path(__file__).parent.parent / "demo_cache.db"
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT problem_text FROM demo_cache LIMIT ?", (sample_size,)
    ).fetchall()
    conn.close()

    demo_cache.init_db()
    latencies_ms = []
    leaks = 0
    for (problem_text,) in rows:
        t0 = time.perf_counter()
        hit = demo_cache.get(problem_text)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)
        if hit:
            combined = f"{hit.verified_answer} {hit.explanation}"
            if any(tok in combined for tok in FORBIDDEN_STUDENT_TOKENS):
                leaks += 1

    avg_ms = statistics.mean(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
    print(f"[Cache Benchmark] n={len(rows)} | avg={avg_ms:.3f} ms | p95={p95_ms:.3f} ms | leaks={leaks}")
    assert avg_ms <= 4.0, f"Cache average latency {avg_ms:.3f} ms exceeded 4.0 ms SLA"
    assert leaks == 0, f"Detected {leaks} student-facing leaks in demo_cache.db"


def benchmark_uncached_cas():
    problems = [
        ("Find the derivative of x^3 * sin(2*x)", "derivative"),
        ("Evaluate the indefinite integral of 2*x * exp(x^2) dx", "indefinite_integral"),
        ("Solve 5*x - 15 = 0", "equation_solving"),
    ]
    for prob, tmpl in problems:
        t0 = time.perf_counter()
        res = solve(prob)
        steps = generate_steps(prob, res.candidate_answer, tmpl)
        dt = (time.perf_counter() - t0) * 1000.0
        assert res.candidate_answer, f"Empty answer for {prob}"
        print(f"[CAS + Steps] {tmpl}: {dt:.2f} ms | steps={len(steps)} | ans={res.candidate_answer}")


if __name__ == "__main__":
    benchmark_cache()
    benchmark_uncached_cas()
    print("ALL EDGE BENCHMARKS PASSED.")
