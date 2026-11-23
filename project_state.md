# VeriMath — Project State & Operating Protocol

> **Active Memory Protocol**: This living document stores the comprehensive operating rules, mathematical standards, engineering constraints, and live task queue for the VeriMath project.

---

## 1. Operating Rules (Strict Memory Protocol)

### Core Operational Rules
* **Rule 1 (Mandatory Context Refresh)**: Before initiating any task, silently read [`project_state.md`](file:///d:/New%20folder%20(3)/verimath/project_state.md) to load all mathematical constraints, formatting standards, and the active task queue.
* **Rule 2 (Task Queue & Protocol Maintenance)**: Upon completing any task, immediately update [`project_state.md`](file:///d:/New%20folder%20(3)/verimath/project_state.md): delete the completed item, shift remaining tasks up, and document any new architectural decisions or rules discussed.
* **Rule 3 (Cumulative Progress Tracking)**: Maintain [`progress_tracker.md`](file:///d:/New%20folder%20(3)/verimath/progress_tracker.md) as the immutable historical log from project inception to date, appending each completed milestone.
* **Rule 4 (Strict Open-Response Standard)**: Never present, accept, or generate multiple-choice options (A, B, C, D). All mathematical problems must be formulated and evaluated in open-response format.
* **Rule 5 (Edge Hardware SLA)**: All core components must run offline on edge hardware (Raspberry Pi 4 CPU, 4 threads, 2048 ctx) with $\le 4\text{ ms}$ latency via the pre-seeded SQLite cache ([`demo_cache.db`](file:///d:/New%20folder%20(3)/verimath/demo_cache.db)).
* **Rule 6 (Zero Regression Guarantee)**: The complete test suite ([`backend/tests/`](file:///d:/New%20folder%20(3)/verimath/backend/tests/)) must maintain a 100.0% pass rate at all times (currently 266/266 passing tests).

### Mathematical & Engineering Rules
* **Rule 7 (Neurosymbolic Division of Labor)**: Neural models (Qwen2.5-1.5B) parse informal English intent and populate structured JSON slots; the deterministic CAS engine (SymPy) solves and simplifies; Lean 4 mechanically checks proofs. Never rely on probabilistic LLMs for multi-step algebraic or coefficient calculations.
* **Rule 8 (Two-Way Fine-Tuning Methodology)**: Every concept must be trained via:
  - **Way 1 (Authentic Pedagogy)**: Exact textbook wording, authentic steps, and diverse conversational prompts.
  - **Way 2 (Parametric Generalization)**: Programmatically varied coefficients, function families, powers, and roots.
* **Rule 9 (Hallucination Rejection / Guardrail)**: If required parameters are missing from a problem prompt, the assistant must return `"is_complete": false` rather than inventing numerical values.
* **Rule 10 (Balanced-Parenthesis Coordinate Extraction)**: When extracting evaluation points $(x_0, y_0)$ from prompts, always count parenthesis nesting depth to correctly parse nested function values (e.g., $(4, \sqrt{19})$, $(1, 6e^{-2})$, $(1, \sin(e-2))$).
* **Rule 11 (Point-Free Tangent Slopes)**: Prompts phrasing *"Find the slope of the tangent for the function f(x) = ..."* without an evaluation point $x_0$ represent queries for the derivative function $f'(x)$.
* **Rule 12 (Difference Quotient Limit Definition)**: Limits of difference quotients $\lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ must map to the derivative of the depth-0 subtrahend $f(x)$ to avoid limit branch-cut timeouts.
* **Rule 13 (Constant & Pythagorean Identity Traps)**: Expressions evaluating to constants (e.g. $\sin^2 x + \cos^2 x = 1$, $\cos(3x-\pi) + \cos(3x) = 0$, $10^{10 \log(u)} = u^{10}$) must be recognized and simplified to $f'(x) = 0$ via CAS.
* **Rule 14 (Universal Square Bracket Normalization)**: Normalize `[` `]` to `(` `)` across all mathematical parsing to ensure standard SymPy syntax.
* **Rule 15 (Faithfulness Threshold)**: Statement Judge back-translations must meet a minimum token set similarity threshold ($\ge 0.38$) against the user's natural language prompt.
* **Rule 16 (Cache Key Normalization)**: Always lowercase, collapse spaces, strip terminal punctuation (`?`, `.`, `!`), and convert Unicode minus/powers (`−` $\to$ `-`, `²` $\to$ `^2`) before checking or writing to [`demo_cache.db`](file:///d:/New%20folder%20(3)/verimath/demo_cache.db).
* **Rule 17 (Implicit Multiplication in CAS Parsing)**: Multi-letter variables (e.g. $xy, yx, xy^2$) in implicitly defined equations must be parsed with `sympy.parsing.sympy_parser.parse_expr` and `standard_transformations + (implicit_multiplication_application, convert_xor)` to avoid creating composite atom symbols like `xy` that zero out partial derivatives $\frac{\partial}{\partial y}$.
* **Rule 18 (Euler Exponential in Lean 4 Back-Conversion)**: Expressions in Lean statements containing $e^{\dots}$ must be normalized to $\exp(\dots)$ in SymPy verifier pipelines to ensure Euler's number $E$ is evaluated rather than treated as a free variable.
* **Rule 19 (Functional Substitution & Identity Resolution)**: Multi-function declarations (`Given g(x) = ..., f(x) = ... find derivative of h(x) = ...`) must extract symbol definitions and evaluate substituted expressions prior to differentiation to capture constant identities (e.g. $\arcsin x + \arccos x = \pi/2 \implies h'(x) = 1$).
* **Rule 20 (Radial Fraction Point Evaluation in Tangent Slopes)**: Evaluation coordinates for point-slopes at angles must parse exact fractional multiples of $\pi$ ($\pi/6, \pi/4, \pi/3, \pi/2$) prior to bare $\pi$, and non-differentiable / pole points (e.g. $y = \frac{6\sin x}{x^2}$ at $x = 0$) must cleanly evaluate to `undefined` rather than propagating `nan`.
* **Rule 21 (Deterministic Pedagogical Step Generation)**: Step-by-step derivations and pedagogical explanations must be generated symbolically via [`backend/pipeline/step_generator.py`](file:///d:/New%20folder%20(3)/verimath/backend/pipeline/step_generator.py) grounded in authentic textbook standards (El-Moasser: General Mathematics, Unit 3). Never rely on LLM decoding on edge hardware for intermediate step algebra to maintain sub-15ms latency, zero hallucination, and 4GB RAM safety.
* **Rule 22 (Lean 4 Formal Syntax Isolation & KaTeX Safety)**: Lean 4 formal syntax (`Real.sin x`, `Real.cos x`, `Real.exp x`, `Real.pi`, etc.) belongs exclusively in the Lean verification panel (`lean_statement`). The student-facing answer fields (`verified_answer`, `candidate_answer`) must strictly use standard human mathematical notation (`sin(x)`, `cos(x)`, `e^x`, `\pi`). `to_latex()` must sanitize any Lean namespaces prior to SymPy parsing, and must never wrap expressions containing mathematical operators (`^`, `*`, `+`, `-`, `/`, `=`) inside `\text{...}`, preventing KaTeX text-mode fatal errors on exponents.
* **Rule 23 (Pedagogical Explanation Cleanliness & Citation Prohibition)**: Student-facing explanations and derivation step texts must strictly avoid external textbook names, curriculum titles, authors, or page numbers (e.g., no "According to El-Moasser", "The Book of Integrals, p. 58", "Senior 3 Calculus"). Mathematical derivations must stand on their own objective logical grounds using standard universal mathematical terminology (Power Rule, Product Rule, Quotient Rule, U-Substitution, Polynomial Long Division, Integration by Parts, etc.).

---

## 2. Formatting & Engineering Standards

### A. Fine-Tune Dataset Schema (ChatML JSON)
File: [`verimath_finetune_dataset.jsonl`](file:///d:/New%20folder%20(3)/verimath/verimath_finetune_dataset.jsonl)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are VeriMath, a rigorous mathematical AI assistant. Solve the user's math problem with complete formal verification. Output your final response as valid JSON matching the VeriMath schema."
    },
    {
      "role": "user",
      "content": "<Problem Statement in Natural Language>"
    },
    {
      "role": "assistant",
      "content": "{\n  \"template\": \"<template_id>\",\n  \"is_complete\": true,\n  \"parameters\": {\n    \"expr\": \"...\",\n    \"answer\": \"...\"\n  },\n  \"lean_formalization\": \"example (x : ℝ) : deriv (fun x => ...) x = ... := by ...\",\n  \"confidence\": \"high\"\n}"
    }
  ]
}
```

### B. Lean 4 Formalization Standards
- Real type annotations on domain variables: `(x : ℝ)`, `(t : ℝ)`.
- Mathlib standard real namespaces: `Real.sin`, `Real.cos`, `Real.tan`, `Real.exp`, `Real.log`, `Real.sqrt`, `Real.arctan`, `Real.arcsin`, `Real.arccos`.
- Standard tactics: `simp [...]`, `ring`, `norm_num`, `linarith`, `nlinarith`.

### C. Edge Cache Schema
SQLite Database: [`demo_cache.db`](file:///d:/New%20folder%20(3)/verimath/demo_cache.db)
Table: `demo_cache`
Columns: `(problem_key, problem_text, template_id, verified_answer, lean_statement, back_translation, explanation, elapsed_ms, cached_at)`.

---

## 3. System Metrics & Telemetry

| Metric | Current Status | Notes |
| :--- | :--- | :--- |
| **Registered Templates** | **244** | Complete calculus coverage: arithmetic, algebra, related rates, optimization, integrals, derivatives, higher-order, kinematics, implicit differentiation, geometric volume rates |
| **Fine-Tuning Dataset Size** | **10,465 entries** | 244/244 templates (100% coverage) + 154 generalized higher-math & non-math guardrail entries, 0 `sorry` placeholders |
| **Edge Cache Entries** | **1,828 entries** | Sanitized of all Lean syntax in student fields, pre-seeded for $\le 4\text{ ms}$ offline edge demo |
| **Pedagogical Step Engine** | **Remediated & Verified (100% Complete)** | Deterministic CAS step generator + interactive React & Lean 4 `calc` student derivation grader (`POST /grade-steps` with implicit domain / dropped-root diagnostics), automated Rule 22 & 23 safety linting, zero KaTeX errors |
| **External Connectors & Vision OCR** | **Active (RapidOCR + Tesseract 5 + Camera)** | WebRTC browser camera & Pi CSI/USB hardware capture (`/scan/ocr`, `/scan/solve`, `/scan/grade-steps`, `/scan/hardware-capture`), RapidOCR PP-OCRv4 + Tesseract 5 + TrOCR/Pix2Tex/Gemini Vision adapters, MQTT v2 / Firestore sync |
| **Unit Test Suite** | **273 / 273 passing** | 100.0% pass rate (0 failures, 0 regressions across all 16 test suites including camera/OCR scan pipelines, Lean 4 `calc` derivation certification, Lean-certified counterexamples, and AST anti-cheating guards) |
| **Verification Architecture** | **Bidirectional Zero-Trust + AST Isomorphism** | True claims: AST Faithfulness Audit + Lean 4.36.0 proof; False claims: Minimal Witness Counterexample Synthesis + Lean 4.36.0 disproof theorem |
| **Textbook Verification** | **100.0% Verified** | Cross-verified against official Answer Keys (El-Moasser & *Book of Integrals*) with zero discrepancies |
| **Target Hardware** | **Raspberry Pi 4 (8 GB RAM)** | 4 threads, 4096 context, Qwen2.5-1.5B or 3B Q4_K_M GGUF + Lean 4 Mathlib |

---

## 4. Live Task Tracker

### Active / Next Task
- [ ] **Colab Fine-Tuning Execution**: Execute fine-tuning notebook ([`scripts/VeriMath_Colab_FineTuning.ipynb`](file:///d:/New%20folder%20(3)/verimath/scripts/VeriMath_Colab_FineTuning.ipynb)) on Google Colab GPU using Unsloth LoRA (4-bit QLoRA, rank 16, alpha 32) on the completed 9,888 verified ChatML dataset.

### Queued Tasks (Shift Up as Work Progresses)
1. **Quantization & GGUF Export**: Export fine-tuned model checkpoint into Q4_K_M GGUF format (`models/qwen2_5_1_5b_verimath_pi4-q4_k_m.gguf`).
2. **Edge Hardware Benchmarking on Raspberry Pi 4**: Transfer model GGUF and pre-seeded SQLite cache ([`demo_cache.db`](file:///d:/New%20folder%20(3)/verimath/demo_cache.db)) to physical Raspberry Pi 4 CPU, measure inference tokens/sec and cache lookup latency ($\le 4\text{ ms}$).
3. **End-to-End Production Verification**: Verify offline Lean 4 type-checking and automated fallback on live hardware.
