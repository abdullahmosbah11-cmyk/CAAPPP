# VeriMath 🧮✅

**A Lean-Verified AI Math Tutor for Reliable Educational Assistance**

VeriMath is an AI-powered math tutoring system that uses **Lean 4** as an independent correctness oracle. The LLM proposes answers. Lean proves them. Students only see verified answers.

---

## How It Works

```
Student types problem → LLM solves it → LLM formalizes into Lean 4 template
→ Rule-based judge checks faithfulness → Lean 4 verifies the statement
→ ✅ Verified: student sees answer + intuitive explanation
→ ❌ Rejected: student sees "this answer could not be verified"
→ 🚫 Unsupported: "this problem type is outside the verified domain"
```

## Hardware

- **Raspberry Pi 4 (8 GB RAM)** — runs Lean 4 + Python backend + Qwen2.5-1.5B/3B Q4_K_M + MQTT broker
- **HDMI Monitor + USB keyboard/mouse** — student interface
- **Internet** (optional) — for Firebase Firestore telemetry sync

## Quick Start (Development on Laptop)

```bash
# 1. Clone and set up Python environment
git clone <repo>
cd verimath
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Copy .env.example to .env and fill in your keys
cp .env.example .env

# 3. Install Lean 4 (via elan)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# 4. Build Mathlib (DO THIS ON LAPTOP, NOT PI — takes 30-90 min first time)
cd lean/VeriMath
lake update && lake build
cd ../..

# 5. Download Phi-3 Mini model
python scripts/download_model.py

# 6. Start backend
uvicorn backend.main:app --reload --port 8000

# 7. Start frontend (in a new terminal)
cd frontend && npm install && npm start

# 8. Open http://localhost:3000
```

## Project Structure

```
verimath/
├── backend/          # Python FastAPI backend + pipeline modules
├── frontend/         # React student UI + teacher dashboard
├── lean/             # Lean 4 project (VeriMath package)
├── scripts/          # Utility scripts (warmup, dataset, eval)
├── deployment/       # Raspberry Pi install scripts + systemd units
└── backend/tests/dataset/  # 200-problem dataset (train + test)
```

## Supported Problem Types

| Type | Example | Lean Tactic |
|------|---------|-------------|
| Arithmetic | "What is 47 × 89 + 123?" | `norm_num` |
| Algebraic simplification | "Simplify (x+1)² − x²" | `ring` |
| Equation solving | "Solve 3x + 7 = 22" | `omega` |
| Derivative | "Find the derivative of x² sin(x)" | `simp` + `ring` |
| Definite integral | "Evaluate ∫₀¹ x² dx" | `norm_num` |
| Number theory | "Is 97 prime?" | `decide` |
| Geometry (distance) | "Distance between (1,2) and (4,6)?" | `norm_num` |

## Design Requirements

| Requirement | Target |
|---|---|
| Response time | ≤ 30 seconds |
| System accuracy (held-out test) | ≥ 80% |
| Hallucination rejection rate | ≥ 90% |
| Formalization fidelity | ≥ 85% |
| Battery life (Pi 4 + power bank) | ≥ 2 hours |
| Student comprehension | ≥ 75% agreement |

## Guarantee

> Any answer shown to a student has been formally verified by Lean 4's kernel — a proof assistant with a mathematically certified core. The AI proposes; Lean decides.
