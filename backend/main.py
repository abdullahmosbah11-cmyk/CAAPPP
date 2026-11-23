"""
VeriMath — FastAPI Main Application
=====================================
The central orchestrator that runs the full 6-module pipeline for each
student problem and exposes REST endpoints for the React frontend.

Pipeline stages (per problem):
  1. Classify → detect template type (or refuse if unsupported)
  2. Solve    → LLM produces candidate answer
  3. Formalize → LLM fills template slots
  4. Judge    → rule-based faithfulness check
  5. Verify   → Lean 4 proves or rejects the statement
  6. Explain  → LLM generates intuitive explanation
  7. Publish  → MQTT telemetry event

Cache: If problem is in demo cache, steps 2–6 are skipped.

Key guarantee:
  Students ONLY see answers that passed step 5 (Lean verification).
  Steps 1–4 and 6 use LLMs but have no bearing on mathematical correctness.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Pipeline modules
from backend.templates.registry import classify_problem, TEMPLATES
from backend.pipeline.solver import solve
from backend.pipeline.formalizer import formalize
from backend.pipeline.statement_judge import judge
from backend.pipeline.lean_runner import verify_async, warmup, LeanVerdict
from backend.pipeline.explainer import explain, build_intuitive_guide
from backend.pipeline import cache as demo_cache
from backend.pipeline.step_generator import generate_steps

# IoT
from backend.iot.mqtt_publisher import PipelineEvent, publish, disconnect

# Dashboard
from backend.dashboard.router import router as dashboard_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.getenv("MAX_FORMALIZATION_RETRIES", "3"))


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle events."""
    logger.info("🚀 VeriMath backend starting up...")

    # Initialize demo cache DB
    demo_cache.init_db()
    logger.info("Demo cache initialized.")

    try:
        lean_ok = await warmup()
        if not lean_ok:
            logger.warning(
                "⚠️  Lean warmup failed or Lean is not installed locally. "
                "Backend will run with mock/demo mode for development."
            )
    except Exception as e:
        logger.warning(f"⚠️  Lean warmup skipped: {e}")

    logger.info("✅ VeriMath backend ready.")

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down VeriMath backend...")
    disconnect()  # MQTT cleanup


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VeriMath API",
    description="Lean-verified AI math tutor backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local edge computing and local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount teacher dashboard router
app.include_router(dashboard_router)

from fastapi.responses import RedirectResponse

# Serve React build (production)
FRONTEND_BUILD = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
STATIC_DIR = os.path.join(FRONTEND_BUILD, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.isdir(FRONTEND_BUILD):
    app.mount("/app", StaticFiles(directory=FRONTEND_BUILD, html=True), name="frontend")

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/app/")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    problem: str
    student_id: str = "anonymous"


class SolveResponse(BaseModel):
    status: str                        # VERIFIED | REJECTED | UNSUPPORTED | ERROR
    verified_answer: Optional[str] = None
    latex_answer: Optional[str] = None
    lean_statement: Optional[str] = None
    back_translation: Optional[str] = None
    explanation: Optional[str] = None
    mental_model: Optional[str] = None
    pitfall_tip: Optional[str] = None
    steps: Optional[List[Dict[str, Any]]] = None
    rejection_reason: Optional[str] = None
    template_id: Optional[str] = None
    elapsed_ms: int = 0
    from_cache: bool = False
    generalized_mode: bool = False
    counterexample: Optional[Dict[str, Any]] = None
    ast_verified: bool = True
    ast_checks: Optional[List[str]] = None


def _clean_lean_syntax(s: Optional[str]) -> Optional[str]:
    """Strip Lean 4 namespaces and function application syntax into standard math notation."""
    if not s or "Real." not in s:
        return s
    # Convert Real.func arg (without parens) -> func(arg)
    # e.g. Real.sin x -> sin(x), Real.cos x -> cos(x), Real.exp x -> exp(x), Real.sqrt 2 -> sqrt(2)
    res = re.sub(r"\bReal\.([a-zA-Z]+)\s+([a-zA-Z0-9]+)\b", r"\1(\2)", s)
    # Convert remaining Real.identifier -> identifier (e.g. Real.pi -> pi, Real.sin(x) -> sin(x))
    res = re.sub(r"\bReal\.([a-zA-Z]+)\b", r"\1", res)
    return res


def to_latex(answer: Optional[str]) -> Optional[str]:
    """Convert a mathematical answer string to clean, readable LaTeX."""
    if not answer:
        return None
    try:
        raw_res = _to_latex_inner(str(answer).strip())
        if not raw_res:
            return raw_res
        # Clean any unit exponents inside \text{...} so KaTeX does not trigger parse errors:
        # e.g. \text{ cm^2} -> \text{ cm}^2, \text{cm^3} -> \text{cm}^3
        cleaned = re.sub(r"\\text\{\s*([a-zA-Z]+)\^(\d+)\s*\}", r"\\text{ \1}^{\2}", raw_res)
        cleaned = re.sub(r"\\text\{\s*([a-zA-Z]+)\s*\^(\d+)\s*\}", r"\\text{ \1}^{\2}", cleaned)
        return cleaned
    except Exception as e:
        logger.warning(f"to_latex conversion error for '{answer}': {e}")
        return str(answer).strip()


def _to_latex_inner(ans_str: str) -> str:
    ans_str = _clean_lean_syntax(ans_str) or ans_str
    # 0a. Related rates: dV/dt, dA/dt, dy/dt, dh/dt, ds/dt
    m_rate = re.match(r"^d([VAyhs])\/dt\s*=\s*(.+?)(?:\s*\((?:~|\approx)?\s*(.+?)\))?$", ans_str)
    if m_rate:
        var, val, approx = m_rate.groups()
        val_clean = val.replace("*pi", r"\pi").replace("*", r"\cdot ")
        val_clean = re.sub(r"([a-zA-Z]+)\^(\d+)\/([a-zA-Z]+)", r"\\text{ \1}^{\2}/\\text{\3}", val_clean)
        approx_tex = f" \\approx {approx}" if approx else ""
        return rf"\frac{{d{var}}}{{dt}} = {val_clean}{approx_tex}"

    # 0b. Implicit derivative: dy/dx = ...
    m_dydx = re.match(r"^dy\/dx\s*=\s*(.+)$", ans_str)
    if m_dydx:
        val = m_dydx.group(1).strip()
        m_frac = re.match(r"^\((.+?)\)\/\((.+?)\)$", val)
        if m_frac:
            num, den = m_frac.groups()
            num_tex = num.replace("*", "").replace("^2", "^{2}").replace("^3", "^{3}")
            den_tex = den.replace("*", "").replace("^2", "^{2}").replace("^3", "^{3}")
            return rf"\frac{{dy}}{{dx}} = \frac{{{num_tex}}}{{{den_tex}}}"
        return rf"\frac{{dy}}{{dx}} = {val}"

    # 0c. Symbolic inscribed cylinder: r = (2/3)R, h = (1/3)H, maximum volume = (4/27)*pi*R^2*H
    m_sym_cyl = re.match(r"^r\s*=\s*\(2\/3\)\*?R,\s*h\s*=\s*\(1\/3\)\*?H,\s*maximum\s+volume\s*=\s*\(4\/27\)\*?pi\*?R\^2\*?H$", ans_str, re.I)
    if m_sym_cyl:
        return r"r = \frac{2}{3}R, \quad h = \frac{1}{3}H, \quad V_{\max} = \frac{4}{27}\pi R^2 H"

    # 0d. Riverbank pasture fencing: 30 m by 60 m, max area = 1800 m^2
    m_river = re.match(r"^(\d+(?:\.\d+)?)\s*m\s+by\s+(\d+(?:\.\d+)?)\s*m,\s*max(?:imum)?\s+area\s*=\s*(\d+(?:\.\d+)?)\s*m\^?2$", ans_str, re.I)
    if m_river:
        x_v, y_v, a_v = m_river.groups()
        return rf"{x_v}\text{{ m}} \times {y_v}\text{{ m}}, \quad A_{{\max}} = {a_v}\text{{ m}}^2"

    # 0e. Partition fence cost: x = ... m (~... m), y = ... m (~... m), minimum cost = $... (~$...)
    m_part = re.match(r"^x\s*=\s*(.+?)\s+([a-zA-Z]+)\s*\((?:~|\approx)?\s*(.+?)\s+[a-zA-Z]+\),\s*y\s*=\s*(.+?)\s+([a-zA-Z]+)\s*\((?:~|\approx)?\s*(.+?)\s+[a-zA-Z]+\),\s*minimum\s+cost\s*=\s*\$(.+?)\s*\((?:~|\approx)?\s*\$(.+?)\)$", ans_str, re.I)
    if m_part:
        x_e, u1, x_v, y_e, u2, y_v, c_e, c_v = m_part.groups()
        x_tex = x_e.replace("sqrt", r"\sqrt").replace("*", "")
        y_tex = y_e.replace("sqrt", r"\sqrt").replace("*", "")
        if "/" in y_tex:
            ynum, yden = y_tex.split("/", 1)
            y_tex = rf"\frac{{{ynum}}}{{{yden}}}"
        c_tex = c_e.replace("sqrt", r"\sqrt").replace("*", "")
        return rf"x = {x_tex}\text{{ {u1}}} \approx {x_v}\text{{ {u1}}}, \quad y = {y_tex}\text{{ {u2}}} \approx {y_v}\text{{ {u2}}}, \quad C_{{\min}} = \${c_tex} \approx \${c_v}"

    # 0f. Open-top box cost: base side x = ... m (~... m), height y = ... m (~... m), minimum cost = $... (~$...)
    m_box_cost = re.match(r"^base\s+side\s+x\s*=\s*(.+?)\s+([a-zA-Z]+)\s*\((?:~|\approx)?\s*(.+?)\s+[a-zA-Z]+\),\s*height\s+y\s*=\s*(.+?)\s+([a-zA-Z]+)\s*\((?:~|\approx)?\s*(.+?)\s+[a-zA-Z]+\),\s*minimum\s+cost\s*=\s*\$(.+?)\s*\((?:~|\approx)?\s*\$(.+?)\)$", ans_str, re.I)
    if m_box_cost:
        x_e, u1, x_v, y_e, u2, y_v, c_e, c_v = m_box_cost.groups()
        x_tex = x_e.replace("^(2/3)", "^{2/3}").replace("^(1/3)", "^{1/3}")
        y_tex = y_e.replace("^(2/3)", "^{2/3}").replace("^(1/3)", "^{1/3}")
        c_tex = c_e.replace("*(2^(1/3))", r"\sqrt[3]{2}").replace("*2^(1/3)", r"\sqrt[3]{2}").replace("2^(1/3)", r"\sqrt[3]{2}")
        return rf"x = {x_tex}\text{{ {u1}}} \approx {x_v}\text{{ {u1}}}, \quad y = {y_tex}\text{{ {u2}}} \approx {y_v}\text{{ {u2}}}, \quad C_{{\min}} = \${c_tex} \approx \${c_v}"

    # Handle multi-part (a) and (b) answers
    if ans_str.startswith("(a)") and "(b)" in ans_str:
        idx_b = ans_str.find("(b)")
        part_a = ans_str[3:idx_b].strip(" ,;")
        part_b = ans_str[idx_b+3:].strip(" ,;")
        latex_a = to_latex(part_a)
        latex_b = to_latex(part_b)
        return rf"\text{{(a) }}{latex_a} \quad \text{{(b) }}{latex_b}"

    # Handle closest point answers with two points: "(1, 1) and (-1, 1) with minimum distance sqrt(5)/2 (~1.1180)"
    m_closest_pts = re.match(r"^\((.+?),\s*(.+?)\)\s+and\s+\((.+?),\s*(.+?)\)\s+with\s+minimum\s+distance\s+([^~]+?)(?:\s*\((?:~|\approx)\s*([0-9\.]+)\))?$", ans_str)
    if m_closest_pts:
        x1, y1, x2, y2, d_val, d_approx = m_closest_pts.groups()
        x1_tex = to_latex(x1)
        y1_tex = to_latex(y1)
        x2_tex = to_latex(x2)
        y2_tex = to_latex(y2)
        d_tex = to_latex(d_val.strip())
        approx_tex = f" \\approx {d_approx}" if d_approx else ""
        return rf"\left({x1_tex}, {y1_tex}\right) \text{{ and }} \left({x2_tex}, {y2_tex}\right), \quad d_{{\min}} = {d_tex}{approx_tex}"

    # Handle closest point optimization answers: "(x, y) with minimum distance d (~val)"
    m_parab = re.match(r"^\((.+?),\s*(.+?)\)\s+with\s+minimum\s+distance\s+(.+?)(?:\s*\((?:~|\approx)?\s*([0-9\.\-]+)\))?$", ans_str)
    if m_parab:
        pt_x, pt_y, dist_str, approx_val = m_parab.groups()
        x_tex = to_latex(pt_x)
        y_tex = to_latex(pt_y)
        d_tex = to_latex(dist_str)
        if approx_val:
            return rf"\left({x_tex}, {y_tex}\right), \quad d_{{\min}} = {d_tex} \approx {approx_val}"
        return rf"\left({x_tex}, {y_tex}\right), \quad d_{{\min}} = {d_tex}"

    # Handle rectangular sheet box volume: "x = ... cm (~... cm), maximum volume V = ... cm^3 (~... cm^3)"
    m_box_vol = re.match(
        r"^x\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*[a-zA-Z]+\),\s*maximum\s+volume\s+V\s*=\s*(.+?)\s*([a-zA-Z0-9\^]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*[a-zA-Z0-9\^]+\)$",
        ans_str,
        re.I
    )
    if m_box_vol:
        x_e, u1, x_v, v_e, u2, v_v = m_box_vol.groups()
        u2_clean = u2.replace("^3", "").replace("^2", "").strip()
        return (
            rf"x = {to_latex(x_e.strip())}\text{{ {u1}}} \approx {x_v}\text{{ {u1}}}, \quad "
            rf"V_{{\max}} = {to_latex(v_e.strip())}\text{{ {u2_clean}}}^3 \approx {v_v}\text{{ {u2_clean}}}^3"
        )

    # Handle Norman window fixed area min perimeter: "r = ... m, h = ... m, minimum perimeter P = ... m (~... m)"
    m_norman_fixed = re.match(
        r"^r\s*=\s*(.+?)\s*([a-zA-Z]+),\s*h\s*=\s*(.+?)\s*([a-zA-Z]+),\s*minimum\s+perimeter\s+P\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*[a-zA-Z]+\)$",
        ans_str,
        re.I
    )
    if m_norman_fixed:
        r_e, u1, h_e, u2, p_e, u3, p_v = m_norman_fixed.groups()
        return (
            rf"r = {to_latex(r_e.strip())}\text{{ {u1}}}, \quad "
            rf"h = {to_latex(h_e.strip())}\text{{ {u2}}}, \quad "
            rf"P_{{\min}} = {to_latex(p_e.strip())}\text{{ {u3}}} \approx {p_v}\text{{ {u3}}}"
        )

    # Handle inscribed inverted cone in cone: "r = ... cm, h = ... cm, maximum volume V = ... cm^3 (~... cm^3)"
    m_cone_inscribed = re.match(
        r"^r\s*=\s*(.+?)\s*([a-zA-Z]+),\s*h\s*=\s*(.+?)\s*([a-zA-Z]+),\s*maximum\s+volume\s+V\s*=\s*(.+?)\s*([a-zA-Z0-9\^]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*[a-zA-Z0-9\^]+\)$",
        ans_str,
        re.I
    )
    if m_cone_inscribed:
        r_e, u1, h_e, u2, v_e, u3, v_v = m_cone_inscribed.groups()
        u3_clean = u3.replace("^3", "").replace("^2", "").strip()
        return (
            rf"r = {to_latex(r_e.strip())}\text{{ {u1}}}, \quad "
            rf"h = {to_latex(h_e.strip())}\text{{ {u2}}}, \quad "
            rf"V_{{\max}} = {to_latex(v_e.strip())}\text{{ {u3_clean}}}^3 \approx {v_v}\text{{ {u3_clean}}}^3"
        )

    # Handle hyperbola tangent min segment length: "(x_0, y_0) = (..., ...) (~(..., ...)), minimum length AB = ..."
    m_hyp_tangent = re.match(
        r"^\(x_0,\s*y_0\)\s*=\s*\((.+?),\s*(.+?)\)\s*\((?:~|\approx)?\s*\((.+?),\s*(.+?)\)\),\s*minimum\s+length\s+AB\s*=\s*(.+?)$",
        ans_str,
        re.I
    )
    if m_hyp_tangent:
        x0_e, y0_e, x0_v, y0_v, l_e = m_hyp_tangent.groups()
        return (
            rf"(x_0, y_0) = \left({to_latex(x0_e.strip())}, {to_latex(y0_e.strip())}\right) \approx ({x0_v}, {y0_v}), \quad "
            rf"L_{{\min}} = {to_latex(l_e.strip())}"
        )

    # Handle cable pipeline cost minimization: "x = ... km, minimum cost = $..."
    m_cable_cost = re.match(
        r"^x\s*=\s*(.+?)\s*km,\s*minimum\s+cost\s*=\s*\$([0-9,]+)$",
        ans_str,
        re.I
    )
    if m_cable_cost:
        x_e, c_v = m_cable_cost.groups()
        return rf"x = {to_latex(x_e.strip())}\text{{ km}}, \quad C_{{\min}} = \${c_v}"

    # Handle trapezoid three equal sides max area:
    m_trap = re.match(
        r"^theta\s*=\s*pi/3.*?top\s+base\s+B\s*=\s*(\d+)\s*([a-zA-Z]+),\s*height\s+h\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?([0-9\.]+)\s*[a-zA-Z]+\),\s*maximum\s+area\s+A\s*=\s*(.+?)\s*([a-zA-Z0-9\^]+)\s*\((?:~|\approx)?([0-9\.]+)\s*[a-zA-Z0-9\^]+\)$",
        ans_str,
        re.I
    )
    if m_trap:
        b_val, u1, h_e, u2, h_v, a_e, u3, a_v = m_trap.groups()
        u3_clean = u3.replace("^2", "").strip()
        return (
            rf"\theta = \frac{{\pi}}{{3}} = 60^\circ \text{{ (horiz)}}, \quad "
            rf"B = {b_val}\text{{ {u1}}}, \quad "
            rf"h = {to_latex(h_e.strip())}\text{{ {u2}}} \approx {h_v}\text{{ {u2}}}, \quad "
            rf"A_{{\max}} = {to_latex(a_e.strip())}\text{{ {u3_clean}}}^2 \approx {a_v}\text{{ {u3_clean}}}^2"
        )

    # Handle ellipse circumscribed rectangle min area:
    m_ellipse_rect = re.match(
        r"^semi-axes\s+a\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?([0-9\.]+)\s*[a-zA-Z]+\),\s*b\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?([0-9\.]+)\s*[a-zA-Z]+\),\s*minimum\s+area\s+A\s*=\s*(.+?)\s*([a-zA-Z0-9\^]+)\s*\((?:~|\approx)?([0-9\.]+)\s*[a-zA-Z0-9\^]+\)$",
        ans_str,
        re.I
    )
    if m_ellipse_rect:
        a_e, u1, a_v, b_e, u2, b_v, a_min_e, u3, a_min_v = m_ellipse_rect.groups()
        u3_clean = u3.replace("^2", "").strip()
        return (
            rf"a = {to_latex(a_e.strip())}\text{{ {u1}}} \approx {a_v}\text{{ {u1}}}, \quad "
            rf"b = {to_latex(b_e.strip())}\text{{ {u2}}} \approx {b_v}\text{{ {u2}}}, \quad "
            rf"A_{{\min}} = {to_latex(a_min_e.strip())}\text{{ {u3_clean}}}^2 \approx {a_min_v}\text{{ {u3_clean}}}^2"
        )

    # Handle parabola tangent line intercepts sum min:
    m_parab_tan = re.match(
        r"^P\(x0,\s*y0\)\s*=\s*\(([0-9\.]+),\s*([0-9\.]+)\),\s*intercepts\s+OA\s*=\s*([0-9\.]+),\s*OB\s*=\s*([0-9\.]+),\s*minimum\s+sum\s+OA\s*\+\s*OB\s*=\s*([0-9\.]+)$",
        ans_str,
        re.I
    )
    if m_parab_tan:
        x0_v, y0_v, oa_v, ob_v, s_v = m_parab_tan.groups()
        return (
            rf"P\left(x_0, y_0\right) = \left({x0_v}, {y0_v}\right), \quad "
            rf"OA \approx {oa_v}, \quad OB \approx {ob_v}, \quad "
            rf"\left(OA + OB\right)_{{\min}} \approx {s_v}"
        )

    # Handle hemisphere inscribed cone max volume:
    m_hemi_cone = re.match(
        r"^radius\s+r\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?([0-9\.]+)\s*[a-zA-Z]+\),\s*height\s+h\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?([0-9\.]+)\s*[a-zA-Z]+\),\s*maximum\s+volume\s+V\s*=\s*(.+?)\s*([a-zA-Z0-9\^]+)\s*\((?:~|\approx)?([0-9\.]+)\s*[a-zA-Z0-9\^]+\)$",
        ans_str,
        re.I
    )
    if m_hemi_cone:
        r_e, u1, r_v, h_e, u2, h_v, v_e, u3, v_v = m_hemi_cone.groups()
        u3_clean = u3.replace("^3", "").replace("^2", "").strip()
        return (
            rf"r = {to_latex(r_e.strip())}\text{{ {u1}}} \approx {r_v}\text{{ {u1}}}, \quad "
            rf"h = {to_latex(h_e.strip())}\text{{ {u2}}} \approx {h_v}\text{{ {u2}}}, \quad "
            rf"V_{{\max}} = {to_latex(v_e.strip())}\text{{ {u3_clean}}}^3 \approx {v_v}\text{{ {u3_clean}}}^3"
        )

    # Handle inventory EOQ batch size total cost minimization:
    m_eoq = re.match(
        r"^C\(x\)\s*=\s*(.+?),\s*optimal\s+batch\s+size\s+x\s*=\s*(.+?)\s*\((?:~|\approx)?([0-9\.]+),\s*or\s*(\d+)\s*([a-zA-Z]+)\),\s*minimum\s+annual\s+cost\s+C\s*=\s*\$(.+?)\s*\((?:~|\approx)?\$([0-9\.]+)\)$",
        ans_str,
        re.I
    )
    if m_eoq:
        c_eq, x_e, x_v, x_r, item, c_e, c_v = m_eoq.groups()
        return (
            rf"C(x) = {c_eq}, \quad "
            rf"x^* = {to_latex(x_e.strip())} \approx {x_v}\text{{ ({x_r} {item})}}, \quad "
            rf"C_{{\min}} = \text{{\$}}{to_latex(c_e.strip())} \approx \text{{\$}}{c_v}"
        )

    # Handle friction pull angle minimization: "tan(theta) = ... = ... (theta ≈ ...° or ... rad), minimum force F = ... N"
    m_fric = re.match(
        r"^tan\(theta\)\s*=\s*(.+?)\s*=\s*(.+?)\s*\(theta\s*≈\s*([^°]+)°\s*or\s*(.+?)\s*rad\),\s*minimum\s+force\s+F\s*=\s*(.+?)\s*([a-zA-Z]+)$",
        ans_str,
        re.I
    )
    if m_fric:
        t_e, t_v, deg_v, rad_v, f_e, u_f = m_fric.groups()
        return rf"\tan\theta = {to_latex(t_e.strip())} = {t_v} \quad (\theta \approx {deg_v}^\circ \text{{ or }} {rad_v}\text{{ rad}}), \quad F_{{\min}} = {to_latex(f_e.strip())}\text{{ {u_f}}}"

    # Handle rectangle between two horizontal parabolas: "width W = ..., height H = ... (~...), maximum area A = ... (~...)"
    m_two_parab = re.match(
        r"^width\s+W\s*=\s*([^,]+),\s*height\s+H\s*=\s*(.+?)\s*\(\s*~\s*([0-9.]+)\s*\),\s*maximum\s+area\s+A\s*=\s*(.+?)\s*\(\s*~\s*([0-9.]+)\s*\)$",
        ans_str,
        re.I
    )
    if m_two_parab:
        w_e, h_e, h_v, a_e, a_v = m_two_parab.groups()
        return rf"W = {to_latex(w_e.strip())}, \quad H = {to_latex(h_e.strip())} \approx {h_v}, \quad A_{{\max}} = {to_latex(a_e.strip())} \approx {a_v}"

    # Handle cone fixed slant height max volume: "r = ... cm (~... cm), h = ... cm, maximum volume V = ... cm^3 (~... cm^3)"
    m_cone_slant = re.match(
        r"^r\s*=\s*(.+?)\s+([a-zA-Z]+)\s*\((?:~|\approx)?\s*(.+?)\s+[a-zA-Z]+\),\s*h\s*=\s*(.+?)\s+[a-zA-Z]+,\s*maximum\s+volume\s+V\s*=\s*(.+?)\s+([a-zA-Z0-9\^]+)\s*\((?:~|\approx)?\s*(.+?)\s+[a-zA-Z0-9\^]+\)$",
        ans_str,
        re.I
    )
    if m_cone_slant:
        r_e, u1, r_v, h_e, v_e, u3, v_v = m_cone_slant.groups()
        u3_clean = u3.replace("^3", "").replace("^2", "").strip()
        return rf"r = {to_latex(r_e.strip())}\text{{ {u1}}} \approx {r_v}\text{{ {u1}}}, \quad h = {to_latex(h_e.strip())}\text{{ {u1}}}, \quad V_{{\max}} = {to_latex(v_e.strip())}\text{{ {u3_clean}}}^3 \approx {v_v}\text{{ {u3_clean}}}^3"

    # Handle curves maximum vertical distance: "x ≈ ..., maximum vertical distance D_max ≈ ..."
    m_vert_dist = re.match(
        r"^x\s*(?:≈|~|=)\s*(.+?),\s*maximum\s+vertical\s+distance\s+D_max\s*(?:≈|~|=)\s*(.+?)$",
        ans_str,
        re.I
    )
    if m_vert_dist:
        x_v, d_v = m_vert_dist.groups()
        return rf"x \approx {x_v.strip()}, \quad D_{{\max}} \approx {d_v.strip()}"

    # Handle painting viewing angle maximization: "x = ... meters, tan(theta) = ... (~..., theta ≈ ...° or ... rad)"
    m_paint_opt = re.match(
        r"^x\s*=\s*(.+?)\s+([a-zA-Z]+),\s*tan\(theta\)\s*=\s*(.+?)\s*\((?:~|\approx)?\s*(.+?),\s*theta\s*≈\s*([^°]+)°\s*or\s*(.+?)\s*rad\)$",
        ans_str,
        re.I
    )
    if m_paint_opt:
        x_e, u, t_e, t_v, deg_v, rad_v = m_paint_opt.groups()
        return rf"x = {to_latex(x_e.strip())}\text{{ {u}}}, \quad \tan\theta = {to_latex(t_e.strip())} \approx {t_v} \quad (\theta \approx {deg_v}^\circ \text{{ or }} {rad_v}\text{{ rad}})"

    # Handle sphere inscribed closed cylinder total surface area: "r = ... cm (~... cm), h = ... cm (~... cm), maximum total surface area S = ... cm^2 (~... cm^2)"
    _u_cyl = r"(?:cm|m|in|ft|meters?|inches?|units?)"
    m_cyl_tot = re.match(
        rf"^r\s*=\s*(.+?)\s+({_u_cyl})\s*\((?:~|\approx)?\s*(.+?)\s+{_u_cyl}\),\s*h\s*=\s*(.+?)\s+({_u_cyl})\s*\((?:~|\approx)?\s*(.+?)\s+{_u_cyl}\),\s*maximum\s+total\s+surface\s+area\s+S\s*=\s*(.+?)\s+({_u_cyl}\^?2?)\s*\((?:~|\approx)?\s*(.+?)\s+{_u_cyl}\^?2?\)$",
        ans_str,
        re.I
    )
    if m_cyl_tot:
        r_e, u1, r_v, h_e, u2, h_v, s_e, u3, s_v = m_cyl_tot.groups()
        u3_clean = u3.replace("^2", "").strip()
        return rf"r = {to_latex(r_e.strip())}\text{{ {u1}}} \approx {r_v}\text{{ {u1}}}, \quad h = {to_latex(h_e.strip())}\text{{ {u2}}} \approx {h_v}\text{{ {u2}}}, \quad S_{{\max}} = {to_latex(s_e.strip())}\text{{ {u3_clean}}}^2 \approx {s_v}\text{{ {u3_clean}}}^2"

    # Handle curve tangent intercepts min sum: "(x_0, y_0) = (..., ...) (~(..., ...)), minimum sum of intercepts S = ... (~...)"
    m_tan_int = re.match(
        r"^\(x_0,\s*y_0\)\s*=\s*\((.+?),\s*(.+?)\)\s*\((?:~|\approx)?\s*\((.+?),\s*(.+?)\)\),\s*minimum\s+sum\s+of\s+intercepts\s+S\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\)$",
        ans_str,
        re.I
    )
    if m_tan_int:
        x_e, y_e, x_v, y_v, s_e, s_v = m_tan_int.groups()
        return rf"(x_0, y_0) = \left({to_latex(x_e.strip())}, {to_latex(y_e.strip())}\right) \approx ({x_v}, {y_v}), \quad S_{{\min}} = {to_latex(s_e.strip())} \approx {s_v}"

    # Handle delivery drone headwind energy: "v = ... km/h, minimum energy E = ... Wh (... Watt-hours)"
    m_drone_e = re.match(
        r"^v\s*=\s*(\d+(?:\.\d+)?)\s*([a-zA-Z/]+),\s*minimum\s+energy\s+E\s*=\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)(?:\s*\(.*\))?$",
        ans_str,
        re.I
    )
    if m_drone_e:
        v_v, u_v, e_v, u_e = m_drone_e.groups()
        return rf"v = {v_v}\text{{ {u_v}}}, \quad E_{{\min}} = {e_v}\text{{ {u_e}}}"

    # Handle semicircle circumscribed isosceles triangle: "theta = pi/4 (45 deg, tan(theta) = 1), base = ... cm (~... cm), height = ... cm (~... cm), minimum area A = ... cm^2"
    m_semi_tri = re.match(
        r"^theta\s*=\s*(.+?)\s*\((.+?)\s*deg,\s*tan\(theta\)\s*=\s*(.+?)\),\s*base\s*=\s*(.+?)\s+([a-zA-Z]+)\s*\((?:~|\approx)?\s*(.+?)\s+[a-zA-Z]+\),\s*height\s*=\s*(.+?)\s+([a-zA-Z]+)\s*\((?:~|\approx)?\s*(.+?)\s+[a-zA-Z]+\),\s*minimum\s+area\s+A\s*=\s*(.+?)\s+([a-zA-Z0-9\^]+)$",
        ans_str,
        re.I
    )
    if m_semi_tri:
        th_e, th_d, tan_v, b_e, u1, b_v, h_e, u2, h_v, a_v, u3 = m_semi_tri.groups()
        u3_clean = u3.replace("^2", "").strip()
        return rf"\theta = {to_latex(th_e.strip())} = {th_d}^\circ \quad (\tan\theta = {tan_v}), \quad \text{{base }} B = {to_latex(b_e.strip())}\text{{ {u1}}} \approx {b_v}\text{{ {u1}}}, \quad \text{{height }} H = {to_latex(h_e.strip())}\text{{ {u2}}} \approx {h_v}\text{{ {u2}}}, \quad A_{{\min}} = {a_v}\text{{ {u3_clean}}}^2"

    # Handle rectangle garden answers: "length = X m, width = Y m, minimum perimeter = Z m"
    m_rect = re.match(r"^length\s*=\s*(.+?)\s*m,\s*width\s*=\s*(.+?)\s*m,\s*minimum\s+perimeter\s*=\s*(.+?)\s*m$", ans_str)
    if m_rect:
        l_val, w_val, p_val = m_rect.groups()
        l_tex = to_latex(l_val)
        w_tex = to_latex(w_val)
        p_tex = to_latex(p_val)
        return rf"\text{{length }} = {l_tex}\text{{ m, width }} = {w_tex}\text{{ m}}, \quad P_{{\min}} = {p_tex}\text{{ m}}"

    # Handle rectangle area maximization answers: "length = L m, width = W m, maximum area = A m^2"
    m_rec_area = re.match(r"^length\s*=\s*(.+?)\s*m,\s*width\s*=\s*(.+?)\s*m,\s*maximum\s+area\s*=\s*(.+?)\s*m\^?2$", ans_str)
    if m_rec_area:
        l_val, w_val, a_val = m_rec_area.groups()
        l_tex = to_latex(l_val)
        w_tex = to_latex(w_val)
        a_tex = to_latex(a_val)
        return rf"\text{{length }} = {l_tex}\text{{ m, width }} = {w_tex}\text{{ m}}, \quad A_{{\max}} = {a_tex}\text{{ m}}^2"

    # Handle divided pens answers: "outer dimensions = X m by Y m, maximum area = Z m^2"
    m_pen = re.match(r"^outer\s+dimensions\s*=\s*(.+?)\s*m\s+by\s+(.+?)\s*m,\s*maximum\s+area\s*=\s*(.+?)\s*m\^?2$", ans_str)
    if m_pen:
        x_val, y_val, a_val = m_pen.groups()
        x_tex = to_latex(x_val)
        y_tex = to_latex(y_val)
        a_tex = to_latex(a_val)
        return rf"\text{{dimensions }} = {x_tex}\text{{ m}} \times {y_tex}\text{{ m}}, \quad A_{{\max}} = {a_tex}\text{{ m}}^2"

    # Handle poster margin answers: "width = W cm, height = H cm, minimum total area = A cm^2"
    m_post = re.match(r"^width\s*=\s*(.+?)\s*cm,\s*height\s*=\s*(.+?)\s*cm,\s*minimum\s+total\s+area\s*=\s*(.+?)\s*cm\^?2$", ans_str)
    if m_post:
        w_val, h_val, a_val = m_post.groups()
        w_tex = to_latex(w_val)
        h_tex = to_latex(h_val)
        a_tex = to_latex(a_val)
        return rf"\text{{width }} = {w_tex}\text{{ cm, height }} = {h_tex}\text{{ cm}}, \quad A_{{\min}} = {a_tex}\text{{ cm}}^2"

    # Handle open top cylinder answers: "r = X cm, h = Y cm, minimum surface area = Z*pi cm^2 (~approx)"
    m_cyl = re.match(r"^r\s*=\s*(.+?)\s*cm,\s*h\s*=\s*(.+?)\s*cm,\s*minimum\s+surface\s+area\s*=\s*(.+?)(?:\s*\(~?([0-9\.\-]+)\))?$", ans_str)
    if m_cyl:
        r_val, h_val, s_val, approx_val = m_cyl.groups()
        r_tex = to_latex(r_val)
        h_tex = to_latex(h_val)
        s_clean = re.sub(r"\*?\s*pi\b", r"\\pi", s_val).replace("cm^2", "").strip()
        approx_tex = f" \\approx {approx_val}" if approx_val else ""
        return rf"r = {r_tex}\text{{ cm, }} h = {h_tex}\text{{ cm}}, \quad S_{{\min}} = {s_clean}\text{{ cm}}^2{approx_tex}"

    # Handle Norman window answers: "r = X m (~approx_r m), h = Y m (~approx_h m), maximum area = A m^2 (~approx_A m^2)"
    m_norman = re.match(r"^r\s*=\s*(.+?)\s*m\s*(?:\(.*?\))?,\s*h\s*=\s*(.+?)\s*m\s*(?:\(.*?\))?,\s*maximum\s+area\s*=\s*(.+?)\s*m\^?2(?:\s*\(.*?\))?$", ans_str)
    if m_norman:
        r_val, h_val, a_val = m_norman.groups()
        r_tex = to_latex(r_val)
        h_tex = to_latex(h_val)
        a_tex = to_latex(a_val)
        return rf"r = {r_tex}\text{{ m, }} h = {h_tex}\text{{ m}}, \quad A_{{\max}} = {a_tex}\text{{ m}}^2"

    # Handle inscribed cylinder (sphere or cone) answers: "r = X cm, h = Y cm, maximum volume = V*pi cm^3 (~approx)"
    m_inscr_cyl = re.match(r"^r\s*=\s*(.+?)\s*cm,\s*h\s*=\s*(.+?)\s*cm,\s*maximum\s+volume\s*=\s*(.+?)(?:\s*\(~?([0-9\.\-]+)\s*(?:cm\^?3)?\))?$", ans_str)
    if m_inscr_cyl:
        r_val, h_val, v_val, approx_val = m_inscr_cyl.groups()
        r_tex = to_latex(r_val)
        h_tex = to_latex(h_val)
        v_clean = re.sub(r"\*?\s*pi\b", r"\\pi", v_val).replace("cm^3", "").strip()
        approx_tex = f" \\approx {approx_val}\\text{{ cm}}^3" if approx_val else ""
        return rf"r = {r_tex}\text{{ cm, }} h = {h_tex}\text{{ cm}}, \quad V_{{\max}} = {v_clean}\text{{ cm}}^3{approx_tex}"

    # Handle local extrema answers
    m_loc_both = re.match(r"^(?:local\s+)?maximum\s+value\s*=\s*([^,]+?)\s+at\s+([xt])\s*=\s*([^,]+?),\s*(?:local\s+)?minimum\s+value\s*=\s*([^,]+?)\s+at\s+([xt])\s*=\s*([^,]+?)$", ans_str, re.I)
    if m_loc_both:
        m_val, v_m, x_m, n_val, v_n, x_n = m_loc_both.groups()
        return rf"\text{{max }} = {to_latex(m_val)}\text{{ at }} {v_m} = {to_latex(x_m)}, \quad \text{{min }} = {to_latex(n_val)}\text{{ at }} {v_n} = {to_latex(x_n)}"

    m_loc_min_only = re.match(r"^(?:local\s+)?minimum\s+value\s*=\s*([^,]+?)\s+at\s+([xt])\s*=\s*([^,]+?)(?:,\s*no\s+local\s+maximum)?$", ans_str, re.I)
    if m_loc_min_only:
        n_val, v_n, x_n = m_loc_min_only.groups()
        return rf"\text{{min }} = {to_latex(n_val)}\text{{ at }} {v_n} = {to_latex(x_n)}"

    m_loc_max_only = re.match(r"^(?:local\s+)?maximum\s+value\s*=\s*([^,]+?)\s+at\s+([xt])\s*=\s*([^,]+?)(?:,\s*no\s+local\s+minimum)?$", ans_str, re.I)
    if m_loc_max_only:
        m_val, v_m, x_m = m_loc_max_only.groups()
        return rf"\text{{max }} = {to_latex(m_val)}\text{{ at }} {v_m} = {to_latex(x_m)}"

    # Handle absolute extrema on closed interval answers
    m_abs_ext = re.match(r"^absolute\s+maximum\s*=\s*([^,]+?)\s+at\s+x\s*=\s*([^,]+?),\s*absolute\s+minimum\s*=\s*([^,]+?)\s+at\s+x\s*=\s*([^,]+?)$", ans_str)
    if m_abs_ext:
        m_val, x_m, n_val, x_n = m_abs_ext.groups()
        return rf"\text{{absolute max }} = {to_latex(m_val)}\text{{ at }} x = {to_latex(x_m)}, \quad \text{{absolute min }} = {to_latex(n_val)}\text{{ at }} x = {to_latex(x_n)}"

    # Handle beam strength optimization answers: "width w = W cm (~approx_w cm), depth d = D cm (~approx_d cm)"
    m_beam = re.match(r"^(?:width\s+)?w\s*=\s*(.+?)\s*([a-zA-Z]+)(?:\s*\([^)]*\))?,\s*(?:depth\s+)?d\s*=\s*(.+?)\s*([a-zA-Z]+)(?:\s*\([^)]*\))?$", ans_str)
    if m_beam:
        w_val, u1, d_val, u2 = m_beam.groups()
        w_tex = to_latex(w_val.strip())
        d_tex = to_latex(d_val.strip())
        return rf"w = {w_tex}\text{{ {u1}}}, \quad d = {d_tex}\text{{ {u2}}}"

    # Handle cubic parameters answers: "a = -3, b = 0, local maximum value = 5 at x = 0"
    m_cubic_ext = re.match(r"^a\s*=\s*([^,]+?),\s*b\s*=\s*([^,]+?),\s*(?:local\s+)?(maximum|minimum)\s+value\s*=\s*([^,]+?)\s+at\s+x\s*=\s*([^,]+?)$", ans_str, re.I)
    if m_cubic_ext:
        a_val, b_val, ext_type, ext_val, x_val = m_cubic_ext.groups()
        lbl = r"\text{local max}" if "max" in ext_type.lower() else r"\text{local min}"
        return rf"a = {to_latex(a_val)}, \quad b = {to_latex(b_val)}, \quad {lbl} = {to_latex(ext_val)}\text{{ at }} x = {to_latex(x_val)}"

    # Handle number square diff answers: "x = X, maximum value = M"
    m_num_sq = re.match(r"^x\s*=\s*([^,]+?),\s*maximum\s+value\s*=\s*(.+?)$", ans_str)
    if m_num_sq:
        x_val, m_val = m_num_sq.groups()
        x_tex = to_latex(x_val)
        m_tex = to_latex(m_val)
        return rf"x = {x_tex}, \quad \text{{maximum value }} = {m_tex}"

    # Handle number reciprocal sum answers: "x = X, minimum sum = S"
    m_num_recip = re.match(r"^x\s*=\s*([^,]+?),\s*minimum\s+sum\s*=\s*(.+?)$", ans_str)
    if m_num_recip:
        x_val, s_val = m_num_recip.groups()
        x_tex = to_latex(x_val)
        s_tex = to_latex(s_val)
        return rf"x = {x_tex}, \quad S_{{\min}} = {s_tex}"

    # Handle two numbers product maximization: "first number = X, second number = Y, maximum product = P"
    m_two_prod = re.match(r"^(?:first\s+number\s*=\s*|x\s*=\s*)([^,]+?),\s*(?:second\s+number\s*=\s*|y\s*=\s*)([^,]+?),\s*maximum\s+product\s*=\s*(.+?)$", ans_str)
    if m_two_prod:
        x_val, y_val, p_val = m_two_prod.groups()
        x_tex = to_latex(x_val)
        y_tex = to_latex(y_val)
        p_tex = to_latex(p_val)
        return rf"x = {x_tex}, \quad y = {y_tex}, \quad P_{{\max}} = {p_tex}"

    # Handle two numbers sum minimization: "first number = X, second number = Y, minimum sum = S"
    m_two_sum = re.match(r"^(?:first\s+number\s*=\s*|x\s*=\s*)([^,]+?),\s*(?:second\s+number\s*=\s*|y\s*=\s*)([^,]+?),\s*minimum\s+sum\s*=\s*(.+?)$", ans_str)
    if m_two_sum:
        x_val, y_val, s_val = m_two_sum.groups()
        x_tex = to_latex(x_val)
        y_tex = to_latex(y_val)
        s_tex = to_latex(s_val)
        return rf"x = {x_tex}, \quad y = {y_tex}, \quad S_{{\min}} = {s_tex}"

    # Handle right triangle legs area answers: "base = X cm, height = Y cm, maximum area = A cm^2"
    m_tri_legs = re.match(r"^base\s*=\s*(.+?)\s*cm,\s*height\s*=\s*(.+?)\s*cm,\s*maximum\s+area\s*=\s*(.+?)\s*cm\^?2$", ans_str)
    if m_tri_legs:
        a_val, b_val, area_val = m_tri_legs.groups()
        a_tex = to_latex(a_val)
        b_tex = to_latex(b_val)
        area_tex = to_latex(area_val)
        return rf"\text{{base }} = {a_tex}\text{{ cm, height }} = {b_tex}\text{{ cm}}, \quad A_{{\max}} = {area_tex}\text{{ cm}}^2"

    # Handle product power answers: "x = X, y = Y, maximum product = P"
    m_prod = re.match(r"^x\s*=\s*(.+?),\s*y\s*=\s*(.+?),\s*maximum\s+product\s*=\s*(.+?)$", ans_str)
    if m_prod:
        x_val, y_val, p_val = m_prod.groups()
        x_tex = to_latex(x_val)
        y_tex = to_latex(y_val)
        p_tex = to_latex(p_val)
        return rf"x = {x_tex}, \quad y = {y_tex}, \quad P_{{\max}} = {p_tex}"

    # Handle garden cost answers: "front side = X m, other side = Y m, minimum cost = $C"
    m_cost = re.match(r"^front\s+side\s*=\s*(.+?)\s*m,\s*other\s+side\s*=\s*(.+?)\s*m,\s*minimum\s+cost\s*=\s*\$?(.+?)$", ans_str)
    if m_cost:
        x_val, y_val, c_val = m_cost.groups()
        x_tex = to_latex(x_val)
        y_tex = to_latex(y_val)
        c_tex = to_latex(c_val)
        return rf"\text{{front }} = {x_tex}\text{{ m, other }} = {y_tex}\text{{ m}}, \quad C_{{\min}} = \${c_tex}"

    # Handle profit maximization answers: "production level x = X units, maximum profit = $P"
    m_profit = re.match(r"^(?:production\s+level\s+)?(?:x\s*=\s*)?(\d+(?:\.\d+)?)\s*(?:units)?,\s*maximum\s+profit\s*=\s*\$?(\d+(?:\.\d+)?)$", ans_str, re.I)
    if m_profit:
        x_val, p_val = m_profit.groups()
        x_tex = to_latex(x_val)
        p_tex = f"{int(float(p_val)):,}" if float(p_val).is_integer() else p_val
        return rf"x = {x_tex}\text{{ units}}, \quad P_{{\max}} = \${p_tex}"

    # Handle inscribed cone in sphere answers: "r = R_disp cm (~approx), h = H cm, maximum volume = V*pi cm^3 (~approx)"
    m_inscr_cone = re.match(r"^r\s*=\s*(.+?)\s*cm(?:\s*\(~?[0-9\.\-]+\s*cm\))?,\s*h\s*=\s*(.+?)\s*cm,\s*maximum\s+volume\s*=\s*(.+?)(?:\s*\(~?([0-9\.\-]+)\s*(?:cm\^?3)?\))?$", ans_str)
    if m_inscr_cone:
        r_val, h_val, v_val, approx_val = m_inscr_cone.groups()
        r_tex = to_latex(r_val)
        h_tex = to_latex(h_val)
        v_clean = re.sub(r"\*?\s*pi\b", r"\\pi", v_val).replace("cm^3", "").strip()
        approx_tex = f" \\approx {approx_val}\\text{{ cm}}^3" if approx_val else ""
        return rf"r = {r_tex}\text{{ cm, }} h = {h_tex}\text{{ cm}}, \quad V_{{\max}} = {v_clean}\text{{ cm}}^3{approx_tex}"

    # Handle closest point answers: "(X, Y), minimum distance = D (~approx)"
    m_line_pt = re.match(r"^\((.+?),\s*(.+?)\)(?:\s*\([^)]*\))?,\s*minimum\s+distance\s*=\s*(.+?)(?:\s*\((?:~|\approx)?\$?([0-9\.]+)\))?$", ans_str, re.I)
    if m_line_pt:
        x_val, y_val, d_val, approx_d = m_line_pt.groups()
        x_tex = to_latex(x_val.strip())
        y_tex = to_latex(y_val.strip())
        d_tex = to_latex(d_val.strip())
        approx_tex = f" \\approx {approx_d}" if approx_d else ""
        return rf"\left({x_tex}, {y_tex}\right), \quad d_{{\min}} = {d_tex}{approx_tex}"

    # Handle average cost answers: "production level x = X units, minimum average cost = $C"
    m_avg_cost = re.match(r"^(?:production\s+level\s+)?(?:x\s*=\s*)?(\d+(?:\.\d+)?)\s*(?:units)?,\s*minimum\s+average\s+cost\s*=\s*\$?(\d+(?:\.\d+)?)$", ans_str, re.I)
    if m_avg_cost:
        x_val, c_val = m_avg_cost.groups()
        x_tex = to_latex(x_val)
        c_tex = f"{int(float(c_val)):,}" if float(c_val).is_integer() else c_val
        return rf"x = {x_tex}\text{{ units}}, \quad AC_{{\min}} = \${c_tex}"

    # Handle lifeguard travel time: "x = 22.5 m (45/2 m), minimum travel time = 18 s"
    m_lifeguard = re.match(r"^x\s*=\s*(.+?)\s*([a-zA-Z]+)?\s*\((.+?)\s*([a-zA-Z]+)?\),\s*minimum\s+travel\s+time\s*=\s*(.+?)\s*([a-zA-Z]+)$", ans_str, re.I)
    if m_lifeguard:
        x_dec, u1, x_frac, u2, t_val, u_t = m_lifeguard.groups()
        x_dec_tex = to_latex(x_dec)
        x_frac_tex = to_latex(x_frac)
        t_tex = to_latex(t_val)
        unit_str = u1 or u2 or "m"
        return rf"x = {x_dec_tex}\text{{ {unit_str}}} = {x_frac_tex}\text{{ {unit_str}}}, \quad T_{{\min}} = {t_tex}\text{{ {u_t}}}"

    # Handle moving objects closest approach: "t = 1.6 hours (8/5 hours), minimum distance = 6 km"
    m_closest_approach = re.match(r"^t\s*=\s*(.+?)\s*([a-zA-Z]+)?(?:\s*\((.+?)\s*([a-zA-Z]+)?\))?,\s*minimum\s+distance\s*=\s*(.+?)\s*([a-zA-Z]+)$", ans_str, re.I)
    if m_closest_approach:
        t_val, u1, t_frac, u2, d_val, u_d = m_closest_approach.groups()
        t_tex = to_latex(t_val)
        u_t = u1 or u2 or "hours"
        t_part = rf"t = {t_tex}\text{{ {u_t}}}"
        if t_frac and t_frac != t_val:
            t_part += rf" = {to_latex(t_frac)}\text{{ {u_t}}}"
        d_tex = to_latex(d_val)
        return rf"{t_part}, \quad d_{{\min}} = {d_tex}\text{{ {u_d}}}"

    # Handle illumination height: "h = 2*sqrt(2) m (~2.8284 m)"
    m_illum = re.match(r"^h\s*=\s*(.+?)\s*([a-zA-Z]+)?\s*\((?:~|\approx)\s*([0-9\.]+)\s*([a-zA-Z]+)?\)$", ans_str, re.I)
    if m_illum:
        h_exact, u1, h_approx, u2 = m_illum.groups()
        h_tex = to_latex(h_exact)
        unit_str = u1 or u2 or "m"
        return rf"h = {h_tex}\text{{ {unit_str}}} \approx {h_approx}\text{{ {unit_str}}}"

    # Handle rain gutter cross section: "theta = pi/3 rad (60 deg), maximum cross-sectional area = 75*sqrt(3) cm^2 (~129.9038 cm^2)"
    m_gutter = re.match(r"^theta\s*=\s*(.+?)\s*rad\s*\((.+?)\s*deg\),\s*maximum\s+cross-sectional\s+area\s*=\s*(.+?)\s*([a-zA-Z]+)\^?2(?:\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\^?2\))?$", ans_str, re.I)
    if m_gutter:
        th_rad, th_deg, a_exact, u1, a_approx, u2 = m_gutter.groups()
        th_tex = to_latex(th_rad)
        a_tex = to_latex(a_exact)
        unit_str = u1 or "cm"
        approx_str = f" \\approx {a_approx}\\text{{ {unit_str}}}^2" if a_approx else ""
        return rf"\theta = {th_tex}\text{{ rad }} ({th_deg}^\circ), \quad A_{{\max}} = {a_tex}\text{{ {unit_str}}}^2{approx_str}"

    # Handle line intercepts sum: "a = 1 + 2*sqrt(2) (~3.8284), b = 8 + 2*sqrt(2) (~10.8284), minimum sum = 9 + 4*sqrt(2) (~14.6569)"
    m_line_int = re.match(r"^a\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\),\s*b\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\),\s*minimum\s+sum\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\)$", ans_str, re.I)
    if m_line_int:
        a_val, a_app, b_val, b_app, s_val, s_app = m_line_int.groups()
        a_tex = to_latex(a_val)
        b_tex = to_latex(b_val)
        s_tex = to_latex(s_val)
        return rf"a = {a_tex} \approx {a_app}, \quad b = {b_tex} \approx {b_app}, \quad S_{{\min}} = {s_tex} \approx {s_app}"

    # Handle line triangle minimum area: "minimum area = 12 (with a = 4, b = 6)"
    m_line_min_area = re.match(r"^minimum\s+area\s*=\s*(.+?)\s*\(with\s+a\s*=\s*(.+?),\s*b\s*=\s*(.+?)\)$", ans_str, re.I)
    if m_line_min_area:
        a_min_val, a_val, b_val = m_line_min_area.groups()
        a_min_tex = to_latex(a_min_val)
        a_tex = to_latex(a_val)
        b_tex = to_latex(b_val)
        return rf"A_{{\min}} = {a_min_tex}, \quad (a = {a_tex}, b = {b_tex})"

    # Handle rental unit / hotel revenue maximization: "nightly rate = $140, maximum revenue = $9800"
    m_hotel_rev = re.match(r"^(?:nightly\s+rate|rate)\s*=\s*\$?(\d+(?:\.\d+)?),\s*maximum\s+revenue\s*=\s*\$?(\d+(?:\.\d+)?)$", ans_str, re.I)
    if m_hotel_rev:
        r_val, rev_val = m_hotel_rev.groups()
        r_tex = to_latex(r_val)
        rev_tex = f"{int(float(rev_val)):,}" if float(rev_val).is_integer() else rev_val
        return rf"\text{{rate }} = \${r_tex}, \quad R_{{\max}} = \${rev_tex}"

    # Handle transport trip cost minimization: "speed v = 50 km/h, minimum total cost = $2000"
    m_trip_cost = re.match(r"^speed\s+v\s*=\s*(.+?)\s*([a-zA-Z/]+)?,\s*minimum\s+total\s+cost\s*=\s*\$?(\d+(?:\.\d+)?)$", ans_str, re.I)
    if m_trip_cost:
        v_val, u_spd, c_val = m_trip_cost.groups()
        v_tex = to_latex(v_val)
        u_str = u_spd or "km/h"
        c_tex = f"{int(float(c_val)):,}" if float(c_val).is_integer() else c_val
        return rf"v = {v_tex}\text{{ {u_str}}}, \quad C_{{\min}} = \${c_tex}"

    # Handle open box minimum surface area: "base side length x = X cm, height = Y cm, minimum surface area = S cm^2"
    m_box_min = re.match(r"^(?:base\s+side\s+length\s+)?(?:x\s*=\s*)?(.+?)\s*([a-zA-Z]+)?,\s*height\s*=\s*(.+?)\s*([a-zA-Z]+)?,\s*minimum\s+surface\s+area\s*=\s*(.+?)\s*([a-zA-Z]+)\^?2$", ans_str, re.I)
    if m_box_min:
        x_val, u1, h_val, u2, s_val, u3 = m_box_min.groups()
        x_tex = to_latex(x_val)
        h_tex = to_latex(h_val)
        s_tex = to_latex(s_val)
        u_str = u1 or u3 or "cm"
        return rf"x = {x_tex}\text{{ {u_str}, }} h = {h_tex}\text{{ {u_str}}}, \quad S_{{\min}} = {s_tex}\text{{ {u_str}}}^2"

    # Handle open box square base answers: "base side = X cm, height = Y cm, maximum volume = V cm^3"
    m_boxsq = re.match(r"^base\s+side\s*=\s*(.+?)\s*cm,\s*height\s*=\s*(.+?)\s*cm,\s*maximum\s+volume\s*=\s*(.+?)\s*cm\^?3$", ans_str)
    if m_boxsq:
        x_val, h_val, v_val = m_boxsq.groups()
        x_tex = to_latex(x_val)
        h_tex = to_latex(h_val)
        v_tex = to_latex(v_val)
        return rf"\text{{base }} = {x_tex}\text{{ cm, height }} = {h_tex}\text{{ cm}}, \quad V_{{\max}} = {v_tex}\text{{ cm}}^3"

    # Handle inscribed rectangle answers (semicircle or triangle): "width = W cm, height = H cm, maximum area = A cm^2 (~...)"
    m_inscr_rect = re.match(r"^width\s*=\s*(.+?)\s*([a-zA-Z]+)?,\s*height\s*=\s*(.+?)\s*([a-zA-Z]+)?,\s*maximum\s+area\s*=\s*(.+?)\s*([a-zA-Z]+)\^?2(?:\s*\(.*?\))?$", ans_str)
    if m_inscr_rect:
        w_val, u1, h_val, u2, a_val, u3 = m_inscr_rect.groups()
        w_tex = to_latex(w_val)
        h_tex = to_latex(h_val)
        a_tex = to_latex(a_val)
        unit_str = u1 or u3 or "cm"
        return rf"\text{{width }} = {w_tex}\text{{ {unit_str}, height }} = {h_tex}\text{{ {unit_str}}}, \quad A_{{\max}} = {a_tex}\text{{ {unit_str}}}^2"

    # Handle inscribed rectangle answers: "width = W, height = H, maximum area = A (~approx)"
    m_inscr = re.match(r"^width\s*=\s*(.+?),\s*height\s*=\s*(.+?),\s*maximum\s+area\s*=\s*(.+?)(?:\s*\([^)]*\))?$", ans_str)
    if m_inscr:
        w_val, h_val, a_val = m_inscr.groups()
        w_tex = to_latex(w_val)
        h_tex = to_latex(h_val)
        a_tex = to_latex(a_val)
        return rf"\text{{width }} = {w_tex}, \quad \text{{height }} = {h_tex}, \quad A_{{\max}} = {a_tex}"

    # Handle ray from origin maximum slope answers: "(X, Y), maximum slope = M"
    m_ray_slope = re.match(r"^\((.+?),\s*(.+?)\),\s*maximum\s+slope\s*=\s*(.+?)$", ans_str, re.I)
    if m_ray_slope:
        x_v, y_v, m_v = m_ray_slope.groups()
        return rf"\left({to_latex(x_v.strip())}, {to_latex(y_v.strip())}\right), \quad m_{{\max}} = {to_latex(m_v.strip())}"

    # Handle weighted sum of squares answers: "x = X, y = Y, minimum sum = S"
    m_sumsq = re.match(r"^x\s*=\s*(.+?),\s*y\s*=\s*(.+?),\s*minimum\s+sum\s*=\s*(.+?)$", ans_str)
    if m_sumsq:
        x_val, y_val, s_val = m_sumsq.groups()
        x_tex = to_latex(x_val)
        y_tex = to_latex(y_val)
        s_tex = to_latex(s_val)
        return rf"x = {x_tex}, \quad y = {y_tex}, \quad S_{{\min}} = {s_tex}"

    # Handle closed box cost optimization: "x = X m (~approx_x m), h = Y m (~approx_h m), minimum total cost = $C (~$approx_C)"
    m_box_cost = re.match(
        r"^x\s*=\s*(.+?)\s*([a-zA-Z]+)(?:\s*\([^)]*\))?,\s*h\s*=\s*(.+?)\s*([a-zA-Z]+)(?:\s*\([^)]*\))?,\s*minimum\s+total\s+cost\s*=\s*\$?(.+?)(?:\s*\((?:~|\approx)?\$?([0-9\.]+)\))?$",
        ans_str,
        re.I
    )
    if m_box_cost:
        x_val, u1, h_val, u2, c_val, approx_c = m_box_cost.groups()
        x_tex = to_latex(x_val.strip())
        h_tex = to_latex(h_val.strip())
        c_tex = to_latex(c_val.strip())
        u_str = u1 or u2 or "m"
        approx_tex = f" \\approx \\${approx_c}" if approx_c else ""
        return rf"x = {x_tex}\text{{ {u_str}, }} h = {h_tex}\text{{ {u_str}}}, \quad C_{{\min}} = \${c_tex}{approx_tex}"

    # Handle cubic two-parameter answers: "a = A, b = B"
    m_ab = re.match(r"^a\s*=\s*([^,]+?),\s*b\s*=\s*(.+?)$", ans_str, re.I)
    if m_ab:
        a_val, b_val = m_ab.groups()
        return rf"a = {to_latex(a_val.strip())}, \quad b = {to_latex(b_val.strip())}"

    # Handle cylinder radius and height dimensions: "r = R cm (~approx_r cm), h = H cm (~approx_h cm)"
    m_rh = re.match(
        r"^r\s*=\s*(.+?)\s*([a-zA-Z]+)(?:\s*\([^)]*\))?,\s*h\s*=\s*(.+?)\s*([a-zA-Z]+)(?:\s*\([^)]*\))?$",
        ans_str,
        re.I
    )
    if m_rh:
        r_val, u1, h_val, u2 = m_rh.groups()
        r_tex = to_latex(r_val.strip())
        h_tex = to_latex(h_val.strip())
        u_str = u1 or u2 or "cm"
        return rf"r = {r_tex}\text{{ {u_str}}}, \quad h = {h_tex}\text{{ {u_str}}}"

    # Handle curve tangent triangle min area: "x_0 = X (~approx_x), minimum area = A (~approx_A)"
    m_tangent_tri = re.match(
        r"^x_0\s*=\s*(.+?)(?:\s*\([^)]*\))?,\s*minimum\s+area\s*=\s*(.+?)(?:\s*\([^)]*\))?$",
        ans_str,
        re.I
    )
    if m_tangent_tri:
        x0_val, a_val = m_tangent_tri.groups()
        x0_tex = to_latex(x0_val.strip())
        a_tex = to_latex(a_val.strip())
        return rf"x_0 = {x0_tex}, \quad A_{{\min}} = {a_tex}"

    # Handle semicircle inscribed trapezoid: "height h = H cm (~approx_h cm), maximum area = A cm^2 (~approx_A cm^2)"
    m_trap = re.match(
        r"^(?:height\s+)?h\s*=\s*(.+?)\s*([a-zA-Z]+)(?:\s*\([^)]*\))?,\s*maximum\s+area\s*=\s*(.+?)\s*([a-zA-Z]+)\^?2(?:\s*\([^)]*\))?$",
        ans_str,
        re.I
    )
    if m_trap:
        h_val, u1, a_val, u2 = m_trap.groups()
        h_tex = to_latex(h_val.strip())
        a_tex = to_latex(a_val.strip())
        u_str = u1 or u2 or "cm"
        return rf"h = {h_tex}\text{{ {u_str}}}, \quad A_{{\max}} = {a_tex}\text{{ {u_str}}}^2"

    # Handle curve secant max vertical distance: "x = X (~approx), maximum vertical distance = D (~approx)"
    m_secant_dist = re.match(
        r"^x\s*=\s*(.+?)(?:\s*\([^)]*\))?,\s*maximum\s+vertical\s+distance\s*=\s*(.+?)(?:\s*\([^)]*\))?$",
        ans_str,
        re.I
    )
    if m_secant_dist:
        x_val, d_val = m_secant_dist.groups()
        x_tex = to_latex(x_val.strip())
        d_tex = to_latex(d_val.strip())
        return rf"x = {x_tex}, \quad d_{{\max}} = {d_tex}"

    # Handle tour group revenue answers: "group size = N passengers, ticket price = $P, maximum revenue = $R"
    m_tour = re.match(
        r"^group\s+size\s*=\s*(\d+)\s*passengers?,\s*ticket\s+price\s*=\s*\$?([0-9\.]+),\s*maximum\s+revenue\s*=\s*\$?([0-9\.]+)$",
        ans_str,
        re.I
    )
    if m_tour:
        n_val, p_val, r_val = m_tour.groups()
        p_num = f"{float(p_val):.2f}"
        r_num = f"{float(r_val):,.2f}" if "." in r_val and not float(r_val).is_integer() else f"{int(float(r_val)):,}"
        return rf"\text{{group size }} = {n_val}\text{{ passengers}}, \quad \text{{price }} = \${p_num}, \quad R_{{\max}} = \${r_num}"

    # Handle sphere and cube volume ratio: "ratio x/r = 2 (cube side to sphere radius ratio = 2:1)"
    m_ratio = re.match(r"^ratio\s+(?:x/r\s*=\s*)?(\d+(?:\.\d+)?)(?:\s*\(.*\))?$", ans_str, re.I)
    if m_ratio:
        val = m_ratio.group(1)
        return rf"\frac{{x}}{{r}} = {val}"

    # Handle corridor / hypotenuse length: "L = 5*sqrt(5) m (~11.1803 m) (with intercepts a = 5 m, b = 10 m)"
    m_corridor = re.match(
        r"^L\s*=\s*(.+?)\s*([a-zA-Z]+)?(?:\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)?\))?\s*\(with\s+intercepts\s+a\s*=\s*(.+?),\s*b\s*=\s*(.+?)\)$",
        ans_str,
        re.I
    )
    if m_corridor:
        L_expr, u1, approx_val, u2, a_val, b_val = m_corridor.groups()
        L_latex = to_latex(L_expr.strip())
        u = u1 or u2 or ""
        u_str = rf"\text{{ {u}}}" if u else ""
        approx_str = rf" \approx {approx_val}{u_str}" if approx_val else ""

        def _fmt_dim(dim_str, default_u):
            m = re.match(r"^([-\w\.\/\*\^\(\)\+]+?)(?:\s*([a-zA-Z]+))?$", dim_str.strip())
            if m:
                v, unit = m.groups()
                v_tex = to_latex(v.strip())
                dim_u = unit or default_u
                return rf"{v_tex}\text{{ {dim_u}}}" if dim_u else v_tex
            return to_latex(dim_str.strip())

        a_latex = _fmt_dim(a_val, u)
        b_latex = _fmt_dim(b_val, u)
        return rf"L_{{\min}} = {L_latex}{u_str}{approx_str}, \quad (a = {a_latex}, \; b = {b_latex})"

    # Handle farmer harvest revenue: "t = 3.5 weeks, maximum revenue = $722.50"
    m_farmer = re.match(
        r"^t\s*=\s*([0-9\./]+)\s*([a-zA-Z]+)?,\s*maximum\s+revenue\s*=\s*\$?([0-9\.,]+)$",
        ans_str,
        re.I
    )
    if m_farmer:
        t_val, time_unit, r_val = m_farmer.groups()
        u_str = time_unit or "weeks"
        return rf"t = {t_val}\text{{ {u_str}}}, \quad R_{{\max}} = \${r_val}"

    # Handle cone circumscribed sphere: "h = 12 cm (base radius r = 3*sqrt(2) cm, minimum volume V = 72*pi cm^3)"
    m_cone_circ = re.match(
        r"^h\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\(base\s+radius\s+r\s*=\s*(.+?)\s*([a-zA-Z]+),\s*minimum\s+volume\s+V\s*=\s*(.+?)\s*([a-zA-Z]+)\^?3\)$",
        ans_str,
        re.I
    )
    if m_cone_circ:
        h_val, u1, r_val, u2, v_val, u3 = m_cone_circ.groups()
        h_latex = to_latex(h_val.strip())
        r_latex = to_latex(r_val.strip())
        clean_v = v_val.replace("*pi", "").strip()
        v_latex = to_latex(clean_v) + r"\pi" if "*pi" in v_val else to_latex(v_val.strip())
        u_str = u1 or "cm"
        return rf"h = {h_latex}\text{{ {u_str}}}, \quad r = {r_latex}\text{{ {u_str}}}, \quad V_{{\min}} = {v_latex}\text{{ {u_str}}}^3"

    # Handle two light sources illumination: "x = 20/3 m (~6.6667 m) from light source A"
    m_light_src = re.match(
        r"^x\s*=\s*(.+?)\s*([a-zA-Z]+)?(?:\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)?\))?\s*from\s+light\s+source\s+([A-Za-z])$",
        ans_str,
        re.I
    )
    if m_light_src:
        x_expr, u1, approx_val, u2, src = m_light_src.groups()
        x_latex = to_latex(x_expr.strip())
        u = u1 or u2 or "m"
        approx_str = rf" \approx {approx_val}\text{{ {u}}}" if approx_val else ""
        return rf"x = {x_latex}\text{{ {u}}}{approx_str} \quad \text{{(from light source {src})}}"

    # Handle triangle included angle max area: "theta = pi/2 (90 deg), maximum area = 30 cm^2"
    m_tri_ang = re.match(
        r"^theta\s*=\s*(.+?)\s*\((.+?)\s*deg\),\s*maximum\s+area\s*=\s*(.+?)\s*([a-zA-Z]+)\^?2$",
        ans_str,
        re.I
    )
    if m_tri_ang:
        th_val, th_deg, a_val, u = m_tri_ang.groups()
        th_tex = r"\frac{\pi}{2}" if "pi/2" in th_val else to_latex(th_val.strip())
        a_tex = to_latex(a_val.strip())
        return rf"\theta = {th_tex} \; ({th_deg}^\circ), \quad A_{{\max}} = {a_tex}\text{{ {u}}}^2"

    # Handle open-top box length twice width: "width W = ... cm (~... cm), length L = ... cm (~... cm), height H = ... cm (~... cm), minimum surface area S = ... cm^2 (~... cm^2)"
    m_box2w = re.match(
        r"^width\s+W\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*length\s+L\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*height\s+H\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*minimum\s+surface\s+area\s+S\s*=\s*(.+?)\s*([a-zA-Z]+)\^?2\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\^?2\)$",
        ans_str,
        re.I
    )
    if m_box2w:
        w_e, u1, w_v, _, l_e, u2, l_v, _, h_e, u3, h_v, _, s_e, u4, s_v, _ = m_box2w.groups()
        u = u1 or "cm"
        return (
            rf"W = {to_latex(w_e.strip())}\text{{ {u}}} \approx {w_v}\text{{ {u}}}, \quad "
            rf"L = {to_latex(l_e.strip())}\text{{ {u}}} \approx {l_v}\text{{ {u}}}, \quad "
            rf"H = {to_latex(h_e.strip())}\text{{ {u}}} \approx {h_v}\text{{ {u}}}, \quad "
            rf"S_{{\min}} = {to_latex(s_e.strip())}\text{{ {u}}}^2 \approx {s_v}\text{{ {u}}}^2"
        )

    # Handle projectile launch angle: "theta = pi/4 (45 deg), maximum range = 20 m"
    m_proj = re.match(
        r"^theta\s*=\s*(.+?)\s*\((.+?)\s*deg\),\s*maximum\s+range\s*=\s*(.+?)\s*([a-zA-Z]+)$",
        ans_str,
        re.I
    )
    if m_proj:
        th_val, th_deg, r_val, u = m_proj.groups()
        th_tex = r"\frac{\pi}{4}" if "pi/4" in th_val else to_latex(th_val.strip())
        r_tex = to_latex(r_val.strip())
        return rf"\theta = {th_tex} \; ({th_deg}^\circ), \quad R_{{\max}} = {r_tex}\text{{ {u}}}"

    # Handle horizontal parabola inscribed rectangle: "width W = 6, height H = 2*sqrt(3) (~3.4641), maximum area = 12*sqrt(3) (~20.7846)"
    m_h_parab = re.match(
        r"^width\s+W\s*=\s*(.+?),\s*height\s+H\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\),\s*maximum\s+area\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\)$",
        ans_str,
        re.I
    )
    if m_h_parab:
        w_val, h_exact, h_val, a_exact, a_val = m_h_parab.groups()
        return (
            rf"W = {to_latex(w_val.strip())}, \quad "
            rf"H = {to_latex(h_exact.strip())} \approx {h_val}, \quad "
            rf"A_{{\max}} = {to_latex(a_exact.strip())} \approx {a_val}"
        )

    # Handle cubic tangent minimum slope: "(2, 12), minimum slope m = 3"
    m_cub_slope = re.match(
        r"^\((.+?),\s*(.+?)\),\s*minimum\s+slope\s+m\s*=\s*(.+?)$",
        ans_str,
        re.I
    )
    if m_cub_slope:
        x_val, y_val, m_val = m_cub_slope.groups()
        return rf"\left({to_latex(x_val.strip())}, {to_latex(y_val.strip())}\right), \quad m_{{\min}} = {to_latex(m_val.strip())}"

    # Handle fish swimming upstream: "v = 6 km/h"
    m_fish = re.match(r"^v\s*=\s*(\d+(?:\.\d+)?)\s*([a-zA-Z/]+)$", ans_str, re.I)
    if m_fish:
        v_val, u_spd = m_fish.groups()
        return rf"v = {to_latex(v_val)}\text{{ {u_spd}}}"

    # Handle right triangle fixed area min leg sum: "legs x = 6 cm, y = 6 cm, hypotenuse h = 6*sqrt(2) cm (~8.4853 cm)"
    m_rt_leg_sum = re.match(
        r"^legs\s+x\s*=\s*(.+?)\s*([a-zA-Z]+),\s*y\s*=\s*(.+?)\s*([a-zA-Z]+),\s*hypotenuse\s+h\s*=\s*(.+?)\s*([a-zA-Z]+)(?:\s*\((?:~|\approx)?\s*([0-9\.]+)\s*[a-zA-Z]+\))?$",
        ans_str,
        re.I
    )
    if m_rt_leg_sum:
        x_val, u1, y_val, u2, h_val, u3, h_approx = m_rt_leg_sum.groups()
        approx_str = rf" \approx {h_approx}\text{{ {u3}}}" if h_approx else ""
        return rf"x = {to_latex(x_val)}\text{{ {u1}}}, \quad y = {to_latex(y_val)}\text{{ {u2}}}, \quad h = {to_latex(h_val)}\text{{ {u3}}}{approx_str}"

    # Handle light ray reflection Fermat path: "x = 2, minimum path length L = 6*sqrt(2) (~8.4853)"
    m_fermat = re.match(
        r"^x\s*=\s*(.+?),\s*minimum\s+path\s+length\s+L\s*=\s*(.+?)(?:\s*\((?:~|\approx)?\s*([0-9\.]+)\))?$",
        ans_str,
        re.I
    )
    if m_fermat:
        x_val, L_val, L_approx = m_fermat.groups()
        approx_str = f" \\approx {L_approx}" if L_approx else ""
        return rf"x = {to_latex(x_val)}, \quad L_{{\min}} = {to_latex(L_val)}{approx_str}"

    # Handle cylinder in sphere lateral area: "r = 5*sqrt(2)/2 cm (~3.5355 cm), H = 5*sqrt(2) cm (~7.0711 cm), maximum lateral area S = 50*pi cm^2 (~157.0796 cm^2)"
    m_cyl_lat = re.match(
        r"^r\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*H\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*maximum\s+lateral\s+area\s+S\s*=\s*(.+?)\s*([a-zA-Z]+)\^?2\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\^?2\)$",
        ans_str,
        re.I
    )
    if m_cyl_lat:
        r_val, u1, r_app, _, h_val, u2, h_app, _, s_val, u3, s_app, _ = m_cyl_lat.groups()
        clean_s = s_val.replace("*pi", "").strip()
        s_latex = to_latex(clean_s) + r"\pi" if "*pi" in s_val else to_latex(s_val.strip())
        return (
            rf"r = {to_latex(r_val)}\text{{ {u1}}} \approx {r_app}\text{{ {u1}}}, \quad "
            rf"H = {to_latex(h_val)}\text{{ {u2}}} \approx {h_app}\text{{ {u2}}}, \quad "
            rf"S_{{\max}} = {s_latex}\text{{ {u3}}}^2 \approx {s_app}\text{{ {u3}}}^2"
        )

    # Handle chemical tank hemispherical ends: "r = 5 / 2^(1/3) m (~3.9685 m), h = 10 * 2^(2/3) m (~15.8740 m)"
    m_chem_tank = re.match(
        r"^r\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*h\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\)$",
        ans_str,
        re.I
    )
    if m_chem_tank:
        r_val, u1, r_app, _, h_val, u2, h_app, _ = m_chem_tank.groups()
        return (
            rf"r = {to_latex(r_val.strip())}\text{{ {u1}}} \approx {r_app}\text{{ {u1}}}, \quad "
            rf"h = {to_latex(h_val.strip())}\text{{ {u2}}} \approx {h_app}\text{{ {u2}}}"
        )

    # Handle damped harmonic oscillator extrema: "Absolute maximum at t = pi/4 with value sqrt(2)/2 * exp(-pi/4) (~0.3224); Absolute minimum at t = 5*pi/4 with value -sqrt(2)/2 * exp(-5*pi/4) (~-0.0139)"
    m_damped = re.match(
        r"^Absolute\s+maximum\s+at\s+t\s*=\s*(.+?)\s+with\s+value\s+(.+?)\s*\((?:~|\approx)?\s*([0-9\.\-]+)\);\s*Absolute\s+minimum\s+at\s+t\s*=\s*(.+?)\s+with\s+value\s+(.+?)\s*\((?:~|\approx)?\s*([0-9\.\-]+)\)$",
        ans_str,
        re.I
    )
    if m_damped:
        t_max, v_max, app_max, t_min, v_min, app_min = m_damped.groups()
        t_max_tex = r"\frac{\pi}{4}" if "pi/4" in t_max else to_latex(t_max.strip())
        t_min_tex = r"\frac{5\pi}{4}" if "5*pi/4" in t_min else to_latex(t_min.strip())
        return (
            rf"\text{{Abs. Max: }} t = {t_max_tex}, \; f\left({t_max_tex}\right) = {to_latex(v_max.strip())} \approx {app_max}; \quad "
            rf"\text{{Abs. Min: }} t = {t_min_tex}, \; f\left({t_min_tex}\right) = {to_latex(v_min.strip())} \approx {app_min}"
        )

    # Handle paper folding diagonal crease: "x = 9 cm, minimum crease length L = 9*sqrt(3) cm (~15.5885 cm)"
    m_fold = re.match(
        r"^x\s*=\s*(.+?)\s*([a-zA-Z]+),\s*minimum\s+crease\s+length\s+L\s*=\s*(.+?)\s*([a-zA-Z]+)(?:\s*\((?:~|\approx)?\s*([0-9\.]+)\s*[a-zA-Z]+\))?$",
        ans_str,
        re.I
    )
    if m_fold:
        x_val, u1, L_val, u2, L_app = m_fold.groups()
        approx_str = rf" \approx {L_app}\text{{ {u2}}}" if L_app else ""
        return rf"x = {to_latex(x_val.strip())}\text{{ {u1}}}, \quad L_{{\min}} = {to_latex(L_val.strip())}\text{{ {u2}}}{approx_str}"

    # Handle cubic cost average cost: "x = 6 units, AC(6) = 24, MC(6) = 24 (verifying C'(6) = AC(6) = 24)"
    m_cubic_cost = re.match(
        r"^x\s*=\s*(.+?)\s*units,\s*AC\(\d+\)\s*=\s*(.+?),\s*MC\(\d+\)\s*=\s*(.+?)\s*\(verifying\s+C'\(\d+\)\s*=\s*AC\(\d+\)\s*=\s*(.+?)\)$",
        ans_str,
        re.I
    )
    if m_cubic_cost:
        x_val, ac_val, mc_val, ver_val = m_cubic_cost.groups()
        return (
            rf"x = {to_latex(x_val.strip())}\text{{ units}}, \quad "
            rf"AC({to_latex(x_val.strip())}) = {ac_val}, \quad "
            rf"MC({to_latex(x_val.strip())}) = {mc_val} \quad "
            rf"\left(C'({to_latex(x_val.strip())}) = AC({to_latex(x_val.strip())}) = {ver_val}\right)"
        )

    # Handle semicircle inscribed rectangle perimeter: "width 2x = ... cm (~... cm), height y = ... cm (~... cm), maximum perimeter P = ... cm (~... cm)"
    m_semi_p = re.match(
        r"^width\s+2x\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*height\s+y\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*maximum\s+perimeter\s+P\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\)$",
        ans_str,
        re.I
    )
    if m_semi_p:
        w_e, u1, w_v, _, y_e, u2, y_v, _, p_e, u3, p_v, _ = m_semi_p.groups()
        u = u1 or "cm"
        return (
            rf"2x = {to_latex(w_e.strip())}\text{{ {u}}} \approx {w_v}\text{{ {u}}}, \quad "
            rf"y = {to_latex(y_e.strip())}\text{{ {u}}} \approx {y_v}\text{{ {u}}}, \quad "
            rf"P_{{\max}} = {to_latex(p_e.strip())}\text{{ {u}}} \approx {p_v}\text{{ {u}}}"
        )

    # Handle triangle circumscribed around rectangle: "height H = ... cm, base B = ... cm, minimum area A = ... cm^2"
    m_tri_rect = re.match(
        r"^height\s+H\s*=\s*(.+?)\s*([a-zA-Z]+),\s*base\s+B\s*=\s*(.+?)\s*([a-zA-Z]+),\s*minimum\s+area\s+A\s*=\s*(.+?)\s*([a-zA-Z]+)\^?2$",
        ans_str,
        re.I
    )
    if m_tri_rect:
        h_val, u1, b_val, u2, a_val, u3 = m_tri_rect.groups()
        return (
            rf"H = {to_latex(h_val.strip())}\text{{ {u1}}}, \quad "
            rf"B = {to_latex(b_val.strip())}\text{{ {u2}}}, \quad "
            rf"A_{{\min}} = {to_latex(a_val.strip())}\text{{ {u3}}}^2"
        )

    # Handle ellipse tangent intercepted length: "minimum length L = ... cm (with intercepts X = ... cm, Y = ... cm)"
    m_ell_tan = re.match(
        r"^minimum\s+length\s+L\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\(with\s+intercepts\s+X\s*=\s*(.+?)\s*([a-zA-Z]+),\s*Y\s*=\s*(.+?)\s*([a-zA-Z]+)\)$",
        ans_str,
        re.I
    )
    if m_ell_tan:
        l_val, u1, x_val, u2, y_val, u3 = m_ell_tan.groups()
        return (
            rf"L_{{\min}} = {to_latex(l_val.strip())}\text{{ {u1}}} \quad "
            rf"\left(X = {to_latex(x_val.strip())}\text{{ {u2}}}, \; Y = {to_latex(y_val.strip())}\text{{ {u3}}}\right)"
        )

    # Handle Y-shaped road network: "x = ... km (~... km), minimum total road length L = ... km (~... km)"
    m_y_road = re.match(
        r"^x\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*minimum\s+total\s+road\s+length\s+L\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\)$",
        ans_str,
        re.I
    )
    if m_y_road:
        x_e, u1, x_v, _, l_e, u2, l_v, _ = m_y_road.groups()
        return (
            rf"x = {to_latex(x_e.strip())}\text{{ {u1}}} \approx {x_v}\text{{ {u1}}}, \quad "
            rf"L_{{\min}} = {to_latex(l_e.strip())}\text{{ {u2}}} \approx {l_v}\text{{ {u2}}}"
        )

    # Handle curve and line max vertical distance: "x = ... (~...), maximum vertical distance D = ... (~...)"
    m_curve_dist = re.match(
        r"^x\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\),\s*maximum\s+vertical\s+distance\s+D\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\)$",
        ans_str,
        re.I
    )
    if m_curve_dist:
        x_e, x_v, d_e, d_v = m_curve_dist.groups()
        return (
            rf"x = {to_latex(x_e.strip())} \approx {x_v}, \quad "
            rf"D_{{\max}} = {to_latex(d_e.strip())} \approx {d_v}"
        )

    # Handle cone lateral surface area or cylinder cost: "radius r = ... m (~... m), height h = ... m (~... m)"
    m_r_h_dim = re.match(
        r"^radius\s+r\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\),\s*height\s+h\s*=\s*(.+?)\s*([a-zA-Z]+)\s*\((?:~|\approx)?\s*([0-9\.]+)\s*([a-zA-Z]+)\)$",
        ans_str,
        re.I
    )
    if m_r_h_dim:
        r_e, u1, r_v, _, h_e, u2, h_v, _ = m_r_h_dim.groups()
        return (
            rf"r = {to_latex(r_e.strip())}\text{{ {u1}}} \approx {r_v}\text{{ {u1}}}, \quad "
            rf"h = {to_latex(h_e.strip())}\text{{ {u2}}} \approx {h_v}\text{{ {u2}}}"
        )

    # Handle rancher corral budget: "dimensions L = ... m, W = ... m, maximum area A = ... m^2"
    m_corral_dim = re.match(
        r"^dimensions\s+L\s*=\s*(\d+)\s*([a-zA-Z]+),\s*W\s*=\s*(\d+)\s*([a-zA-Z]+),\s*maximum\s+area\s+A\s*=\s*(\d+)\s*([a-zA-Z]+)\^?2$",
        ans_str,
        re.I
    )
    if m_corral_dim:
        l_v, u1, w_v, u2, a_v, u3 = m_corral_dim.groups()
        return rf"L = {l_v}\text{{ {u1}}}, \quad W = {w_v}\text{{ {u2}}}, \quad A_{{\max}} = {a_v}\text{{ {u3}}}^2"

    # Handle parabola inscribed triangle apex: "vertex (x, y) = (..., ...) (~(..., ...)), maximum area A = ... (~...)"
    m_parab_apex = re.match(
        r"^vertex\s+\(x,\s*y\)\s*=\s*\((.+?),\s*(.+?)\)\s*\((?:~|\approx)?\s*\((.+?),\s*(.+?)\)\),\s*maximum\s+area\s+A\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\)$",
        ans_str,
        re.I
    )
    if m_parab_apex:
        x_e, y_e, x_v, y_v, a_e, a_v = m_parab_apex.groups()
        return (
            rf"(x, y) = \left({to_latex(x_e.strip())}, {to_latex(y_e.strip())}\right) \approx ({x_v}, {y_v}), \quad "
            rf"A_{{\max}} = {to_latex(a_e.strip())} \approx {a_v}"
        )

    # Handle curve closest point to line: "point (0, 1), minimum distance d = ... (~...)"
    m_curve_pt_d = re.match(
        r"^point\s+\((.+?),\s*(.+?)\),\s*minimum\s+distance\s+d\s*=\s*(.+?)\s*\((?:~|\approx)?\s*([0-9\.]+)\)$",
        ans_str,
        re.I
    )
    if m_curve_pt_d:
        px, py, d_e, d_v = m_curve_pt_d.groups()
        return (
            rf"(x_0, y_0) = \left({px}, {py}\right), \quad "
            rf"d_{{\min}} = {to_latex(d_e.strip())} \approx {d_v}"
        )




    # Check if answer has unit / description suffix e.g. "-3/2 ft/s (sliding down at 1.5 ft/s)", "-1/(8*pi) m/min (dropping at 1/(8*pi) m/min)", "50 mph"
    unit_pat = (
        r"(?:cm|ft|m|in)\^?(?:2|²|3|³)/(?:min|s|sec)|"
        r"[a-zA-Z]+/[a-zA-Z]+|"
        r"mph|knots|km/h|m/s|ft/s|ft/min|m/min|cm/s|cm/min|in/s|in/min|rad/s|rad/min|"
        r"sq\s+[a-zA-Z]+/[a-zA-Z]+"
    )
    m_units = re.match(
        rf"^([-\w\.\/\s\*\^\(\)\+]+?)\s*({unit_pat})\s*(\(.*\))?$",
        ans_str
    )
    if m_units:
        num = m_units.group(1).strip()
        unit = m_units.group(2).strip()
        desc = m_units.group(3)
        num_latex = to_latex(num)
        unit_latex = unit.replace("³", "^3").replace("²", "^2")
        if desc:
            m_combined = re.match(
                r"^\((?:~|\approx)\s*([0-9\.\-]+)\s*([a-zA-Z/]+)?,\s*(decreasing|increasing)\s+at\s+([-\w\.\/\s\*\^\(\)]+?)\s*([a-zA-Z/]+)?\)$",
                desc
            )
            if m_combined:
                val, u1, action, sub_num, u2 = m_combined.groups()
                u1 = (u1 or unit).strip().replace("³", "^3").replace("²", "^2")
                u2 = (u2 or unit).strip().replace("³", "^3").replace("²", "^2")
                sub_num_latex = to_latex(sub_num.strip())
                return rf"{num_latex}\text{{ {unit_latex}}} \quad \text{{(}}\approx {val}\text{{ {u1}, {action} at }}{sub_num_latex}\text{{ {u2})}}"
            m_desc = re.match(
                r"^\((dropping|falling|sliding down|deflating|rising|filling|climbing|expanding|decreasing|increasing)\s+at\s+([-\w\.\/\s\*\^\(\)]+?)\s*([a-zA-Z]+/[a-zA-Z]+|mph|knots|km/h|m/s|ft/s|ft/min|m/min|cm/s|cm/min|in/s|in/min|rad/s|rad/min)?\)$",
                desc
            )
            if m_desc:
                action = m_desc.group(1)
                sub_num = m_desc.group(2).strip()
                sub_unit = m_desc.group(3)
                sub_num_latex = to_latex(sub_num)
                if sub_unit:
                    desc_latex = rf"\text{{({action} at }}{sub_num_latex}\text{{ {sub_unit})}}"
                else:
                    desc_latex = rf"\text{{({action} at }}{sub_num_latex}\text{{)}}"
                return rf"{num_latex}\text{{ {unit_latex}}} \quad {desc_latex}"
            m_approx = re.match(
                r"^\((?:~|\approx)\s*([0-9\.\-]+)\s*(.+?)?\)$",
                desc
            )
            if m_approx:
                val = m_approx.group(1)
                u = (m_approx.group(2) or unit).strip().replace("³", "^3").replace("²", "^2")
                return rf"{num_latex}\text{{ {unit_latex}}} \quad \text{{(}}\approx {val}\text{{ {u})}}"
            m_alt_val = re.match(r"^\(([0-9\.\-]+)\s*(.+?)?\)$", desc)
            if m_alt_val:
                val = m_alt_val.group(1)
                u = (m_alt_val.group(2) or unit).strip().replace("³", "^3").replace("²", "^2")
                return rf"{num_latex}\text{{ {unit_latex}}} \quad \text{{(}}= {val}\text{{ {u})}}"
            return rf"{num_latex}\text{{ {unit_latex}}} \quad \text{{{desc}}}"
        return rf"{num_latex}\text{{ {unit_latex}}}"

    # Handle parenthetical descriptions / notes e.g. "~ -0.1475 (no elementary closed form)" or "pi/4 (~0.7854)"
    m_note = re.match(r"^(?:~|\approx)?\s*(.+?)\s+\(((?:~|\approx|[a-zA-Z\s]{3,}|no\s+elementary)[^)]*)\)$", ans_str)
    if m_note:
        expr_part = m_note.group(1).strip()
        note_part = m_note.group(2).strip()
        is_approx_prefix = ans_str.startswith("~") or ans_str.startswith("≈") or ans_str.startswith(r"\approx")
        expr_latex = to_latex(expr_part)

        m_approx_note = re.match(r"^(?:~|\approx)\s*([-\d\.]+)\s*(.*)$", note_part)
        if m_approx_note:
            num = m_approx_note.group(1)
            u = m_approx_note.group(2)
            u_str = f" {u}" if u else ""
            return rf"{expr_latex} \quad \text{{(}}\approx {num}{u_str}\text{{)}}"
        elif is_approx_prefix:
            return rf"\approx {expr_latex} \quad \text{{({note_part})}}"
        else:
            return rf"{expr_latex} \quad \text{{({note_part})}}"

    m_approx_simple = re.match(r"^(?:~|\approx)\s*([-\d\.\/]+)$", ans_str)
    if m_approx_simple:
        num_latex = to_latex(m_approx_simple.group(1).strip())
        return rf"\approx {num_latex}"

    # Split disjunctions (e.g. quadratic solutions like 'x = -4 \lor x = 3' or '... or ...')
    parts = re.split(r'\s+(?:\\lor|\\vee|or)\s+', ans_str)
    if len(parts) > 1:
        latex_parts = [to_latex(p) for p in parts]
        return r" \quad\text{or}\quad ".join(filter(None, latex_parts))

    # Split equations 'lhs = rhs'
    if "=" in ans_str:
        sides = ans_str.split("=", 1)
        lhs = to_latex(sides[0].strip())
        rhs = to_latex(sides[1].strip())
        return f"{lhs} = {rhs}"

    # Handle integration constant '+ C' or '+ c'
    has_c = False
    m_c = re.search(r'\s*\+\s*[Cc]$', ans_str)
    if m_c:
        ans_str = ans_str[:m_c.start()].strip()
        has_c = True

    # Standardize function names for sympy
    ans_str = ans_str.replace("arctan", "atan")
    ans_str = ans_str.replace("arcsin", "asin")
    ans_str = ans_str.replace("arccos", "acos")

    try:
        import sympy
        ans_clean = ans_str.replace("^", "**")
        expr = sympy.sympify(ans_clean)
        l = sympy.latex(expr)
        l = l.replace(r"\operatorname{atan}", r"\arctan")
        l = l.replace(r"\operatorname{asin}", r"\arcsin")
        l = l.replace(r"\operatorname{acos}", r"\arccos")
        l = l.replace(r"\log", r"\ln")
        if has_c:
            l += r" + C"
        return l
    except Exception:
        # Never wrap mathematical expressions with operators (+, -, *, /, ^, =) in \text{}
        # because exponents like ^2 inside \text{} cause fatal KaTeX parse errors.
        if " " in ans_str and re.search(r'[a-zA-Z]{3,}', ans_str):
            if not re.search(r'[\^+\-*/=]', ans_str):
                return rf"\text{{{ans_str}}}"
        return ans_str


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/solve", response_model=SolveResponse)
async def solve_endpoint(req: SolveRequest) -> SolveResponse:
    """
    Full VeriMath pipeline: classify → solve → formalize → judge → verify → explain.

    Returns a verified answer or a clear rejection/unsupported message.
    """
    problem = re.sub(r"\s+", " ", req.problem).strip()
    if not problem:
        raise HTTPException(status_code=400, detail="Problem text cannot be empty.")

    pipeline_start = time.monotonic()

    # ── Step 0: Check demo cache ──────────────────────────────────────────────
    cached = demo_cache.get(problem)
    if cached:
        elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)
        _publish_event(
            problem=problem,
            verdict="VERIFIED",
            elapsed_ms=elapsed_ms,
            template_id=cached.template_id,
            candidate_answer=cached.verified_answer,
            lean_statement=cached.lean_statement,
            judge_verdict="MATCH",
            retry_count=0,
            from_cache=True,
            student_id=req.student_id,
        )
        cached_steps = [s.model_dump() for s in generate_steps(cached.template_id, problem, cached.verified_answer)]
        cached_guide = build_intuitive_guide(
            problem=problem,
            verified_answer=cached.verified_answer,
            reasoning=cached.explanation or "",
            template_id=cached.template_id,
        )
        return SolveResponse(
            status="VERIFIED",
            verified_answer=_clean_lean_syntax(cached.verified_answer),
            latex_answer=to_latex(cached.verified_answer),
            lean_statement=cached.lean_statement,
            back_translation=_clean_lean_syntax(cached.back_translation),
            explanation=cached_guide.full_explanation,
            mental_model=cached_guide.mental_model,
            pitfall_tip=cached_guide.pitfall_tip,
            steps=cached_steps,
            template_id=cached.template_id,
            elapsed_ms=elapsed_ms,
            from_cache=True,
        )

    # ── Step 1: Classify ──────────────────────────────────────────────────────
    template_id = classify_problem(problem)

    if template_id is None:
        # Novel Out-of-Domain Generalized Propose → AST Anti-Cheating Audit → Lean Verify / Disprove
        from backend.pipeline.generalized_prover import propose_generalized
        from backend.pipeline.statement_judge import _check_numbers_match, _check_semantic_inversions
        from backend.pipeline.ast_faithfulness import verify_ast_faithfulness
        from backend.pipeline.counterexample import synthesize_counterexample_proposal

        retry_fb: Optional[str] = None
        last_reason = "Could not formally verify a generalized solution for this out-of-domain problem."
        last_lean_stmt: Optional[str] = None

        for attempt in range(1, MAX_RETRIES + 1):
            proposal = propose_generalized(problem, feedback=retry_fb)
            if not proposal.is_math_problem:
                elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)
                _publish_event(problem=problem, verdict="UNSUPPORTED", elapsed_ms=elapsed_ms, student_id=req.student_id)
                return SolveResponse(
                    status="UNSUPPORTED",
                    rejection_reason=(
                        "This prompt does not appear to be a verifiable mathematical problem."
                    ),
                    elapsed_ms=elapsed_ms,
                )

            if not proposal.lean_statement:
                last_reason = proposal.reasoning or last_reason
                break

            last_lean_stmt = proposal.lean_statement

            num_chk = _check_numbers_match(problem, proposal.back_translation)
            inv_chk = _check_semantic_inversions(problem, proposal.back_translation)
            ast_chk = verify_ast_faithfulness(problem, proposal.lean_statement, subkind=proposal.subkind)
            if not num_chk.passed or not inv_chk.passed or not ast_chk.passed:
                last_reason = (
                    ast_chk.reason
                    or num_chk.reason
                    or inv_chk.reason
                    or "Faithfulness mismatch on generalized proposal."
                )
                retry_fb = last_reason
                continue

            lean_res = await verify_async(proposal.lean_statement)
            logger.info(
                f"Generalized Lean [{attempt}] ({proposal.subkind}): "
                f"{lean_res.verdict} in {lean_res.elapsed_ms}ms"
            )

            if lean_res.verdict == LeanVerdict.VERIFIED:
                elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)
                guide = build_intuitive_guide(
                    problem=problem,
                    verified_answer=proposal.candidate_answer,
                    reasoning=proposal.reasoning,
                    template_id=proposal.subkind,
                )
                explanation = explain(
                    problem=problem,
                    verified_answer=proposal.candidate_answer,
                    reasoning=proposal.reasoning,
                    template_id=proposal.subkind,
                )
                _publish_event(
                    problem=problem,
                    verdict="VERIFIED",
                    elapsed_ms=elapsed_ms,
                    template_id=proposal.subkind,
                    candidate_answer=proposal.candidate_answer,
                    lean_statement=proposal.lean_statement,
                    judge_verdict="MATCH",
                    retry_count=attempt - 1,
                    student_id=req.student_id,
                )
                return SolveResponse(
                    status="VERIFIED",
                    verified_answer=_clean_lean_syntax(proposal.candidate_answer),
                    latex_answer=to_latex(proposal.candidate_answer),
                    lean_statement=proposal.lean_statement,
                    back_translation=_clean_lean_syntax(proposal.back_translation),
                    explanation=explanation,
                    mental_model=guide.mental_model,
                    pitfall_tip=guide.pitfall_tip,
                    steps=[
                        {
                            "step_number": 1,
                            "title": "Generalized Concept & Proposal",
                            "explanation": guide.intuitive_body,
                            "math": to_latex(proposal.candidate_answer),
                        }
                    ],
                    template_id=proposal.subkind,
                    elapsed_ms=elapsed_ms,
                    generalized_mode=True,
                    ast_verified=True,
                    ast_checks=ast_chk.checks_run,
                )
            else:
                last_reason = (
                    f"Generalized proposal '{proposal.candidate_answer}' was rejected by formal verification: "
                    f"{lean_res.lean_output[:200]}"
                )
                retry_fb = last_reason

        # Attempt Lean-certified counterexample synthesis when a false claim is rejected
        ce_dict: Optional[Dict[str, Any]] = None
        ce_prop = synthesize_counterexample_proposal(problem, last_lean_stmt)
        if ce_prop and ce_prop.found and ce_prop.disproof_lean_statement:
            ce_lean = await verify_async(ce_prop.disproof_lean_statement)
            logger.info(
                f"Counterexample Lean disproof verification: {ce_lean.verdict} in {ce_lean.elapsed_ms}ms "
                f"(witness={ce_prop.witness_display})"
            )
            if ce_lean.verdict == LeanVerdict.VERIFIED:
                ce_prop.lean_verified = True
                ce_dict = ce_prop.to_dict()
                last_reason = (
                    f"Mathematically Disproven (Counterexample verified in Lean 4 at "
                    f"{ce_prop.witness_display}): {ce_prop.explanation}"
                )

        # Strictly reject false or unverified out-of-domain proposals
        elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)
        _publish_event(
            problem=problem,
            verdict="REJECTED",
            elapsed_ms=elapsed_ms,
            template_id="generalized_rejected",
            student_id=req.student_id,
        )
        return SolveResponse(
            status="REJECTED",
            rejection_reason=last_reason,
            template_id="generalized_rejected",
            elapsed_ms=elapsed_ms,
            generalized_mode=True,
            counterexample=ce_dict,
        )

    logger.info(f"Problem classified as: '{template_id}'")

    # ── Steps 2–6: Full pipeline with retries ─────────────────────────────────
    retry_count = 0
    last_rejection_reason: Optional[str] = None
    retry_feedback: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Step 2: Solve (incorporating feedback from previous attempt if rejected)
            solver_result = solve(problem, feedback=retry_feedback)
            logger.info(
                f"Solver [{attempt}]: answer='{solver_result.candidate_answer}', "
                f"confidence={solver_result.confidence}"
            )

            # Step 3: Formalize
            formalizer_result = formalize(problem, solver_result, template_id)
            logger.info(f"Formalizer [{attempt}]: {formalizer_result.lean_statement}")

            # Step 4: Statement judge
            judge_result = judge(problem, formalizer_result)
            logger.info(
                f"Judge [{attempt}]: {judge_result.verdict} "
                f"(score={judge_result.similarity_score:.2f})"
            )

            if judge_result.verdict == "MISMATCH":
                retry_count += 1
                last_rejection_reason = (
                    f"Formalization mismatch: {judge_result.reason}"
                )
                retry_feedback = (
                    f"Your previous candidate answer was '{solver_result.candidate_answer}', but formalization "
                    f"failed faithfulness validation with reason: {judge_result.reason}. "
                    "Ensure your answer matches all problem parameters and constraints."
                )
                logger.warning(f"Statement judge MISMATCH on attempt {attempt}. Retrying with feedback...")
                continue  # Try again with feedback passed back to solver

            # Step 5: Lean verification
            lean_result = await verify_async(formalizer_result.lean_statement)
            logger.info(f"Lean [{attempt}]: {lean_result.verdict} in {lean_result.elapsed_ms}ms")

            if lean_result.verdict == LeanVerdict.VERIFIED:
                # Step 6: Explain (Intuitive Textbook Pedagogy Layer)
                guide = build_intuitive_guide(
                    problem=problem,
                    verified_answer=solver_result.candidate_answer,
                    reasoning=solver_result.reasoning,
                    template_id=template_id,
                )
                explanation = explain(
                    problem=problem,
                    verified_answer=solver_result.candidate_answer,
                    reasoning=solver_result.reasoning,
                    template_id=template_id,
                )

                elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)

                # Publish telemetry
                _publish_event(
                    problem=problem,
                    verdict="VERIFIED",
                    elapsed_ms=elapsed_ms,
                    template_id=template_id,
                    candidate_answer=solver_result.candidate_answer,
                    lean_statement=formalizer_result.lean_statement,
                    judge_verdict=judge_result.verdict,
                    retry_count=retry_count,
                    student_id=req.student_id,
                )

                # Cache verified answer for instant future edge retrieval (<10ms)
                try:
                    demo_cache.put(
                        problem=problem,
                        template_id=template_id,
                        verified_answer=solver_result.candidate_answer,
                        lean_statement=formalizer_result.lean_statement,
                        back_translation=judge_result.back_translation,
                        explanation=explanation,
                        elapsed_ms=elapsed_ms,
                    )
                except Exception as cache_err:
                    logger.warning(f"Failed to cache verified result: {cache_err}")

                steps_objs = generate_steps(
                    template_id=template_id,
                    problem=problem,
                    answer=solver_result.candidate_answer,
                    slots=getattr(formalizer_result, "slots", None),
                )
                steps_dicts = [s.model_dump() for s in steps_objs]

                return SolveResponse(
                    status="VERIFIED",
                    verified_answer=_clean_lean_syntax(solver_result.candidate_answer),
                    latex_answer=to_latex(solver_result.candidate_answer),
                    lean_statement=formalizer_result.lean_statement,
                    back_translation=_clean_lean_syntax(judge_result.back_translation),
                    explanation=explanation,
                    mental_model=guide.mental_model,
                    pitfall_tip=guide.pitfall_tip,
                    steps=steps_dicts,
                    template_id=template_id,
                    elapsed_ms=elapsed_ms,
                    from_cache=False,
                    ast_verified=True,
                    ast_checks=judge_result.ast_checks,
                )

            else:
                # Lean rejected — pass feedback back to LLM for next attempt
                retry_count += 1
                verdict_label = lean_result.verdict.value
                last_rejection_reason = (
                    f"Lean could not verify the proposed answer "
                    f"({verdict_label}). "
                    "The proposed answer may be incorrect."
                )
                retry_feedback = (
                    f"The candidate answer '{solver_result.candidate_answer}' with formalization "
                    f"'{formalizer_result.lean_statement}' was REJECTED by Lean 4 ({verdict_label}).\n"
                    f"Lean output / error diagnostic:\n{lean_result.lean_output[:300]}\n"
                    "Please re-analyze the problem, find the error in your reasoning, and produce the corrected solution."
                )
                logger.info(
                    f"Lean rejected on attempt {attempt}/{MAX_RETRIES}: "
                    f"{lean_result.lean_output[:200]}. Cycling back to solver with self-correction feedback..."
                )
                continue

        except ValueError as e:
            # Solver or formalizer failed
            retry_count += 1
            last_rejection_reason = str(e)
            logger.warning(f"Pipeline error on attempt {attempt}: {e}")
            continue

    # All retries exhausted — return rejection
    elapsed_ms = int((time.monotonic() - pipeline_start) * 1000)
    _publish_event(
        problem=problem,
        verdict="REJECTED",
        elapsed_ms=elapsed_ms,
        template_id=template_id,
        retry_count=retry_count,
        student_id=req.student_id,
    )

    return SolveResponse(
        status="REJECTED",
        template_id=template_id,
        rejection_reason=(
            last_rejection_reason or
            "VeriMath could not verify an answer for this problem. "
            "The proposed answer was rejected by Lean 4."
        ),
        elapsed_ms=elapsed_ms,
    )


class GradeStepsRequest(BaseModel):
    problem: str
    student_steps: List[str]
    operation: str = "auto"
    lean_certify: bool = False


@app.post("/grade-steps")
async def grade_steps_endpoint(req: GradeStepsRequest):
    """Grade and diagnose a student's own multi-step derivation, optionally certifying in Lean 4."""
    from backend.pipeline.step_grader import grade_student_steps

    res = grade_student_steps(
        problem=req.problem,
        student_steps=req.student_steps,
        operation=req.operation,
    )
    if req.lean_certify and res.overall_correct and res.lean_certificate:
        lean_res = await verify_async(res.lean_certificate)
        res.lean_verified = lean_res.verdict == LeanVerdict.VERIFIED
    return res.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# External Camera, Handwriting & Math OCR Connectors
# ─────────────────────────────────────────────────────────────────────────────

async def _extract_scan_request(request: Request) -> tuple[bytes, dict]:
    """Read raw image bytes + options from either multipart/form-data upload or JSON base64 payload."""
    ctype = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ctype:
        form = await request.form()
        up = form.get("file")
        if up is None or not hasattr(up, "read"):
            raise HTTPException(status_code=400, detail="Missing 'file' upload in multipart form data.")
        img_bytes = await up.read()
        opts = {
            "scan_mode": str(form.get("scan_mode") or "auto"),
            "engine": str(form.get("engine") or "auto"),
            "enhance_handwriting": str(form.get("enhance_handwriting") or "true").lower() in ("1", "true", "yes"),
            "lean_certify": str(form.get("lean_certify") or "false").lower() in ("1", "true", "yes"),
            "operation": str(form.get("operation") or "auto"),
            "student_id": str(form.get("student_id") or "anonymous"),
        }
    else:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")
        b64 = body.get("image_base64") or ""
        if not b64:
            raise HTTPException(status_code=400, detail="Missing 'image_base64' field.")
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
        opts = {
            "scan_mode": str(body.get("scan_mode", "auto")),
            "engine": str(body.get("engine", "auto")),
            "enhance_handwriting": bool(body.get("enhance_handwriting", True)),
            "lean_certify": bool(body.get("lean_certify", False)),
            "operation": str(body.get("operation", "auto")),
            "student_id": str(body.get("student_id", "anonymous")),
        }

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload.")
    return img_bytes, opts


@app.post("/scan/ocr")
async def scan_ocr_endpoint(request: Request):
    """Extract math question & handwritten/printed steps from an uploaded or camera-captured image."""
    from backend.connectors.vision_ocr import extract_math_from_image

    img_bytes, opts = await _extract_scan_request(request)
    try:
        res = extract_math_from_image(
            img_bytes,
            engine=opts["engine"],
            scan_mode=opts["scan_mode"],
            enhance_handwriting=opts["enhance_handwriting"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode or process image: {e}")
    return res.to_dict()


@app.post("/scan/solve")
async def scan_solve_endpoint(request: Request):
    """Scan a math question image, extract the problem via OCR, and verify/solve it in Lean 4."""
    from backend.connectors.vision_ocr import extract_math_from_image

    img_bytes, opts = await _extract_scan_request(request)
    try:
        ocr_res = extract_math_from_image(
            img_bytes,
            engine=opts["engine"],
            scan_mode="question",
            enhance_handwriting=opts["enhance_handwriting"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image OCR failed: {e}")

    if not ocr_res.problem_text.strip():
        return {
            "ocr": ocr_res.to_dict(),
            "solve": SolveResponse(
                status="UNSUPPORTED",
                rejection_reason="No readable mathematical text detected in scanned image.",
                elapsed_ms=ocr_res.elapsed_ms,
            ).model_dump(),
        }

    solve_res = await solve_endpoint(
        SolveRequest(problem=ocr_res.problem_text, student_id=opts["student_id"])
    )
    return {
        "ocr": ocr_res.to_dict(),
        "solve": solve_res.model_dump(),
    }


@app.post("/scan/grade-steps")
async def scan_grade_steps_endpoint(request: Request):
    """Scan a handwritten problem + solution sheet and grade every derivation step."""
    from backend.connectors.vision_ocr import extract_math_from_image
    from backend.pipeline.step_grader import grade_student_steps

    img_bytes, opts = await _extract_scan_request(request)
    try:
        ocr_res = extract_math_from_image(
            img_bytes,
            engine=opts["engine"],
            scan_mode="question_and_answer",
            enhance_handwriting=opts["enhance_handwriting"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image OCR failed: {e}")

    steps = ocr_res.detected_steps if ocr_res.detected_steps else ([ocr_res.problem_text] if ocr_res.problem_text else [])
    grade_res = grade_student_steps(
        problem=ocr_res.problem_text or "Handwritten derivation",
        student_steps=steps,
        operation=opts["operation"],
    )
    if opts["lean_certify"] and grade_res.overall_correct and grade_res.lean_certificate:
        lean_res = await verify_async(grade_res.lean_certificate)
        grade_res.lean_verified = lean_res.verdict == LeanVerdict.VERIFIED

    return {
        "ocr": ocr_res.to_dict(),
        "grade": grade_res.to_dict(),
    }


@app.post("/scan/hardware-capture")
async def scan_hardware_capture_endpoint(
    device: Optional[str] = None,
    scan_mode: str = "auto",
    engine: str = "auto",
):
    """Capture a frame from an attached Raspberry Pi CSI or USB V4L2 camera and run OCR."""
    from backend.connectors.vision_ocr import capture_hardware_camera_frame, extract_math_from_image

    try:
        img_bytes = capture_hardware_camera_frame(device=device)
        ocr_res = extract_math_from_image(img_bytes, engine=engine, scan_mode=scan_mode)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hardware capture error: {e}")

    payload = ocr_res.to_dict()
    payload["captured_image_base64"] = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("ascii")
    return payload


@app.get("/connectors/status")
async def connectors_status_endpoint():
    """Report readiness of all external camera, OCR/handwriting, MQTT, Firebase, and Gemini connectors."""
    from backend.connectors.vision_ocr import get_connectors_status
    return get_connectors_status()


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """System health check — used by install script and monitoring."""
    import shutil
    from backend.connectors.vision_ocr import get_connectors_status

    lean_installed = shutil.which("lean") is not None or shutil.which("lake") is not None
    models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
    model_path = os.getenv("LLM_MODEL_PATH")
    if not model_path:
        candidates = [
            os.path.join(models_dir, "qwen2_5_1_5b_verimath_pi4-q4_k_m.gguf"),
            os.path.join(models_dir, "Qwen2.5-1.5B-Instruct.Q4_K_M.gguf"),
            os.path.join(models_dir, "qwen2.5-3b-instruct-q4_k_m.gguf"),
            os.path.join(models_dir, "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            os.path.join(models_dir, "phi-3-mini-4k-instruct-q4.gguf"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    model_exists = os.path.exists(model_path)

    cache_count = len(demo_cache.list_all())
    conn_status = get_connectors_status()

    return {
        "status": "ok",
        "lean_installed": lean_installed,
        "strict_lean_mode": os.getenv("VERIMATH_STRICT_LEAN", "false").lower() in ("1", "true", "yes"),
        "model_loaded": model_exists,
        "target_hardware": "Raspberry Pi 4 (8 GB RAM)",
        "demo_cache_size": cache_count,
        "supported_templates": list(TEMPLATES.keys()),
        "connectors": conn_status,
    }


@app.get("/cache/list")
async def list_cache():
    """List all pre-cached demo problems."""
    return [
        {
            "problem": c.problem,
            "template_id": c.template_id,
            "verified_answer": c.verified_answer,
            "cached_at": c.cached_at,
        }
        for c in demo_cache.list_all()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _publish_event(
    problem: str,
    verdict: str,
    elapsed_ms: int,
    template_id: Optional[str] = None,
    candidate_answer: Optional[str] = None,
    lean_statement: Optional[str] = None,
    judge_verdict: Optional[str] = None,
    retry_count: int = 0,
    from_cache: bool = False,
    student_id: str = "anonymous",
) -> None:
    """Build and publish a PipelineEvent (non-blocking, non-fatal)."""
    try:
        event = PipelineEvent.create(
            problem_text=problem,
            lean_verdict=verdict,
            total_elapsed_ms=elapsed_ms,
            template_id=template_id,
            candidate_answer=candidate_answer,
            lean_statement=lean_statement,
            judge_verdict=judge_verdict,
            retry_count=retry_count,
            from_cache=from_cache,
            student_id=student_id,
        )
        publish(event)
    except Exception as e:
        logger.warning(f"Telemetry publish failed (non-fatal): {e}")
