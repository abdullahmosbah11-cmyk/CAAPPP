"""
VeriMath — Module 5: Intuitive Textbook-Inspired Explainer Layer
==================================================================
Provides student-friendly conceptual intuition, verbal rules, geometric/physical
mental models, and common pitfall tips inspired by structured calculus and
algebra curricula (General Mathematics / Senior Calculus / Integration techniques).

Key design guarantees:
  1. Purely pedagogical & intuitive — NEVER mentions Lean, formal proofs,
     verification systems, or machine checking.
  2. Strictly complies with Rule 22 (zero Lean 4 syntax like `Real.sin`) and
     Rule 23 (zero external textbook names, authors, or page numbers in
     student-facing text).
  3. Sub-millisecond deterministic intuition by default (zero intermediate LLM
     hallucination on Raspberry Pi 4), with optional `EXPLAINER_USE_LLM=true`
     prose polishing when a local GGUF model is loaded.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional

from backend.pipeline.solver import _get_llm

logger = logging.getLogger(__name__)


@dataclass
class IntuitiveGuide:
    concept_title: str
    mental_model: str
    intuitive_body: str
    pitfall_tip: str
    full_explanation: str


# ─────────────────────────────────────────────────────────────────────────────
# Textbook-Inspired Conceptual Knowledge Base (Rule 22 & 23 Compliant)
# ─────────────────────────────────────────────────────────────────────────────

_CONCEPT_GUIDES: Dict[str, Dict[str, str]] = {
    "derivative_product": {
        "title": "Product Rule — Two Factors Changing Together",
        "mental_model": "Verbal Rule: (First × derivative of Second) + (Second × derivative of First).",
        "intuition": (
            "Think of u(x)·v(x) as the area of a rectangle whose width u and height v both grow as x changes. "
            "The total change isn't just u'·v' — you get one strip from stretching the height while holding the width steady (u·v'), "
            "plus a second strip from stretching the width while holding the height steady (v·u')."
        ),
        "pitfall": "Never multiply the derivatives directly: (u·v)' is NOT u'·v'. Always keep each original factor paired with the other factor's derivative.",
    },
    "derivative_quotient": {
        "title": "Quotient Rule — Balancing Numerator Growth vs. Denominator Scaling",
        "mental_model": "Verbal Rule: (Denominator × derivative of Numerator − Numerator × derivative of Denominator) ÷ Denominator².",
        "intuition": (
            "In a fraction u(x)/v(x), increasing the numerator pushes the fraction up, while increasing the denominator pulls it down. "
            "That opposing pull explains the minus sign in (v·u' − u·v')/v²: the numerator's growth helps, while the denominator's growth works against the ratio."
        ),
        "pitfall": "Order matters because of the subtraction! Always start with the denominator times the numerator's derivative, and never forget to square the denominator (leaving it unexpanded is usually cleanest).",
    },
    "derivative_chain": {
        "title": "Chain Rule — Peeling Layers Outside-In",
        "mental_model": "Outer derivative (evaluated at the untouched inner expression) × Inner derivative.",
        "intuition": (
            "Picture nested gears: if the outer gear turns 3 times faster than the middle gear, and the middle gear turns 2x faster than x, "
            "the overall rate multiplies together (3 × 2x). Differentiate the outer shell first while keeping the inside locked, then multiply by the rate of change of the inner core."
        ),
        "pitfall": "Do not differentiate inside the parentheses during the first step, and never drop the trailing inner-derivative factor at the end.",
    },
    "derivative_trig": {
        "title": "Trigonometric Differentiation — Rates of Oscillation",
        "mental_model": "Co-functions pick up a minus sign: sin' = cos, cos' = −sin, tan' = sec², cot' = −csc², sec' = sec·tan, csc' = −csc·cot.",
        "intuition": (
            "On the unit circle, sin(x) starts at 0 heading upward (+1 slope, matching cos(0) = 1), whereas cos(x) starts at its peak 1 and immediately drops (negative slope, matching −sin). "
            "Every co-function (cosine, cotangent, cosecant) carries a negative derivative."
        ),
        "pitfall": "Watch angles like sin(3x) or cos²(x): always apply the chain rule to the inner angle or outer power, and check whether trig identities (like sin²x + cos²x = 1) simplify the expression to a constant first.",
    },
    "derivative_log_exp": {
        "title": "Exponential & Logarithmic Rates — Self-Scaling Growth",
        "mental_model": "(e^u)' = e^u · u'  and  [ln(u)]' = u' / u.",
        "intuition": (
            "Exponential functions grow proportionally to their own current size, so e^x is its own derivative. "
            "Logarithms do the inverse: ln(u) measures relative/percentage growth, which is why its derivative is simply the rate of change divided by the current value (u'/u)."
        ),
        "pitfall": "For bases other than e (such as a^x or log_a(x)), remember the natural-log scaling factor ln(a): (a^x)' = a^x·ln(a) and (log_a x)' = 1/(x·ln(a)).",
    },
    "derivative_general": {
        "title": "Instantaneous Rate of Change (Power & Linearity Rules)",
        "mental_model": "Bring the exponent down in front and reduce the exponent by 1: d/dx[xⁿ] = n·xⁿ⁻¹.",
        "intuition": (
            "The derivative zooms into a curve until it looks like a straight line and measures that line's slope. "
            "Constants have flat slope 0, linear terms a·x have constant slope a, and higher powers bend at rates proportional to their lowered power."
        ),
        "pitfall": "Rewrite roots (√x = x^(1/2)) and reciprocals (1/xⁿ = x⁻ⁿ) as fractional or negative exponents before applying the power rule.",
    },
    "implicit_derivative": {
        "title": "Implicit Differentiation — Treating y as a Hidden Function of x",
        "mental_model": "Differentiate both sides with respect to x; multiply every y-derivative term by dy/dx, then isolate dy/dx.",
        "intuition": (
            "Even when an equation mixes x and y (like a circle or tilted curve) so you cannot easily isolate y = f(x), "
            "y still varies smoothly along the curve as x moves. By the chain rule, differentiating yⁿ wrt x produces n·yⁿ⁻¹·(dy/dx)."
        ),
        "pitfall": "Remember the Product Rule on mixed terms like x·y (its derivative is 1·y + x·dy/dx), and remember that the derivative of a standalone constant on the right-hand side is 0.",
    },
    "tangent_normal": {
        "title": "Tangent & Normal Lines — Local Linear Geometry",
        "mental_model": "Tangent slope m = f'(x₀); Normal slope m⊥ = −1/m; Point-slope line: y − y₀ = m(x − x₀).",
        "intuition": (
            "A tangent line just kisses the curve at (x₀, y₀) heading in the exact instantaneous direction f'(x₀). "
            "The normal line cuts straight across the curve at a 90° right angle, so its slope is the negative reciprocal −1/m."
        ),
        "pitfall": "If m = 0 (horizontal tangent), the normal is a vertical line x = x₀. Also double-check whether the prompt asks for the tangent slope f'(x₀), the full tangent equation, or the perpendicular normal.",
    },
    "critical_extrema_inflection": {
        "title": "Curve Shape — Critical Points, Peaks, Valleys & Inflection",
        "mental_model": "f'(x) = 0 finds flat turning points; f''(x) < 0 means concave down (local max), f''(x) > 0 means concave up (local min).",
        "intuition": (
            "Walk along the graph: where the ground flattens out (f'(x) = 0), you may be at a hilltop (local max), a valley floor (local min), or a flat terrace. "
            "The second derivative f''(x) tells you which way the curve bends—cupped upward (smile = min) or downward (frown = max)—and inflection points occur where that bending switches sign."
        ),
        "pitfall": "Finding f''(x) = 0 is only a candidate for an inflection point: always verify that f''(x) actually changes sign across the point, and check interval endpoints when finding absolute extrema.",
    },
    "integral_u_sub": {
        "title": "U-Substitution — Reversing the Chain Rule",
        "mental_model": "Pick an inner expression u = g(x) whose derivative g'(x) dx appears elsewhere in the integrand.",
        "intuition": (
            "Whenever an integral contains a composite function multiplied by the derivative of its inside, it was born from the chain rule. "
            "Replacing g(x) with u and g'(x) dx with du collapses the messy integral into a basic power, exponential, or trig integral."
        ),
        "pitfall": "Always convert dx completely into du before integrating, remember any missing constant multiplier (like 1/2 or 1/3), and substitute back to x (with + C for indefinite integrals).",
    },
    "integral_ibp": {
        "title": "Integration by Parts (LIATE) — Reversing the Product Rule",
        "mental_model": "∫ u dv = u·v − ∫ v du. Pick u using LIATE: Log, Inverse trig, Algebraic, Trig, Exponential.",
        "intuition": (
            "When two unrelated function types are multiplied (like x·e^x or x·sin(x)) and substitution fails, "
            "integration by parts trades a hard integral for an easier one by differentiating u (making it simpler) and integrating dv."
        ),
        "pitfall": "Follow LIATE carefully—choosing u backwards makes the new integral harder instead of simpler. For e^(ax)·sin(bx), integrate by parts twice and solve algebraically for the original integral I.",
    },
    "integral_rational": {
        "title": "Rational Integrals — Long Division & Partial Fractions",
        "mental_model": "If deg(top) ≥ deg(bottom), do polynomial long division first; then factor the denominator into partial fractions.",
        "intuition": (
            "A top-heavy polynomial fraction hides a plain polynomial plus a smaller remainder—long division separates them immediately. "
            "Once the numerator has lower degree than the denominator, factoring the bottom lets you split one big fraction into simple 1/(x − r) logs and 1/(x² + a²) arctangents."
        ),
        "pitfall": "Never jump straight into partial fractions when deg(numerator) ≥ deg(denominator): always perform long division first, and include both A/(x−r) and B/(x−r)² for repeated linear roots.",
    },
    "integral_kings": {
        "title": "Symmetry & Reflection Rule — Pairing Opposite Ends of [a, b]",
        "mental_model": "King's Property: I = ∫[a..b] f(x) dx = ∫[a..b] f(a + b − x) dx, then add 2I.",
        "intuition": (
            "Evaluating f(a + b − x) reads the curve backward from right to left over [a, b]. The area under the curve stays identical! "
            "Adding the original integral and the reflected integral together (I + I = 2I) often makes messy trig or radical denominators cancel straight to 1."
        ),
        "pitfall": "Check that the bounds match a symmetric complement (like 0 to π/2 where sin(π/2 − x) = cos(x)) and don't forget to divide by 2 at the end since you computed 2I.",
    },
    "indefinite_integral": {
        "title": "Antiderivatives — Reversing Differentiation",
        "mental_model": "Power Rule in reverse: ∫ xⁿ dx = x^(n+1)/(n+1) + C (for n ≠ −1); ∫ 1/x dx = ln|x| + C.",
        "intuition": (
            "Integration asks: 'Which function has this expression as its rate of change?' "
            "Because constants vanish when you differentiate, every indefinite integral carries an arbitrary vertical shift + C."
        ),
        "pitfall": "You can always check your antiderivative mentally by differentiating it back! Also watch the special case n = −1 (1/x integrates to ln|x|, not x⁰/0).",
    },
    "definite_integral": {
        "title": "Definite Integrals — Net Signed Area & Accumulation",
        "mental_model": "Fundamental Theorem of Calculus: ∫[a..b] f(x) dx = F(b) − F(a).",
        "intuition": (
            "Integrating a rate of change from a to b adds up infinitely many tiny slices f(x)·dx to give the net accumulated change—visually, the signed area between the curve and the x-axis."
        ),
        "pitfall": "Subtract in the exact order F(upper) − F(lower), and watch signs carefully when substituting negative lower bounds.",
    },
    "related_rates": {
        "title": "Related Rates — Linking Dynamic Rates via Implicit Time Differentiation",
        "mental_model": "1) Draw & write the static geometry equation → 2) Differentiate wrt time t → 3) Plug in the snapshot values last.",
        "intuition": (
            "In real physical systems (draining cones, sliding ladders, expanding balloons, moving cars), variables are locked together by a geometric law like x² + y² = L² or similar triangles r/h = R/H. "
            "Because everything changes with time t, differentiating that geometric link turns static lengths into a live equation connecting their speeds."
        ),
        "pitfall": "Never plug in snapshot numbers (like depth h = 6 or distance x = 3) before differentiating! Only genuinely fixed constants (like a ladder's fixed length L) stay constant prior to d/dt.",
    },
    "optimization": {
        "title": "Applied Optimization — Constraint Elimination & Critical Extrema",
        "mental_model": "1) Write the Objective (to max/min) → 2) Use the Constraint to reduce to 1 variable → 3) Solve f'(x) = 0.",
        "intuition": (
            "Optimization problems balance competing effects (e.g., more width leaves less height for a fixed perimeter). "
            "Substituting the constraint expresses the objective as a single smooth curve whose peak or trough sits where the derivative is zero."
        ),
        "pitfall": "Make sure you answer what the question actually asks for (dimensions vs. maximum area/cost), and verify that your critical point falls inside the physical domain (positive lengths).",
    },
    "equation_solving": {
        "title": "Equation Solving — Preserving Balance on Both Sides",
        "mental_model": "Apply identical inverse operations to both sides (or factor / use the quadratic formula for degree 2).",
        "intuition": (
            "An equation acts like a balanced scale: undoing addition/subtraction first and multiplication/division second isolates x without tilting the equality. "
            "For quadratics, setting one side to 0 and factoring uses the zero-product principle: a·b = 0 means a = 0 or b = 0."
        ),
        "pitfall": "Check signs when moving terms across '=' and substitute your root back into the original equation as a quick sanity check.",
    },
    "arithmetic_geometry_number": {
        "title": "Foundational Arithmetic, Number Structure & Geometry",
        "mental_model": "Break expressions into clear order-of-operations steps, prime factors, or right-triangle Pythagoras legs.",
        "intuition": (
            "Distance in the coordinate plane is simply the Pythagorean hypotenuse √((Δx)² + (Δy)²), while primality checks whether any integer up to √n divides n evenly."
        ),
        "pitfall": "Watch negative coordinates when computing Δx = x₂ − x₁ and apply order of operations (exponents, multiplication/division, then addition/subtraction).",
    },
    "generalized_identity": {
        "title": "Algebraic Equivalence & Structural Identities",
        "mental_model": "Two expressions are identically equal when expanding and collecting like terms produces the exact same polynomial.",
        "intuition": (
            "An identity holds for every real value of its variables because it only rearranges terms using distributivity, commutativity, and associativity—not by solving for one special number."
        ),
        "pitfall": "Watch cross-terms in powers like (a + b)² = a² + 2ab + b²: distributing powers across addition without cross-terms is the most common false-identity trap.",
    },
    "generalized_inequality": {
        "title": "Order & Inequality Comparison",
        "mental_model": "Compare magnitudes on the real number line (a < b means b − a > 0); multiplying by a negative flips the inequality.",
        "intuition": (
            "Checking an inequality asks whether one quantity sits strictly to the left or right of another on the number line, or shares the same boundary."
        ),
        "pitfall": "Double-check strict (<, >) versus inclusive (≤, ≥) bounds, and verify signs before cross-multiplying fractions.",
    },
    "generalized_number_theory": {
        "title": "Divisibility, Modular Arithmetic & Greatest Common Divisors",
        "mental_model": "Division algorithm: a = q·m + r with 0 ≤ r < m; a divides b iff remainder r = 0.",
        "intuition": (
            "Modular arithmetic wraps integers around a clock of size m, keeping only the remainder, while gcd(a, b) is the largest shared factor found by repeated Euclidean division."
        ),
        "pitfall": "Remember that divisibility 'a divides b' (a ∣ b) means b is a multiple of a—never reverse the order of a and b.",
    },
    "generalized_series_combinatorics": {
        "title": "Finite Sums & Combinatorial Counting",
        "mental_model": "Pair symmetric terms in arithmetic sums (Gauss pairing) and count unordered subsets via C(n, k) = n! / (k!(n − k)!).",
        "intuition": (
            "Summing 1 + 2 + ... + n pairs first and last terms into equal buckets of (n + 1), while C(n, k) counts how many distinct committees of size k can be chosen from n items."
        ),
        "pitfall": "Check inclusive endpoints carefully (whether summation starts at 0 or 1) and keep track of factorial cancellations.",
    },
    "generalized_partial_derivative": {
        "title": "Multivariable Partial Differentiation — Holding Other Axes Still",
        "mental_model": "To compute ∂f/∂x, freeze every other variable (y, z, ...) as a constant and differentiate wrt x alone.",
        "intuition": (
            "A partial derivative slices a multidimensional surface along one coordinate direction and measures the slope of that slice while keeping all other coordinates locked."
        ),
        "pitfall": "Terms containing only the frozen variables (such as y³ when differentiating wrt x) act like constants and differentiate to 0.",
    },
    "generalized_linear_algebra": {
        "title": "Linear Algebra — Determinants & Vector Dot Products",
        "mental_model": "For 2×2 [[a, b], [c, d]], det = a·d − b·c; dot product u·v = Σ uᵢ·vᵢ.",
        "intuition": (
            "A 2×2 determinant measures the signed area scaling factor of the linear transformation, while the dot product projects one vector onto another by summing component-wise products."
        ),
        "pitfall": "Always subtract the anti-diagonal (b·c) from the main diagonal (a·d) in determinants, and keep negative entries inside parentheses.",
    },
    "generalized_llm_synthesis": {
        "title": "Generalized Mathematical Reasoning & Theorem Checking",
        "mental_model": "Translate the statement into an exact symbolic claim and test every algebraic/numeric step against mathematical axioms.",
        "intuition": (
            "Even for unfamiliar problems outside standard templates, the underlying claim must hold logically without contradiction from definitions and arithmetic laws."
        ),
        "pitfall": "Verify every quantifier, domain condition, and boundary case so a single counterexample cannot invalidate the claim.",
    },
    "generalized_linear_system": {
        "title": "Simultaneous Linear Systems — Elimination & Back-Substitution",
        "mental_model": "Eliminate one variable at a time by scaling and adding equations, then substitute back to find every unknown.",
        "intuition": (
            "Each linear equation represents a line (or plane in 3D). Solving the system finds the exact intersection point where every equation holds at the exact same time."
        ),
        "pitfall": "Always substitute your solution back into ALL original equations—not just the last one—to catch arithmetic or sign slips.",
    },
    "generalized_gradient": {
        "title": "Multivariable Gradient — Direction of Steepest Ascent",
        "mental_model": "∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z): collect each variable's partial derivative into one vector.",
        "intuition": (
            "On a multidimensional surface, the gradient vector points straight uphill in the direction where the function increases fastest, and its length measures that maximum slope."
        ),
        "pitfall": "When computing each component ∂f/∂xᵢ, treat all other variables as frozen constants.",
    },
    "generalized_trig_identity": {
        "title": "Trigonometric Identities — Unit-Circle Structure",
        "mental_model": "Anchor transformations on sin²(x) + cos²(x) = 1, sin(2x) = 2sin(x)cos(x), and cos(2x) = cos²(x) − sin²(x).",
        "intuition": (
            "Trigonometric identities express geometric invariants on the unit circle, so they hold identically for every angle x."
        ),
        "pitfall": "Watch sign changes in cos(2x) = cos²x − sin²x = 1 − 2sin²x = 2cos²x − 1 and never drop the factor of 2 in sin(2x).",
    },
    "generalized_symbolic_inequality": {
        "title": "Universal Symbolic Inequalities & Sum-of-Squares",
        "mental_model": "Rewrite A ≥ B as A − B ≥ 0, then complete the square so every term is a non-negative square (x − y)² ≥ 0.",
        "intuition": (
            "Since any real square u² ≥ 0, showing that the difference between two sides is a sum of squares guarantees the inequality holds for all real numbers."
        ),
        "pitfall": "Test extreme or negative values (like x = −1, y = 1) mentally to make sure an inequality isn't only true for positive numbers.",
    },
    "generalized_complex": {
        "title": "Complex Number Arithmetic — Combining Real & Imaginary Parts",
        "mental_model": "Treat i just like an algebraic variable, distribute normally, and replace every i² with −1; group real terms separately from i terms.",
        "intuition": (
            "A complex number a + b·i lives on a 2D plane (horizontal real axis, vertical imaginary axis). Multiplying rotates and scales vectors in that plane, with 90° × 90° = 180° flip accounting for i² = −1."
        ),
        "pitfall": "Always convert b·i · d·i = b·d·i² = −b·d (watch the minus sign when two imaginary terms multiply!).",
    },
    "generalized_log_identity": {
        "title": "Logarithmic Laws — Turning Products into Sums",
        "mental_model": "log(a·b) = log(a) + log(b) and log(a/b) = log(a) − log(b) for positive quantities a, b > 0; logarithms DO NOT distribute over addition.",
        "intuition": (
            "Because logarithms measure exponents (how many times you multiply by the base), multiplying numbers corresponds to adding their exponents, and dividing subtracts them."
        ),
        "pitfall": "Never expand log(a + b) into log(a) + log(b) — logarithm laws only turn multiplication and division into addition and subtraction.",
    },
    "generalized_exponential_equation": {
        "title": "Exponential Equations — Matching Bases",
        "mental_model": "Rewrite both sides with the same positive base (b^x = b^k) and equate exponents directly: x = k.",
        "intuition": (
            "Exponential curves with base b > 1 strictly climb without ever repeating a height, so each output height corresponds to one unique exponent."
        ),
        "pitfall": "Check that the exponential term is isolated before equating exponents, and remember b⁰ = 1 when the right-hand side is 1.",
    },
}


def _select_guide_key(problem: str, template_id: Optional[str]) -> str:
    tid = (template_id or "").lower()
    if tid in _CONCEPT_GUIDES:
        return tid
    p_l = problem.lower()

    if "king" in tid or "kings_rule" in tid:
        return "integral_kings"
    if any(k in tid for k in ("partial_fraction", "long_division", "rational")):
        return "integral_rational"
    if any(k in tid for k in ("ibp", "tabular", "cyclic")):
        return "integral_ibp"
    if "u_sub" in tid or "substitution" in tid or "euler" in tid or "weierstrass" in tid:
        return "integral_u_sub"
    if "definite_integral" in tid or "wallis" in tid:
        return "definite_integral"
    if "integral" in tid:
        if any(k in p_l for k in ("partial fraction", "long division")):
            return "integral_rational"
        if any(k in p_l for k in ("by parts", "liate")):
            return "integral_ibp"
        if any(k in p_l for k in ("u-sub", "u sub", "substitution")):
            return "integral_u_sub"
        return "indefinite_integral"

    if "implicit" in tid:
        return "implicit_derivative"
    if "tangent_line" in tid or "normal_line" in tid:
        return "tangent_normal"
    if any(k in tid for k in ("critical", "extrema", "inflection", "absolute_extrema")):
        return "critical_extrema_inflection"
    if "optimization" in tid or "minimization" in tid or "maximization" in tid:
        return "optimization"
    if "rates" in tid or "related_rates" in tid:
        return "related_rates"
    if "equation" in tid:
        return "equation_solving"
    if any(k in tid for k in ("arithmetic", "number_theory", "geometry", "algebraic")):
        return "arithmetic_geometry_number"

    if "deriv" in tid or any(k in p_l for k in ("derivative", "differentiate", "d/dx", "dy/dx", "f'(x)")):
        if "quotient" in p_l or ("/" in p_l and "x" in p_l.split("/")[-1]):
            return "derivative_quotient"
        if "product" in p_l or re.search(
            r"(?:x(?:\^\d+)?|\)|e\^x|exp\([^)]+\))\s*\*?\s*(?:\(|(?:sin|cos|tan|sec|csc|cot|exp|e\^|ln|log)\b)",
            p_l,
        ):
            return "derivative_product"
        if any(k in p_l for k in ("ln(", "log(", "e^", "exp(")):
            return "derivative_log_exp"
        if any(k in p_l for k in ("sin", "cos", "tan", "sec", "csc", "cot")):
            return "derivative_trig"
        if "chain" in p_l or re.search(r"\([^)]*x[^)]*\)\s*\^", p_l):
            return "derivative_chain"
        return "derivative_general"

    if any(k in p_l for k in ("integrate", "integral", "antiderivative", "∫")):
        return "indefinite_integral"
    if any(k in p_l for k in ("solve", "=")):
        return "equation_solving"

    return "derivative_general"


def _clean_student_text(text: str) -> str:
    """Enforce Rule 22 (zero Lean syntax) and Rule 23 (zero book citations)."""
    if not text:
        return ""
    res = text
    # Rule 22: clean Lean namespaces
    res = res.replace("Complex.I", "i")
    res = re.sub(r"\bReal\.([a-zA-Z]+)\s+([a-zA-Z0-9]+)\b", r"\1(\2)", res)
    res = re.sub(r"\bReal\.([a-zA-Z]+)\b", r"\1", res)
    # Remove any accidental Lean / proof-assistant jargon
    for word in (
        "lean 4", "lean", "mathlib", "formal proof", "proof assistant",
        "formally verified", "lean kernel", "theorem prover",
    ):
        res = re.sub(rf"\b{re.escape(word)}\b", "symbolic verification", res, flags=re.IGNORECASE)
    # Rule 23: scrub textbook names / page citations
    for cite in (
        r"according to\s+el-?moasser[^.,;]*[.,;]?",
        r"el-?moasser(?:\s*(?:unit|lesson|senior|general|calculus|202\d)[^.,;)]*)?",
        r"the\s+book\s+of\s+(?:1000\s+)?(?:integrals|derivatives)[^.,;)]*",
        r"miguel\s+santiago",
        r"\bsenior\s+3\s+calculus\b",
    ):
        res = re.sub(cite, "", res, flags=re.IGNORECASE)
    res = re.sub(r"\s{2,}", " ", res).strip()
    return res


def build_intuitive_guide(
    problem: str,
    verified_answer: str,
    reasoning: str = "",
    template_id: Optional[str] = None,
) -> IntuitiveGuide:
    """
    Build a structured, textbook-pedagogy intuitive explanation grounded
    in the verified answer (sub-millisecond, zero hallucination).
    """
    guide_key = _select_guide_key(problem, template_id)
    guide = _CONCEPT_GUIDES[guide_key]

    clean_ans = _clean_student_text(verified_answer)
    clean_reasoning = _clean_student_text(reasoning)

    # Extract concise working takeaway without repeating raw machine logs
    working_note = ""
    if clean_reasoning and len(clean_reasoning) > 12:
        # Take up to first two informative sentences of CAS reasoning
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_reasoning) if s.strip()]
        snippet = " ".join(sentences[:2])
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."
        working_note = f"Applied here: {snippet}"
    else:
        working_note = f"Applying this method step by step leads directly to {clean_ans}."

    full_explanation = (
        f"{guide['title']}: {guide['intuition']} "
        f"Mental model — {guide['mental_model']} "
        f"{working_note} "
        f"Key insight: Final result is {clean_ans}."
    )

    return IntuitiveGuide(
        concept_title=guide["title"],
        mental_model=guide["mental_model"],
        intuitive_body=guide["intuition"],
        pitfall_tip=guide["pitfall"],
        full_explanation=_clean_student_text(full_explanation),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional LLM prose polish prompt (when EXPLAINER_USE_LLM=true)
# ─────────────────────────────────────────────────────────────────────────────

_EXPLAINER_SYSTEM_PROMPT = """\
You are a friendly and clear mathematics tutor. Explain the mathematical intuition behind a solved problem.
RULES:
- Address a first-year university or senior high school student.
- Ground your explanation in the intuitive mental model provided.
- NEVER mention "Lean", "formal proof", "verification system", "proof assistant", or any textbook names/page numbers.
- Keep it concise (under 160 words) and end with the key takeaway.
"""


def explain(
    problem: str,
    verified_answer: str,
    reasoning: str = "",
    template_id: Optional[str] = None,
) -> str:
    """
    Generate an intuitive pedagogical explanation for a verified answer.

    By default uses the fast, hallucination-free textbook-inspired conceptual
    explainer (`build_intuitive_guide`). Set `EXPLAINER_USE_LLM=true` to also
    polish prose with the local GGUF model if loaded.
    """
    start = time.monotonic()
    guide = build_intuitive_guide(
        problem=problem,
        verified_answer=verified_answer,
        reasoning=reasoning,
        template_id=template_id,
    )

    use_llm = os.getenv("EXPLAINER_USE_LLM", "false").lower() in ("1", "true", "yes")
    if use_llm:
        llm = _get_llm()
        if llm.__class__.__name__ != "_MockLLM":
            prompt = _build_prompt(problem, verified_answer, guide.full_explanation)
            try:
                response = llm(
                    prompt,
                    max_tokens=240,
                    temperature=0.25,
                    stop=["Problem:", "<|im_end|>", "<|user|>"],
                    echo=False,
                )
                candidate = _clean_student_text(response["choices"][0]["text"].strip())
                if len(candidate) >= 40:
                    elapsed_ms = int((time.monotonic() - start) * 1000)
                    logger.info(f"LLM-polished intuitive explanation generated in {elapsed_ms}ms")
                    return candidate
            except Exception as e:
                logger.warning(f"Optional LLM explainer polish failed, using deterministic guide: {e}")

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(f"Intuitive explanation generated in {elapsed_ms}ms ({len(guide.full_explanation)} chars)")
    return guide.full_explanation


def _build_prompt(problem: str, verified_answer: str, guide_text: str) -> str:
    """Build ChatML prompt for optional local LLM explanation polishing."""
    user_msg = (
        f"Problem: {problem}\n"
        f"Correct answer: {verified_answer}\n"
        f"Conceptual intuition guide: {guide_text}\n\n"
        "Rewrite this as a warm, intuitive explanation:"
    )
    return (
        f"<|im_start|>system\n{_EXPLAINER_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _scrub_lean_mentions(text: str) -> str:
    """Remove any accidental references to Lean, formal verification, or book citations."""
    return _clean_student_text(text)
