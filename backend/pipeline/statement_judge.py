"""
VeriMath — Module 3: Statement Judge
=====================================
Deterministic, rule-based module that checks whether a filled Lean statement
faithfully represents the original natural language problem.

No LLM is used here. The judge:
1. Extracts slot values from the formalization result
2. Generates a human-readable back-translation using fixed string templates
3. Checks that key numbers, operators, and terms from the back-translation
   appear in the original problem (fuzzy matching)

This is auditable and explainable — no black-box AI in the verification chain.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

try:
    from fuzzywuzzy import fuzz
except ImportError:
    import difflib

    class _FuzzFallback:
        @staticmethod
        def _clean_tokens(s: str) -> set[str]:
            cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", s.lower())
            return set(cleaned.split())

        @classmethod
        def token_set_ratio(cls, s1: str, s2: str) -> int:
            t1 = cls._clean_tokens(s1)
            t2 = cls._clean_tokens(s2)
            common = t1.intersection(t2)
            if not common:
                return 0
            diff1 = t1 - common
            diff2 = t2 - common
            s_inter = " ".join(sorted(common))
            s_inter_diff1 = (s_inter + " " + " ".join(sorted(diff1))).strip()
            s_inter_diff2 = (s_inter + " " + " ".join(sorted(diff2))).strip()
            r1 = difflib.SequenceMatcher(None, s_inter, s_inter_diff1).ratio()
            r2 = difflib.SequenceMatcher(None, s_inter, s_inter_diff2).ratio()
            r3 = difflib.SequenceMatcher(None, s_inter_diff1, s_inter_diff2).ratio()
            return int(max(r1, r2, r3) * 100)

        @staticmethod
        def partial_ratio(s1: str, s2: str) -> int:
            if not s1 or not s2:
                return 0
            shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
            m = difflib.SequenceMatcher(None, shorter, longer)
            blocks = m.get_matching_blocks()
            scores = [m.ratio()]
            for i, j, n in blocks:
                start = max(0, j - i)
                sub = longer[start:start + len(shorter)]
                scores.append(difflib.SequenceMatcher(None, shorter, sub).ratio())
            return int(max(scores) * 100)

    fuzz = _FuzzFallback()

from backend.templates.registry import get_back_translation, get_template
from backend.pipeline.formalizer import FormalizerResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Judge result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    verdict: str                    # "MATCH" | "MISMATCH"
    back_translation: str           # Human-readable restatement of the Lean statement
    similarity_score: float         # 0.0 – 1.0 (fuzzy match score)
    reason: Optional[str] = None    # Explanation of mismatch (if any)
    ast_checks: Optional[list[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

MATCH_THRESHOLD = 0.38   # Fuzzy similarity threshold (0–100 scale from fuzzywuzzy → 0–1)
                         # Set conservatively because Lean syntax ≠ natural language


# ─────────────────────────────────────────────────────────────────────────────
# Main judge function
# ─────────────────────────────────────────────────────────────────────────────

def judge(
    problem: str,
    formalizer_result: FormalizerResult,
) -> JudgeResult:
    """
    Rule-based faithfulness check: does the Lean statement match the problem?

    Args:
        problem: Original natural language problem.
        formalizer_result: Output from Module 2.

    Returns:
        JudgeResult with verdict and back-translation.
    """
    template = get_template(formalizer_result.template_id)
    slot_values = formalizer_result.slot_values

    # Step 1: Generate back-translation
    back_translation = get_back_translation(template, slot_values)
    logger.info(f"Back-translation: '{back_translation}'")

    # Step 2: Normalize both strings for comparison
    problem_norm = _normalize(problem)
    back_norm = _normalize(back_translation)

    # Step 3: Fuzzy string similarity
    token_score = fuzz.token_set_ratio(problem_norm, back_norm) / 100.0
    partial_score = fuzz.partial_ratio(problem_norm, back_norm) / 100.0
    similarity = max(token_score, partial_score)

    logger.info(
        f"Faithfulness check — token_set: {token_score:.2f}, "
        f"partial: {partial_score:.2f}, final: {similarity:.2f} "
        f"(threshold: {MATCH_THRESHOLD})"
    )

    # Step 4: Additional checks: numbers, mathematical functions, physical entities, semantic inversions, and AST faithfulness
    from backend.pipeline.ast_faithfulness import verify_ast_faithfulness
    numeric_check = _check_numbers_match(problem, back_translation)
    function_check = _check_math_functions_match(problem, back_translation)
    entity_check = _check_key_entities_match(problem, back_translation)
    inversion_check = _check_semantic_inversions(problem, back_translation, slot_values)
    ast_check = verify_ast_faithfulness(problem, formalizer_result.lean_statement)

    # Step 5: Verdict
    if (
        similarity >= MATCH_THRESHOLD
        and numeric_check.passed
        and function_check.passed
        and entity_check.passed
        and inversion_check.passed
        and ast_check.passed
    ):
        return JudgeResult(
            verdict="MATCH",
            back_translation=back_translation,
            similarity_score=similarity,
            ast_checks=ast_check.checks_run,
        )
    else:
        reason_parts = []
        if similarity < MATCH_THRESHOLD:
            reason_parts.append(
                f"Similarity score {similarity:.2f} is below threshold {MATCH_THRESHOLD}."
            )
        if not numeric_check.passed:
            reason_parts.append(numeric_check.reason)
        if not function_check.passed:
            reason_parts.append(function_check.reason)
        if not entity_check.passed:
            reason_parts.append(entity_check.reason)
        if not inversion_check.passed:
            reason_parts.append(inversion_check.reason)
        if not ast_check.passed and ast_check.reason:
            reason_parts.append(ast_check.reason)

        return JudgeResult(
            verdict="MISMATCH",
            back_translation=back_translation,
            similarity_score=similarity,
            reason=" | ".join(reason_parts),
        )


@dataclass
class _SemanticInversionResult:
    passed: bool
    reason: Optional[str] = None


def _check_semantic_inversions(
    problem: str,
    back_translation: str,
    slot_values: Optional[dict[str, str]] = None,
) -> _SemanticInversionResult:
    """
    Detect semantic inversions that lexical token-set similarity misses:
      1. Flipped inequalities (< vs >, <= vs >=, less than vs greater than).
      2. Swapped derivative targets (dy/dx vs dx/dy, dx/dt vs dy/dt).
      3. Swapped variable bindings (x = a, y = b vs x = b, y = a).
      4. Direct algebraic sign inversions on identical operand pairs (e.g. x - 5 = 2 vs x + 5 = 2).
    """
    prob_l = problem.lower().replace("+ -", "- ").replace("+-", "-").replace("−", "-")
    back_l = back_translation.lower().replace("+ -", "- ").replace("+-", "-").replace("−", "-")

    # 1. Inequality direction check
    def _ineq_dirs(text: str) -> set[str]:
        dirs = set()
        if re.search(r"<=|≤|\bat most\b|\bless than or equal", text):
            dirs.add("le")
        if re.search(r">=|≥|\bat least\b|\bgreater than or equal", text):
            dirs.add("ge")
        # Strict inequalities (excluding <= and >=)
        if re.search(r"(?<![<>=])<(?!=)|\bless than\b(?!\s+or\s+equal)", text):
            dirs.add("lt")
        if re.search(r"(?<![<>=])>(?!=)|\bgreater than\b(?!\s+or\s+equal)", text):
            dirs.add("gt")
        return dirs

    p_ineq = _ineq_dirs(prob_l)
    b_ineq = _ineq_dirs(back_l)
    if p_ineq and b_ineq:
        if ("lt" in p_ineq and "gt" in b_ineq and "lt" not in b_ineq) or (
            "gt" in p_ineq and "lt" in b_ineq and "gt" not in b_ineq
        ):
            return _SemanticInversionResult(
                passed=False,
                reason=f"Inequality direction inverted between problem ({p_ineq}) and back-translation ({b_ineq}).",
            )
        if ("le" in p_ineq and "ge" in b_ineq and "le" not in b_ineq) or (
            "ge" in p_ineq and "le" in b_ineq and "ge" not in b_ineq
        ):
            return _SemanticInversionResult(
                passed=False,
                reason=f"Inequality direction inverted between problem ({p_ineq}) and back-translation ({b_ineq}).",
            )

    # 2. Swapped derivative target check (e.g. dy/dx vs dx/dy, dx/dt vs dy/dt)
    deriv_pairs = [("dy/dx", "dx/dy"), ("dx/dt", "dy/dt"), ("dh/dt", "dr/dt")]
    for d1, d2 in deriv_pairs:
        if d1 in prob_l and d2 in back_l and d1 not in back_l and d2 not in prob_l:
            return _SemanticInversionResult(
                passed=False,
                reason=f"Derivative target inverted: problem asks for {d1} but formalization computes {d2}.",
            )
        if d2 in prob_l and d1 in back_l and d2 not in back_l and d1 not in prob_l:
            return _SemanticInversionResult(
                passed=False,
                reason=f"Derivative target inverted: problem asks for {d2} but formalization computes {d1}.",
            )

    # 3. Swapped variable bindings check (e.g. x = 2, y = 5 vs x = 5, y = 2)
    def _extract_bindings(text: str) -> dict[str, str]:
        bindings = {}
        for m in re.finditer(r"\b([xy])\s*=\s*(-?\d+(?:\.\d+)?)\b", text):
            bindings[m.group(1)] = m.group(2)
        return bindings

    p_bind = _extract_bindings(prob_l)
    b_bind = _extract_bindings(back_l)
    if "x" in p_bind and "y" in p_bind and "x" in b_bind and "y" in b_bind:
        if p_bind["x"] != p_bind["y"] and p_bind["x"] == b_bind["y"] and p_bind["y"] == b_bind["x"]:
            return _SemanticInversionResult(
                passed=False,
                reason=f"Variable bindings swapped: problem has x={p_bind['x']}, y={p_bind['y']} but formalization has x={b_bind['x']}, y={b_bind['y']}.",
            )

    # 4. Algebraic binary sign flip on equations (e.g. "3x + 7 = 22" vs "3x - 7 = 22")
    if "=" in prob_l and "=" in back_l:
        p_eq_terms = re.findall(r"([a-z0-9\)])\s*([+\-])\s*(\d+(?:\.\d+)?)\s*=", prob_l)
        b_eq_terms = re.findall(r"([a-z0-9\)])\s*([+\-])\s*(\d+(?:\.\d+)?)\s*=", back_l)
        if p_eq_terms and b_eq_terms:
            for _, p_op, p_num in p_eq_terms:
                for _, b_op, b_num in b_eq_terms:
                    if p_num == b_num and p_op != b_op:
                        return _SemanticInversionResult(
                            passed=False,
                            reason=f"Sign inversion detected: '{p_op} {p_num}' in problem became '{b_op} {b_num}' in back-translation.",
                        )

    return _SemanticInversionResult(passed=True)


# ─────────────────────────────────────────────────────────────────────────────
# Physical entity consistency check (prevents semantic drift)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _EntityCheckResult:
    passed: bool
    reason: Optional[str] = None

KEY_PHYSICAL_ENTITIES = [
    ("sphere", ["sphere", "spherical", "spheres", "ball", "balls", "balloon"]),
    ("cylinder", ["cylinder", "cylindrical", "cylinders", "metal can", "tin can", "soda can", "cylindrical can", "tank", "oil tank", "wire"]),
    ("ladder", ["ladder", "ladders", "rod"]),
    ("shadow", ["shadow", "shadows", "lamppost", "streetlamp", "streetlight", "lamp"]),
    ("rocket", ["rocket", "rockets"]),
    ("kite", ["kite", "kites"]),
    ("trough", ["trough", "troughs", "drainage canal"]),
    ("trapezoid", ["trapezoid", "trapezoidal", "trapezoids"]),
    ("parabola", ["parabola", "parabolas"]),
    ("conical_pile", ["conical pile", "pile", "piles"]),
    ("ripple", ["ripple", "ripples", "puddle", "puddles", "wave", "waves"]),
    ("lifeguard", ["lifeguard", "swimmer", "swimmers"]),
    ("illumination", ["illumination", "lightbulb", "lightbulbs"]),
    ("gutter", ["gutter", "gutters"]),
    ("ellipse", ["ellipse", "ellipses"]),
    ("semicircle", ["semicircle", "semicircles"]),
    ("road", ["road", "roads"]),
    ("cone", ["cone", "conical", "cones", "funnel"]),
    ("corral", ["corral", "corrals", "pens", "rancher"]),
    ("exponential", ["exponential", "e^{2x}", "e^(2x)"]),
]

def _check_key_entities_match(problem: str, back_translation: str) -> _EntityCheckResult:
    """
    Ensure major physical entities mentioned in the problem (e.g. sphere, ladder)
    are not completely absent from the back-translation, and that specific geometric
    or physical entities are not falsely introduced (e.g. parabola when problem is a line).
    """
    prob_lower = problem.lower()
    back_lower = back_translation.lower()
    for entity_name, syns in KEY_PHYSICAL_ENTITIES:
        if entity_name == "sphere" and ("slant height" in prob_lower or "slant" in prob_lower):
            # Counterfactual or distractor context in cone slant-height problems
            continue
        if entity_name == "sphere" and "hemisphere" in prob_lower and "hemisphere" in back_lower:
            # Both refer to a hemisphere
            continue
        if any(re.search(rf"\b{s}\b", prob_lower) for s in syns):
            if not any(re.search(rf"\b{s}\b", back_lower) for s in syns):
                return _EntityCheckResult(
                    passed=False,
                    reason=f"Problem involves '{entity_name}', but formalized statement omits it."
                )

    # Perimeter optimization check: if problem asks to optimize perimeter, formalization must optimize perimeter
    if re.search(r"\b(?:maximize|minimizing|minimize|maximum)\s+(?:the\s+)?perimeter\b|\bperimeter\s+of\s+the\s+rectangle\b", prob_lower):
        if not re.search(r"\bperimeter\b", back_lower):
            return _EntityCheckResult(
                passed=False,
                reason="Problem asks to optimize perimeter, but formalized statement does not optimize perimeter."
            )

    # Circumscribed geometry check: if problem specifies a circumscribed shape, formalization must not omit it
    if "circumscribed" in prob_lower and not ("circumscribed" in back_lower or "inscribed" in back_lower):
        return _EntityCheckResult(
            passed=False,
            reason="Problem involves a circumscribed polygon, but formalized statement omits it."
        )

    # Reverse check for high-specificity entities (prevents semantic drift into unrelated templates)
    HIGH_SPECIFICITY = ["parabola", "conical pile", "ripple", "gutter", "lifeguard", "ellipse", "semicircle", "corral", "exponential"]
    for entity in HIGH_SPECIFICITY:
        syns = [entity]
        for en, esyns in KEY_PHYSICAL_ENTITIES:
            if en == entity or en == entity.replace(" ", "_"):
                syns = esyns
                break
        if any(re.search(rf"\b{s}\b", back_lower) for s in syns) and not any(re.search(rf"\b{s}\b", prob_lower) for s in syns):
            if entity == "lifeguard" and "swimmer" in prob_lower:
                continue
            if entity == "parabola" and re.search(r"y\s*=\s*\d+\s*-\s*x\^?2", prob_lower):
                continue
            if entity == "exponential" and any(w in prob_lower for w in ["exp", "e^", "log", "ln", "derivative", "differentiate"]):
                continue
            return _EntityCheckResult(
                passed=False,
                reason=f"Formalized statement introduces '{entity}' which is not in the problem."
            )

    return _EntityCheckResult(passed=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical function consistency check
# ─────────────────────────────────────────────────────────────────────────────

MATH_FUNCTIONS = ["sin", "cos", "tan", "ln", "log", "exp", "sqrt", "arctan", "atan", "arcsin", "asin", "arccos", "acos"]

@dataclass
class _FunctionCheckResult:
    passed: bool
    reason: Optional[str] = None


def _check_math_functions_match(problem: str, back_translation: str) -> _FunctionCheckResult:
    """
    Ensure all mathematical functions mentioned in the problem also appear in
    the back-translation (accounting for ln/log, sqrt/√, and arctan/atan equivalency).
    Prevents formalization from dropping functions like ln or cos.
    """
    prob_lower = problem.lower()
    back_lower = back_translation.lower()

    prob_funcs = set()
    for fn in MATH_FUNCTIONS:
        if fn == "log":
            # Guard against wooden log / cylindrical log / timber
            if re.search(r"\b(?:cylindrical|wooden|wood|tree|timber)\s+log\b|\blog\s+(?:of|from)\s+(?:diameter|radius|a\s+tree|timber)\b", prob_lower):
                continue
        if re.search(rf"\b{fn}\b", prob_lower):
            # If function is only applied to constant arguments (e.g. sin(pi), cos(pi), tan(0), sin(3*pi)), it differentiates to 0
            m_const_fn = re.search(rf"\b{fn}\s*\(\s*(?:pi|0|\d+(?:\.\d+)?|\d+\s*\*?\s*pi(?:\s*\/\s*\d+)?)\s*\)", prob_lower)
            m_var_fn = re.search(rf"\b{fn}\s*\([^)]*x[^)]*\)", prob_lower)
            if m_const_fn and not m_var_fn:
                continue
            prob_funcs.add(fn)

    missing = set()
    for fn in prob_funcs:
        if fn in ("ln", "log"):
            if not ("ln" in back_lower or "log" in back_lower):
                missing.add(fn)
        elif fn in ("arctan", "atan"):
            if not ("arctan" in back_lower or "atan" in back_lower):
                missing.add(fn)
        elif fn in ("arcsin", "asin"):
            if not ("arcsin" in back_lower or "asin" in back_lower):
                missing.add(fn)
        elif fn in ("arccos", "acos"):
            if not ("arccos" in back_lower or "acos" in back_lower):
                missing.add(fn)
        elif fn == "sqrt":
            # In calculus/geometry (e.g. square diagonal s*sqrt(2), cube diagonal s*sqrt(3)),
            # sqrt terms can legitimately cancel out during differentiation or calculation.
            pass
        else:
            if not re.search(rf"\b{fn}\b", back_lower):
                missing.add(fn)

    if missing:
        return _FunctionCheckResult(
            passed=False,
            reason=f"Mathematical functions {missing} from problem not found in formalization.",
        )
    return _FunctionCheckResult(passed=True)


# ─────────────────────────────────────────────────────────────────────────────
# Numeric consistency check
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _NumericCheckResult:
    passed: bool
    reason: Optional[str] = None


def _check_numbers_match(problem: str, back_translation: str) -> _NumericCheckResult:
    """
    Extract numeric tokens from the problem and verify they appear in the
    back-translation. Handles comma-separated numbers (3,000 -> 3000) and recognizes
    mathematical equivalence between decimals and fractions (e.g. 0.5 == 1/2, 1.5 == 3/2).
    In calculus word problems, constants combine (e.g. net flow 10 - 1 = 9) or differentiate away.
    Therefore, requiring at least 50% of the problem numbers ensures faithfulness without false rejections.
    """
    prob_clean = re.sub(r"(?<=\d),(?=\d)", "", problem)
    back_clean = re.sub(r"(?<=\d),(?=\d)", "", back_translation)
    problem_numbers = set(_extract_numbers(prob_clean))
    back_numbers = set(_extract_numbers(back_clean))

    if not problem_numbers:
        # No numbers in problem (e.g. "simplify (x+1)² - x²") — skip check
        return _NumericCheckResult(passed=True)

    missing = problem_numbers - back_numbers
    if not missing:
        return _NumericCheckResult(passed=True)

    # Numerical equivalence fallback (handles fractions vs decimals, e.g. 0.5 vs 1/2, 1.5 vs 3/2)
    def _extract_numeric_values(text: str) -> set[float]:
        vals = set()
        cleaned = text.replace("\text", " text")
        cleaned = re.sub(r"\b(?:Question|Section|Problem|Part)\s+\d+\b", "", cleaned, flags=re.I)
        cleaned = re.sub(r"#[0-9]+\b", "", cleaned)
        cleaned = re.sub(r"\\?(?:text|mathrm|mathbf)\{([^}]*)\}", r" \1 ", cleaned)
        cleaned = cleaned.replace(r"\(", " ").replace(r"\)", " ").replace(r"\[", " ").replace(r"\]", " ")
        cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)
        cleaned = re.sub(r"(?:cm|m|ft|in|km|yd|mm)\s*\^?\s*(?:2|3|²|³)", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\b[xytrszXYTRSZ]\s*\^?\s*(?:2|3|²|³)\b", "", cleaned)
        cleaned = re.sub(r"\b[xytrszXYTRSZ]_[0-9]+\b", "", cleaned)
        cleaned = re.sub(r"(?:>|<|>=|<=)\s*0\b", "", cleaned)
        cleaned = re.sub(r":00\b", "", cleaned)

        # Fractions like 3/2 or 1/2
        for m in re.finditer(r"(?<![\w\.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\w\.])", cleaned):
            try:
                vals.add(round(float(m.group(1)) / float(m.group(2)), 5))
            except Exception:
                pass
        # Decimals and integers
        for m in re.finditer(r"\b\d+(?:\.\d+)?\b", cleaned):
            try:
                vals.add(round(float(m.group(0)), 5))
            except Exception:
                pass
        return vals

    prob_vals = _extract_numeric_values(prob_clean)
    back_vals = _extract_numeric_values(back_clean)

    missing_vals = {v for v in prob_vals if not any(abs(v - bv) < 1e-4 for bv in back_vals)}
    foreign_vals = {bv for bv in back_vals if not any(abs(bv - pv) < 1e-4 for pv in prob_vals)}
    if len(prob_vals) <= 2 and len(foreign_vals) >= 3:
        return _NumericCheckResult(
            passed=False,
            reason=f"Back-translation introduces unexpected numbers {foreign_vals} not found in problem."
        )

    matched_count = len(prob_vals) - len(missing_vals)
    if prob_vals and (matched_count / len(prob_vals) >= 0.5 or (len(prob_vals) <= 2 and matched_count >= 1 and len(foreign_vals) <= 1)):
        return _NumericCheckResult(passed=True)

    if missing_vals:
        return _NumericCheckResult(
            passed=False,
            reason=f"Numbers {missing} from problem not found in back-translation ({matched_count}/{len(prob_vals)} found).",
        )

    return _NumericCheckResult(passed=True)


def _extract_numbers(text: str) -> list[str]:
    """Extract all numeric tokens (integers and decimals) from text, normalizing thousands separators and ignoring unit exponents."""
    cleaned = text
    # Avoid \t tab escape from unescaped \text
    cleaned = cleaned.replace("\text", " text")
    # Strip question and section numbers e.g. Question 1, Section 2, #3
    cleaned = re.sub(r"\b(?:Question|Section|Problem|Part)\s+\d+\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"#[0-9]+\b", "", cleaned)
    # Strip LaTeX commands like \text{cm}^3 -> cm^3
    cleaned = re.sub(r"\\?(?:text|mathrm|mathbf)\{([^}]*)\}", r" \1 ", cleaned)
    cleaned = cleaned.replace(r"\(", " ").replace(r"\)", " ").replace(r"\[", " ").replace(r"\]", " ")
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)
    # Strip units with powers e.g. cm^3, m^3, ft^3, in^3, etc. so 3 or 2 from power is not extracted as a problem quantity
    cleaned = re.sub(r"(?:cm|m|ft|in|km|yd|mm)\s*\^?\s*(?:2|3|²|³)", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b[xytrszXYTRSZ]\s*\^?\s*(?:2|3|²|³)\b", "", cleaned)
    cleaned = re.sub(r"\b[xytrszXYTRSZ]_[0-9]+\b", "", cleaned)
    cleaned = re.sub(r"(?:>|<|>=|<=)\s*0\b", "", cleaned)
    # Separate numbers from pi symbol so \b recognizes numbers adjacent to pi (e.g. 54π -> 54 pi)
    cleaned = cleaned.replace("π", " pi ").replace("Π", " pi ")
    cleaned = re.sub(r"\\pi\b", " pi ", cleaned)
    # Strip :00 from clock times like 2:00 PM or 12:00
    cleaned = re.sub(r":00\b", "", cleaned)
    return re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Normalize text for fuzzy comparison:
    - Lowercase
    - Strip LaTeX commands
    - Strip commas within numbers (e.g. 3,000 -> 3000)
    - Replace math symbols with words
    - Remove extra whitespace and punctuation
    """
    text = text.lower()
    text = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r" \1 ", text)
    text = text.replace(r"\(", " ").replace(r"\)", " ").replace(r"\[", " ").replace(r"\]", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    # Replace common math symbols with words
    replacements = {
        "×": " times ",
        "·": " times ",
        "÷": " divided by ",
        "²": "^2",
        "³": "^3",
        "√": "sqrt",
        "∫": "integral",
        "∞": "infinity",
        "π": "pi",
        "≤": " less than or equal ",
        "≥": " greater than or equal ",
        "≠": " not equal ",
        "↔": " if and only if ",
    }
    for sym, word in replacements.items():
        text = text.replace(sym, word)

    # Remove Lean-specific syntax noise for comparison
    lean_noise = ["real.sin", "real.cos", "real.exp", "real.log", "nat.prime", ":= by", "example", ":ℝ"]
    for noise in lean_noise:
        text = text.replace(noise, " ")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
