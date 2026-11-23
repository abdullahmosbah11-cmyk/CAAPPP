"""
VeriMath — Camera, Handwriting & Math OCR Connector
====================================================
Untrusted sensory front-end connector that captures camera frames or uploaded
handwritten/printed math photos and decodes them into normalized problem and
step lines using existing pretrained models.

Safety Guarantee (Zero-Trust Neurosymbolic Architecture):
  OCR / handwriting output is treated strictly as untrusted user input: every
  recognized problem or step still passes through AST Faithfulness, SymPy CAS,
  and the Lean 4.36.0 kernel before any VERIFIED result is shown.

Supported Existing Engines (in offline-first priority order):
  1. `rapidocr`:  RapidOCR (PP-OCRv4 ONNX) neural detector + recognizer (edge fast)
  2. `tesseract`: Tesseract 5 LSTM OCR engine
  3. `trocr`:     HuggingFace `microsoft/trocr-base-handwritten` (if transformers + weights available)
  4. `pix2tex`:   LaTeX-OCR `pix2tex` formula decoder (if installed)
  5. `gemini_vision`: Optional Gemini 1.5 Flash vision extraction (when GOOGLE_API_KEY is set)
"""

from __future__ import annotations

import base64
import glob
import io
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

_RAPID_OCR_INSTANCE = None
_TROCR_PROCESSOR = None
_TROCR_MODEL = None
_PIX2TEX_INSTANCE = None


@dataclass
class OCRScanResult:
    raw_text: str
    normalized_text: str
    lines: List[str]
    problem_text: str
    detected_steps: List[str]
    engine_used: str
    confidence: float
    elapsed_ms: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "lines": self.lines,
            "problem_text": self.problem_text,
            "detected_steps": self.detected_steps,
            "engine_used": self.engine_used,
            "confidence": round(float(self.confidence), 4),
            "elapsed_ms": self.elapsed_ms,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# OCR Math & Handwriting Post-Processing
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_ocr_line(line: str) -> str:
    s = line.strip()
    if not s:
        return ""
    s = s.replace("−", "-").replace("—", "-").replace("–", "-")
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/")
    s = s.replace("²", "^2").replace("³", "^3")
    # Fix RapidOCR caret confusion (`x~2`, `x~3`, `2~10`)
    s = re.sub(r"([a-zA-Z0-9\)])\s*~\s*(\d+)", r"\1^\2", s)
    # Fix missing space after glued command verbs (e.g. "Solve3*x+7=22" -> "Solve 3*x+7=22")
    s = re.sub(
        r"\b(Solve|Find|Prove|Verify|Show|Simplify|Expand|Evaluate|Compute|Integrate|Differentiate|Factor)(?=[0-9a-zA-Z\(])",
        r"\1 ",
        s,
        flags=re.I,
    )
    # Normalize common OCR zero/oh confusion inside equations when adjacent to digits/operators
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_problem_and_steps(lines: List[str], scan_mode: str) -> Tuple[str, List[str]]:
    clean_lines = [l for l in (_normalize_ocr_line(x) for x in lines) if l]
    if not clean_lines:
        return "", []

    if scan_mode == "question":
        # Treat all scanned text as one cohesive problem prompt
        return " ".join(clean_lines), []

    if scan_mode == "steps":
        return clean_lines[0], clean_lines

    if len(clean_lines) == 1:
        return clean_lines[0], [clean_lines[0]]

    # Heuristic for "auto" / "question_and_answer":
    # First line (or prose prefix ending in ? or :) is the problem; remaining lines are steps/answers.
    first = clean_lines[0]
    if any(w in first.lower() for w in ("solve", "find", "prove", "evaluate", "simplify", "expand", "compute", "derivative", "integral", "integrate", "?")) or len(clean_lines) >= 2:
        return first, clean_lines[1:]
    return " ".join(clean_lines), clean_lines


# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing (Deskew, Contrast Boost, Adaptive Clean-Up)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_math_image(pil_img: Image.Image, enhance_handwriting: bool = True) -> Image.Image:
    """
    Prepare raw camera or handwritten notebook photos for neural/LSTM OCR:
      - Applies EXIF auto-rotation
      - Converts to clean RGB/grayscale
      - Upscales small images cleanly with LANCZOS so exponents (^, 2, 3) are preserved
      - Enhances ink-to-paper contrast & sharpness
    """
    img = ImageOps.exif_transpose(pil_img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    if not enhance_handwriting:
        return img

    w, h = img.size
    if w < 1000:
        scale = min(2.5, 1200.0 / max(1, w))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    gray = ImageEnhance.Sharpness(gray).enhance(1.4)
    return gray.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Engine Adapters
# ─────────────────────────────────────────────────────────────────────────────

def _get_rapid_ocr():
    global _RAPID_OCR_INSTANCE
    if _RAPID_OCR_INSTANCE is None:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID_OCR_INSTANCE = RapidOCR()
    return _RAPID_OCR_INSTANCE


def _run_rapidocr(img: Image.Image) -> Tuple[List[str], float]:
    ocr = _get_rapid_ocr()
    result, _ = ocr(img)
    if not result:
        return [], 0.0

    # Group bounding boxes into horizontal baselines sorted top-to-bottom, left-to-right
    boxes = []
    confs = []
    for item in result:
        pts, text, conf = item[0], str(item[1]).strip(), float(item[2])
        if not text:
            continue
        ys = [p[1] for p in pts]
        xs = [p[0] for p in pts]
        y_mid = sum(ys) / len(ys)
        line_h = max(12.0, max(ys) - min(ys))
        boxes.append((y_mid, min(xs), line_h, text))
        confs.append(conf)

    if not boxes:
        return [], 0.0

    boxes.sort(key=lambda b: (b[0], b[1]))
    lines_grouped: List[List[Tuple[float, float, float, str]]] = []
    for box in boxes:
        if not lines_grouped:
            lines_grouped.append([box])
        else:
            ref_y = lines_grouped[-1][0][0]
            tol = max(14.0, 0.55 * lines_grouped[-1][0][2])
            if abs(box[0] - ref_y) <= tol:
                lines_grouped[-1].append(box)
            else:
                lines_grouped.append([box])

    out_lines = []
    for grp in lines_grouped:
        grp.sort(key=lambda b: b[1])
        joined = " ".join(b[3] for b in grp).strip()
        if joined:
            out_lines.append(joined)

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return out_lines, avg_conf


def _run_tesseract(img: Image.Image) -> Tuple[List[str], float]:
    import pytesseract

    raw = pytesseract.image_to_string(img, config="--psm 6")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines, (0.78 if lines else 0.0)


def _segment_handwriting_lines(pil_img: Image.Image) -> List[Image.Image]:
    """Slice multi-line handwritten images by horizontal ink projection for line-level TrOCR."""
    gray = ImageOps.grayscale(pil_img)
    w, h = gray.size
    px = gray.load()
    row_has_ink = []
    for y in range(h):
        dark = 0
        for x in range(0, w, max(1, w // 250)):
            if px[x, y] < 185:
                dark += 1
        row_has_ink.append(dark >= 2)

    bands: List[Tuple[int, int]] = []
    in_band = False
    start_y = 0
    for y, active in enumerate(row_has_ink):
        if active and not in_band:
            in_band = True
            start_y = max(0, y - 6)
        elif not active and in_band:
            in_band = False
            end_y = min(h, y + 6)
            if end_y - start_y >= 12:
                bands.append((start_y, end_y))
    if in_band:
        bands.append((start_y, h))

    if not bands:
        return [pil_img]
    return [pil_img.crop((0, y0, w, y1)) for (y0, y1) in bands]


def _run_trocr(img: Image.Image) -> Tuple[List[str], float]:
    global _TROCR_PROCESSOR, _TROCR_MODEL
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_id = os.getenv("TROCR_MODEL_ID", "microsoft/trocr-base-handwritten")
    if _TROCR_PROCESSOR is None or _TROCR_MODEL is None:
        local_only = os.getenv("TROCR_LOCAL_FILES_ONLY", "true").lower() in ("1", "true", "yes")
        _TROCR_PROCESSOR = TrOCRProcessor.from_pretrained(model_id, local_files_only=local_only)
        _TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained(model_id, local_files_only=local_only)

    crops = _segment_handwriting_lines(img)
    lines = []
    for crop in crops:
        pixel_values = _TROCR_PROCESSOR(images=crop.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = _TROCR_MODEL.generate(pixel_values, max_new_tokens=96)
        txt = _TROCR_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if txt:
            lines.append(txt)
    return lines, (0.88 if lines else 0.0)


def _run_pix2tex(img: Image.Image) -> Tuple[List[str], float]:
    global _PIX2TEX_INSTANCE
    from pix2tex.cli import LatexOCR

    if _PIX2TEX_INSTANCE is None:
        _PIX2TEX_INSTANCE = LatexOCR()
    latex_txt = str(_PIX2TEX_INSTANCE(img)).strip()
    return ([latex_txt] if latex_txt else []), (0.85 if latex_txt else 0.0)


def _run_gemini_vision(img: Image.Image) -> Tuple[List[str], float]:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured for gemini_vision OCR.")
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "Transcribe the mathematical question and any handwritten derivation steps from this image line by line.\n"
        "Use clean plain ASCII math notation (e.g., x^2 * sin(x), 3*x + 7 = 22, d/dx).\n"
        "Output ONLY the transcribed lines, one line per row, with no extra commentary."
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = model.generate_content([prompt, {"mime_type": "image/png", "data": buf.getvalue()}])
    lines = [l.strip() for l in (resp.text or "").splitlines() if l.strip()]
    return lines, (0.92 if lines else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Public Unified Image-to-Math Extractor
# ─────────────────────────────────────────────────────────────────────────────

def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


def extract_math_from_image(
    image_bytes: bytes,
    engine: str = "auto",
    scan_mode: str = "auto",
    enhance_handwriting: bool = True,
) -> OCRScanResult:
    t0 = time.monotonic()
    raw_img = decode_image_bytes(image_bytes)
    proc_img = preprocess_math_image(raw_img, enhance_handwriting=enhance_handwriting)

    req_engine = (engine or os.getenv("OCR_BACKEND", "auto")).lower().strip()
    warnings: List[str] = []
    lines: List[str] = []
    conf = 0.0
    used = req_engine

    def _try_engine(name: str) -> bool:
        nonlocal lines, conf, used
        try:
            if name == "rapidocr":
                l, c = _run_rapidocr(proc_img)
            elif name == "tesseract":
                l, c = _run_tesseract(proc_img)
            elif name == "trocr":
                l, c = _run_trocr(proc_img)
            elif name == "pix2tex":
                l, c = _run_pix2tex(proc_img)
            elif name == "gemini_vision":
                l, c = _run_gemini_vision(proc_img)
            else:
                return False
            if l:
                lines, conf, used = l, c, name
                return True
        except Exception as e:
            warnings.append(f"{name}: {str(e)[:120]}")
        return False

    if req_engine == "auto":
        order = ["rapidocr", "tesseract"]
        for cand in order:
            if _try_engine(cand):
                break
        else:
            used = "none"
            warnings.append("No text detected in image. Try brighter lighting or clearer ink contrast.")
    else:
        if not _try_engine(req_engine):
            warnings.append(f"Requested engine '{req_engine}' produced no output; trying offline fallback.")
            for cand in ("rapidocr", "tesseract"):
                if cand != req_engine and _try_engine(cand):
                    break

    raw_joined = "\n".join(lines)
    norm_lines = [_normalize_ocr_line(l) for l in lines if _normalize_ocr_line(l)]
    problem_text, detected_steps = _split_problem_and_steps(norm_lines, scan_mode=scan_mode)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return OCRScanResult(
        raw_text=raw_joined,
        normalized_text="\n".join(norm_lines),
        lines=norm_lines,
        problem_text=problem_text,
        detected_steps=detected_steps,
        engine_used=used,
        confidence=conf,
        elapsed_ms=elapsed_ms,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Raspberry Pi / USB Hardware Camera Capture Connector
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware_cameras() -> List[str]:
    devices = sorted(glob.glob("/dev/video*"))
    if shutil.which("rpicam-still") or shutil.which("libcamera-still"):
        devices.insert(0, "rpi-csi-camera")
    return devices


def capture_hardware_camera_frame(device: Optional[str] = None) -> bytes:
    """
    Capture a single still JPEG/PNG frame from an attached Raspberry Pi CSI camera
    (`rpicam-still` / `libcamera-still`) or USB V4L2 webcam (`fswebcam` / `cv2`).
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        out_path = tf.name

    try:
        if (device in (None, "rpi-csi-camera")) and (shutil.which("rpicam-still") or shutil.which("libcamera-still")):
            cmd_bin = shutil.which("rpicam-still") or shutil.which("libcamera-still")
            proc = subprocess.run(
                [cmd_bin, "-o", out_path, "-t", "500", "--nopreview"],  # type: ignore[list-item]
                capture_output=True,
                text=True,
                timeout=12,
            )
            if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                with open(out_path, "rb") as f:
                    return f.read()

        if shutil.which("fswebcam"):
            dev_path = device if (device and device.startswith("/dev/video")) else "/dev/video0"
            proc = subprocess.run(
                ["fswebcam", "-d", dev_path, "-r", "1280x720", "--no-banner", out_path],
                capture_output=True,
                text=True,
                timeout=12,
            )
            if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                with open(out_path, "rb") as f:
                    return f.read()

        try:
            import cv2  # type: ignore

            idx = 0
            if device and device.startswith("/dev/video"):
                idx = int(device.replace("/dev/video", "") or "0")
            cap = cv2.VideoCapture(idx)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                ok_enc, buf = cv2.imencode(".jpg", frame)
                if ok_enc:
                    return buf.tobytes()
        except ImportError:
            pass

        raise RuntimeError(
            "No local hardware camera backend detected (install rpicam-apps, fswebcam, or opencv-python, "
            "or use the browser live WebRTC camera capture button)."
        )
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# External Connector Readiness Inspector
# ─────────────────────────────────────────────────────────────────────────────

def _pkg_available(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def get_connectors_status() -> Dict[str, Any]:
    """Return real-time availability of all external & local hardware/AI connectors."""
    mqtt_host = os.getenv("MQTT_BROKER_HOST", "localhost")
    mqtt_port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
    mqtt_reachable = False
    try:
        with socket.create_connection((mqtt_host, mqtt_port), timeout=0.4):
            mqtt_reachable = True
    except OSError:
        mqtt_reachable = False

    fb_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")
    firebase_configured = bool(fb_path and os.path.exists(fb_path))

    ocr_engines = {
        "rapidocr": _pkg_available("rapidocr_onnxruntime"),
        "tesseract": shutil.which("tesseract") is not None and _pkg_available("pytesseract"),
        "trocr": _pkg_available("transformers") and _pkg_available("torch"),
        "pix2tex": _pkg_available("pix2tex"),
        "gemini_vision": bool(os.getenv("GOOGLE_API_KEY", "")) and _pkg_available("google.generativeai"),
    }

    cameras = detect_hardware_cameras()
    return {
        "ocr_engines": ocr_engines,
        "available_ocr_list": [k for k, v in ocr_engines.items() if v],
        "hardware_cameras": cameras,
        "browser_camera_supported": True,
        "mqtt": {
            "host": mqtt_host,
            "port": mqtt_port,
            "reachable": mqtt_reachable,
        },
        "firebase": {
            "configured": firebase_configured,
            "project_id": os.getenv("FIREBASE_PROJECT_ID", ""),
        },
        "gemini_summary_configured": bool(os.getenv("GOOGLE_API_KEY", "")),
    }
