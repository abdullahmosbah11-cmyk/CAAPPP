"""
VeriMath External Connectors Package
====================================
Provides hardware camera capture, handwriting/printed math OCR adapters
(RapidOCR ONNX, Tesseract 5, TrOCR, Pix2Tex, Gemini Vision), and connector
health/readiness telemetry.
"""

from backend.connectors.vision_ocr import (
    OCRScanResult,
    capture_hardware_camera_frame,
    extract_math_from_image,
    get_connectors_status,
    preprocess_math_image,
)

__all__ = [
    "OCRScanResult",
    "capture_hardware_camera_frame",
    "extract_math_from_image",
    "get_connectors_status",
    "preprocess_math_image",
]
