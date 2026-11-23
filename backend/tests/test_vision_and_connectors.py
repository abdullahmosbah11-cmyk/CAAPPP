"""
Unit & Integration Tests for External Connectors, Camera Capture & Handwriting/Math OCR
========================================================================================
Verifies:
  1. Image preprocessing & math OCR normalization (`x~2` -> `x^2`, glued verbs).
  2. Live pretrained RapidOCR neural ONNX & Tesseract 5 LSTM math extraction on rendered images.
  3. `/scan/ocr` (both multipart upload & base64 camera capture payloads).
  4. `/scan/solve` (scanned question -> full AST & verification pipeline).
  5. `/scan/grade-steps` (scanned handwritten question + derivation steps -> step grader).
  6. `/connectors/status`, `/health` connector telemetry, and paho-mqtt v2 callbacks in `firestore_sync.py`.
"""

import base64
import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from backend.connectors.vision_ocr import (
    _normalize_ocr_line,
    extract_math_from_image,
    get_connectors_status,
)
from backend.iot.firestore_sync import on_connect, on_disconnect
from backend.main import app


def _render_math_sheet(lines: list[str]) -> bytes:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    img = Image.new("RGB", (980, max(180, 90 + 72 * len(lines))), "white")
    draw = ImageDraw.Draw(img)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += 72
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestExternalConnectorsAndOCR(unittest.TestCase):
    def test_ocr_math_postprocessing_fixes_common_confusions(self):
        self.assertEqual(_normalize_ocr_line("2*x * sin(x) + x~2 * cos(x)"), "2*x * sin(x) + x^2 * cos(x)")
        self.assertEqual(_normalize_ocr_line("Solve3*x+7=22"), "Solve 3*x+7=22")
        self.assertEqual(_normalize_ocr_line("4×x − 2²"), "4*x - 2^2")

    def test_rapidocr_and_tesseract_extract_rendered_math(self):
        img_bytes = _render_math_sheet([
            "Solve 3*x + 7 = 22",
            "3*x = 15",
            "x = 5",
        ])
        res_rapid = extract_math_from_image(img_bytes, engine="rapidocr", scan_mode="question_and_answer")
        self.assertEqual(res_rapid.engine_used, "rapidocr")
        self.assertIn("3*x", res_rapid.normalized_text)
        self.assertGreaterEqual(len(res_rapid.detected_steps), 2)

        res_tess = extract_math_from_image(img_bytes, engine="tesseract", scan_mode="question")
        self.assertEqual(res_tess.engine_used, "tesseract")
        self.assertIn("22", res_tess.normalized_text)

    def test_scan_endpoints_multipart_and_base64_solve_and_grade(self):
        question_bytes = _render_math_sheet(["Solve 3*x + 7 = 22"])
        b64_payload = "data:image/png;base64," + base64.b64encode(question_bytes).decode("ascii")

        sheet_bytes = _render_math_sheet([
            "Expand (x + 2)*(x - 3)",
            "(x + 2)*(x - 3)",
            "x^2 - x - 6",
        ])

        with TestClient(app) as client:
            # 1. Multipart upload /scan/ocr
            r_up = client.post(
                "/scan/ocr",
                files={"file": ("question.png", question_bytes, "image/png")},
                data={"scan_mode": "question", "engine": "rapidocr"},
            )
            self.assertEqual(r_up.status_code, 200)
            self.assertIn("3*x", r_up.json()["problem_text"])

            # 2. Base64 camera snapshot /scan/solve
            r_solve = client.post(
                "/scan/solve",
                json={"image_base64": b64_payload, "engine": "rapidocr"},
            )
            self.assertEqual(r_solve.status_code, 200)
            body = r_solve.json()
            self.assertEqual(body["solve"]["status"], "VERIFIED")
            self.assertIn("5", body["solve"]["verified_answer"])

            # 3. Multi-line derivation sheet /scan/grade-steps
            r_grade = client.post(
                "/scan/grade-steps",
                files={"file": ("sheet.png", sheet_bytes, "image/png")},
                data={"engine": "rapidocr", "operation": "algebra"},
            )
            self.assertEqual(r_grade.status_code, 200)
            g_body = r_grade.json()
            self.assertTrue(g_body["grade"]["overall_correct"], f"Unexpected grade result: {g_body}")

            # 4. Connector readiness telemetry endpoints
            r_conn = client.get("/connectors/status")
            self.assertEqual(r_conn.status_code, 200)
            c_data = r_conn.json()
            self.assertIn("rapidocr", c_data["available_ocr_list"])
            self.assertIn("tesseract", c_data["available_ocr_list"])

            r_health = client.get("/health")
            self.assertEqual(r_health.status_code, 200)
            self.assertIn("connectors", r_health.json())

    def test_firestore_sync_uses_paho_v2_callbacks(self):
        class _DummyClient:
            subscribed = None
            def subscribe(self, topic, qos=1):
                self.subscribed = (topic, qos)

        c = _DummyClient()
        on_connect(c, None, {}, 0, None)
        self.assertIsNotNone(c.subscribed)
        on_disconnect(c, None, None, 0, None)


if __name__ == "__main__":
    unittest.main()
