"""
VeriMath — MQTT Event Publisher
================================
Publishes telemetry events to the local Mosquitto broker after each
pipeline run. The Firestore sync service subscribes and forwards to cloud.

Topic: verimath/events
QoS: 1 (at least once delivery)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
TOPIC = os.getenv("MQTT_TOPIC", "verimath/events")


# ─────────────────────────────────────────────────────────────────────────────
# Event data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineEvent:
    problem_id: str
    timestamp: str
    problem_text: str
    template_id: Optional[str]
    candidate_answer: Optional[str]
    lean_statement: Optional[str]
    lean_verdict: str                   # VERIFIED | REJECTED | TIMEOUT | UNSUPPORTED | ERROR
    judge_verdict: Optional[str]        # MATCH | MISMATCH | None
    retry_count: int
    total_elapsed_ms: int
    from_cache: bool
    student_id: str = "anonymous"

    @classmethod
    def create(
        cls,
        problem_text: str,
        lean_verdict: str,
        total_elapsed_ms: int,
        template_id: Optional[str] = None,
        candidate_answer: Optional[str] = None,
        lean_statement: Optional[str] = None,
        judge_verdict: Optional[str] = None,
        retry_count: int = 0,
        from_cache: bool = False,
        student_id: str = "anonymous",
    ) -> "PipelineEvent":
        return cls(
            problem_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            problem_text=problem_text,
            template_id=template_id,
            candidate_answer=candidate_answer,
            lean_statement=lean_statement,
            lean_verdict=lean_verdict,
            judge_verdict=judge_verdict,
            retry_count=retry_count,
            total_elapsed_ms=total_elapsed_ms,
            from_cache=from_cache,
            student_id=student_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MQTT client (singleton)
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[Any] = None
_connected: bool = False


def _get_client() -> Optional[Any]:
    global _client, _connected

    if mqtt is None:
        return None

    if _client is not None and _connected:
        return _client

    try:
        if hasattr(mqtt, "CallbackAPIVersion"):
            _client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="verimath-backend",
                protocol=mqtt.MQTTv311,
            )
        else:
            _client = mqtt.Client(client_id="verimath-backend", protocol=mqtt.MQTTv311)
        _client.on_connect = _on_connect
        _client.on_disconnect = _on_disconnect

        _client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        _client.loop_start()
        time.sleep(0.2)  # Brief wait for connection
        logger.info(f"MQTT client connected to {BROKER_HOST}:{BROKER_PORT}")
    except Exception as e:
        logger.warning(f"MQTT connection failed: {e}. Events will not be published.")

    return _client


def _on_connect(client, userdata, flags, reason_code, properties=None):
    global _connected
    if reason_code == 0:
        _connected = True
        logger.info("MQTT broker connected (rc=0)")
    else:
        _connected = False
        logger.warning(f"MQTT connection failed with code {reason_code}")


def _on_disconnect(client, userdata, disconnect_flags, reason_code=0, properties=None):
    global _connected
    _connected = False
    logger.warning(f"MQTT disconnected (rc={reason_code})")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def publish(event: PipelineEvent) -> bool:
    """
    Publish a pipeline event to the local event store and MQTT broker.

    Args:
        event: PipelineEvent dataclass instance.

    Returns:
        True if published successfully, False if broker unavailable (non-fatal).
    """
    # 1. Always persist to local SQLite store so Teacher Dashboard works offline
    try:
        from backend.iot.event_store import record_event
        record_event(asdict(event))
    except Exception as e:
        logger.warning(f"Local telemetry record failed: {e}")

    # 2. Attempt MQTT network dispatch
    if mqtt is None:
        return False

    try:
        client = _get_client()
        if client is None:
            return False

        payload = json.dumps(asdict(event), ensure_ascii=False)
        result = client.publish(TOPIC, payload, qos=1)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.debug(f"Published event {event.problem_id} to {TOPIC}")
            return True
        else:
            logger.warning(f"MQTT publish returned rc={result.rc}")
            return False

    except Exception as e:
        # MQTT errors are non-fatal — the pipeline continues without telemetry
        logger.warning(f"MQTT publish failed (non-fatal): {e}")
        return False


def disconnect() -> None:
    """Gracefully disconnect the MQTT client. Call on server shutdown."""
    global _client, _connected
    if mqtt is None:
        return

    if _client:
        try:
            _client.loop_stop()
            _client.disconnect()
        except Exception:
            pass
        _connected = False
        logger.info("MQTT client disconnected.")
