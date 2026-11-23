"""
VeriMath — Firestore Sync Service
===================================
Subscribes to the local MQTT broker and forwards events to Firebase Firestore.
Runs as a separate process (see deployment/systemd/verimath-mqtt-sync.service).

Architecture:
  [Pi backend] → MQTT publish → [Mosquitto broker] → [this service] → Firestore

This separation means:
- The main FastAPI backend never makes cloud calls (offline-first).
- If internet is unavailable at exhibition, events are simply not forwarded
  to Firestore — the pipeline still works 100%.
- When internet returns, no automatic backfill (events published while offline are lost).
  This is acceptable for the capstone demo.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import paho.mqtt.client as mqtt
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [firestore-sync] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
TOPIC = os.getenv("MQTT_TOPIC", "verimath/events")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")

# ─────────────────────────────────────────────────────────────────────────────
# Firebase initialization
# ─────────────────────────────────────────────────────────────────────────────

_db = None  # Firestore client singleton


def _init_firebase():
    global _db
    if _db is not None:
        return _db

    if not SERVICE_ACCOUNT_PATH or not os.path.exists(SERVICE_ACCOUNT_PATH):
        logger.warning(
            f"Firebase service account not found at '{SERVICE_ACCOUNT_PATH}'. "
            "Firestore sync disabled — events will be logged locally only."
        )
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        logger.info(f"Firebase Firestore connected (project: {FIREBASE_PROJECT_ID})")
        return _db

    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MQTT callbacks
# ─────────────────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties=None):
    rc = getattr(reason_code, "value", reason_code)
    if rc == 0:
        logger.info(f"Connected to MQTT broker at {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC, qos=1)
        logger.info(f"Subscribed to topic: {TOPIC}")
    else:
        logger.error(f"MQTT connection failed with code {reason_code}")


def on_message(client, userdata, msg):
    """Handle an incoming MQTT event and forward it to Firestore."""
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        problem_id = payload.get("problem_id", "unknown")

        logger.info(f"Received event {problem_id} — forwarding to Firestore...")

        db = _init_firebase()
        if db is None:
            logger.warning(f"Firestore unavailable — event {problem_id} dropped.")
            return

        # Write to Firestore collection
        db.collection("verimath_events").document(problem_id).set(payload)
        logger.info(f"Event {problem_id} written to Firestore ✅")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse MQTT message: {e} — raw: {msg.payload[:200]}")
    except Exception as e:
        logger.error(f"Firestore write failed: {e}")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    logger.warning(f"Disconnected from MQTT broker (rc={reason_code}). Will reconnect...")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("VeriMath Firestore Sync Service starting...")
    _init_firebase()  # Pre-initialize (fail-safe)

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="verimath-firestore-sync",
        protocol=mqtt.MQTTv311,
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    # Reconnect on disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    while True:
        try:
            client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
            client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down Firestore sync...")
            client.disconnect()
            sys.exit(0)
        except Exception as e:
            logger.error(f"Connection error: {e}. Retrying in 5s...")
            time.sleep(5)


if __name__ == "__main__":
    main()
