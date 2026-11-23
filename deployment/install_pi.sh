#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# VeriMath — Automated Raspberry Pi 4 Installation Script
# ─────────────────────────────────────────────────────────────────────────────
# Sets up the entire VeriMath runtime on Raspberry Pi OS (64-bit):
#   1. System packages & build tools
#   2. Lean 4 (elan version manager)
#   3. Python 3.11 virtual environment & dependencies
#   4. llama.cpp / llama-cpp-python
#   5. Model download (Qwen2.5-1.5B Q4_K_M GGUF)
#   6. Mosquitto MQTT broker configuration
#   7. React frontend build
#   8. Systemd service registration & enablement
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "=========================================================="
echo "  VeriMath — Raspberry Pi 4 Automated Deployment"
echo "=========================================================="

WORKDIR="/home/pi/verimath"
cd "$WORKDIR"

# Step 1: System packages
echo ""
echo "[1/8] Updating package lists and installing dependencies..."
sudo apt update
sudo apt install -y curl git python3-pip python3-venv mosquitto mosquitto-clients nodejs npm build-essential cmake tesseract-ocr fswebcam

# Step 2: Mosquitto configuration
echo ""
echo "[2/8] Configuring Mosquitto MQTT broker..."
sudo cp deployment/mosquitto.conf /etc/mosquitto/conf.d/verimath.conf
sudo systemctl restart mosquitto
sudo systemctl enable mosquitto

# Step 3: Lean 4 installation via elan
echo ""
echo "[3/8] Checking Lean 4 installation..."
if ! command -v elan &> /dev/null; then
    echo "Installing elan (Lean 4 version manager)..."
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    source "$HOME/.elan/env"
fi
echo "Lean version: $(lean --version || echo 'Not in PATH yet - source ~/.elan/env')"

# Step 4: Python Virtual Environment
echo ""
echo "[4/8] Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Step 5: Download Qwen2.5-1.5B Model
echo ""
echo "[5/8] Downloading local model (Qwen2.5-1.5B Q4_K_M GGUF)..."
python3 scripts/download_model.py

# Step 6: Frontend Build
echo ""
echo "[6/8] Building React frontend..."
cd frontend
npm install
npm run build
sudo npm install -g serve
cd ..

# Step 7: Environment Setup
echo ""
echo "[7/8] Checking .env configuration..."
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env to insert your GOOGLE_API_KEY and Firebase paths if needed."
fi

# Step 8: Systemd service registration
echo ""
echo "[8/8] Registering systemd services..."
sudo cp deployment/systemd/verimath-backend.service /etc/systemd/system/
sudo cp deployment/systemd/verimath-frontend.service /etc/systemd/system/
sudo cp deployment/systemd/verimath-mqtt-sync.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable verimath-backend
sudo systemctl enable verimath-frontend
sudo systemctl enable verimath-mqtt-sync

sudo systemctl restart verimath-backend
sudo systemctl restart verimath-frontend
sudo systemctl restart verimath-mqtt-sync

echo ""
echo "=========================================================="
echo "  ✅ VeriMath Deployment Complete!"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  Services: sudo systemctl status verimath-backend"
echo "=========================================================="
