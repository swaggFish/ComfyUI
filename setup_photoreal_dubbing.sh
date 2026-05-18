#!/bin/bash

# setup_photoreal_dubbing.sh
# Automated installer for LatentSync and Impact Pack in ComfyUI

set -e

COMFY_DIR="/home/mikeyb/Documents/AI/ComfyUI"
CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"
VENV_PYTHON="$COMFY_DIR/venv/bin/python"
VENV_PIP="$COMFY_DIR/venv/bin/pip"

echo "=========================================================="
echo "🎬 Initializing Photorealistic Dubbing Node Installation 🎬"
echo "=========================================================="

cd "$CUSTOM_NODES_DIR"

# 1. Install ComfyUI-Impact-Pack (Face Detailer & YOLO logic)
echo "----------------------------------------------------------"
echo "📦 Installing ComfyUI-Impact-Pack (Face Detailer)..."
if [ ! -d "ComfyUI-Impact-Pack" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
    cd ComfyUI-Impact-Pack
    git submodule update --init --recursive
    $VENV_PYTHON install.py || echo "Impact Pack install script skipped or warned, continuing..."
    cd ..
else
    echo "✅ ComfyUI-Impact-Pack is already installed!"
fi

# 2. Install ComfyUI-LatentSyncWrapper (ByteDance Lip-Sync)
echo "----------------------------------------------------------"
echo "👄 Installing ComfyUI-LatentSyncWrapper..."
if [ ! -d "ComfyUI-LatentSyncWrapper" ]; then
    git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git
    cd ComfyUI-LatentSyncWrapper
    echo "⚙️ Installing Python requirements for LatentSync..."
    $VENV_PIP install -r requirements.txt
    cd ..
else
    echo "✅ ComfyUI-LatentSyncWrapper is already installed!"
fi

echo "=========================================================="
echo "✅ Custom Nodes Installed Successfully!"
echo "Note: The LatentSync model checkpoints and GFPGAN/CodeFormer models"
echo "will automatically download via the ComfyUI-Manager or upon first run,"
echo "depending on your setup."
echo "Please restart your ComfyUI server to load the new nodes."
echo "=========================================================="
