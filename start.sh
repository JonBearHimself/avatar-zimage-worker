#!/bin/bash
# Z-Image Turbo RunPod Serverless Worker — Startup Script
#
# Base image: pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime (has torch pre-installed)
# This script installs extra Python deps, downloads LoRA, then runs the handler.

set -e

MODEL_DIR="/models"
LORA_URL="https://github.com/JonBearHimself/avatar-zimage-worker/releases/download/v1.0/sg_style_v2.safetensors"

echo "=== Z-Image Turbo Worker Starting ==="
echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"

# ---------------------------------------------------------------------------
# HF cache on network volume
# ---------------------------------------------------------------------------
if [ -d "/runpod-volume" ]; then
    export HF_HOME="/runpod-volume/hf_cache"
    mkdir -p "$HF_HOME"
    echo "HF cache: $HF_HOME (network volume)"
else
    export HF_HOME="/root/.cache/huggingface"
    echo "HF cache: $HF_HOME (local)"
fi

# ---------------------------------------------------------------------------
# Install extra Python packages (torch is already in the base image)
# Cache on network volume for fast warm starts
# ---------------------------------------------------------------------------
PIP_TARGET="/runpod-volume/pip_packages"

install_deps() {
    local target="$1"
    if [ -n "$target" ]; then
        pip install --no-cache-dir --target="$target" \
            diffusers transformers accelerate safetensors \
            sentencepiece peft runpod Pillow 2>&1 | tail -3
        export PYTHONPATH="$target:${PYTHONPATH:-}"
    else
        pip install --no-cache-dir \
            diffusers transformers accelerate safetensors \
            sentencepiece peft runpod Pillow 2>&1 | tail -3
    fi
}

if [ -d "/runpod-volume" ]; then
    if [ -f "$PIP_TARGET/.deps_v2" ]; then
        echo "[cached] Python packages"
        export PYTHONPATH="$PIP_TARGET:${PYTHONPATH:-}"
    else
        echo "[installing] Python packages to $PIP_TARGET ..."
        mkdir -p "$PIP_TARGET"
        install_deps "$PIP_TARGET"
        touch "$PIP_TARGET/.deps_v2"
        echo "[done] Packages installed"
    fi
else
    echo "[installing] Python packages ..."
    install_deps ""
    echo "[done] Packages installed"
fi

# ---------------------------------------------------------------------------
# Download LoRA
# ---------------------------------------------------------------------------
echo "=== Checking LoRA model ==="
mkdir -p "$MODEL_DIR"

download_lora() {
    local dest="$1"
    if [ -f "$dest" ]; then
        echo "  [cached] LoRA ($(du -h "$dest" | cut -f1))"
        return
    fi
    echo "  [downloading] LoRA from GitHub releases ..."
    python3 -c "
import urllib.request, sys, os
url = '$LORA_URL'
dest = '$dest'
print(f'  Downloading to {dest}...')
sys.stdout.flush()
urllib.request.urlretrieve(url, dest)
size_mb = os.path.getsize(dest) / (1024*1024)
print(f'  Done ({size_mb:.0f}MB)')
"
}

if [ -d "/runpod-volume" ]; then
    CACHE_DIR="/runpod-volume/models"
    mkdir -p "$CACHE_DIR"
    download_lora "$CACHE_DIR/sg_style_v2.safetensors"
    ln -sf "$CACHE_DIR/sg_style_v2.safetensors" "$MODEL_DIR/sg_style_v2.safetensors"
else
    download_lora "$MODEL_DIR/sg_style_v2.safetensors"
fi

echo "=== Starting handler ==="
cd /app
exec python3 handler.py
