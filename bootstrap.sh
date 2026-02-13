#!/bin/bash
# Bootstrap script for Z-Image Turbo RunPod serverless worker.
# Used with a pre-built PyTorch base image (no custom Docker build needed).
#
# This script:
#   1. Installs Python deps (cached on network volume for fast restarts)
#   2. Downloads handler.py from GitHub
#   3. Downloads LoRA from GitHub releases
#   4. Sets up HF cache on network volume
#   5. Runs the handler

set -e

GITHUB_RAW="https://raw.githubusercontent.com/JonBearHimself/avatar-zimage-worker/main"
LORA_URL="https://github.com/JonBearHimself/avatar-zimage-worker/releases/download/v1.0/sg_style_v2.safetensors"

NV="/runpod-volume"
PIP_CACHE="$NV/pip_packages"
APP_CACHE="$NV/app"
MODEL_DIR="/models"

echo "=== Z-Image Turbo Bootstrap ==="
echo "Network volume: $([ -d "$NV" ] && echo 'MOUNTED' || echo 'NOT MOUNTED')"

# ---------------------------------------------------------------------------
# 1. Install Python dependencies (cached on network volume)
# ---------------------------------------------------------------------------
if [ -d "$NV" ]; then
    export HF_HOME="$NV/hf_cache"
    mkdir -p "$HF_HOME" "$PIP_CACHE" "$APP_CACHE" "$MODEL_DIR"

    # Check if packages are already installed on network volume
    if [ -f "$PIP_CACHE/.installed" ]; then
        echo "[cached] Python packages already installed"
    else
        echo "[installing] Python packages to $PIP_CACHE ..."
        pip install --no-cache-dir --target="$PIP_CACHE" \
            diffusers transformers accelerate safetensors \
            sentencepiece peft runpod Pillow numpy
        touch "$PIP_CACHE/.installed"
        echo "[done] Python packages installed"
    fi
    export PYTHONPATH="$PIP_CACHE:${PYTHONPATH:-}"

    # Download handler.py (always refresh to pick up updates)
    echo "[downloading] handler.py ..."
    wget -q -O "$APP_CACHE/handler.py" "$GITHUB_RAW/handler.py"
    echo "[done] handler.py"

    # Download LoRA if not cached
    if [ -f "$NV/models/sg_style_v2.safetensors" ]; then
        echo "[cached] LoRA model"
    else
        echo "[downloading] LoRA model ..."
        mkdir -p "$NV/models"
        wget -q --show-progress -O "$NV/models/sg_style_v2.safetensors" "$LORA_URL"
        echo "[done] LoRA model ($(du -h "$NV/models/sg_style_v2.safetensors" | cut -f1))"
    fi
    ln -sf "$NV/models/sg_style_v2.safetensors" "$MODEL_DIR/sg_style_v2.safetensors"

    HANDLER="$APP_CACHE/handler.py"
else
    # No network volume — download everything fresh
    export HF_HOME="/root/.cache/huggingface"
    mkdir -p "$MODEL_DIR" /app

    echo "[installing] Python packages ..."
    pip install --no-cache-dir \
        diffusers transformers accelerate safetensors \
        sentencepiece peft runpod Pillow numpy

    echo "[downloading] handler.py ..."
    wget -q -O /app/handler.py "$GITHUB_RAW/handler.py"

    if [ -n "$MODEL_URL_LORA" ]; then
        echo "[downloading] LoRA model ..."
        wget -q --show-progress -O "$MODEL_DIR/sg_style_v2.safetensors" "$LORA_URL"
    fi

    HANDLER="/app/handler.py"
fi

echo "=== Bootstrap complete — starting handler ==="
python3 "$HANDLER"
