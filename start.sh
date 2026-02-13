#!/bin/bash
# Download LoRA model, then launch the RunPod handler.
# Z-Image Turbo base model is downloaded by diffusers from_pretrained
# and cached in HF_HOME on the network volume.

set -e

MODEL_DIR="/models"
CACHE_DIR="/runpod-volume/models"

# ---------------------------------------------------------------------------
# Set HF cache to network volume for persistent model caching
# ---------------------------------------------------------------------------
if [ -d "/runpod-volume" ]; then
    export HF_HOME="/runpod-volume/hf_cache"
    mkdir -p "$HF_HOME"
    echo "HF cache: $HF_HOME (network volume — persistent)"
else
    export HF_HOME="/root/.cache/huggingface"
    echo "HF cache: $HF_HOME (local — will re-download on cold start)"
fi

# ---------------------------------------------------------------------------
# Download LoRA if not cached
# ---------------------------------------------------------------------------
download_if_missing() {
    local dest="$1"
    local url="$2"
    local name="$(basename "$dest")"

    if [ -f "$dest" ]; then
        echo "  [cached] $name"
        return
    fi

    echo "  [downloading] $name ..."
    mkdir -p "$(dirname "$dest")"
    if [ -n "$HF_TOKEN" ]; then
        wget -q --show-progress --header="Authorization: Bearer $HF_TOKEN" -O "$dest" "$url"
    else
        wget -q --show-progress -O "$dest" "$url"
    fi
    echo "  [done] $name ($(du -h "$dest" | cut -f1))"
}

echo "=== Checking LoRA model ==="

if [ -d "/runpod-volume" ]; then
    echo "Network volume detected — caching LoRA"
    mkdir -p "$CACHE_DIR"

    if [ -n "$MODEL_URL_LORA" ]; then
        download_if_missing "$CACHE_DIR/sg_style_v2.safetensors" "$MODEL_URL_LORA"
    fi

    # Symlink to /models
    ln -sf "$CACHE_DIR/sg_style_v2.safetensors" "$MODEL_DIR/sg_style_v2.safetensors"
else
    echo "No network volume — downloading directly"
    if [ -n "$MODEL_URL_LORA" ]; then
        download_if_missing "$MODEL_DIR/sg_style_v2.safetensors" "$MODEL_URL_LORA"
    fi
fi

echo "=== LoRA ready ==="

echo "=== Starting Z-Image Turbo handler ==="
cd /app
python3 handler.py
