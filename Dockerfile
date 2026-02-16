# RunPod Serverless Worker — Z-Image Turbo + StrongGirls LoRA
#
# Uses PyTorch 2.5+ for enable_gqa support required by Z-Image attention.
# LoRA + HF model downloaded on first request by handler.py.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.5+ (required for enable_gqa in Z-Image attention)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# diffusers + Z-Image deps + RunPod
RUN pip3 install --no-cache-dir \
    diffusers transformers accelerate safetensors \
    sentencepiece peft scipy \
    runpod \
    Pillow numpy \
    && rm -rf /root/.cache/pip

# Model directory (LoRA downloaded at runtime by handler.py)
RUN mkdir -p /models

# Handler runs directly — no start.sh needed
WORKDIR /app
COPY handler.py .

CMD ["python3", "handler.py"]
