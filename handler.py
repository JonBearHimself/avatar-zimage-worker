"""
RunPod Serverless Handler — Z-Image Turbo + StrongGirls LoRA.

Natural language prompting. 8-step generation. ~3-5s per image on A-series GPUs.
"""

import base64
import io
import os
import sys
import time
import traceback

import runpod
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
LORA_URL = "https://github.com/JonBearHimself/avatar-zimage-worker/releases/download/v2.0/zit_sg_2_000002500.safetensors"

# Use network volume for caching if available
if os.path.isdir("/runpod-volume"):
    HF_CACHE = "/runpod-volume/hf_cache"
    LORA_PATH = "/runpod-volume/models/zit_sg_2_000002500.safetensors"
    os.makedirs(HF_CACHE, exist_ok=True)
    os.makedirs("/runpod-volume/models", exist_ok=True)
else:
    HF_CACHE = None  # use diffusers default
    LORA_PATH = "/models/zit_sg_2_000002500.safetensors"

# ---------------------------------------------------------------------------
# Character definitions — natural language for Z-Image Turbo
# ---------------------------------------------------------------------------
CHARACTERS = {
    "kaori": {
        "appearance": "a young petite muscular girl with short pink hair and pink eyes, wearing black headphones, small chest",
        "outfit": "pink crop top and pink shorts",
    },
    "yuka": {
        "appearance": "a muscular woman with long black hair in a side ponytail with a red scrunchie, red eyes, ahoge",
        "outfit": "black tank top and black cargo pants",
    },
    "haruka": {
        "appearance": "a mature muscular woman with orange hair in a messy bun with long side locks, green eyes, bangs",
        "outfit": "green sundress with low neckline",
    },
    "kasumi": {
        "appearance": "a mature muscular woman with short red hair and a black eyepatch over her right eye, one yellow eye",
        "outfit": "unbuttoned black jacket over a red tank top and black pants",
    },
    "manami": {
        "appearance": "a muscular woman with long wavy light green hair with braided locks tied with red ribbons, light green eyes, medium chest",
        "outfit": "white seifuku with navy sailor collar and navy pleated skirt",
    },
    "miyu": {
        "appearance": "a muscular woman with dark purple wavy hair in medium twintails with black bows, purple eyes, medium chest",
        "outfit": "black sports jacket unzipped over white sports bra and black shorts",
    },
    "naomi": {
        "appearance": "a mature feral-looking muscular woman with short messy white hair and red eyes, sharp teeth",
        "outfit": "black tank top and red shorts",
    },
    "saya": {
        "appearance": "a muscular woman with long straight light blue hair, open light blue eyes, bangs, energetic",
        "outfit": "white button-up shirt with black choker, plaid skirt, and blue tie",
    },
    "hino": {
        "appearance": "a muscular woman with long blonde hair, blue eyes, bangs, large chest",
        "outfit": "seifuku with white shirt, pink ribbon bowtie, pink sailor collar, pink mini skirt",
    },
}

MUSCLE_LEVELS = {
    "default": "muscular with defined biceps, abs, and strong shoulders",
    "athletic": "athletic",
    "muscular": "muscular with defined biceps, abs, and strong shoulders",
    "highly_muscular": "highly muscular female bodybuilder woman with massive thighs, neck muscles, and powerful build",
}

STYLE_ANCHOR = "Clean anime linework, vibrant colors, detailed muscle definition, cel-shaded lighting."

# ---------------------------------------------------------------------------
# Global pipeline
# ---------------------------------------------------------------------------
pipe = None


def download_lora():
    """Download LoRA from GitHub releases if not present."""
    if os.path.exists(LORA_PATH):
        print(f"LoRA already at {LORA_PATH}")
        return
    os.makedirs(os.path.dirname(LORA_PATH), exist_ok=True)
    print(f"Downloading LoRA from {LORA_URL}...")
    sys.stdout.flush()
    import urllib.request
    urllib.request.urlretrieve(LORA_URL, LORA_PATH)
    size_mb = os.path.getsize(LORA_PATH) / (1024 * 1024)
    print(f"  LoRA downloaded ({size_mb:.0f}MB)")
    sys.stdout.flush()


def load_pipeline():
    """Load Z-Image Turbo pipeline + LoRA."""
    global pipe

    from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler

    download_lora()

    t0 = time.time()

    print(f"Loading Z-Image Turbo pipeline...")
    sys.stdout.flush()

    kwargs = {"torch_dtype": torch.bfloat16}
    if HF_CACHE:
        kwargs["cache_dir"] = HF_CACHE
    pipe = ZImagePipeline.from_pretrained(MODEL_ID, **kwargs)
    pipe.to("cuda")

    # Override scheduler: shift=7 + Euler Beta (recommended by LoRA trainer)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        shift=7.0,
        use_beta_sigmas=True,
    )
    print("  Scheduler: FlowMatchEuler, shift=7.0, beta sigmas")
    sys.stdout.flush()

    print(f"  Pipeline loaded in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    # Load and fuse LoRA
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA from {LORA_PATH}...")
        sys.stdout.flush()
        pipe.load_lora_weights(LORA_PATH)
        pipe.fuse_lora(lora_scale=0.8)
        print("  LoRA fused at strength 0.8")
        sys.stdout.flush()
    else:
        print(f"WARNING: LoRA not found at {LORA_PATH} — running without style LoRA")
        sys.stdout.flush()

    print(f"Pipeline fully loaded in {time.time() - t0:.1f}s")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(description: str, character: str = "kaori",
                 muscle_size: str = "default") -> str:
    """Build a natural language prompt for Z-Image Turbo.

    Separates appearance from action/pose for cleaner results.
    """
    char = CHARACTERS.get(character, CHARACTERS["kaori"])
    muscle = MUSCLE_LEVELS.get(muscle_size, MUSCLE_LEVELS["default"])

    # Build the prompt in parts: trigger, appearance, muscle, then scene
    parts = [
        f"anime illustration of {char['appearance']}, {muscle}.",
    ]

    # If description mentions clothing, use that; otherwise use default outfit
    clothing_words = [
        "dress", "shirt", "top", "bikini", "swimsuit", "uniform", "jacket",
        "hoodie", "skirt", "pants", "jeans", "bra", "lingerie", "wearing",
        "bodysuit", "armor", "kimono", "sweater", "naked", "nude", "topless",
        "seifuku", "sundress", "towel", "crop top", "shorts", "leggings",
    ]
    desc_lower = description.lower()
    has_clothing = any(w in desc_lower for w in clothing_words)

    if has_clothing:
        parts.append(description + ".")
    else:
        parts.append(f"Wearing {char['outfit']}, {description}.")

    parts.append(STYLE_ANCHOR)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
def generate_image(job_input: dict) -> dict:
    global pipe

    if pipe is None:
        load_pipeline()

    description = job_input.get("description", "standing, looking at viewer, smile")
    character = job_input.get("character", "kaori")
    muscle_size = job_input.get("muscle_size", "default")
    width = job_input.get("width", 832)
    height = job_input.get("height", 1216)
    seed = job_input.get("seed", -1)

    prompt = build_prompt(description, character, muscle_size)

    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    print(f"PROMPT: {prompt}")
    print(f"PARAMS: char={character}, muscle={muscle_size}, {width}x{height}, seed={seed}")
    sys.stdout.flush()

    t0 = time.time()

    image = pipe(
        prompt=prompt,
        num_inference_steps=8,
        guidance_scale=1.0,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    gen_time = time.time() - t0

    # Encode as base64 PNG
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()

    print(f"Generated: {gen_time:.1f}s, seed={seed}, char={character}")
    sys.stdout.flush()

    return {
        "image_base64": image_b64,
        "seed": seed,
        "prompt": prompt,
        "gen_time": gen_time,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def get_diagnostics() -> dict:
    import subprocess

    diag = {}
    diag["lora_exists"] = os.path.exists(LORA_PATH)
    if diag["lora_exists"]:
        diag["lora_size_mb"] = round(os.path.getsize(LORA_PATH) / (1024 * 1024))
    diag["pipeline_loaded"] = pipe is not None
    diag["network_volume"] = "MOUNTED" if os.path.isdir("/runpod-volume") else "NOT MOUNTED"
    diag["runpod_version"] = runpod.__version__ if hasattr(runpod, "__version__") else "unknown"
    diag["torch_version"] = torch.__version__

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        diag["gpu"] = result.stdout.strip() if result.returncode == 0 else "nvidia-smi failed"
    except Exception as e:
        diag["gpu"] = str(e)

    return diag


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job):
    job_input = job.get("input", {})

    if job_input.get("diagnostic"):
        return get_diagnostics()

    try:
        return generate_image(job_input)
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


print("Z-Image Turbo handler starting — pipeline loads on first request")
runpod.serverless.start({"handler": handler})
