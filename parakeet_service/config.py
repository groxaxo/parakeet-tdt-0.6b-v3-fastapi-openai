"""Configuration for the optimized Parakeet v3 service."""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & model
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Point HF cache at local models dir
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "true")

MODEL_CONFIGS = {
    "parakeet-tdt-0.6b-v3": {
        "hf_id": "nemo-parakeet-tdt-0.6b-v3",
        "quantization": "int8",
        "description": "INT8 (fastest)",
    },
    "istupakov/parakeet-tdt-0.6b-v3-onnx": {
        "hf_id": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "quantization": None,
        "description": "FP32",
    },
    "grikdotnet/parakeet-tdt-0.6b-fp16": {
        "hf_id": "grikdotnet/parakeet-tdt-0.6b-fp16",
        "quantization": "fp16",
        "description": "FP16",
    },
}
DEFAULT_MODEL = "parakeet-tdt-0.6b-v3"

# ---------------------------------------------------------------------------
# Performance knobs
# ---------------------------------------------------------------------------
TARGET_SR = 16_000

# Auto-chunking targets (seconds)
CHUNK_TARGET_SEC = float(os.getenv("PARAKEET_CHUNK_TARGET_SEC", "60"))
CHUNK_MAX_SEC = float(os.getenv("PARAKEET_CHUNK_MAX_SEC", "75"))
CHUNK_MIN_SEC = float(os.getenv("PARAKEET_CHUNK_MIN_SEC", "20"))

# VAD parameters (Silero)
VAD_THRESHOLD = float(os.getenv("PARAKEET_VAD_THRESHOLD", "0.5"))
VAD_MIN_SILENCE_MS = int(os.getenv("PARAKEET_VAD_MIN_SILENCE_MS", "400"))
VAD_SPEECH_PAD_MS = int(os.getenv("PARAKEET_VAD_SPEECH_PAD_MS", "120"))

# Micro-batch worker
MAX_BATCH_SIZE = int(os.getenv("PARAKEET_MAX_BATCH_SIZE", "16"))
BATCH_WINDOW_MS = float(os.getenv("PARAKEET_BATCH_WINDOW_MS", "8"))

# Providers
USE_GPU = os.getenv("PARAKEET_USE_GPU", "auto").lower()  # auto|true|false
GPU_DEVICE_ID = int(os.getenv("PARAKEET_GPU_DEVICE_ID", "0"))


def _get_env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(minimum, default)


# ORT threading (CPU)
import os as _os
try:
    _available_logical = len(_os.sched_getaffinity(0))
except (AttributeError, OSError):
    _available_logical = _os.cpu_count() or 1

try:
    import psutil  # type: ignore
    _physical = psutil.cpu_count(logical=False) or _available_logical
except Exception:
    _physical = _available_logical

DEFAULT_INTRA = min(_physical, _available_logical)
ORT_INTRA_THREADS = _get_env_int("PARAKEET_ORT_INTRA_THREADS", DEFAULT_INTRA)
ORT_INTER_THREADS = _get_env_int("PARAKEET_ORT_INTER_THREADS", 1)

# Audio preprocessing pool
AUDIO_WORKERS = _get_env_int("PARAKEET_AUDIO_WORKERS", min(8, _physical))

# Keep numeric libs from creating competing thread pools
for _e in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_e, "1")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("parakeet_v3")

CPU_INFO = {
    "physical": _physical,
    "logical": _available_logical,
    "ort_intra": ORT_INTRA_THREADS,
    "ort_inter": ORT_INTER_THREADS,
    "audio_workers": AUDIO_WORKERS,
}
