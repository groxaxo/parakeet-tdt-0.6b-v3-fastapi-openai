from contextlib import asynccontextmanager
import contextlib
import gc
import torch, asyncio
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

from .config import MODEL_NAME, MODEL_NAMES, MODEL_PRECISION, DEVICE, logger

from parakeet_service.batchworker import batch_worker


# Cache for loaded models
_loaded_models = {}


def _to_builtin(obj):
    """torch/NumPy → pure-Python (JSON-safe)."""
    import numpy as np
    import torch as th

    if isinstance(obj, (th.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj


def load_model_for_language(language: str = "default"):
    """Load and cache model for a specific language."""
    # Map language code to model
    if language == "ja":
        model_key = "ja"
    else:
        model_key = "default"
    
    # Return cached model if already loaded
    if model_key in _loaded_models:
        logger.info("Using cached model for language: %s", language)
        return _loaded_models[model_key]
    
    # Load new model
    model_name = MODEL_NAMES[model_key]
    logger.info("Loading %s for language %s...", model_name, language)
    
    with torch.inference_mode():
        dtype = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name, 
            map_location=DEVICE
        ).to(dtype=dtype)
        logger.info("Loaded %s with %s weights on %s", model_name, MODEL_PRECISION.upper(), DEVICE)
    
    _loaded_models[model_key] = model
    gc.collect()
    torch.cuda.empty_cache()
    
    return model


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; free GPU on shutdown."""
    logger.info("Loading %s with optimized memory...", MODEL_NAME)
    
    # Pre-load the default model
    model = load_model_for_language("default")
    
    app.state.asr_model = model
    logger.info("Model ready on %s", next(model.parameters()).device)

    app.state.worker = asyncio.create_task(batch_worker(model), name="batch_worker")
    logger.info("batch_worker scheduled")

    try:
        yield
    finally:
        app.state.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.worker

        logger.info("Releasing GPU memory and shutting down worker")
        
        # Clean up all loaded models
        for model_key in list(_loaded_models.keys()):
            del _loaded_models[model_key]
        
        if hasattr(app.state, 'asr_model'):
            del app.state.asr_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free cache but keep driver


def reset_fast_path(model):
    """Restore low-latency decoding flags."""
    with open_dict(model.cfg.decoding):
        if getattr(model.cfg.decoding, "compute_timestamps", False):
            model.cfg.decoding.compute_timestamps = False
        if getattr(model.cfg.decoding, "preserve_alignments", False):
            model.cfg.decoding.preserve_alignments = False
    model.change_decoding_strategy(model.cfg.decoding)
