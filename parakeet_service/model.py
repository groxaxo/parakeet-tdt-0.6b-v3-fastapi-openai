"""ONNX Runtime model loader for Parakeet TDT 0.6B v3.

Wraps `onnx_asr` so we can:
  * choose CPU or CUDA providers
  * tune ORT session options
  * lazy-load alternate precision models
  * call `recognize(list_of_waveforms)` for true batched inference
"""
from __future__ import annotations
from typing import Any, Dict, List

import onnxruntime as ort
import onnx_asr

from .config import (
    DEFAULT_MODEL,
    MODEL_CONFIGS,
    ORT_INTRA_THREADS,
    ORT_INTER_THREADS,
    USE_GPU,
    GPU_DEVICE_ID,
    logger,
)

_MODELS: Dict[str, object] = {}
_CUDA_PRELOADED = False


def _preload_cuda_libraries() -> None:
    """Load CUDA/cuDNN shared libraries from Python packages when available."""
    global _CUDA_PRELOADED
    if _CUDA_PRELOADED or USE_GPU not in ("true", "auto"):
        return
    _CUDA_PRELOADED = True
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(cuda=True, cudnn=True, msvc=False)
        logger.info("Preloaded CUDA/cuDNN libraries for ONNX Runtime")
    except Exception as exc:
        if USE_GPU == "true":
            raise RuntimeError("failed to preload CUDA/cuDNN libraries") from exc
        logger.warning("CUDA/cuDNN preload failed; GPU may be unavailable: %s", exc)


def _build_sess_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_INTRA_THREADS
    so.inter_op_num_threads = ORT_INTER_THREADS
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.set_denormal_as_zero", "1")
    so.add_session_config_entry("session.intra_op.allow_spinning", "1")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return so


def _resolve_providers() -> List:
    _preload_cuda_libraries()
    available = ort.get_available_providers()
    want_gpu = USE_GPU in ("true", "auto") and "CUDAExecutionProvider" in available
    providers: List = []
    if want_gpu:
        providers.append((
            "CUDAExecutionProvider",
            {
                "device_id": GPU_DEVICE_ID,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "cudnn_conv_use_max_workspace": "1",
                "do_copy_in_default_stream": True,
            },
        ))
    if USE_GPU != "true":  # always include CPU fallback unless forced GPU
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


def _session_provider_report(model: Any) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {}
    candidates = [("model", model), ("asr", getattr(model, "asr", None))]
    for prefix, obj in candidates:
        if obj is None:
            continue
        for attr in ("_model", "_encoder", "_decoder", "_decoder_joint"):
            session = getattr(obj, attr, None)
            get_providers = getattr(session, "get_providers", None)
            if callable(get_providers):
                report[f"{prefix}.{attr}"] = list(get_providers())
    return report


def _validate_gpu_binding(name: str, model: Any) -> None:
    report = _session_provider_report(model)
    if not report:
        logger.warning("Loaded %s but could not inspect ORT session providers", name)
        return
    logger.info("Session providers for %s: %s", name, report)
    if USE_GPU != "true":
        return
    gpu_bound = all(
        providers and providers[0] in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        for providers in report.values()
    )
    if not gpu_bound:
        raise RuntimeError(
            f"PARAKEET_USE_GPU=true but {name} did not bind all sessions to GPU: {report}"
        )


def load_model(name: str = DEFAULT_MODEL, *, with_timestamps: bool = True):
    if name in _MODELS:
        return _MODELS[name]
    if name not in MODEL_CONFIGS:
        logger.warning("Unknown model %r, falling back to %s", name, DEFAULT_MODEL)
        name = DEFAULT_MODEL
    cfg = MODEL_CONFIGS[name]
    providers = _resolve_providers()
    so = _build_sess_options()
    logger.info("Loading %s (quant=%s) providers=%s intra=%d inter=%d",
                cfg["hf_id"], cfg["quantization"], providers,
                ORT_INTRA_THREADS, ORT_INTER_THREADS)
    m = onnx_asr.load_model(
        cfg["hf_id"],
        quantization=cfg["quantization"],
        providers=providers,
        sess_options=so,
    )
    if with_timestamps:
        m = m.with_timestamps()
    _validate_gpu_binding(name, m)
    _MODELS[name] = m
    logger.info("Loaded %s", name)
    return m


def get_model(name: str = DEFAULT_MODEL):
    if name not in _MODELS:
        return load_model(name)
    return _MODELS[name]


def loaded_models() -> List[str]:
    return list(_MODELS.keys())
