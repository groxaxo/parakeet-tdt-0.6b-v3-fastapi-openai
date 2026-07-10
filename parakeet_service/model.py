"""Thread-safe ONNX Runtime model loading for Parakeet TDT."""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

# Import config before ONNX Runtime so thread-pool environment limits are active.
from .config import (
    DEFAULT_MODEL,
    GPU_DEVICE_ID,
    MODEL_CONFIGS,
    ORT_INTER_THREADS,
    ORT_INTRA_THREADS,
    USE_GPU,
    logger,
)

import onnx_asr
import onnxruntime as ort

_ModelKey = Tuple[str, bool]
_MODELS: Dict[_ModelKey, object] = {}
_MODEL_LOCK = threading.RLock()
_CUDA_PRELOADED = False


def _preload_cuda_libraries() -> bool:
    """Load CUDA/cuDNN libraries before creating any ORT session."""
    global _CUDA_PRELOADED
    if USE_GPU not in {"true", "auto"}:
        return False
    if _CUDA_PRELOADED:
        return True
    with _MODEL_LOCK:
        if _CUDA_PRELOADED:
            return True
        preload = getattr(ort, "preload_dlls", None)
        if preload is None:
            if USE_GPU == "true":
                raise RuntimeError(
                    "onnxruntime-gpu does not expose preload_dlls; install a "
                    "compatible ONNX Runtime GPU package"
                )
            return True
        try:
            # Empty directory explicitly prefers NVIDIA runtime wheels installed
            # by the [cuda,cudnn] extra over potentially incompatible system libs.
            preload(cuda=True, cudnn=True, msvc=False, directory="")
            _CUDA_PRELOADED = True
            logger.info("Preloaded CUDA/cuDNN libraries for ONNX Runtime")
            return True
        except Exception as exc:
            if USE_GPU == "true":
                raise RuntimeError("failed to preload CUDA/cuDNN libraries") from exc
            logger.warning("CUDA/cuDNN preload failed; using CPU fallback: %s", exc)
            return False


def _build_sess_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = ORT_INTRA_THREADS
    options.inter_op_num_threads = ORT_INTER_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def _resolve_providers() -> List[Any]:
    cuda_runtime_ready = _preload_cuda_libraries()
    available = set(ort.get_available_providers())
    has_cuda = cuda_runtime_ready and "CUDAExecutionProvider" in available

    if USE_GPU == "true" and not has_cuda:
        raise RuntimeError(
            "PARAKEET_USE_GPU=true but CUDAExecutionProvider is unavailable; "
            f"available providers: {sorted(available)}"
        )
    if USE_GPU == "false" or not has_cuda:
        return ["CPUExecutionProvider"]

    cuda = (
        "CUDAExecutionProvider",
        {
            "device_id": GPU_DEVICE_ID,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "1",
        },
    )
    return [cuda] if USE_GPU == "true" else [cuda, "CPUExecutionProvider"]


def _session_provider_report(model: Any) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {}
    candidates = [("model", model), ("asr", getattr(model, "asr", None))]
    for prefix, candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("_model", "_encoder", "_decoder", "_decoder_joint"):
            session = getattr(candidate, attribute, None)
            get_providers = getattr(session, "get_providers", None)
            if callable(get_providers):
                report[f"{prefix}.{attribute}"] = list(get_providers())
    return report


def _validate_gpu_binding(name: str, model: Any) -> None:
    report = _session_provider_report(model)
    if report:
        logger.info("Session providers for %s: %s", name, report)
    if USE_GPU != "true":
        return
    if not report:
        raise RuntimeError(
            f"PARAKEET_USE_GPU=true but ORT providers for {name} could not be inspected"
        )
    if not all(
        providers
        and providers[0] in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        for providers in report.values()
    ):
        raise RuntimeError(
            f"PARAKEET_USE_GPU=true but {name} did not bind all sessions to GPU: {report}"
        )


def load_model(name: str = DEFAULT_MODEL, *, with_timestamps: bool = True):
    normalized = (name or DEFAULT_MODEL).strip().lower()
    if normalized not in MODEL_CONFIGS:
        raise ValueError(f"unknown model {name!r}; choose one of {sorted(MODEL_CONFIGS)}")
    key = (normalized, with_timestamps)

    with _MODEL_LOCK:
        cached = _MODELS.get(key)
        if cached is not None:
            return cached

        config = MODEL_CONFIGS[normalized]
        providers = _resolve_providers()
        session_options = _build_sess_options()
        logger.info(
            "Loading %s (quant=%s) providers=%s intra=%d inter=%d",
            config["hf_id"],
            config["quantization"],
            providers,
            ORT_INTRA_THREADS,
            ORT_INTER_THREADS,
        )
        model = onnx_asr.load_model(
            config["hf_id"],
            quantization=config["quantization"],
            providers=providers,
            sess_options=session_options,
        )
        if with_timestamps:
            model = model.with_timestamps()
        _validate_gpu_binding(normalized, model)
        _MODELS[key] = model
        logger.info("Loaded %s", normalized)
        return model


def get_model(name: str = DEFAULT_MODEL):
    return load_model(name, with_timestamps=True)


def loaded_models() -> List[str]:
    with _MODEL_LOCK:
        return sorted({name for name, _timestamps in _MODELS})
