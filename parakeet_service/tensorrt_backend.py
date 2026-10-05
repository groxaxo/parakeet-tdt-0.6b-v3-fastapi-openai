"""Encoder-only TensorRT integration for the pinned onnx-asr 0.12.0 runtime.

The upstream constructor gives both encoder and decoder the same providers.
This small subclass reuses its TDT decoding and timestamps while creating the
two sessions separately. No process-wide monkeypatching or duplicate encoder.
"""
from __future__ import annotations

import ctypes
import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx_asr.asr import _AsrWithDecoding
from onnx_asr.loader import Manager
from onnx_asr.models.nemo import NemoConformerTdt
from onnx_asr.resolver import Resolver

from .config import GPU_DECODER_STATE, GPU_DEVICE_ID, TRT_CACHE_DIR, TRT_MAX_FRAMES, TRT_WORKSPACE_MB


@dataclass(frozen=True)
class _DecoderState:
    request: "_DecoderRequest"
    slot: int


class _DecoderRequest:
    """One utterance's binding, host scalars/logits, and accepted/spare state."""
    def __init__(self, session, initial, token_dtypes):
        self.binding = session.io_binding()
        self.states = (
            tuple(ort.OrtValue.ortvalue_from_numpy(v, "cuda", GPU_DEVICE_ID) for v in initial),
            tuple(ort.OrtValue.ortvalue_from_shape_and_type(v.shape, v.dtype, "cuda", GPU_DEVICE_ID) for v in initial),
        )
        self.targets = np.zeros((1, 1), token_dtypes["targets"])
        self.targets_value = ort.OrtValue.ortvalue_from_numpy(self.targets, "cuda", GPU_DEVICE_ID)
        self.target_length = np.ones(1, token_dtypes["target_length"])
        self.logits = None
        self.logits_value = None
        self.binding.bind_ortvalue_input("targets", self.targets_value)
        self.binding.bind_cpu_input("target_length", self.target_length)
        self.binding.bind_output("outputs", "cpu")
        self.frame = None
        self.frame_value = None
        self.bound_slot = None


def _preload_tensorrt() -> None:
    # NVIDIA's wheel stores libraries outside the OS search path. Load its
    # dependencies by absolute path before ONNX Runtime dlopens the provider.
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is not None and spec.submodule_search_locations:
        directory = Path(next(iter(spec.submodule_search_locations)))
        for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
            ctypes.CDLL(str(directory / name), mode=ctypes.RTLD_GLOBAL)
    else:
        for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)


class TensorRTParakeet(NemoConformerTdt):
    def __init__(self, files, preprocessor_factory, encoder_options, cuda_options):
        # Initialize upstream metadata/tokenizer without allocating its sessions.
        _AsrWithDecoding.__init__(self, files, preprocessor_factory, encoder_options)
        self._encoder_lock = threading.Lock()
        self._encoder = ort.InferenceSession(files["encoder"], **encoder_options)
        if self._encoder.get_providers()[0] != "TensorrtExecutionProvider":
            raise RuntimeError("TensorRT encoder initialization fell back to another provider")
        self._decoder_joint = ort.InferenceSession(files["decoder_joint"], **cuda_options)
        self.gpu_decoder_state = GPU_DECODER_STATE
        self._token_dtypes = {
            value.name: {"tensor(int32)": np.int32, "tensor(int64)": np.int64}[value.type]
            for value in self._decoder_joint.get_inputs()
            if value.name in {"targets", "target_length"}
        }
        if self.gpu_decoder_state and self._decoder_joint.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("GPU decoder state requires the CUDA decoder provider")

    def _create_state(self):
        state = super()._create_state()
        if not self.gpu_decoder_state:
            return state
        request = _DecoderRequest(self._decoder_joint, state, self._token_dtypes)
        return _DecoderState(request, 0)

    def _decode(self, prev_tokens, prev_state, encoder_out):
        if not self.gpu_decoder_state:
            return super()._decode(prev_tokens, prev_state, encoder_out)
        request, accepted = prev_state.request, prev_state.slot
        binding = request.binding
        if request.frame is None:
            request.frame = np.zeros((1, encoder_out.size, 1), np.float32)
            request.frame_value = ort.OrtValue.ortvalue_from_numpy(request.frame, "cuda", GPU_DEVICE_ID)
            binding.bind_ortvalue_input("encoder_outputs", request.frame_value)
        request.frame[0, :, 0] = encoder_out
        request.targets[0, 0] = prev_tokens[-1] if prev_tokens else self._blank_idx
        # CPU inputs may be copied at BindInput time by ORT. Updating their
        # numpy arrays alone leaves stale device data. Upload explicitly.
        request.frame_value.update_inplace(request.frame)
        request.targets_value.update_inplace(request.targets)
        candidate = 1 - accepted
        if request.bound_slot != accepted:
            for index in range(2):
                binding.bind_ortvalue_input(f"input_states_{index + 1}", request.states[accepted][index])
                binding.bind_ortvalue_output(f"output_states_{index + 1}", request.states[candidate][index])
            request.bound_slot = accepted
        # A blank keeps prev_state's slot, so the next step overwrites only the
        # rejected candidate. Only upstream's nonblank accept switches slots.
        self._decoder_joint.run_with_iobinding(binding)
        if request.logits is None:
            request.logits = binding.get_outputs()[0].numpy()
            request.logits_value = ort.OrtValue.ortvalue_from_numpy(request.logits)
            binding.bind_ortvalue_output("outputs", request.logits_value)
        logits = request.logits.reshape(-1)
        return logits[:self._vocab_size], int(logits[self._vocab_size:].argmax()), _DecoderState(request, candidate)

    def _encode(self, features, features_lens):
        if features.shape[2] > TRT_MAX_FRAMES:
            raise ValueError("audio exceeds the TensorRT encoder profile; chunk audio first")
        # Very short clips can produce fewer frames than the convolution needs.
        if features.shape[2] < 8:
            features = np.pad(features, ((0, 0), (0, 0), (0, 8 - features.shape[2])))
        outputs, lengths = [], []
        # Batch APIs still work, but each encoder call uses the measured batch-1
        # engine. Serialize its context even if users enable multiple workers.
        with self._encoder_lock:
            for index in range(features.shape[0]):
                output, length = super()._encode(features[index:index + 1], features_lens[index:index + 1])
                outputs.append(output)
                lengths.append(length)
        return np.concatenate(outputs), np.concatenate(lengths)


def load_tensorrt_model(hf_id, session_options, cuda_providers):
    _preload_tensorrt()
    cache = TRT_CACHE_DIR / "encoder-fp16-16s-v1"
    cache.mkdir(parents=True, exist_ok=True)
    trt_options = {
        "device_id": GPU_DEVICE_ID,
        "trt_fp16_enable": True,
        "trt_max_workspace_size": TRT_WORKSPACE_MB * 1024**2,
        "trt_builder_optimization_level": 2,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache),
        "trt_force_sequential_engine_build": True,
        "trt_min_subgraph_size": 5,
        "trt_profile_min_shapes": "audio_signal:1x128x8,length:1",
        "trt_profile_opt_shapes": "audio_signal:1x128x800,length:1",
        "trt_profile_max_shapes": f"audio_signal:1x128x{TRT_MAX_FRAMES},length:1",
    }
    cuda_options = {"sess_options": session_options, "providers": cuda_providers}
    encoder_options = {
        "sess_options": session_options,
        "providers": [("TensorrtExecutionProvider", trt_options), *cuda_providers],
    }
    cpu_options = {"sess_options": session_options, "providers": ["CPUExecutionProvider"]}
    manager = Manager(
        **cuda_options,
        preprocessor_config={**cpu_options, "use_numpy_preprocessors": False, "use_conv_preprocessors": False},
        resampler_config=cpu_options,
    )
    files = Resolver(TensorRTParakeet, hf_id).resolve_model()
    asr = TensorRTParakeet(files, manager._create_preprocessor, encoder_options, cuda_options)
    return manager._create_asr_adapter(asr)
