import threading

import numpy as np
import pytest

from parakeet_service import model
from parakeet_service import tensorrt_backend as trt


class Encoder:
    def __init__(self, provider="TensorrtExecutionProvider"):
        self.provider = provider
        self.inputs = []

    def get_providers(self):
        return [self.provider, "CPUExecutionProvider"]

    def run(self, names, inputs):
        self.inputs.append(inputs)
        features = inputs["audio_signal"]
        return (np.zeros((1, 4, (features.shape[2] + 7) // 8), dtype=np.float32),
                (inputs["length"] + 7) // 8)


def bare_model():
    asr = trt.TensorRTParakeet.__new__(trt.TensorRTParakeet)
    asr._encoder_lock = threading.Lock()
    asr._encoder = Encoder()
    return asr


def test_short_audio_is_padded_without_changing_real_length():
    asr = bare_model()
    outputs, lengths = asr._encode(np.zeros((1, 128, 3), np.float32), np.array([3], np.int64))
    assert asr._encoder.inputs[0]["audio_signal"].shape == (1, 128, 8)
    assert asr._encoder.inputs[0]["length"].tolist() == [3]
    assert outputs.shape == (1, 1, 4)
    assert lengths.tolist() == [1]


def test_batch_is_serialized_into_batch_one_engine_calls():
    asr = bare_model()
    outputs, lengths = asr._encode(np.zeros((2, 128, 1600), np.float32), np.array([1501, 700], np.int64))
    assert len(asr._encoder.inputs) == 2
    assert all(i["audio_signal"].shape[0] == 1 for i in asr._encoder.inputs)
    assert outputs.shape == (2, 200, 4)
    assert lengths.tolist() == [188, 88]


def test_oversized_features_fail_before_engine_rebuild():
    asr = bare_model()
    with pytest.raises(ValueError, match="chunk audio first"):
        asr._encode(np.zeros((1, 128, 1601), np.float32), np.array([1601], np.int64))
    assert not asr._encoder.inputs


def test_silent_provider_fallback_is_detected(monkeypatch):
    monkeypatch.setattr(trt._AsrWithDecoding, "__init__", lambda *args: None)
    monkeypatch.setattr(trt.ort, "InferenceSession", lambda *args, **kwargs: Encoder("CUDAExecutionProvider"))
    with pytest.raises(RuntimeError, match="fell back"):
        trt.TensorRTParakeet({"encoder": "encoder.onnx"}, None, {}, {})


def test_failed_trt_load_uses_cuda_and_reports_reason(monkeypatch):
    monkeypatch.setattr(model, "_MODELS", {})
    monkeypatch.setattr(model, "_RUNTIMES", {})
    monkeypatch.setattr(model, "GPU_BACKEND", "tensorrt")
    monkeypatch.setattr(model, "USE_GPU", "true")
    monkeypatch.setattr(model, "_resolve_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(model, "_build_sess_options", lambda: None)
    def unavailable(*args):
        raise OSError("TensorRT library absent")
    monkeypatch.setattr(trt, "load_tensorrt_model", unavailable)
    class Model:
        _encoder = Encoder("CUDAExecutionProvider")
        def with_timestamps(self): return self
    fallback = Model()
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return fallback
    monkeypatch.setattr(model.onnx_asr, "load_model", load)
    name = "istupakov/parakeet-tdt-0.6b-v3-onnx"
    assert model.load_model(name) is fallback
    assert model.load_model(name) is fallback
    assert len(calls) == 1
    assert calls[0]["quantization"] is None
    assert model.runtime_status()[name]["backend"] == "cuda"
    assert model.runtime_status()[name]["fallback_reason"] == "TensorRT library absent"


def test_auto_cpu_fallback_reports_actual_cpu_backend(monkeypatch):
    monkeypatch.setattr(model, "_MODELS", {})
    monkeypatch.setattr(model, "_RUNTIMES", {})
    monkeypatch.setattr(model, "GPU_BACKEND", "cuda")
    monkeypatch.setattr(model, "USE_GPU", "auto")
    monkeypatch.setattr(model, "_resolve_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(model, "_build_sess_options", lambda: None)
    class Model:
        _encoder = Encoder("CPUExecutionProvider")
        def with_timestamps(self): return self
    monkeypatch.setattr(model.onnx_asr, "load_model", lambda *args, **kwargs: Model())
    name = "istupakov/parakeet-tdt-0.6b-v3-onnx"
    model.load_model(name)
    status = model.runtime_status()[name]
    assert status["backend"] == "cpu"
    assert status["fallback_reason"] == "ONNX Runtime selected CPU instead of CUDA"
