import os
import subprocess
import sys


def check_config(**overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PARAKEET_")}
    env.update(overrides)
    return subprocess.run([sys.executable, "-c", "from parakeet_service.config import *; print(GPU_BACKEND, BATCHED, INFER_WORKERS, CHUNK_MAX_SEC)"], env=env, capture_output=True, text=True)


def test_gpu_default_is_serial_tensorrt_with_bounded_chunks():
    result = check_config()
    assert result.returncode == 0
    assert "tensorrt False 1 15.0" in result.stdout


def test_explicit_oversized_trt_chunks_are_rejected():
    result = check_config(PARAKEET_CHUNK_MAX_SEC="75")
    assert result.returncode != 0
    assert "TensorRT requires" in result.stderr


def test_cuda_rollback_keeps_original_chunk_range():
    result = check_config(PARAKEET_GPU_BACKEND="cuda")
    assert result.returncode == 0
    assert "75.0" in result.stdout


def test_cpu_override_keeps_original_chunk_range():
    result = check_config(PARAKEET_USE_GPU="false")
    assert result.returncode == 0
    assert "75.0" in result.stdout
