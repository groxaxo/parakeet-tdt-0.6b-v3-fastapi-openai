#!/usr/bin/env python3
import ast
import json
import os
from pathlib import Path


APP_PATH = Path(__file__).with_name("app.py")


def load_helpers():
    source = APP_PATH.read_text()
    tree = ast.parse(source, filename=str(APP_PATH))
    helper_names = {
        "get_env_int",
        "is_openvino_provider_selected",
        "build_provider_chain",
        "build_session_options",
        "get_default_concurrency",
        "load_asr_model",
    }
    helper_nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=helper_nodes, type_ignores=[])
    namespace = {"json": json, "os": os}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


class FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.execution_mode = None
        self.graph_optimization_level = None


class FakeExecutionMode:
    ORT_SEQUENTIAL = "ORT_SEQUENTIAL"


class FakeGraphOptimizationLevel:
    ORT_ENABLE_ALL = "ORT_ENABLE_ALL"
    ORT_DISABLE_ALL = "ORT_DISABLE_ALL"


class FakeOrt:
    SessionOptions = FakeSessionOptions
    ExecutionMode = FakeExecutionMode
    GraphOptimizationLevel = FakeGraphOptimizationLevel

    def __init__(self, providers):
        self._providers = providers

    def get_available_providers(self):
        return list(self._providers)


def provider_name(provider):
    return provider[0] if isinstance(provider, tuple) else provider


def with_env(**updates):
    class EnvContext:
        def __enter__(ctx):
            ctx.previous = {}
            for key, value in updates.items():
                ctx.previous[key] = os.environ.get(key)
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        def __exit__(ctx, exc_type, exc, tb):
            for key, value in ctx.previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return EnvContext()


helpers = load_helpers()


def test_get_env_int():
    get_env_int = helpers["get_env_int"]
    with with_env(TEST_INT_VALUE=None):
        assert get_env_int("TEST_INT_VALUE", 7) == 7
    with with_env(TEST_INT_VALUE="12"):
        assert get_env_int("TEST_INT_VALUE", 7) == 12
    with with_env(TEST_INT_VALUE="invalid"):
        assert get_env_int("TEST_INT_VALUE", 7) == 7
    print("✅ get_env_int test passed")


def test_auto_backend_priority():
    build_provider_chain = helpers["build_provider_chain"]
    ort = FakeOrt(
        [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "OpenVINOExecutionProvider",
        ]
    )
    providers = build_provider_chain(ort, "auto")
    assert providers == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    print("✅ auto backend priority test passed")


def test_openvino_provider_chain():
    build_provider_chain = helpers["build_provider_chain"]
    is_openvino_provider_selected = helpers["is_openvino_provider_selected"]
    ort = FakeOrt(["CPUExecutionProvider", "OpenVINOExecutionProvider"])

    with with_env(
        OV_DEVICE="GPU",
        OV_HINT="LATENCY",
        OV_EXECUTION_MODE="ACCURACY",
        OV_CACHE_DIR="/tmp/ov-cache",
    ):
        providers = build_provider_chain(ort, "openvino")

    assert is_openvino_provider_selected(providers) is True
    provider_name, provider_options = providers[0]
    assert provider_name == "OpenVINOExecutionProvider"
    assert provider_options["device_type"] == "GPU"
    assert provider_options["cache_dir"] == "/tmp/ov-cache"
    load_config = json.loads(provider_options["load_config"])
    assert load_config["GPU"]["PERFORMANCE_HINT"] == "LATENCY"
    assert load_config["GPU"]["EXECUTION_MODE_HINT"] == "ACCURACY"
    assert providers[1] == "CPUExecutionProvider"
    print("✅ openvino provider chain test passed")


def test_openvino_requires_provider():
    build_provider_chain = helpers["build_provider_chain"]
    ort = FakeOrt(["CPUExecutionProvider"])
    try:
        build_provider_chain(ort, "openvino")
    except RuntimeError as exc:
        assert "OpenVINOExecutionProvider" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when OpenVINO provider is unavailable")
    print("✅ openvino provider availability test passed")


def test_session_options_and_concurrency():
    build_session_options = helpers["build_session_options"]
    get_default_concurrency = helpers["get_default_concurrency"]

    with with_env(WEB_THREADS="8"):
        cpu_options = build_session_options(
            FakeOrt(["CPUExecutionProvider"]),
            "cpu",
            ["CPUExecutionProvider"],
        )
        ov_options = build_session_options(
            FakeOrt(["CPUExecutionProvider", "OpenVINOExecutionProvider"]),
            "openvino",
            [("OpenVINOExecutionProvider", {"device_type": "GPU"}), "CPUExecutionProvider"],
        )

    assert cpu_options.intra_op_num_threads == 4
    assert cpu_options.inter_op_num_threads == 1
    assert cpu_options.execution_mode == "ORT_SEQUENTIAL"
    assert cpu_options.graph_optimization_level == "ORT_ENABLE_ALL"

    assert ov_options.intra_op_num_threads == 1
    assert ov_options.inter_op_num_threads == 1
    assert ov_options.graph_optimization_level == "ORT_DISABLE_ALL"
    assert get_default_concurrency("auto", 8) == 8
    assert get_default_concurrency("openvino", 8) == 1
    print("✅ session tuning and concurrency test passed")


def test_openvino_model_load_falls_back_to_cpu():
    load_asr_model = helpers["load_asr_model"]
    calls = []

    class FakeLoadedModel:
        def with_timestamps(self):
            return self

    def fake_loader(hf_id, quantization=None, providers=None, sess_options=None):
        assert providers, "Expected provider list during model load"
        calls.append(providers)
        if provider_name(providers[0]) == "OpenVINOExecutionProvider":
            raise RuntimeError("dynamic rank MatMul unsupported")
        return FakeLoadedModel()

    with with_env(
        OV_DEVICE="GPU",
        OV_HINT="LATENCY",
        OV_EXECUTION_MODE="ACCURACY",
    ):
        model, backend = load_asr_model(
            {"hf_id": "demo/model", "quantization": "int8"},
            FakeOrt(["OpenVINOExecutionProvider", "CPUExecutionProvider"]),
            "openvino",
            loader=fake_loader,
        )

    assert isinstance(model, FakeLoadedModel)
    assert backend == "cpu"
    assert len(calls) == 2
    assert calls[1] == ["CPUExecutionProvider"]
    print("✅ openvino model load fallback test passed")


if __name__ == "__main__":
    test_get_env_int()
    test_auto_backend_priority()
    test_openvino_provider_chain()
    test_openvino_requires_provider()
    test_session_options_and_concurrency()
    test_openvino_model_load_falls_back_to_cpu()
    print("\n✅ Backend configuration tests passed successfully!")
