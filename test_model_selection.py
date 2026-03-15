#!/usr/bin/env python3
"""
Test script to verify model selection implementation.
This tests the MODEL_CONFIGS, get_model function logic without loading actual models.
"""
import ast
from pathlib import Path


APP_PATH = Path(__file__).with_name("app.py")


def read_app_content():
    return APP_PATH.read_text()

def test_model_configs():
    """Test that MODEL_CONFIGS is properly structured"""
    # Import the actual MODEL_CONFIGS from app.py to avoid duplication
    import sys
    import os
    
    # Add app.py directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Since we can't import app.py without triggering model loading,
    # we'll verify it by reading the file content
    content = read_app_content()
    
    # Verify MODEL_CONFIGS structure exists
    assert 'MODEL_CONFIGS = {' in content
    assert '"parakeet-tdt-0.6b-v3"' in content
    assert '"istupakov/parakeet-tdt-0.6b-v3-onnx"' in content
    assert '"grikdotnet/parakeet-tdt-0.6b-fp16"' in content
    
    # Verify quantization settings are present
    assert '"quantization": "int8"' in content
    assert '"quantization": None' in content
    assert '"quantization": "fp16"' in content
    
    # Verify HuggingFace IDs are present
    assert '"hf_id": "nemo-parakeet-tdt-0.6b-v3"' in content
    assert '"hf_id": "istupakov/parakeet-tdt-0.6b-v3-onnx"' in content
    assert '"hf_id": "grikdotnet/parakeet-tdt-0.6b-fp16"' in content
    
    print("✅ MODEL_CONFIGS structure test passed")


def test_model_fallback_logic():
    """Test the fallback logic when unknown model is requested"""
    # Read app.py to verify fallback logic
    content = read_app_content()
    
    # Verify fallback is implemented
    assert 'if model_name not in MODEL_CONFIGS:' in content
    assert 'parakeet-tdt-0.6b-v3' in content  # Default fallback model
    
    # Verify warning is logged
    assert 'Unknown model' in content or 'unknown model' in content.lower()
    
    print("✅ Model fallback logic test passed")


def test_lazy_loading_caching():
    """Test that lazy loading and caching are implemented"""
    content = read_app_content()
    
    # Verify model_cache exists
    assert 'model_cache = {}' in content
    
    # Verify get_model function exists
    assert 'def get_model(model_name):' in content
    
    # Verify caching logic
    assert 'if model_name in model_cache:' in content
    assert 'model_cache[model_name] = model' in content
    
    print("✅ Lazy loading and caching test passed")


def test_openai_compatibility():
    """Test OpenAI compatible parameter defaults"""
    content = read_app_content()
    
    # Default model should be parakeet variant
    assert 'model", "parakeet-tdt-0.6b-v3"' in content
    
    # Verify model_to_use is called
    assert 'model_to_use = get_model(model_name)' in content
    assert 'model_to_use.recognize(chunk_path)' in content
    
    print("✅ OpenAI compatibility test passed")


def test_startup_backend_reporting():
    """Test startup prints the active runtime backend, not a removed variable name"""
    tree = ast.parse(read_app_content(), filename=str(APP_PATH))

    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            comparators = node.test.comparators
            if (
                isinstance(left, ast.Name)
                and left.id == "__name__"
                and comparators
                and isinstance(comparators[0], ast.Constant)
                and comparators[0].value == "__main__"
            ):
                break
    else:
        raise AssertionError("Could not find __main__ block in app.py")

    backend_print_found = False
    for stmt in node.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Name) or call.func.id != "print" or not call.args:
            continue
        arg = call.args[0]
        if not isinstance(arg, ast.JoinedStr):
            continue
        text_parts = [
            value.value for value in arg.values if isinstance(value, ast.Constant) and isinstance(value.value, str)
        ]
        if "ASR backend: " not in "".join(text_parts):
            continue
        assert len(arg.values) >= 2, "Expected backend print to interpolate a variable"
        formatted_value = arg.values[1]
        assert isinstance(formatted_value, ast.FormattedValue)
        assert isinstance(formatted_value.value, ast.Name)
        assert formatted_value.value.id == "ACTIVE_BACKEND"
        backend_print_found = True
        break

    assert backend_print_found, "Expected __main__ block to print the active backend"

    print("✅ Startup backend reporting test passed")


if __name__ == "__main__":
    test_model_configs()
    test_model_fallback_logic()
    test_openai_compatibility()
    test_startup_backend_reporting()
    print("\n✅ All tests passed successfully!")
