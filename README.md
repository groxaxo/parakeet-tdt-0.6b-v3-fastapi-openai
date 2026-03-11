# Parakeet TDT Transcription with ONNX Runtime

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Parakeet TDT** is a high-performance implementation of NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model using [ONNX Runtime](https://onnxruntime.ai/), designed for ultra-fast inference on CPU.

This implementation achieves exceptional real-time speeds, outperforming standard [openai/whisper](https://github.com/openai/whisper) and competing directly with GPU-accelerated [faster-whisper](https://github.com/SYSTRAN/faster-whisper) implementations while running entirely on consumer CPUs. The efficiency is achieved through the architectural advantages of the Token-and-Duration Transducer (TDT) model combined with 8-bit quantization.

## 🌍 Multilingual Support

**Parakeet TDT 0.6B v3** features robust multilingual capabilities with **automatic language detection**. The model can automatically identify and transcribe speech in any of the **25 supported languages** without requiring manual language specification:

English, Spanish, French, Russian, German, Italian, Polish, Ukrainian, Romanian, Dutch, Hungarian, Greek, Swedish, Czech, Bulgarian, Portuguese, Slovak, Croatian, Danish, Finnish, Lithuanian, Slovenian, Latvian, Estonian, Maltese

Simply send audio in any of these languages, and the model will automatically detect and transcribe it with high accuracy, including proper punctuation and capitalization.

## Benchmark

### LibriSpeech test-clean (Verified Ground Truth) ⭐

Benchmarked on **LibriSpeech test-clean** dataset with professionally verified human transcriptions. This provides reliable, reproducible accuracy metrics.

**Test Environment:** CPU-only inference, 50 samples (~350 seconds of audio)

| Model | Precision | Accuracy | WER | CER | Speedup (RTF) |
|-------|-----------|----------|-----|-----|---------------|
| **Parakeet TDT 0.6B v3** | INT8 | **97.84%** | 2.16% | 0.56% | **18.41x** (0.054) |
| **Parakeet TDT 0.6B v3** | FP16 | **97.84%** | 2.16% | 0.56% | **18.82x** (0.053) |
| **Parakeet TDT 0.6B v3** | FP32 | **97.84%** | 2.16% | 0.56% | **19.42x** (0.052) |
| Whisper Large v3* | FP16 | ~95-96% | ~4-5% | ~2-3% | varies |

> *Whisper Large v3 benchmarks from published literature on LibriSpeech test-clean. Actual results vary by implementation and hardware.

**Key Findings:**
- All Parakeet precision variants achieve **identical accuracy** (97.84%)
- INT8 quantization has **zero accuracy loss** vs FP32
- Real-time factor (RTF) of ~0.05 means 20x faster than real-time
- Competitive with Whisper Large v3 accuracy with significantly faster CPU inference

---

### Parakeet TDT vs Faster Whisper

We compare the performance of **Parakeet TDT (CPU)** against **faster-whisper (GPU & CPU)**.

The metric used is **Speedup Factor** (Audio Duration / Processing Time). Higher is better.

| Implementation | Hardware | Model | Precision | Speedup |
| --- | --- | --- | --- | --- |
| **Parakeet TDT** (Ours) | **CPU** (i7-12700KF) | **TDT 0.6B v3** | **int8** | **~29.7x** |
| **Parakeet TDT** (Ours) | **CPU** (i7-4790) | **TDT 0.6B v3** | **int8** | **~17.0x** |
| faster-whisper | GPU (RTX 3070 Ti) | Large-v2 | int8 | 13.2x |
| faster-whisper | GPU (RTX 3070 Ti) | Large-v2 | fp16 | 12.4x |
| faster-whisper | CPU (i7-12700K) | Small | int8 | 7.6x |
| faster-whisper | CPU (i7-12700K) | Small | fp32 | 4.9x |

*   **Parakeet TDT**: Benchmarked on Intel Core i7-12700K with ONNX Runtime INT8.
*   **faster-whisper**: Benchmarks from [official faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper).

### Detailed Parakeet Performance

| Metrics | Result |
| --- | --- |
| **Average Speedup** | **29.7x** |
| **Real Time Factor (RTF)** | **0.033** |
| **Max Speedup** | **~30x** |

### Extended Multilingual Benchmark (YouTube Samples)

Additional benchmark on real-world YouTube content across multiple languages:

| Language | Model Variant | Latency (s) | Speedup (RTF) | WER | CER |
| --- | --- | ---: | ---: | ---: | ---: |
| English | INT8 (`parakeet-tdt-0.6b-v3`) | 70.60 | 20.32x (0.049) | 5.13% | 2.35% |
| English | FP16 (`grikdotnet/parakeet-tdt-0.6b-fp16`) | 135.43 | 10.59x (0.094) | 5.48% | 2.83% |
| English | FP32 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 112.80 | 12.72x (0.079) | 5.53% | 2.85% |
| English | Whisper-Large-v3 (DeepInfra) | 53.45 | 26.84x (0.037) | 4.25% | 3.91% |
| Spanish | INT8 (`parakeet-tdt-0.6b-v3`) | 29.92 | 18.64x (0.054) | 19.45% | 13.79% |
| Spanish | FP16 (`grikdotnet/parakeet-tdt-0.6b-fp16`) | 48.52 | 11.49x (0.087) | 15.31% | 11.33% |
| Spanish | FP32 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 38.99 | 14.30x (0.070) | 15.31% | 11.33% |
| Spanish | Whisper-Large-v3 (DeepInfra) | 15.79 | 35.30x (0.028) | 20.70% | 18.05% |

> ⚠️ **Note:** YouTube subtitle references may contain errors. For verified accuracy, see LibriSpeech benchmark above.

## Requirements

*   [Docker](https://docs.docker.com/get-docker/) (Recommended)
*   Or: Python 3.10+ and [FFmpeg](https://ffmpeg.org/)

### CPU Optimization
For hybrid CPUs (like Intel 12th-14th Gen), performance is significantly improved by pinning the process to Performance cores (P-cores).

## Installation

### 🐳 Docker (Recommended)

The easiest way to get started. No dependencies to install!

**CPU Deployment:**
```bash
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai
cd parakeet-tdt-0.6b-v3-fastapi-openai
docker compose up parakeet-cpu -d
```

**GPU Deployment** (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
```bash
docker compose up parakeet-gpu -d
```

**Optional Intel iGPU / OpenVINO Deployment** (opt-in only, requires compatible Intel hardware and drivers):
```bash
docker compose --profile openvino up parakeet-openvino -d
```

The server will be available at `http://localhost:5092`. See [DOCKER.md](DOCKER.md) for more options.

---

### Conda (Alternative)

For development or customization:

```bash
conda create -n parakeet-onnx python=3.10
conda activate parakeet-onnx
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai
cd parakeet-tdt-0.6b-v3-fastapi-openai
pip install -r requirements.txt
```

**Optional OpenVINO install path for Intel iGPU users:**
```bash
pip install -r requirements.openvino.txt
ASR_BACKEND=openvino python app.py
```

## Usage

### Start the Server

Parakeet TDT provides an OpenAI-compatible API server.

```bash
conda activate parakeet-onnx
python app.py
```
*   **Port**: 5092
*   **Docs**: [http://127.0.0.1:5092/docs](http://127.0.0.1:5092/docs)

### Backend Selection

Runtime backend selection is controlled with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5092` | HTTP listen port |
| `WEB_THREADS` | `8` | Waitress worker thread count |
| `ASR_BACKEND` | `auto` | `auto`, `cpu`, `cuda`, `tensorrt`, or `openvino` |
| `MAX_CONCURRENT_INFERENCES` | backend-aware | Optional override for concurrent transcription requests |
| `OV_DEVICE` | `GPU` | OpenVINO target device when `ASR_BACKEND=openvino` |
| `OV_HINT` | `LATENCY` | OpenVINO performance hint when `ASR_BACKEND=openvino` |
| `OV_EXECUTION_MODE` | `ACCURACY` | OpenVINO execution mode hint when `ASR_BACKEND=openvino` |
| `OV_CACHE_DIR` | empty | Optional OpenVINO model cache directory |

`ASR_BACKEND=auto` preserves the existing default provider priority: TensorRT → CUDA → CPU. OpenVINO is never selected automatically; it is only used when you explicitly set `ASR_BACKEND=openvino`.
These environment variables are read when the server starts, so restart the process or container after changing them.

### Client Example (Python)

You can use the standard `openai` Python library to interact with the server.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="sk-no-key-required"
)

audio_file = open("audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="parakeet-tdt-0.6b-v3",  # or "istupakov/parakeet-tdt-0.6b-v3-onnx" or "grikdotnet/parakeet-tdt-0.6b-fp16"
  file=audio_file,
  response_format="text"
)

print(transcript)
```

### Model Selection

The API supports multiple model variants with different precision levels:

| Model Name | Precision | Speed | Description |
|------------|-----------|-------|-------------|
| `parakeet-tdt-0.6b-v3` | INT8 | Fastest | Default model with 8-bit quantization (recommended) |
| `istupakov/parakeet-tdt-0.6b-v3-onnx` | FP32 | Slower | Full precision for maximum accuracy |
| `grikdotnet/parakeet-tdt-0.6b-fp16` | FP16 | Medium | Half precision, balanced speed and accuracy |

Models are lazy-loaded on first use and cached for subsequent requests. The default INT8 model is pre-loaded at startup.

**To select a model via API:**
```python
transcript = client.audio.transcriptions.create(
  model="grikdotnet/parakeet-tdt-0.6b-fp16",  # Select FP16 model
  file=audio_file,
  response_format="text"
)
```

### Web Interface

The server includes a built-in web interface for testing and easy drag-and-drop transcription.
Access it at: **[http://127.0.0.1:5092](http://127.0.0.1:5092)**

The web interface includes a dropdown menu to select between INT8, FP16, and FP32 model variants.

## 🔌 Open WebUI Integration

**This project provides out-of-the-box compatibility with [Open WebUI](https://openwebui.com/)**, serving as a drop-in replacement for OpenAI's speech-to-text API. Experience lightning-fast, local transcription across 25 languages with automatic language detection!

### Setup Instructions

1.  **Start the Parakeet Server** (if not already running):
    ```bash
    conda activate parakeet-onnx
    python app.py
    ```
    The server will be available at `http://127.0.0.1:5092`

2.  **Configure Open WebUI**:
    - Navigate to **Open WebUI Settings -> Audio**
    - Set **STT Engine** to `OpenAI`
    - Set **OpenAI Base URL** to `http://127.0.0.1:5092/v1`
    - Set **OpenAI API Key** to `sk-no-key-required`
    - Set **STT Model** to `parakeet-tdt-0.6b-v3`
    - Click **Save**

3.  **Start Using Voice!**
    - All voice interactions in Open WebUI will now be transcribed locally
    - Enjoy real-time transcription speeds (up to 30x faster than real-time on modern CPUs)
    - Automatic language detection across all 25 supported languages
    - Complete privacy - all processing happens locally on your machine

## Model details

When running the application, the ONNX models are automatically loaded from the `models/` directory. The primary model used is the **Parakeet TDT 0.6B v3** converted to ONNX with INT8 quantization, providing the optimal balance of speed and accuracy for multilingual speech recognition across 25 European languages.

## 🙏 Acknowledgments

This project stands on the shoulders of giants and wouldn't be possible without:

- **[Shadowfita](https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi)** - For the original FastAPI implementation that served as the foundation for this project
- **[NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** - For developing and open-sourcing the exceptional Parakeet TDT model family
- **[groxaxo](https://github.com/groxaxo)** - The mastermind behind this project, bringing together ONNX optimization, multilingual support, and seamless OpenAI API compatibility

Thank you to all contributors and the open-source community for making high-performance, local speech recognition accessible to everyone!
