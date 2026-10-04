# Parakeet TDT Transcription with TensorRT and ONNX Runtime

OpenAI-compatible speech recognition using NVIDIA [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), with 25 supported languages, punctuation, and timestamps.

## Default GPU backend: TensorRT FP16 encoder

`server.py` defaults to **TensorRT mixed-FP16 encoder + CUDA FP32 decoder** using the existing `istupakov/parakeet-tdt-0.6b-v3-onnx` model ID. The model weights and API model name have not changed. CPU INT8 and explicit CUDA FP32 execution remain available. `app.py` is the legacy CPU-oriented server; use `server.py` for TensorRT.

On an RTX 3060 12 GB, a paired 50-clip clean-English trial measured:

| Runtime | Warm speed | Word error rate | Additional sampled peak GPU memory |
| --- | ---: | ---: | ---: |
| CUDA FP32 | 128.80× real time | 3.53% | 3.06 GiB |
| TensorRT FP16 encoder + CUDA FP32 decoder | 159.76× real time | 3.53% | 1.93 GiB |

All 50 transcripts were identical. This is a five-minute, single-speaker smoke test, not a multilingual accuracy guarantee. See [the measured trial](docs/tensorrt-trial.md) for methods and limitations. Historical CPU/3090 results below concern different hardware and configurations.

```bash
python -m pip install -r requirements.txt
python server.py                  # :5092, TensorRT preferred

# Explicit CUDA rollback (also restores larger default chunk sizes)
PARAKEET_GPU_BACKEND=cuda python server.py

# CPU override (see Dockerfile.cpu for installation without GPU dependencies)
PARAKEET_USE_GPU=false PARAKEET_DEFAULT_MODEL=parakeet-tdt-0.6b-v3 python server.py
```

The first launch builds a GPU/version-specific engine and can take several minutes. Engines persist under `models/tensorrt/`; keep that volume across container restarts. Missing TensorRT libraries or a failed engine build trigger a logged CUDA fallback. `/health` reports the **actual** backend, session providers, encoder precision, and any fallback reason under `runtime`.

TensorRT defaults to one inference worker and a batch-1 encoder. The batch HTTP API still accepts multiple files; encoder work is serialized. Long recordings are split near pauses into chunks no longer than 15 seconds; very short inputs are padded to the engine's minimum feature shape. Chunk overrides above 15 seconds are rejected in TensorRT mode to prevent unbounded engine rebuilding. The decoder and timestamp handling reuse onnx-asr 0.12.0, which is pinned for this integration.

| Setting | Default | Purpose |
| --- | --- | --- |
| `PARAKEET_GPU_BACKEND` | `tensorrt` | `tensorrt` or `cuda` |
| `PARAKEET_TRT_CACHE_DIR` | `models/tensorrt` | Persistent engine/timing cache |
| `PARAKEET_TRT_WORKSPACE_MB` | `256` | TensorRT builder workspace; not a total VRAM cap |
| `PARAKEET_CHUNK_MIN_SEC` / `TARGET_SEC` / `MAX_SEC` | `5` / `12` / `15` | TensorRT chunking; each variable has the `PARAKEET_CHUNK_` prefix |
| `PARAKEET_BATCHED` | `false` | Optional request micro-batching; TensorRT encoder still runs batch 1 |
| `PARAKEET_INFER_WORKERS` | `1` | TensorRT worker default |
| `PARAKEET_GPU_MEMORY_LIMIT_MB` | `0` | CUDA arena limit only, not TensorRT or total process memory |

CUDA uses heuristic cuDNN selection, bounded cuDNN workspace, and `same_as_requested` arena growth. `PARAKEET_MAX_BATCH_AUDIO_SECONDS=90` limits padded audio per optional CUDA batch. See [OPTIMIZATION.md](OPTIMIZATION.md) for the earlier optimization history and [DOCKER.md](DOCKER.md) for container deployment.

## ⚡ Lower-latency WAV uploads

The server now includes a faster request path for short PCM WAV uploads with no API changes required:

- WAV duration is read directly from the WAV header instead of spawning `ffprobe`
- Already-normalized **16 kHz mono PCM WAV** uploads skip FFmpeg conversion entirely
- Short unchunked PCM WAV uploads can be decoded and resampled **in process** before being passed straight to ONNX Runtime
- FFmpeg remains the fallback for unsupported, compressed, non-WAV, or chunked inputs

On a 20-file English/Spanish Chatterbox WAV benchmark corpus, this reduced endpoint RTF from **0.0459** to **0.0379** and improved effective throughput from **21.80x** to **26.40x** real time, while keeping **20/20** correlation passes.

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
- The three variants produced the same score on this historical 50-sample test (97.84%)
- This historical sample showed no measured INT8 accuracy loss versus FP32; this is not a general guarantee
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
ONNX Runtime's CPU execution provider automatically dispatches AVX2/FMA kernels from the standard wheel when the host CPU supports them. The server now detects AVX2 at startup, reports the result in `/health`, and configures ONNX Runtime threading to use the available physical CPU cores while preventing NumPy/BLAS thread pools from competing with inference.

For hybrid CPUs (like Intel 12th-14th Gen), performance is still improved by pinning the process to Performance cores (P-cores). You can also override the auto-tuned defaults:

* `PARAKEET_ORT_INTRA_THREADS`: ONNX Runtime intra-op worker threads. Defaults to the lower of detected physical CPUs and available logical CPUs in the container/affinity mask. Minimum: `1`.
* `PARAKEET_ORT_INTER_THREADS`: ONNX Runtime inter-op threads. Defaults to `1`, which is best for single-model inference. Minimum: `1`.
* `PARAKEET_WAITRESS_THREADS`: HTTP worker threads. Defaults to a conservative value to avoid oversubscribing ONNX Runtime's AVX2 worker pool. Minimum: `1`.

## Installation

### 🐳 Docker (Recommended)

The easiest way to get started. No dependencies to install!

**GPU Deployment (default; requires NVIDIA Container Toolkit):**
```bash
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai
cd parakeet-tdt-0.6b-v3-fastapi-openai
docker compose up parakeet-gpu -d
```

**CPU alternative** (no NVIDIA GPU required):
```bash
docker compose up parakeet-cpu -d
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

## Usage

### Start the Server

Parakeet TDT provides an OpenAI-compatible API server.

```bash
conda activate parakeet-onnx
python server.py
```
*   **Port**: 5092
*   **Docs**: [http://127.0.0.1:5092/docs](http://127.0.0.1:5092/docs)

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
  model="istupakov/parakeet-tdt-0.6b-v3-onnx",  # TensorRT GPU default
  file=audio_file,
  response_format="text"
)

print(transcript)
```

### Model Selection

The API supports multiple model variants with different precision levels:

| Model Name | Precision | Speed | Description |
|------------|-----------|-------|-------------|
| `parakeet-tdt-0.6b-v3` | INT8 | CPU profile | Explicit CPU-oriented model |
| `istupakov/parakeet-tdt-0.6b-v3-onnx` | FP32 weights | GPU default | TensorRT FP16 encoder, CUDA FP32 decoder; CUDA fallback |
| `grikdotnet/parakeet-tdt-0.6b-fp16` | FP16 | Medium | Half precision, balanced speed and accuracy |

Models are lazy-loaded on first use and cached for subsequent requests. The configured default model is pre-loaded at startup (TensorRT GPU profile by default).

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

The legacy `app.py` web interface includes a precision dropdown; it does not use the optimized TensorRT backend. Use `server.py` and its API for the current default.

## 🔌 Open WebUI Integration

**This project provides out-of-the-box compatibility with [Open WebUI](https://openwebui.com/)**, serving as a drop-in replacement for OpenAI's speech-to-text API. Experience lightning-fast, local transcription across 25 languages with automatic language detection!

### Setup Instructions

1.  **Start the Parakeet Server** (if not already running):
    ```bash
    conda activate parakeet-onnx
    python server.py
    ```
    The server will be available at `http://127.0.0.1:5092`

2.  **Configure Open WebUI**:
    - Navigate to **Open WebUI Settings -> Audio**
    - Set **STT Engine** to `OpenAI`
    - Set **OpenAI Base URL** to `http://127.0.0.1:5092/v1`
    - Set **OpenAI API Key** to `sk-no-key-required`
    - Set **STT Model** to `istupakov/parakeet-tdt-0.6b-v3-onnx`
    - Click **Save**

3.  **Start Using Voice!**
    - All voice interactions in Open WebUI will now be transcribed locally
    - Enjoy real-time transcription speeds (up to 30x faster than real-time on modern CPUs)
    - Automatic language detection across all 25 supported languages
    - Complete privacy - all processing happens locally on your machine

## Model details

When running the application, the ONNX models are automatically loaded from the `models/` directory. The GPU default compiles the FP32 **Parakeet TDT 0.6B v3** ONNX encoder with TensorRT FP16 and keeps the decoder in CUDA FP32. INT8 remains available for CPU use.

## 🙏 Acknowledgments

This project stands on the shoulders of giants and wouldn't be possible without:

- **[Shadowfita](https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi)** - For the original FastAPI implementation that served as the foundation for this project
- **[NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** - For developing and open-sourcing the exceptional Parakeet TDT model family
- **[groxaxo](https://github.com/groxaxo)** - The mastermind behind this project, bringing together ONNX optimization, multilingual support, and seamless OpenAI API compatibility

Thank you to all contributors and the open-source community for making high-performance, local speech recognition accessible to everyone!
