# Docker Deployment Guide

This document covers Docker deployment options for Parakeet TDT transcription service.

## Quick Start

### CPU Deployment (Recommended for most users)

```bash
# Build and run
docker compose up parakeet-cpu -d

# Or build manually
docker build -f Dockerfile.cpu -t parakeet-tdt:cpu .
docker run -d --name parakeet -p 5092:5092 -v parakeet-models:/app/models parakeet-tdt:cpu
```

### GPU Deployment (Requires NVIDIA GPU)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Build and run with Docker Compose
docker compose up parakeet-gpu -d

# Or build manually
docker build -f Dockerfile.gpu -t parakeet-tdt:gpu .
docker run -d --name parakeet-gpu -p 5092:5092 --gpus all \
    -v parakeet-models:/app/models parakeet-tdt:gpu
```

### Optional OpenVINO Deployment (Compatible Intel Hardware)

> [!WARNING]
> The OpenVINO container path is still hardware-dependent. Installing `onnxruntime-openvino` is not always sufficient by itself; some systems also need a full OpenVINO runtime installation and initialized environment variables before `OpenVINOExecutionProvider` is available. Validate this flow on the target Intel host before relying on it in production.

```bash
# Build and run with Docker Compose
docker compose --profile openvino up parakeet-openvino -d

# Or build manually
docker build -f Dockerfile.openvino -t parakeet-tdt:openvino .
docker run -d --name parakeet-openvino -p 5092:5092 \
    -e ASR_BACKEND=openvino \
    -e OV_DEVICE=GPU \
    -e OV_HINT=LATENCY \
    -e OV_EXECUTION_MODE=ACCURACY \
    -e MAX_CONCURRENT_INFERENCES=1 \
    -v parakeet-models:/app/models parakeet-tdt:openvino
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `http://localhost:5092` | Web UI |
| `http://localhost:5092/health` | Health check |
| `http://localhost:5092/v1/audio/transcriptions` | OpenAI-compatible API |
| `http://localhost:5092/docs` | Swagger documentation |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `/app/models` | HuggingFace model cache |
| `HF_HUB_CACHE` | `/app/models` | HuggingFace hub cache |
| `PORT` | `5092` | HTTP listen port |
| `WEB_THREADS` | `8` | Waitress worker thread count |
| `ASR_BACKEND` | `auto` | Backend selection (`auto`, `cpu`, `cuda`, `tensorrt`, `openvino`) |
| `MAX_CONCURRENT_INFERENCES` | backend-aware | Optional concurrent request cap |
| `OV_DEVICE` | `GPU` | OpenVINO target device when `ASR_BACKEND=openvino` |
| `OV_HINT` | `LATENCY` | OpenVINO performance hint |
| `OV_EXECUTION_MODE` | `ACCURACY` | OpenVINO execution mode hint |
| `OV_CACHE_DIR` | empty | Optional OpenVINO cache path |

### Persistent Model Cache

Models are cached in a Docker volume to avoid re-downloading:

```bash
# List volumes
docker volume ls | grep parakeet

# Inspect volume
docker volume inspect parakeet-models

# Remove volume (forces model re-download)
docker volume rm parakeet-models
```

## Files Created

| File | Description |
|------|-------------|
| `Dockerfile.cpu` | CPU-only image (Python 3.10 slim) |
| `Dockerfile.gpu` | NVIDIA CUDA 12.1 image with GPU support |
| `Dockerfile.openvino` | Optional OpenVINO image for compatible Intel hardware |
| `docker-compose.yml` | Orchestration for CPU, GPU, and optional OpenVINO variants |
| `.dockerignore` | Excludes unnecessary files from build |

## Testing

```bash
# Check health
curl http://localhost:5092/health

# Transcribe audio (OpenAI-compatible)
curl -X POST http://localhost:5092/v1/audio/transcriptions \
    -F "file=@audio.mp3" \
    -F "model=parakeet-tdt-0.6b-v3"
```

## Troubleshooting

**Container won't start:**
- Check logs: `docker logs parakeet-cpu`
- First startup takes ~60s to download the model

**GPU not detected:**
- Verify NVIDIA Container Toolkit: `nvidia-smi` should work inside container
- Run: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

**Out of memory:**
- CPU image requires ~2GB RAM
- GPU image requires ~4GB VRAM
