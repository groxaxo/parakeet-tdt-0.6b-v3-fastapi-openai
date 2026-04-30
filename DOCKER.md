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

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `http://localhost:5092` | Web UI |
| `http://localhost:5092/health` | Health check |
| `http://localhost:5092/v1/audio/transcriptions` | OpenAI-compatible API |
| `ws://localhost:5092/v1/realtime` | Realtime websocket (OpenAI-compatible subset) |
| `http://localhost:5092/realtime-demo` | Browser mic realtime demo |
| `http://localhost:5092/docs` | Swagger documentation |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `/app/models` | HuggingFace model cache |
| `HF_HUB_CACHE` | `/app/models` | HuggingFace hub cache |

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
| `docker-compose.yml` | Orchestration for both variants |
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
