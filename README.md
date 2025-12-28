# Parakeet-TDT 0.6B v3 FastAPI STT Service

A production-ready FastAPI service for high-accuracy speech-to-text using NVIDIA's Parakeet-TDT models. Implements both REST and WebSocket endpoints following the [OpenAI Audio API specification](https://platform.openai.com/docs/api-reference/audio) interface.

**Based on the excellent work by [jianchang512/parakeet-api](https://github.com/jianchang512/parakeet-api) - Thank you!**

## Features

- **Multi-language support**
  - English, Japanese, and 24+ European languages
  - Automatic model selection based on language
  - Uses nvidia/parakeet-tdt-0.6b-v3 (default) and nvidia/parakeet-tdt_ctc-0.6b-ja (Japanese)

- **Web UI**
  - Simple and intuitive web interface
  - Drag-and-drop file upload
  - Language selection dropdown
  - Direct SRT subtitle download

- **RESTful transcription**  
  - `POST /v1/audio/transcriptions` with multipart audio uploads
  - SRT and JSON response formats
  - Word/character/segment timestamps
  - OpenAI-compatible response schema

- **WebSocket streaming**  
  - Real-time voice activity detection via Silero VAD
  - Partial/final transcription delivery
  - Supports 16kHz mono PCM input

- **Batch processing**  
  - Micro-batching for efficient GPU utilization
  - Configurable batch size and processing timeout

- **Production-ready deployment**  
  - Docker and Docker Compose support
  - Health checks and configuration endpoints
  - Environment variable configuration

- **Audio preprocessing**  
  - Automatic downmixing and resampling
  - File validation and chunking

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Running the Server](#running-the-server)  
- [Usage](#usage)  
  - [REST API](#rest-api)  
  - [WebSocket Streaming](#websocket-streaming)  
- [Architecture Overview](#architecture-overview)  
- [Environment Variables](#environment-variables)  
- [Contributing](#contributing)  
- [License](#license)  

## Prerequisites

- Python 3.10+  
- NVIDIA GPU with CUDA 12.1+ (recommended)
- Docker Engine 24.0+ (for container deployment)

## Installation

### Local Development
```bash
git clone https://github.com/your-repo/parakeet-fastapi.git
cd parakeet-fastapi

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Deployment
```bash
docker build -t parakeet-stt .
docker run -d -p 8000:8000 --gpus all parakeet-stt
```

### Docker Compose
```bash
docker-compose up --build
```

## Configuration

All configuration is managed through environment variables. Create a `.env` file with your preferences:

```ini
# Model configuration
MODEL_PRECISION=fp16
DEVICE=cuda
BATCH_SIZE=4

# Audio processing
TARGET_SR=16000
MAX_AUDIO_DURATION=30
VAD_THRESHOLD=0.5

# System
LOG_LEVEL=INFO
PROCESSING_TIMEOUT=60
```

## Running the Server

### Local Development
```bash
uvicorn parakeet_service.main:app --host 0.0.0.0 --port 8000
```

### Production
```bash
docker-compose up --build -d
```

## Usage

### Web Interface

1. Start the server (see [Running the Server](#running-the-server))
2. Open your browser to [http://localhost:8000](http://localhost:8000)
3. Select the language from the dropdown
4. Upload your audio/video file
5. Click "Start Transcription"
6. Download the generated SRT file

### REST API

#### Health Check
```bash
curl http://localhost:8000/healthz
# {"status":"ok"}
```

#### Transcription
```bash
# Get SRT format (default for web UI compatibility)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file="@audio.wav" \
  -F prompt="en" \
  -F response_format="srt"

# Get JSON format with timestamps
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file="@audio.wav" \
  -F prompt="en" \
  -F include_timestamps=true

# Get SRT with word-level timestamps appended
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file="@audio.wav" \
  -F prompt="en" \
  -F model="parakeet_srt_words"
```

**Parameters**:
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `file` | `audio/*` | Required | Audio file (wav, mp3, flac) |
| `prompt` | string | "en" | Language code (en, ja, etc.) |
| `response_format` | string | "json" | Response format: "json" or "srt" |
| `model` | string | "parakeet" | "parakeet" or "parakeet_srt_words" for word timestamps |
| `include_timestamps` | bool | false | Return word/segment timestamps (JSON only) |
| `should_chunk` | bool | true | Enable audio chunking for long files |

**Response**:
```json
{
  "text": "Transcribed text content",
  "timestamps": {
    "words": [
      {"text": "Hello", "start": 0.2, "end": 0.5},
      {"text": "world", "start": 0.6, "end": 0.9}
    ],
    "segments": [
      {"text": "Hello world", "start": 0.2, "end": 0.9}
    ]
  }
}
```

### WebSocket Streaming

Connect to `ws://localhost:8000/ws` to stream audio:

- **Input**: 16kHz mono PCM frames (int16)
- **Output**: JSON messages with partial/final transcriptions

**JavaScript Example**:
```javascript
const ws = new WebSocket("ws://localhost:8000/ws");
const audioContext = new AudioContext();
const processor = audioContext.createScriptProcessor(1024, 1, 1);

processor.onaudioprocess = e => {
  const pcmData = e.inputBuffer.getChannelData(0);
  const int16Data = convertFloat32ToInt16(pcmData);
  ws.send(int16Data);
};

ws.onmessage = evt => {
  const data = JSON.parse(evt.data);
  console.log("Transcription:", data.text);
};
```

## Architecture Overview

```mermaid
graph LR
A[Client] -->|HTTP| B[REST API]
A -->|WebSocket| C[Streaming API]
B --> D[Batch Worker]
C --> E[VAD Processing]
E --> F[Chunker]
F --> D
D --> G[ASR Model]
G --> H[Response Formatter]
H --> A
```

**Components**:
1. **`main.py`** - App initialization and lifecycle management
2. **`routes.py`** - REST endpoints implementation
3. **`stream_routes.py`** - WebSocket endpoint handler
4. **`streaming_vad.py`** - Voice activity detection
5. **`chunker.py`** - Audio segmentation
6. **`batchworker.py`** - Micro-batch processing
7. **`model.py`** - ASR model interface
8. **`audio.py`** - Audio preprocessing utilities
9. **`config.py`** - Configuration management

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PRECISION` | fp16 | Model precision (fp16/fp32) |
| `DEVICE` | cuda | Computation device |
| `BATCH_SIZE` | 4 | Processing batch size |
| `TARGET_SR` | 16000 | Target sample rate |
| `MAX_AUDIO_DURATION` | 30 | Max audio length in seconds |
| `VAD_THRESHOLD` | 0.5 | Voice activity threshold |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `PROCESSING_TIMEOUT` | 60 | Processing timeout in seconds |

## Contributing

1. Fork the repository and create your feature branch
2. Submit a pull request with detailed description
