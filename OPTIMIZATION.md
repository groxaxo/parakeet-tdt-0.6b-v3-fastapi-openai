# Optimization Report: `parakeet-tdt-0.6b-v3-fastapi-openai`

## TL;DR

Replaced the legacy Flask + Waitress + `ffmpeg-silencedetect` design with a
new FastAPI service inspired by `parakeet-flash`. On a single i7-12700KF
(8P + 4E, 20 threads), CPU-only ONNX INT8:

| Workload                | Baseline (Flask/Waitress) | Optimized (FastAPI/InferencePool) | Δ          |
|-------------------------|---------------------------|-----------------------------------|------------|
| 10 s file (single)      | 0.690 s  /  14.6× RTFx    | **0.661 s  /  15.2× RTFx**        | +4%        |
| 60 s file (single)      | 3.11 s   /  18.2× RTFx    | **3.02 s   /  18.7× RTFx**        | +3%        |
| **300 s file (single)** | 17.96 s  /  15.7× RTFx    | **10.41 s  /  27.2× RTFx**        | **+73%**   |
| 16× 10 s concurrent     | wall 4.64 s / **34.6×** thrpt | wall 4.10 s / **39.3×** thrpt | +13%       |

> RTFx = audio_seconds / wall_seconds (higher is better).

The biggest CPU win is on long files: parallel Silero‑VAD chunking + a fan-out
inference pool turn a 5-minute clip from 18 s of inference into 10 s.

With `onnxruntime-gpu==1.26.0` installed in the same conda env and CUDA
provider binding validated from the live ORT sessions, the default backend is
now the best stable RTX 3090 profile: FP32 + GPU micro-batching.

| Workload                | CPU optimized      | GPU profile        | Δ        |
|-------------------------|--------------------|--------------------|----------|
| 10 s file (single)      | 0.661 s / 15.2×    | **0.058 s / 174.4×**| +11.5×  |
| 60 s file (single)      | 3.02 s / 18.7×     | **0.229 s / 246.9×**| +13.2×  |
| **300 s file (single)** | 10.41 s / 27.2×    | **1.37 s / 205.9×** | **+7.6×** |
| 16× 10 s concurrent     | 39.3× throughput   | **200.3× throughput** | **+5.1×** |

Default GPU command:

```bash
PARAKEET_USE_GPU=true \
PARAKEET_DEFAULT_MODEL=istupakov/parakeet-tdt-0.6b-v3-onnx \
PARAKEET_BATCHED=1 \
PARAKEET_MAX_BATCH_SIZE=4 \
PARAKEET_BATCH_WINDOW_MS=4 \
PARAKEET_ORT_INTRA_THREADS=1 \
PARAKEET_AUDIO_WORKERS=8 \
python server.py
```

CPU override:

```bash
PARAKEET_USE_GPU=false \
PARAKEET_DEFAULT_MODEL=parakeet-tdt-0.6b-v3 \
PARAKEET_BATCHED=0 \
PARAKEET_ORT_INTRA_THREADS=12 \
python server.py
```

## Method

We followed the user requirement to "establish a measured baseline before
changing code" and "identify bottlenecks with evidence":

1. **Baseline** (`bench_corpus/baseline.json`): legacy `app.py` running on
   Waitress, default settings, warmed up.
2. **Profile**: looked at where time was going during a 300 s clip.
   The legacy server spawns `ffmpeg -loglevel … -filter silencedetect` to
   find chunks, then runs **sequential** per-chunk inference. Inference
   threads in Waitress are 8 but they share the ORT intra-op thread pool,
   so concurrent requests battle for CPU.
3. **Hypotheses tested**:
   - Drop the per-chunk ffmpeg subprocess by decoding once in-process.
   - Replace `silencedetect` with Silero-VAD-based packing into ~60 s
     chunks on pause midpoints.
   - Cross-request micro-batching (`recognize([w1..wN])`).
   - Parallel single-item inference pool (`InferencePool`).
   - Pin to P-cores only via `taskset`.
  - GPU provider setup and model/worker/batch sweeps on RTX 3090.

## Findings

### What worked

- **In-process audio decode** (`parakeet_service/audio.py`): single
  `ffmpeg -i pipe:0 -ac 1 -ar 16000 -f s16le pipe:1` for non-WAV inputs,
  and stdlib `wave`+`audioop` for WAVs. Removes per-chunk subprocess
  fork/exec.
- **Silero-VAD auto-chunking** (`parakeet_service/chunker.py`): pack speech
  segments into 60 s targets, cutting on pause midpoints, with min/max
  guards. Falls back to energy-RMS when silero-vad is unavailable.
  - Bypasses chunking entirely for clips ≤ `CHUNK_MAX_SEC` (75 s by
    default), so 10 s and 60 s files are processed in a single ORT call.
- **Parallel `InferencePool`** (`parakeet_service/batchworker.py`):
  4 worker threads, each calling `model.recognize(single_wav)`. Used for
  both concurrent requests *and* fan-out of multiple chunks from one long
  request, via `asyncio.gather`. This is what produced the 73% jump on
  300 s files.
- **FastAPI + uvicorn**: removes Flask's per-thread blocking model and
  makes the audio pipeline async-friendly without changing the
  OpenAI-compatible response shape.
- **CUDA preload + provider validation** (`parakeet_service/model.py`):
  `onnxruntime-gpu` can list `CUDAExecutionProvider` even when the provider
  later fails to load cuDNN. The loader now calls `ort.preload_dlls()` and,
  when `PARAKEET_USE_GPU=true`, raises if the live encoder/decoder sessions
  do not actually bind to CUDA/TensorRT first.
- **GPU micro-batching**: on RTX 3090 the fastest stable profile was FP32
  model `istupakov/parakeet-tdt-0.6b-v3-onnx` with `PARAKEET_BATCHED=1`,
  `PARAKEET_MAX_BATCH_SIZE=4`, and `PARAKEET_BATCH_WINDOW_MS=4`.

### What did NOT work (and why)

- **Cross-request micro-batching on CPU INT8** (`BatchWorker`):
  initial implementation collected jobs in an 8 ms window then called
  `model.recognize([w1..w8])`. Result: concurrent throughput **dropped
  from 34.6× to 20.6×**. Reason: on CPU INT8, batched `recognize` scales
  near-linearly in wall time per item (you pay padding to the longest
  clip times the batch size). The optimization is GPU-shaped, not
  CPU-shaped. We **kept** `BatchWorker` behind the `PARAKEET_BATCHED=1`
  env flag for users on `onnxruntime-gpu`, where this design is expected
  to win, but the default is `InferencePool`.
- **P-core pinning** (`taskset -c 0-15` via `pin_pcores.sh`): mildly
  *hurt* on this workload — concurrent throughput dropped from
  39.3× to 33.4×. Restricting affinity also blocks ORT's lightweight
  ops and audio I/O from spilling onto the 4 E-cores. Kept the script
  available for users who want predictability but it is not the default.
- **INT8 on CUDA**: the default CPU INT8 model is a poor CUDA target. It
  bound to CUDA after preload, but measured only about 8.6× RTFx on the
  300 s file and 8.7× concurrent throughput. Use it on CPU, not GPU.
- **FP32 pool with 4 workers under sustained concurrent load**: one-shot
  concurrency looked very fast, but the repeated finalist run hit CUDA OOM
  and produced zero successful concurrent requests. Avoid that profile.

## Architecture

```
parakeet-tdt-0.6b-v3-fastapi-openai/
├── app.py                       # Legacy Flask service (kept for reference)
├── server.py                    # New uvicorn entry point (port 5092)
├── pin_pcores.sh                # Optional P-core taskset wrapper
└── parakeet_service/
    ├── config.py                # Env knobs, CPU detection
    ├── audio.py                 # In-process decode (wave / single ffmpeg)
    ├── chunker.py               # Silero-VAD auto-chunking
    ├── model.py                 # ORT session options, providers, cache
    ├── batchworker.py           # InferencePool (default) + BatchWorker (GPU)
    ├── routes.py                # OpenAI-compatible endpoints
    └── main.py                  # FastAPI lifespan
```

Key endpoints (unchanged contract):

- `POST /v1/audio/transcriptions` — multipart `file=`, `model=`,
  `response_format=json|text|srt|vtt|verbose_json`,
  `timestamp_granularities[]=segment|word`.
- `POST /v1/audio/transcriptions/batch` — multiple files in one call.
- `GET /health`, `GET /healthz`.

## Env knobs

All optional. Defaults are tuned for an 8-core CPU.

| Variable                   | Default      | Meaning                                                  |
|----------------------------|--------------|----------------------------------------------------------|
| `PARAKEET_HOST`            | `0.0.0.0`    | bind host                                                |
| `PARAKEET_PORT`            | `5092`       | bind port (matches the legacy service)                   |
| `PARAKEET_DEFAULT_MODEL`   | `istupakov/parakeet-tdt-0.6b-v3-onnx` | default OpenAI model when form field is omitted |
| `PARAKEET_INFER_WORKERS`   | `4`          | parallel ORT workers in `InferencePool` when `PARAKEET_BATCHED=0` |
| `PARAKEET_BATCHED`         | `1`          | `1` → use GPU-friendly `BatchWorker`; set `0` for CPU INT8 |
| `PARAKEET_USE_GPU`         | `true`       | `true` / `auto` / `false`                                |
| `PARAKEET_GPU_DEVICE_ID`   | `0`          | CUDA device for ORT                                      |
| `PARAKEET_CHUNK_TARGET_SEC`| `60`         | preferred chunk length                                   |
| `PARAKEET_CHUNK_MAX_SEC`   | `75`         | hard cap before force-cut; ≤ this skips chunking         |
| `PARAKEET_CHUNK_MIN_SEC`   | `20`         | min chunk length before merge                            |
| `PARAKEET_VAD_THRESHOLD`   | `0.5`        | Silero-VAD speech probability                            |
| `PARAKEET_VAD_MIN_SILENCE_MS` | `400`     | min silence between chunks                               |
| `PARAKEET_VAD_SPEECH_PAD_MS` | `120`      | pad around speech segments                               |
| `PARAKEET_MAX_BATCH_SIZE`  | `4`          | max batch (only used when `PARAKEET_BATCHED=1`)          |
| `PARAKEET_BATCH_WINDOW_MS` | `4`          | batch collection window                                  |
| `PARAKEET_ORT_INTRA_THREADS` | `1` for GPU, physical cores for CPU override | ORT intra-op threads |
| `PARAKEET_ORT_INTER_THREADS` | `1`        | ORT inter-op threads                                     |
| `PARAKEET_AUDIO_WORKERS`   | `min(8, physical)` | audio decode/chunk worker pool                     |

## Running

```bash
# Conda-isolated install (matches the benchmark env)
conda create -n parakeet-v3 python=3.11 -y
conda activate parakeet-v3
pip install -r requirements.txt

# Optional CUDA path, installed only in the benchmark env used for GPU tests
pip install onnxruntime-gpu==1.26.0

# Run
python server.py
# or pinned to P-cores (NOT recommended on this hardware, see Findings)
./pin_pcores.sh python server.py

# Default RTX 3090 profile
PARAKEET_USE_GPU=true \
PARAKEET_DEFAULT_MODEL=istupakov/parakeet-tdt-0.6b-v3-onnx \
PARAKEET_BATCHED=1 \
PARAKEET_MAX_BATCH_SIZE=4 \
PARAKEET_BATCH_WINDOW_MS=4 \
PARAKEET_ORT_INTRA_THREADS=1 \
python server.py
```

## Reproducing the benchmark

```bash
cd bench_corpus
python bench.py --url http://127.0.0.1:5092/v1/audio/transcriptions \
                --label mylabel --warmup --out mylabel.json \
                --sequential-n 3 --concurrency 8 --concurrent-total 16
```

The bench corpus uses three real audio files (10 s, 60 s, 300 s) and
measures sequential mean/p50/p95 plus concurrent wall-clock throughput.

## GPU sweep highlights

All rows below were validated from live ORT session providers with CUDA first.
The final rows use `--sequential-n 3`; earlier exploratory sweeps used one
sequential run per duration.

| Profile | 10 s | 60 s | 300 s | 16×10 s throughput | Notes |
|---------|------|------|-------|--------------------|-------|
| FP32 batch b4/w4ms (final) | 174.4× | 246.9× | **205.9×** | **200.3×** | best stable overall |
| FP16 batch b4/w2ms (final) | **221.4×** | **509.5×** | 143.0× | 91.0× | best short/medium single request |
| FP16 pool w4 (exploratory) | 217.0× | 337.3× | 134.7× | 109.7× | simple low-risk GPU profile |
| FP32 pool w4 (exploratory) | 197.1× | 228.6× | 100.3× | 244.3× | OOMed under final concurrent repeat |
| INT8 pool w1 on CUDA | 8.8× | 8.9× | 8.6× | 8.7× | avoid on GPU |

## Future work

- **Multi-GPU serving**: the host has 3× RTX 3090 + 1× RTX 3060. The current
  service intentionally binds one ORT model to one CUDA device. Running one
  uvicorn process per GPU behind a local load balancer should scale aggregate
  throughput further.
- **Word-level timestamps**: currently exposed via
  `timestamp_granularities[]=word` and returned by the underlying
  `onnx_asr` model — would benefit from an alignment pass for long
  chunks where boundary words can split.
- **Streaming endpoint**: the `parakeet-flash` reference project ships
  WebSocket streaming with VAD; adding it here is a natural follow-up.
