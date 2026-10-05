# GPU decoder-state experiment (2026-10-05)

`PARAKEET_GPU_DECODER_STATE=1` enables request-local CUDA decoder state in the
repository TensorRT adapter. Accepted and candidate state use different
buffers; blank tokens never replace accepted state. Targets use the model's
int32 input type. Explicit OrtValue uploads avoid stale input data when a
binding is reused. Installed `onnx-asr` and ONNX Runtime files are unchanged.

The validated batch-one TensorRT FP16 encoder and its lock are retained.
Decoder computation remains CUDA FP32 on both paths. Model selection,
chunking, timestamps, and CUDA fallback are preserved. `/health` reports
`decoder_state` (`host` or `gpu`) and `inference_workers`; `host` describes
state storage, not the device performing decoder computation.

All 50 existing corpus clips plus nine silence/short/chunk-boundary/120-second
cases matched in text, tokens, timestamps, timestamp indices, words and
segments at concurrency 1/2/4. The final quality harness and regression tests
cover blank-state acceptance and concurrent buffer isolation.

The completed performance trial used 72 thirty-second windows: solo tests
for Parakeet, Supertonic and Nemotron, plus all three together, at concurrency
1/2/4, with two reversed-order repetitions and three variants (host state with
one worker; GPU state with one or two workers). All 57,230 responses were
valid, health checks succeeded, and sampled free GPU memory stayed at or
above 3517 MiB. Steady combined GPU memory increased 22 MiB for one GPU-state
worker and 64 MiB for two.

**Neither optimization is promoted.** At concurrency four, one GPU-state
worker improved solo throughput only 2.22%; mixed throughput regressed
60.95%. Two workers improved solo throughput 75.48% and p95 latency 38.46%,
but mixed throughput regressed 28.57% and p95 latency 44.85%. These exceed
the allowed 5% regression. The selected inference default remains **host
state, one worker**. Only the additive health diagnostics are rolled out.

The final trial was rerun entirely after a host reboot interrupted an earlier
26-window attempt. Partial attempts are excluded from promotion. A uniform
10 GiB process-group RAM limit applied to all final variants; no OOM occurred,
and its reclaim-event count stayed unchanged from phase 15 through phase 51.
The full JSON evidence, CPU/health/memory traces, exact source revisions,
corpus hashes and reproduction scripts are published in Experimentos under
`speech/inference-admission-2026-10-05`.

For experiments, use `PARAKEET_GPU_DECODER_STATE=1` and choose
`PARAKEET_INFER_WORKERS=1` or `2` before starting the service. Roll back with
`PARAKEET_GPU_DECODER_STATE=0` and `PARAKEET_INFER_WORKERS=1`. Drain requests
before restarting. True TensorRT encoder batching remains a separate experiment.
