# Parakeet TensorRT trial — 2026-10-05

Tested on an RTX 3060 12 GB, driver 580.178.04.

## Result

TensorRT FP16 encoder inference works on this machine. On this small test it was **1.24× faster** than the matching CUDA FP32 runner, with **identical transcripts on all 50 clips**. These measurements were collected before deployment, using an isolated feasibility and accuracy-regression test.

| Measured path | Word error rate | Sum of per-clip median inference time | Audio / inference time |
|---|---:|---:|---:|
| Live HTTP endpoint, CUDA FP32 | 3.528% | 2.410 s | 126.35× |
| Standalone CUDA FP32, profiling off | 3.528% | 2.364 s | 128.80× |
| Standalone TensorRT FP16 encoder + CUDA FP32 decoder, profiling off | 3.528% | 1.906 s | 159.76× |

TensorRT reduced measured inference time by 19.4%, equivalent to 24.0% higher throughput. These are warm repeated-clip timings, excluding model/engine loading. The HTTP path additionally includes WAV encoding/upload and the production service pipeline; the standalone pair is the controlled runtime comparison.

Sampled incremental peak GPU use in the standalone speed runs was 3,137 MiB for CUDA versus 1,973 MiB for TensorRT, relative to the same 7,568 MiB occupied baseline. Sampling was approximately every 0.5 seconds, so these are observed peaks, not exact allocator maxima. Initial TensorRT compilation peaked at 2,005 MiB above baseline. Other GPU services remained running.

## Dataset and accuracy scope

- Dataset: `hf-internal-testing/librispeech_asr_dummy`, clean validation, first 50 clips at most 15 seconds long.
- Total: 304.515 seconds of audio, 737 reference words, **26 word edits**.
- All clips are from **one speaker**, speaker 1272. This is a smoke/regression sample, not a representative multilingual or conversational accuracy benchmark.
- WER normalization: lowercase and remove punctuation except apostrophes. No number expansion, spelling harmonization, or contraction expansion. Proper names, `sceptre`/`scepter`, and `I'm`/`I am` can count as differences.
- Each clip ran three times after two initial warmups. The speed runs asserted identical output across all three repetitions. Each reported clip time is its median; total inference time sums those medians.
- CUDA, TensorRT, and the live endpoint produced exactly the same strings, including punctuation and capitalization, on all 50 clips.
- No claims are established here about Spanish, noisy audio, other accents, long recordings, concurrency, or production tail latency.

## TensorRT configuration and proof

- Same cached ONNX weights as production: `istupakov/parakeet-tdt-0.6b-v3-onnx`, snapshot `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`.
- Existing ONNX Runtime 1.23.2 and onnx-asr 0.12.0, plus isolated TensorRT 10.9.0.34 CUDA 12 libraries.
- TensorRT FP16 enabled for the encoder; decoder remains CUDA FP32, preprocessing CPU. FP16 is an optimization setting and does not imply every internal operation uses FP16.
- Batch size 1; encoder profile minimum/optimal/maximum lengths 1/8/16 seconds. This engine is not validated for arbitrary production input lengths.
- Workspace 256 MiB, builder optimization level 2, engine and timing caches enabled.
- Initial load/compilation plus profiled 50-clip run completed in about 110 seconds. Cached 50-clip speed run completed in about 22 seconds including startup.
- Separate execution-profile proof run: encoder profile recorded **5 TensorRT node executions**, corresponding to two warmups and three repetitions of one clip. Decoder profile recorded CUDA execution. This confirms actual TensorRT execution, not just an available provider label or CUDA fallback.
- Engine cache is about 1.2 GiB and lives in `trt/` in the remote test directory.

## Deployment

The GPU default now uses this encoder-only TensorRT approach with a lower minimum profile size for short clips, bounded 15-second chunks for long recordings, actual-provider health reporting, and CUDA fallback. Production setup instructions are in [README.md](../README.md). The original table above is the isolated trial, not a claim that all deployed workloads achieve the same speed.
