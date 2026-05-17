"""FastAPI routes: OpenAI-compatible transcription endpoint + batch helper."""
from __future__ import annotations
import asyncio
import datetime
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse

from .audio import load_audio
from .chunker import auto_chunk, slice_chunks
from .config import (
    CPU_INFO,
    DEFAULT_MODEL,
    MODEL_CONFIGS,
    TARGET_SR,
    logger,
)
from .model import loaded_models

router = APIRouter()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u2581", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.replace(" '", "'")


def _fmt_srt_time(seconds: float) -> str:
    d = datetime.timedelta(seconds=max(0.0, seconds))
    s = str(d)
    if "." in s:
        a, b = s.split(".")
        ms = b[:3].ljust(3, "0")
    else:
        a, ms = s, "000"
    if a.count(":") == 1:
        a = "0:" + a
    return f"{a},{ms}"


def _segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        text = seg["segment"].strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    out = ["WEBVTT", ""]
    for seg in segments:
        text = seg["segment"].strip()
        if not text:
            continue
        s = _fmt_srt_time(seg["start"]).replace(",", ".")
        e = _fmt_srt_time(seg["end"]).replace(",", ".")
        out.extend([f"{s} --> {e}", text, ""])
    return "\n".join(out)


def _extract(result) -> Dict[str, Any]:
    """Convert onnx_asr result (with timestamps) into a plain dict."""
    text = _clean_text(getattr(result, "text", str(result)))
    tokens = list(getattr(result, "tokens", []) or [])
    ts = list(getattr(result, "timestamps", []) or [])
    return {"text": text, "tokens": tokens, "timestamps": ts}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health")
def health():
    return {
        "status": "healthy",
        "models": list(MODEL_CONFIGS.keys()),
        "loaded": loaded_models(),
        "default_model": DEFAULT_MODEL,
        "cpu": CPU_INFO,
    }


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Core OpenAI-compatible transcribe
# ---------------------------------------------------------------------------
@router.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    response_format: str = Form("json"),
    timestamp_granularities: Optional[str] = Form(None),
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    model_name = (model or DEFAULT_MODEL).lower()
    if model_name not in MODEL_CONFIGS:
        logger.warning("Unknown model %r, using default", model_name)
        model_name = DEFAULT_MODEL

    raw = await file.read()
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    t0 = time.perf_counter()
    try:
        wav = load_audio(raw)
    except Exception as exc:
        logger.exception("audio decode failed")
        raise HTTPException(status_code=415, detail=f"audio decode failed: {exc}") from exc
    total_duration = wav.size / TARGET_SR
    if total_duration <= 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    ranges = auto_chunk(wav)
    pieces = slice_chunks(wav, ranges)
    decode_ms = (time.perf_counter() - t0) * 1000

    worker = request.app.state.worker
    t1 = time.perf_counter()
    results = await worker.submit_many(pieces, model_name)
    infer_ms = (time.perf_counter() - t1) * 1000

    # Stitch results back together with absolute timestamps
    all_segments: List[Dict[str, Any]] = []
    all_words: List[Dict[str, Any]] = []
    for (start_s, _end_s), res in zip(ranges, results):
        offset = start_s / TARGET_SR
        info = _extract(res)
        if not info["text"]:
            continue
        starts = info["timestamps"]
        if starts:
            seg_start = starts[0] + offset
            seg_end = (starts[-1] if len(starts) > 1 else starts[0] + 0.1) + offset
        else:
            seg_start = offset
            seg_end = offset + 0.1
        all_segments.append({
            "start": seg_start,
            "end": seg_end,
            "segment": info["text"],
        })
        for i, (tok, ts) in enumerate(zip(info["tokens"], starts)):
            word_end = (starts[i + 1] if i + 1 < len(starts) else seg_end - offset) + offset
            all_words.append({
                "start": ts + offset,
                "end": word_end,
                "word": tok.replace("\u2581", " ").strip(),
            })

    full_text = " ".join(s["segment"] for s in all_segments)
    logger.info("transcribe model=%s dur=%.2fs chunks=%d decode=%.0fms infer=%.0fms total=%.0fms",
                model_name, total_duration, len(pieces), decode_ms, infer_ms,
                (time.perf_counter() - t0) * 1000)

    fmt = (response_format or "json").lower()
    if fmt == "text":
        return PlainTextResponse(full_text)
    if fmt == "srt":
        return PlainTextResponse(_segments_to_srt(all_segments))
    if fmt == "vtt":
        return PlainTextResponse(_segments_to_vtt(all_segments))
    if fmt == "verbose_json":
        return JSONResponse({
            "task": "transcribe",
            "language": "auto",
            "duration": total_duration,
            "text": full_text,
            "segments": [
                {
                    "id": i,
                    "seek": 0,
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["segment"],
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                }
                for i, s in enumerate(all_segments)
            ],
            "words": all_words if (timestamp_granularities and "word" in timestamp_granularities) else None,
        })
    return JSONResponse({"text": full_text})


# ---------------------------------------------------------------------------
# Batch endpoint (non-OpenAI) for high-throughput pipelines
# ---------------------------------------------------------------------------
@router.post("/v1/audio/transcriptions/batch")
async def transcribe_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    model: str = Form(DEFAULT_MODEL),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    model_name = (model or DEFAULT_MODEL).lower()
    if model_name not in MODEL_CONFIGS:
        model_name = DEFAULT_MODEL

    # Decode in parallel using the audio thread pool implicitly through asyncio
    raws = []
    for f in files:
        raws.append(await f.read())
        await f.close()

    loop = asyncio.get_running_loop()
    pool = request.app.state.audio_pool
    wavs = await asyncio.gather(*(loop.run_in_executor(pool, load_audio, r) for r in raws))

    worker = request.app.state.worker
    results = await worker.submit_many(list(wavs), model_name)
    texts = [_clean_text(getattr(r, "text", str(r))) for r in results]
    return {
        "results": [
            {"filename": f.filename, "text": t, "duration": w.size / TARGET_SR}
            for f, t, w in zip(files, texts, wavs)
        ],
        "batch_size": len(files),
    }
