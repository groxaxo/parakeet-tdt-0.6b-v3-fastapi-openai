"""FastAPI routes for OpenAI-compatible transcription."""
from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from .audio import load_audio
from .chunker import auto_chunk, slice_chunks
from .config import (
    CPU_INFO,
    DEFAULT_MODEL,
    MAX_AUDIO_SECONDS,
    MAX_BATCH_BYTES,
    MAX_BATCH_FILES,
    MAX_REQUEST_CHUNKS,
    MAX_UPLOAD_BYTES,
    MODEL_CONFIGS,
    TARGET_SR,
    UPLOAD_READ_CHUNK_BYTES,
    logger,
)
from .model import loaded_models

router = APIRouter()
_ALLOWED_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}


@dataclass(slots=True)
class _PreparedAudio:
    waveform: Any
    ranges: List[Tuple[int, int]]
    pieces: List[Any]
    duration: float


class _AudioTooLong(ValueError):
    pass


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u2581", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.replace(" '", "'")


def _fmt_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    index = 1
    for segment in segments:
        text = segment["segment"].strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{_fmt_srt_time(segment['start'])} --> {_fmt_srt_time(segment['end'])}",
                text,
                "",
            ]
        )
        index += 1
    return "\n".join(lines)


def _segments_to_vtt(segments: Sequence[Dict[str, Any]]) -> str:
    output = ["WEBVTT", ""]
    for segment in segments:
        text = segment["segment"].strip()
        if not text:
            continue
        start = _fmt_srt_time(segment["start"]).replace(",", ".")
        end = _fmt_srt_time(segment["end"]).replace(",", ".")
        output.extend([f"{start} --> {end}", text, ""])
    return "\n".join(output)


def _extract(result: Any) -> Dict[str, Any]:
    text = _clean_text(getattr(result, "text", str(result)))
    tokens = [str(token) for token in (getattr(result, "tokens", []) or [])]
    timestamps: List[float] = []
    for value in list(getattr(result, "timestamps", []) or []):
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(timestamp):
            timestamps.append(max(0.0, timestamp))
    return {"text": text, "tokens": tokens, "timestamps": timestamps}


def _validate_model(model: str) -> str:
    normalized = (model or DEFAULT_MODEL).strip().lower()
    if normalized not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model!r}. Available models: {sorted(MODEL_CONFIGS)}",
        )
    return normalized


def _validate_format(response_format: str) -> str:
    normalized = (response_format or "json").strip().lower()
    if normalized not in _ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format {response_format!r}",
        )
    return normalized


async def _read_upload_limited(upload: UploadFile) -> bytes:
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    declared_size = getattr(upload, "size", None)
    if declared_size is not None and declared_size > MAX_UPLOAD_BYTES:
        await upload.close()
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
        )

    payload = bytearray()
    try:
        while True:
            chunk = await upload.read(UPLOAD_READ_CHUNK_BYTES)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
                )
    finally:
        await upload.close()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")
    return bytes(payload)


def _prepare_audio(raw: bytes) -> _PreparedAudio:
    waveform = load_audio(raw)
    duration = float(waveform.size) / TARGET_SR
    if duration <= 0:
        raise ValueError("decoded audio is empty")
    if duration > MAX_AUDIO_SECONDS:
        raise _AudioTooLong(
            f"audio duration {duration:.1f}s exceeds limit {MAX_AUDIO_SECONDS:.1f}s"
        )
    ranges = auto_chunk(waveform)
    pieces = slice_chunks(waveform, ranges)
    if len(pieces) > MAX_REQUEST_CHUNKS:
        raise _AudioTooLong(
            f"audio produced {len(pieces)} chunks; limit is {MAX_REQUEST_CHUNKS}"
        )
    return _PreparedAudio(
        waveform=waveform,
        ranges=ranges,
        pieces=pieces,
        duration=duration,
    )


async def _prepare_in_pool(request: Request, raw: bytes) -> _PreparedAudio:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            request.app.state.audio_pool, _prepare_audio, raw
        )
    except _AudioTooLong as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("audio decode/preprocessing failed")
        raise HTTPException(status_code=415, detail="Audio could not be decoded") from exc


def _stitch(
    prepared: _PreparedAudio, results: Sequence[Any]
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(results) != len(prepared.ranges):
        raise RuntimeError(
            f"inference returned {len(results)} results for "
            f"{len(prepared.ranges)} chunks"
        )

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    for (start_sample, end_sample), result in zip(prepared.ranges, results):
        chunk_start = start_sample / TARGET_SR
        chunk_end = min(prepared.duration, end_sample / TARGET_SR)
        info = _extract(result)
        if not info["text"]:
            continue

        timestamps = info["timestamps"]
        segment_start = chunk_start
        if timestamps:
            segment_start = min(chunk_end, max(chunk_start, chunk_start + timestamps[0]))
        segment_end = max(segment_start, chunk_end)
        segments.append(
            {
                "start": segment_start,
                "end": segment_end,
                "segment": info["text"],
            }
        )

        for index, (token, timestamp) in enumerate(zip(info["tokens"], timestamps)):
            word = token.replace("\u2581", " ").strip()
            if not word:
                continue
            word_start = min(chunk_end, max(chunk_start, chunk_start + timestamp))
            if index + 1 < len(timestamps):
                word_end = chunk_start + timestamps[index + 1]
            else:
                word_end = segment_end
            word_end = min(chunk_end, max(word_start, word_end))
            words.append({"start": word_start, "end": word_end, "word": word})

    full_text = _clean_text(" ".join(item["segment"] for item in segments))
    return full_text, segments, words


async def _infer_prepared(request: Request, prepared: _PreparedAudio, model_name: str):
    worker = request.app.state.worker
    if worker is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready")
    return await worker.submit_many(prepared.pieces, model_name)


@router.get("/health")
def health(request: Request):
    ready = bool(getattr(request.app.state, "ready", False))
    return {
        "status": "healthy" if ready else "starting",
        "ready": ready,
        "models": list(MODEL_CONFIGS.keys()),
        "loaded": loaded_models(),
        "default_model": DEFAULT_MODEL,
        "cpu": CPU_INFO,
    }


@router.get("/healthz")
def healthz(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok"}


@router.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    response_format: str = Form("json"),
    timestamp_granularities: Optional[List[str]] = Form(
        None, alias="timestamp_granularities[]"
    ),
    timestamp_granularities_plain: Optional[List[str]] = Form(
        None, alias="timestamp_granularities"
    ),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
):
    del language, prompt, temperature  # accepted for OpenAI client compatibility
    model_name = _validate_model(model)
    output_format = _validate_format(response_format)
    raw = await _read_upload_limited(file)

    started = time.perf_counter()
    prepared = await _prepare_in_pool(request, raw)
    decode_ms = (time.perf_counter() - started) * 1000

    infer_started = time.perf_counter()
    results = await _infer_prepared(request, prepared, model_name)
    infer_ms = (time.perf_counter() - infer_started) * 1000
    full_text, segments, words = _stitch(prepared, results)

    logger.info(
        "transcribe model=%s dur=%.2fs chunks=%d decode=%.0fms infer=%.0fms total=%.0fms",
        model_name,
        prepared.duration,
        len(prepared.pieces),
        decode_ms,
        infer_ms,
        (time.perf_counter() - started) * 1000,
    )

    if output_format == "text":
        return PlainTextResponse(full_text)
    if output_format == "srt":
        return Response(_segments_to_srt(segments), media_type="application/x-subrip")
    if output_format == "vtt":
        return Response(_segments_to_vtt(segments), media_type="text/vtt")
    if output_format == "verbose_json":
        granularities = set(timestamp_granularities or []) | set(
            timestamp_granularities_plain or []
        )
        return JSONResponse(
            {
                "task": "transcribe",
                "language": "auto",
                "duration": prepared.duration,
                "text": full_text,
                "segments": [
                    {
                        "id": index,
                        "seek": 0,
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["segment"],
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }
                    for index, segment in enumerate(segments)
                ],
                "words": words if "word" in granularities else None,
            }
        )
    return JSONResponse({"text": full_text})


@router.post("/v1/audio/transcriptions/batch")
async def transcribe_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    model: str = Form(DEFAULT_MODEL),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Batch contains {len(files)} files; limit is {MAX_BATCH_FILES}",
        )
    model_name = _validate_model(model)
    filenames = [upload.filename or "unnamed" for upload in files]

    raws: List[bytes] = []
    total_bytes = 0
    for upload in files:
        raw = await _read_upload_limited(upload)
        total_bytes += len(raw)
        if total_bytes > MAX_BATCH_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Batch exceeds the {MAX_BATCH_BYTES} byte limit",
            )
        raws.append(raw)

    loop = asyncio.get_running_loop()
    futures = [
        loop.run_in_executor(request.app.state.audio_pool, _prepare_audio, raw)
        for raw in raws
    ]
    prepared_or_errors = await asyncio.gather(*futures, return_exceptions=True)
    prepared_files: List[_PreparedAudio] = []
    for filename, item in zip(filenames, prepared_or_errors):
        if isinstance(item, _AudioTooLong):
            raise HTTPException(status_code=413, detail=f"{filename}: {item}")
        if isinstance(item, BaseException):
            logger.exception(
                "batch audio preprocessing failed for %s",
                filename,
                exc_info=(type(item), item, item.__traceback__),
            )
            raise HTTPException(
                status_code=415, detail=f"{filename}: audio could not be decoded"
            )
        prepared_files.append(item)

    total_chunks = sum(len(item.pieces) for item in prepared_files)
    if total_chunks > MAX_REQUEST_CHUNKS:
        raise HTTPException(
            status_code=413,
            detail=f"Batch produced {total_chunks} chunks; limit is {MAX_REQUEST_CHUNKS}",
        )

    flattened = [piece for item in prepared_files for piece in item.pieces]
    worker = request.app.state.worker
    if worker is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready")
    flat_results = await worker.submit_many(flattened, model_name)

    cursor = 0
    response_items = []
    for filename, prepared in zip(filenames, prepared_files):
        count = len(prepared.pieces)
        item_results = flat_results[cursor : cursor + count]
        cursor += count
        text, _segments, _words = _stitch(prepared, item_results)
        response_items.append(
            {"filename": filename, "text": text, "duration": prepared.duration}
        )

    if cursor != len(flat_results):
        raise RuntimeError("inference result accounting mismatch")
    return {"results": response_items, "batch_size": len(response_items)}
