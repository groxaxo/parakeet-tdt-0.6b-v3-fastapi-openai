"""Pause-aware audio chunking with strict size invariants."""
from __future__ import annotations

import threading
from typing import List, Tuple

from .config import (
    CHUNK_MAX_SEC,
    CHUNK_MIN_SEC,
    CHUNK_TARGET_SEC,
    TARGET_SR,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
    logger,
)

import numpy as np

Range = Tuple[int, int]
_vad_model = None
_vad_init_lock = threading.Lock()
_vad_infer_lock = threading.Lock()


def _get_vad():
    global _vad_model
    if _vad_model is not None:
        return _vad_model
    with _vad_init_lock:
        if _vad_model is not None:
            return _vad_model
        try:
            from silero_vad import load_silero_vad  # type: ignore

            _vad_model = load_silero_vad(onnx=True)
            logger.info("Loaded Silero VAD (ONNX backend)")
        except Exception as exc:
            logger.warning(
                "Silero VAD unavailable (%s); falling back to energy VAD", exc
            )
            _vad_model = "energy"
    return _vad_model


def _silero_speech_segments(wav: np.ndarray) -> List[Range]:
    """Return speech spans as half-open sample ranges."""
    model = _get_vad()
    if model == "energy":
        return _energy_speech_segments(wav)

    from silero_vad import get_speech_timestamps  # type: ignore
    import torch

    tensor = torch.from_numpy(wav)
    # Silero resets internal state during timestamp extraction, so serialize
    # access to the shared singleton model across preprocessing threads.
    with _vad_infer_lock:
        timestamps = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=TARGET_SR,
            threshold=VAD_THRESHOLD,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
            return_seconds=False,
        )
    return [(int(item["start"]), int(item["end"])) for item in timestamps]


def _energy_speech_segments(wav: np.ndarray) -> List[Range]:
    """Cheap RMS-based fallback when Silero is unavailable."""
    frame = max(1, int(0.02 * TARGET_SR))
    if wav.size < frame:
        return [(0, wav.size)] if np.any(np.abs(wav) > 1e-4) else []

    frame_count = wav.size // frame
    framed = wav[: frame_count * frame].reshape(frame_count, frame)
    rms = np.sqrt((framed * framed).mean(axis=1) + 1e-12)
    threshold = max(1e-3, float(rms.mean()) * 0.4)
    voiced = rms > threshold
    minimum_silence_frames = max(1, int(VAD_MIN_SILENCE_MS / 20))

    segments: List[Range] = []
    index = 0
    while index < frame_count:
        if not voiced[index]:
            index += 1
            continue
        start = index
        cursor = index
        silence = 0
        while cursor < frame_count:
            if voiced[cursor]:
                silence = 0
            else:
                silence += 1
                if silence >= minimum_silence_frames:
                    break
            cursor += 1
        end = max(start + 1, min(cursor - silence, frame_count))
        segments.append((start * frame, min(end * frame, wav.size)))
        index = max(cursor, index + 1)
    return segments


def _normalize_segments(segments: List[Range], total: int) -> List[Range]:
    normalized: List[Range] = []
    for start, end in sorted(segments):
        start = min(total, max(0, int(start)))
        end = min(total, max(start, int(end)))
        if end <= start:
            continue
        if normalized and start <= normalized[-1][1]:
            previous_start, previous_end = normalized[-1]
            normalized[-1] = (previous_start, max(previous_end, end))
        else:
            normalized.append((start, end))
    return normalized


def _split_oversized(start: int, end: int, target: int, maximum: int) -> List[Range]:
    """Split one non-empty range while guaranteeing every part <= maximum."""
    if end <= start:
        return []
    parts: List[Range] = []
    cursor = start
    while end - cursor > maximum:
        cut = min(end, cursor + target)
        if cut <= cursor:
            cut = min(end, cursor + maximum)
        parts.append((cursor, cut))
        cursor = cut
    if end > cursor:
        parts.append((cursor, end))
    return parts


def auto_chunk(wav: np.ndarray) -> List[Range]:
    """Return ordered, non-empty, bounded ranges in the original waveform.

    Short clips bypass VAD. Long clips with no detected speech return no ranges,
    allowing the API to skip expensive ASR inference for silence.
    """
    total = int(wav.size)
    if total <= 0:
        return []

    target = max(1, int(CHUNK_TARGET_SEC * TARGET_SR))
    maximum = max(target, int(CHUNK_MAX_SEC * TARGET_SR))
    minimum = max(0, int(CHUNK_MIN_SEC * TARGET_SR))
    if total <= maximum:
        return [(0, total)]

    segments = _normalize_segments(_silero_speech_segments(wav), total)
    if not segments:
        return []

    packed: List[Range] = []
    current_start, current_end = segments[0]
    for start, end in segments[1:]:
        if end - current_start <= target:
            current_end = end
            continue

        if current_end - current_start >= minimum:
            cut = min(total, max(current_end, (current_end + start) // 2))
            if cut > current_start:
                packed.append((current_start, cut))
            current_start = cut
            current_end = end
        else:
            current_end = end

    if current_end > current_start:
        packed.append((current_start, current_end))

    output: List[Range] = []
    for start, end in packed:
        output.extend(_split_oversized(start, end, target, maximum))

    # Defensive invariant filter: malformed VAD output must never reach ORT.
    return [
        (start, end)
        for start, end in output
        if 0 <= start < end <= total and end - start <= maximum
    ]


def slice_chunks(wav: np.ndarray, ranges: List[Range]) -> List[np.ndarray]:
    """Return zero-copy contiguous views for the selected ranges."""
    return [wav[start:end] for start, end in ranges if end > start]
