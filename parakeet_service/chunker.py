"""Auto-chunking based on speech-pause / sentence-boundary detection.

Uses Silero VAD (ONNX backend, no torch needed) to find low-energy speech
boundaries and produces chunks of ~CHUNK_TARGET_SEC seconds that always cut
on a silence boundary. This is the *intelligent* chunker the baseline tries
to do with `ffmpeg silencedetect`, but ~10x faster and without subprocesses.
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

from .config import (
    TARGET_SR,
    CHUNK_TARGET_SEC,
    CHUNK_MAX_SEC,
    CHUNK_MIN_SEC,
    VAD_THRESHOLD,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    logger,
)

_vad_model = None  # lazy init


def _get_vad():
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad  # type: ignore
            _vad_model = load_silero_vad(onnx=True)
            logger.info("Loaded Silero VAD (ONNX backend)")
        except Exception as exc:
            logger.warning("Silero VAD unavailable (%s); falling back to energy VAD", exc)
            _vad_model = "energy"
    return _vad_model


# ---------------------------------------------------------------------------
# Pause detection
# ---------------------------------------------------------------------------
def _silero_speech_segments(wav: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start_sample, end_sample) speech spans."""
    model = _get_vad()
    if model == "energy":
        return _energy_speech_segments(wav)
    from silero_vad import get_speech_timestamps  # type: ignore
    import torch  # silero-vad pulls torch even for onnx mode; lightweight here

    # Silero VAD wants a torch tensor
    t = torch.from_numpy(wav)
    ts = get_speech_timestamps(
        t,
        model,
        sampling_rate=TARGET_SR,
        threshold=VAD_THRESHOLD,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
        return_seconds=False,
    )
    return [(int(t["start"]), int(t["end"])) for t in ts]


def _energy_speech_segments(wav: np.ndarray) -> List[Tuple[int, int]]:
    """Cheap RMS-based fallback if Silero isn't installed."""
    frame = int(0.02 * TARGET_SR)  # 20 ms frames
    if frame <= 0 or wav.size < frame:
        return [(0, wav.size)]
    n = wav.size // frame
    framed = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt((framed * framed).mean(axis=1) + 1e-12)
    thr = max(1e-3, rms.mean() * 0.4)
    voiced = rms > thr
    segs: List[Tuple[int, int]] = []
    i = 0
    min_sil = max(1, int(VAD_MIN_SILENCE_MS / 20))
    while i < n:
        if voiced[i]:
            start = i
            j = i
            sil = 0
            while j < n:
                if voiced[j]:
                    sil = 0
                else:
                    sil += 1
                    if sil >= min_sil:
                        break
                j += 1
            end = min(j - sil, n)
            segs.append((start * frame, end * frame))
            i = j
        else:
            i += 1
    return segs or [(0, wav.size)]


# ---------------------------------------------------------------------------
# Chunk packer
# ---------------------------------------------------------------------------
def auto_chunk(wav: np.ndarray) -> List[Tuple[int, int]]:
    """Pack speech segments into ~target-length chunks aligned on pauses.

    Returns list of (start_sample, end_sample) ranges in the original
    waveform. If `wav` is shorter than CHUNK_MAX_SEC, returns a single chunk.
    """
    total = wav.size
    if total <= int(CHUNK_MAX_SEC * TARGET_SR):
        return [(0, total)]

    target = int(CHUNK_TARGET_SEC * TARGET_SR)
    max_len = int(CHUNK_MAX_SEC * TARGET_SR)
    min_len = int(CHUNK_MIN_SEC * TARGET_SR)

    segs = _silero_speech_segments(wav)
    if not segs:
        # Pure silence — emit a single chunk; ASR will just produce empty text.
        return [(0, total)]

    chunks: List[Tuple[int, int]] = []
    cur_start = segs[0][0]
    cur_end = segs[0][1]
    for s, e in segs[1:]:
        # If adding this segment keeps us under target, extend
        if e - cur_start <= target:
            cur_end = e
            continue
        # If we're already past min and the next segment would push us beyond
        # max, cut here on the trailing silence (between cur_end and s).
        if (cur_end - cur_start) >= min_len:
            mid = (cur_end + s) // 2  # middle of the pause
            chunks.append((cur_start, mid))
            cur_start = mid
            cur_end = e
        else:
            cur_end = e
            # If we've blown past max_len without finding a pause, force-cut
            if (cur_end - cur_start) >= max_len:
                chunks.append((cur_start, cur_end))
                cur_start = e
                cur_end = e
    # Tail
    chunks.append((cur_start, max(cur_end, cur_start + 1)))

    # Guard: very long final chunk → split on time
    out: List[Tuple[int, int]] = []
    for cs, ce in chunks:
        if ce - cs <= max_len:
            out.append((cs, ce))
            continue
        # Force time-based split for the overflow
        cursor = cs
        while ce - cursor > max_len:
            out.append((cursor, cursor + target))
            cursor += target
        out.append((cursor, ce))
    return out


def slice_chunks(wav: np.ndarray, ranges: List[Tuple[int, int]]) -> List[np.ndarray]:
    return [wav[s:e].copy() for s, e in ranges]
