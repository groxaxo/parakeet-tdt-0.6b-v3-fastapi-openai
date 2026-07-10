"""In-memory audio decoding and resampling."""
from __future__ import annotations

import audioop
import subprocess
import wave
from io import BytesIO
from pathlib import Path
from typing import Optional

from .config import FFMPEG_TIMEOUT_SEC, TARGET_SR

import numpy as np


def _wav_info(data: bytes) -> Optional[dict]:
    try:
        with wave.open(BytesIO(data), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            return {
                "frames": wav_file.getnframes(),
                "sample_rate": sample_rate,
                "channels": wav_file.getnchannels(),
                "sample_width": wav_file.getsampwidth(),
                "compression": wav_file.getcomptype(),
                "duration": wav_file.getnframes() / sample_rate if sample_rate else 0.0,
            }
    except (wave.Error, EOFError, OSError):
        return None


def _decode_pcm_wav(data: bytes, info: dict) -> Optional[np.ndarray]:
    if info["compression"] != "NONE":
        return None
    sample_width = info["sample_width"]
    channels = info["channels"]
    if sample_width not in (1, 2, 3, 4) or channels not in (1, 2):
        return None
    try:
        with wave.open(BytesIO(data), "rb") as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
        if channels == 2:
            pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
            channels = 1
        if info["sample_rate"] != TARGET_SR:
            pcm, _ = audioop.ratecv(
                pcm,
                sample_width,
                channels,
                info["sample_rate"],
                TARGET_SR,
                None,
            )
        if sample_width == 1:
            result = (
                np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0
            ) / 128.0
        elif sample_width == 2:
            result = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            result = (
                np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
            )
        else:
            pcm16 = audioop.lin2lin(pcm, sample_width, 2)
            result = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        return np.ascontiguousarray(result, dtype=np.float32)
    except (wave.Error, EOFError, OSError, audioop.error, ValueError):
        return None


def _ffmpeg_decode(data: bytes) -> np.ndarray:
    """Decode a container/codec to mono 16 kHz float32 with one FFmpeg call."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        process = subprocess.run(
            command,
            input=data,
            capture_output=True,
            check=False,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not available on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg decode exceeded {FFMPEG_TIMEOUT_SEC:g} seconds"
        ) from exc
    if process.returncode != 0:
        error = process.stderr.decode(errors="replace").strip()[:500]
        raise RuntimeError(f"ffmpeg decode failed: {error or 'unknown error'}")
    if not process.stdout:
        return np.empty(0, dtype=np.float32)
    result = np.frombuffer(process.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return np.ascontiguousarray(result, dtype=np.float32)


def load_audio(data: bytes) -> np.ndarray:
    """Return a finite mono float32 waveform at 16 kHz."""
    if not data:
        return np.empty(0, dtype=np.float32)
    info = _wav_info(data)
    waveform = _decode_pcm_wav(data, info) if info is not None else None
    if waveform is None:
        waveform = _ffmpeg_decode(data)
    if waveform.ndim != 1:
        waveform = np.ravel(waveform)
    if not np.isfinite(waveform).all():
        waveform = np.nan_to_num(waveform, copy=False)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def load_audio_path(path: Path) -> np.ndarray:
    return load_audio(Path(path).read_bytes())
