"""In-memory audio decoding & resampling.

Avoids per-chunk ffmpeg subprocesses. Strategy:
  * 16 kHz mono PCM WAV  -> read with stdlib `wave` straight to float32
  * Other WAV variants    -> stdlib `wave` + `audioop` (channels, sample width, rate)
  * Compressed / non-WAV  -> one ffmpeg subprocess decoding to s16le mono 16 kHz on stdout
"""
from __future__ import annotations
import audioop
import subprocess
import wave
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import TARGET_SR, logger


def _wav_info(data: bytes) -> Optional[dict]:
    try:
        with wave.open(BytesIO(data), "rb") as w:
            return {
                "frames": w.getnframes(),
                "sample_rate": w.getframerate(),
                "channels": w.getnchannels(),
                "sample_width": w.getsampwidth(),
                "compression": w.getcomptype(),
                "duration": (w.getnframes() / w.getframerate()) if w.getframerate() else 0.0,
            }
    except (wave.Error, EOFError, OSError):
        return None


def _decode_pcm_wav(data: bytes, info: dict) -> Optional[np.ndarray]:
    if info["compression"] != "NONE":
        return None
    sw = info["sample_width"]
    ch = info["channels"]
    if sw not in (1, 2, 3, 4) or ch not in (1, 2):
        return None
    try:
        with wave.open(BytesIO(data), "rb") as w:
            pcm = w.readframes(w.getnframes())
        if ch == 2:
            pcm = audioop.tomono(pcm, sw, 0.5, 0.5)
            ch = 1
        if info["sample_rate"] != TARGET_SR:
            pcm, _ = audioop.ratecv(pcm, sw, ch, info["sample_rate"], TARGET_SR, None)
        if sw == 1:
            return (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        if sw == 2:
            return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if sw == 4:
            return np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
        pcm16 = audioop.lin2lin(pcm, sw, 2)
        return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    except (wave.Error, EOFError, OSError, audioop.error, ValueError):
        return None


def _ffmpeg_decode(data: bytes) -> np.ndarray:
    """Decode any container/codec to mono 16 kHz float32 via a single ffmpeg call."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", "pipe:0",
        "-ac", "1", "-ar", str(TARGET_SR),
        "-f", "s16le", "pipe:1",
    ]
    p = subprocess.run(cmd, input=data, capture_output=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {p.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(p.stdout, dtype="<i2").astype(np.float32) / 32768.0


def load_audio(data: bytes) -> np.ndarray:
    """Return mono float32 [-1,1] at 16 kHz."""
    info = _wav_info(data)
    if info is not None:
        wav = _decode_pcm_wav(data, info)
        if wav is not None:
            return wav
    return _ffmpeg_decode(data)


def load_audio_path(path: Path) -> np.ndarray:
    return load_audio(Path(path).read_bytes())
