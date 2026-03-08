from __future__ import annotations

import asyncio
import base64
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app import get_model
from realtime.postprocess import apply_final_punctuation
from realtime.protocol import error_event, server_event


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
DEFAULT_MIN_PARTIAL_SECONDS = 1.0
DEFAULT_PARTIAL_COOLDOWN_SECONDS = 0.8
DEFAULT_MIN_FINAL_SECONDS = 0.25
DEFAULT_MAX_PARTIAL_WINDOW_SECONDS = 5.0
DEFAULT_VAD_SILENCE_RMS_THRESHOLD = 0.012
MIN_PARTIAL_MIN_SECONDS = 0.2
MIN_PARTIAL_MAX_SECONDS = 3.0
PARTIAL_COOLDOWN_MIN_SECONDS = 0.2
PARTIAL_COOLDOWN_MAX_SECONDS = 2.0
MIN_FINAL_MIN_SECONDS = 0.1
MIN_FINAL_MAX_SECONDS = 2.0
MAX_PARTIAL_WINDOW_MIN_SECONDS = 1.0
MAX_PARTIAL_WINDOW_MAX_SECONDS = 12.0
VAD_SILENCE_RMS_THRESHOLD_MIN = 0.002
VAD_SILENCE_RMS_THRESHOLD_MAX = 0.06
SILENCE_COMMIT_MIN_MS = 300.0
SILENCE_COMMIT_MAX_MS = 2000.0
UNSET = object()


@dataclass
class SessionConfig:
    model: str = "parakeet-tdt-0.6b-v3"
    context_utterances: int = 0
    language: str | None = None
    silence_break_ms: float = 900.0
    new_line_per_final: bool = False
    vad_silence_rms_threshold: float = DEFAULT_VAD_SILENCE_RMS_THRESHOLD
    partial_cooldown_seconds: float = DEFAULT_PARTIAL_COOLDOWN_SECONDS
    min_partial_seconds: float = DEFAULT_MIN_PARTIAL_SECONDS
    max_partial_window_seconds: float = DEFAULT_MAX_PARTIAL_WINDOW_SECONDS
    min_final_seconds: float = DEFAULT_MIN_FINAL_SECONDS


class RealtimeSession:
    def __init__(self) -> None:
        self.config = SessionConfig()
        self.buffer = bytearray()
        self.history: deque[str] = deque(maxlen=64)
        self.utterance_id = 0
        self.last_partial_ts = 0.0
        self.had_speech = False
        self.silence_ms = 0.0

    def update_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session = payload.get("session", {})
        if isinstance(session, dict):
            model_name = session.get("model")
            if isinstance(model_name, str) and model_name.strip():
                self.config.model = model_name.strip().lower()

            context_utt = session.get("context_utterances")
            if isinstance(context_utt, int) and context_utt >= 0:
                self.config.context_utterances = min(context_utt, 8)

            language = session.get("language")
            normalized_language = self._normalize_language(language)
            if normalized_language is not UNSET:
                self.config.language = normalized_language

            silence_break_ms = session.get("silence_break_ms")
            if isinstance(silence_break_ms, (int, float)):
                self.config.silence_break_ms = max(
                    SILENCE_COMMIT_MIN_MS,
                    min(float(silence_break_ms), SILENCE_COMMIT_MAX_MS),
                )

            new_line_per_final = self._coerce_bool(session.get("new_line_per_final"))
            if new_line_per_final is not None:
                self.config.new_line_per_final = new_line_per_final

            vad_silence_rms_threshold = self._coerce_float(
                session.get("vad_silence_rms_threshold"),
                lower=VAD_SILENCE_RMS_THRESHOLD_MIN,
                upper=VAD_SILENCE_RMS_THRESHOLD_MAX,
            )
            if vad_silence_rms_threshold is not None:
                self.config.vad_silence_rms_threshold = vad_silence_rms_threshold

            partial_cooldown_seconds = self._coerce_float(
                session.get("partial_cooldown_seconds"),
                lower=PARTIAL_COOLDOWN_MIN_SECONDS,
                upper=PARTIAL_COOLDOWN_MAX_SECONDS,
            )
            if partial_cooldown_seconds is not None:
                self.config.partial_cooldown_seconds = partial_cooldown_seconds

            min_partial_seconds = self._coerce_float(
                session.get("min_partial_seconds"),
                lower=MIN_PARTIAL_MIN_SECONDS,
                upper=MIN_PARTIAL_MAX_SECONDS,
            )
            if min_partial_seconds is not None:
                self.config.min_partial_seconds = min_partial_seconds

            max_partial_window_seconds = self._coerce_float(
                session.get("max_partial_window_seconds"),
                lower=MAX_PARTIAL_WINDOW_MIN_SECONDS,
                upper=MAX_PARTIAL_WINDOW_MAX_SECONDS,
            )
            if max_partial_window_seconds is not None:
                self.config.max_partial_window_seconds = max_partial_window_seconds

            min_final_seconds = self._coerce_float(
                session.get("min_final_seconds"),
                lower=MIN_FINAL_MIN_SECONDS,
                upper=MIN_FINAL_MAX_SECONDS,
            )
            if min_final_seconds is not None:
                self.config.min_final_seconds = min_final_seconds

            if self.config.max_partial_window_seconds < self.config.min_partial_seconds:
                self.config.max_partial_window_seconds = self.config.min_partial_seconds

        return server_event(
            "session.updated",
            session=self.get_public_session(),
        )

    def get_public_session(self) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "context_utterances": self.config.context_utterances,
            "language": self.config.language or "autodetect",
            "silence_break_ms": int(round(self.config.silence_break_ms)),
            "new_line_per_final": self.config.new_line_per_final,
            "vad_silence_rms_threshold": round(self.config.vad_silence_rms_threshold, 4),
            "partial_cooldown_seconds": round(self.config.partial_cooldown_seconds, 2),
            "min_partial_seconds": round(self.config.min_partial_seconds, 2),
            "max_partial_window_seconds": round(self.config.max_partial_window_seconds, 2),
            "min_final_seconds": round(self.config.min_final_seconds, 2),
            "sample_rate": SAMPLE_RATE,
            "input_audio_format": "pcm16",
        }

    async def append_audio(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        b64_audio = payload.get("audio")
        if not isinstance(b64_audio, str) or not b64_audio:
            return [error_event("input_audio_buffer.append requires base64 `audio`")]

        try:
            chunk = base64.b64decode(b64_audio)
        except Exception:
            return [error_event("Invalid base64 audio payload")]

        if len(chunk) % BYTES_PER_SAMPLE != 0:
            return [error_event("Audio payload must contain 16-bit PCM samples")]

        self.buffer.extend(chunk)
        self._update_vad_state(chunk)

        out: List[Dict[str, Any]] = []
        maybe_partial = await self._maybe_emit_partial()
        if maybe_partial:
            out.append(maybe_partial)

        # Local extension: auto-commit on long silence after detected speech.
        if self.had_speech and self.silence_ms >= self.config.silence_break_ms:
            out.extend(await self.commit_audio())

        return out

    async def commit_audio(self) -> List[Dict[str, Any]]:
        duration = self._buffer_duration_seconds()
        if duration < self.config.min_final_seconds:
            self._reset_turn_state(clear_buffer=True)
            return [
                server_event(
                    "input_audio_buffer.committed",
                    duration_seconds=round(duration, 3),
                    x_disposition="discarded_too_short",
                )
            ]

        self.utterance_id += 1
        utt_id = self.utterance_id
        raw_text = await self._transcribe_bytes(bytes(self.buffer))

        context_size = self.config.context_utterances
        context = list(self.history)[-context_size:] if context_size > 0 else []
        final_text = apply_final_punctuation(raw_text, context=context)

        if final_text:
            self.history.append(final_text)

        events = [
            server_event(
                "input_audio_buffer.committed",
                utterance_id=utt_id,
                duration_seconds=round(duration, 3),
            ),
            server_event(
                "response.output_text.done",
                utterance_id=utt_id,
                text=final_text,
                raw_text=raw_text,
                x_phase="final",
            ),
            server_event("response.completed", utterance_id=utt_id),
        ]

        self._reset_turn_state(clear_buffer=True)
        return events

    async def create_response(self) -> List[Dict[str, Any]]:
        maybe_partial = await self._emit_partial(force=True)
        if maybe_partial is None:
            return [server_event("response.completed", x_reason="no_partial")]
        return [maybe_partial]

    async def handle_event(self, event_type: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if event_type == "session.update":
            return [self.update_session(payload)]
        if event_type == "input_audio_buffer.append":
            return await self.append_audio(payload)
        if event_type == "input_audio_buffer.commit":
            return await self.commit_audio()
        if event_type == "response.create":
            return await self.create_response()
        if event_type == "ping":
            return [server_event("pong", ts=time.time())]
        return [error_event(f"Unsupported event type: {event_type}")]

    async def _maybe_emit_partial(self) -> Dict[str, Any] | None:
        if self._buffer_duration_seconds() < self.config.min_partial_seconds:
            return None
        if time.time() - self.last_partial_ts < self.config.partial_cooldown_seconds:
            return None
        return await self._emit_partial(force=False)

    async def _emit_partial(self, force: bool) -> Dict[str, Any] | None:
        total_seconds = self._buffer_duration_seconds()
        if not force and total_seconds < self.config.min_partial_seconds:
            return None

        max_bytes = int(self.config.max_partial_window_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
        pcm = bytes(self.buffer[-max_bytes:]) if len(self.buffer) > max_bytes else bytes(self.buffer)
        text = await self._transcribe_bytes(pcm)
        self.last_partial_ts = time.time()

        return server_event(
            "response.output_text.delta",
            utterance_id=self.utterance_id + 1,
            text=text,
            x_phase="partial",
        )

    async def _transcribe_bytes(self, pcm16_bytes: bytes) -> str:
        if not pcm16_bytes:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, pcm16_bytes)

    def _transcribe_sync(self, pcm16_bytes: bytes) -> str:
        model = get_model(self.config.model)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = Path(tmp.name)

        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(BYTES_PER_SAMPLE)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm16_bytes)

            language = self.config.language
            if language is None:
                result = model.recognize(str(path))
            else:
                try:
                    result = model.recognize(str(path), language=language)
                except (TypeError, KeyError):
                    # Some models may not support language hints.
                    result = model.recognize(str(path))
            text = getattr(result, "text", "")
            return (text or "").replace("\u2581", " ").strip()
        finally:
            if path.exists():
                path.unlink(missing_ok=True)

    def _buffer_duration_seconds(self) -> float:
        return len(self.buffer) / float(SAMPLE_RATE * BYTES_PER_SAMPLE)

    def _reset_turn_state(self, clear_buffer: bool) -> None:
        if clear_buffer:
            self.buffer.clear()
        self.had_speech = False
        self.silence_ms = 0.0
        self.last_partial_ts = 0.0

    def _update_vad_state(self, chunk: bytes) -> None:
        if not chunk:
            return

        arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            return

        rms = float(np.sqrt(np.mean(arr * arr)))
        duration_ms = (arr.size / SAMPLE_RATE) * 1000.0

        if rms >= self.config.vad_silence_rms_threshold:
            self.had_speech = True
            self.silence_ms = 0.0
        else:
            self.silence_ms += duration_ms

    @staticmethod
    def _normalize_language(value: Any) -> str | None | object:
        if not isinstance(value, str):
            return UNSET
        normalized = value.strip().lower()
        if normalized in {"", "autodetect", "auto"}:
            return None
        if normalized in {"en", "english"}:
            return "en"
        if normalized in {"nl", "dutch", "nederlands"}:
            return "nl"
        return UNSET

    @staticmethod
    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return None

    @staticmethod
    def _coerce_float(value: Any, *, lower: float, upper: float) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        return max(lower, min(float(value), upper))
