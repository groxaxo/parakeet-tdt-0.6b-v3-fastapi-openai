from __future__ import annotations

from realtime.session import RealtimeSession


def test_session_update_new_fields_normalization() -> None:
    session = RealtimeSession()
    event = session.update_session(
        {
            "session": {
                "model": "PARAKEET-TDT-0.6B-V3",
                "context_utterances": 999,
                "language": "Dutch",
                "silence_break_ms": 2500,
                "new_line_per_final": True,
                "vad_silence_rms_threshold": 0.5,
                "partial_cooldown_seconds": 0.05,
                "min_partial_seconds": 4.0,
                "max_partial_window_seconds": 0.1,
                "min_final_seconds": 3.0,
            }
        }
    )

    assert event["type"] == "session.updated"
    assert event["session"]["model"] == "parakeet-tdt-0.6b-v3"
    assert event["session"]["context_utterances"] == 8
    assert event["session"]["language"] == "nl"
    assert event["session"]["silence_break_ms"] == 2000
    assert event["session"]["new_line_per_final"] is True
    assert event["session"]["vad_silence_rms_threshold"] == 0.06
    assert event["session"]["partial_cooldown_seconds"] == 0.2
    assert event["session"]["min_partial_seconds"] == 3.0
    assert event["session"]["max_partial_window_seconds"] == 3.0
    assert event["session"]["min_final_seconds"] == 2.0


def test_session_update_invalid_language_keeps_previous() -> None:
    session = RealtimeSession()
    session.update_session({"session": {"language": "en"}})
    session.update_session({"session": {"language": "unsupported-lang"}})
    event = session.update_session({"session": {"silence_break_ms": 100}})

    assert event["session"]["language"] == "en"
    assert event["session"]["silence_break_ms"] == 300


def test_public_session_defaults() -> None:
    session = RealtimeSession()
    payload = session.get_public_session()

    assert payload["language"] == "autodetect"
    assert payload["silence_break_ms"] == 900
    assert payload["new_line_per_final"] is False
    assert payload["vad_silence_rms_threshold"] == 0.012
    assert payload["partial_cooldown_seconds"] == 0.8
    assert payload["min_partial_seconds"] == 1.0
    assert payload["max_partial_window_seconds"] == 5.0
    assert payload["min_final_seconds"] == 0.25


def test_transcribe_sync_language_hint_and_fallback(monkeypatch) -> None:
    class FakeResult:
        text = "hello world"

    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def recognize(self, path: str, language: str | None = None):
            self.calls.append((path, language))
            if language == "nl":
                raise TypeError("language not supported")
            return FakeResult()

    fake_model = FakeModel()
    monkeypatch.setattr("realtime.session.get_model", lambda _name: fake_model)

    session = RealtimeSession()
    session.config.language = "en"
    text = session._transcribe_sync(b"\x00\x00" * 320)
    assert text == "hello world"
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0][1] == "en"

    session.config.language = "nl"
    text = session._transcribe_sync(b"\x00\x00" * 320)
    assert text == "hello world"
    assert len(fake_model.calls) == 3
    assert fake_model.calls[1][1] == "nl"
    assert fake_model.calls[2][1] is None
