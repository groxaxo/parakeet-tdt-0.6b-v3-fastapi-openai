from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


class ProtocolError(ValueError):
    pass


@dataclass
class ParsedEvent:
    event_type: str
    payload: Dict[str, Any]


SUPPORTED_CLIENT_EVENTS = {
    "session.update",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "response.create",
    "ping",
}


def parse_client_event(data: Any) -> ParsedEvent:
    if not isinstance(data, dict):
        raise ProtocolError("Event must be a JSON object")

    event_type = data.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ProtocolError("Event `type` must be a non-empty string")

    if event_type not in SUPPORTED_CLIENT_EVENTS:
        raise ProtocolError(f"Unsupported event type: {event_type}")

    return ParsedEvent(event_type=event_type, payload=data)


def server_event(event_type: str, **kwargs: Any) -> Dict[str, Any]:
    evt = {"type": event_type}
    evt.update(kwargs)
    return evt


def error_event(message: str, code: str = "invalid_request_error") -> Dict[str, Any]:
    return server_event(
        "error",
        error={
            "type": code,
            "message": message,
        },
    )
