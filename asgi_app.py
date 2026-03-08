from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import app as flask_app
from realtime.protocol import ProtocolError, error_event, parse_client_event, server_event
from realtime.session import RealtimeSession


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Parakeet Realtime ASGI Gateway", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/realtime-demo")
async def realtime_demo() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "realtime.html")


@app.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = RealtimeSession()

    await ws.send_json(
        server_event(
            "session.created",
            session={
                **session.get_public_session(),
                "x_protocol": "openai-subset-with-local-extensions",
            },
        )
    )

    try:
        while True:
            payload = await ws.receive_json()
            try:
                parsed = parse_client_event(payload)
            except ProtocolError as exc:
                await ws.send_json(error_event(str(exc)))
                continue

            events = await session.handle_event(parsed.event_type, parsed.payload)
            for event in events:
                await ws.send_json(event)

    except WebSocketDisconnect:
        return


# Mount legacy Flask app so existing endpoints and UI remain available.
app.mount("/", WSGIMiddleware(flask_app))
