from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .model import lifespan
from .routes import router
from .config import logger

from parakeet_service.stream_routes import router as stream_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet-TDT 0.6B v3 STT service",
        version="0.0.1",
        description=(
            "High-accuracy speech-to-text with multi-language support "
            "and optional word/segment timestamps. Based on jianchang512/parakeet-api."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    # TODO: improve streaming and add support for other audio formats (maybe)
    app.include_router(stream_router)
    
    # Serve the web UI
    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        template_path = Path(__file__).parent.parent / "templates" / "index.html"
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    
    logger.info("FastAPI app initialised")
    return app


app = create_app()
