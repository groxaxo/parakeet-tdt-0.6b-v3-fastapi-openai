"""FastAPI app factory + lifespan."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .batchworker import build_worker
from .config import AUDIO_WORKERS, DEFAULT_MODEL, logger
from .model import get_model, load_model
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lifespan startup: loading default model")
    load_model(DEFAULT_MODEL)
    app.state.worker = build_worker(get_model)
    await app.state.worker.start()
    app.state.audio_pool = ThreadPoolExecutor(
        max_workers=AUDIO_WORKERS, thread_name_prefix="audio"
    )
    logger.info("Service ready")
    try:
        yield
    finally:
        logger.info("Lifespan shutdown")
        await app.state.worker.stop()
        app.state.audio_pool.shutdown(wait=False, cancel_futures=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet TDT 0.6B v3 (optimized)",
        version="1.0.0",
        description="High-throughput OpenAI-compatible ASR service for Parakeet TDT 0.6B v3.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
