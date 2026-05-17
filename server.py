#!/usr/bin/env python3
"""Entry point for the optimized Parakeet v3 server.

Run with::

    python server.py                 # uvicorn defaults
    PARAKEET_USE_GPU=true python server.py
    PARAKEET_PORT=5093 python server.py

Or directly with uvicorn::

    uvicorn parakeet_service.main:app --host 0.0.0.0 --port 5092
"""
from __future__ import annotations
import os

import uvicorn


def main() -> None:
    host = os.getenv("PARAKEET_HOST", "0.0.0.0")
    port = int(os.getenv("PARAKEET_PORT", "5092"))
    workers = int(os.getenv("PARAKEET_UVICORN_WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    uvicorn.run(
        "parakeet_service.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
