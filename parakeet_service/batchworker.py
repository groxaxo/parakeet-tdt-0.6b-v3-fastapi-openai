"""Bounded asynchronous inference workers for CPU and GPU deployments."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from .config import (
    BATCHED,
    BATCH_WINDOW_MS,
    INFER_WORKERS,
    MAX_BATCH_SIZE,
    logger,
)

import numpy as np


@dataclass(slots=True)
class _Job:
    wav: np.ndarray
    model_name: str
    future: asyncio.Future[Any]


def _shutdown_executor(executor: ThreadPoolExecutor) -> None:
    executor.shutdown(wait=True, cancel_futures=True)


def _infer_one(get_model_fn: Callable[[str], Any], model_name: str, wav: np.ndarray):
    return get_model_fn(model_name).recognize(wav)


def _infer_named_batch(
    get_model_fn: Callable[[str], Any], model_name: str, wavs: List[np.ndarray]
) -> List[Any]:
    return _infer_batch(get_model_fn(model_name), wavs)


class InferencePool:
    """Parallel single-item ORT calls, optimized for CPU inference."""

    def __init__(self, get_model_fn, *, workers: int = INFER_WORKERS):
        self._get_model = get_model_fn
        self._workers = max(1, workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="ort"
        )
        self._closed = False
        logger.info("InferencePool started (workers=%d)", self._workers)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(_shutdown_executor, self._executor)

    async def submit(self, wav: np.ndarray, model_name: str) -> Any:
        if self._closed:
            raise RuntimeError("inference pool is stopped")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, _infer_one, self._get_model, model_name, wav
        )

    async def submit_many(self, wavs: List[np.ndarray], model_name: str) -> List[Any]:
        if not wavs:
            return []
        return list(
            await asyncio.gather(*(self.submit(wav, model_name) for wav in wavs))
        )


class BatchWorker:
    """Cross-request micro-batching, intended for GPU inference."""

    def __init__(
        self,
        get_model_fn,
        *,
        max_batch: int = MAX_BATCH_SIZE,
        window_ms: float = BATCH_WINDOW_MS,
    ):
        self._get_model = get_model_fn
        self._max_batch = max(1, max_batch)
        self._window_s = max(0.0, window_ms) / 1000.0
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ort")
        self._accepting = False

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._accepting = True
            self._task = asyncio.create_task(self._run(), name="batch_worker")
            logger.info(
                "BatchWorker started (max_batch=%d window=%.1fms)",
                self._max_batch,
                self._window_s * 1000,
            )

    async def stop(self) -> None:
        self._accepting = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._fail_queued(RuntimeError("batch worker stopped"))
        await asyncio.to_thread(_shutdown_executor, self._executor)

    def _new_job(self, wav: np.ndarray, model_name: str) -> _Job:
        if not self._accepting:
            raise RuntimeError("batch worker is not running")
        future = asyncio.get_running_loop().create_future()
        return _Job(wav=wav, model_name=model_name, future=future)

    async def submit(self, wav: np.ndarray, model_name: str) -> Any:
        job = self._new_job(wav, model_name)
        await self._queue.put(job)
        return await job.future

    async def submit_many(self, wavs: List[np.ndarray], model_name: str) -> List[Any]:
        if not wavs:
            return []
        jobs = [self._new_job(wav, model_name) for wav in wavs]
        for job in jobs:
            await self._queue.put(job)
        return list(await asyncio.gather(*(job.future for job in jobs)))

    def _fail_queued(self, exc: BaseException) -> None:
        while True:
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if not job.future.done():
                job.future.set_exception(exc)
            self._queue.task_done()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        carry: Optional[_Job] = None
        try:
            while True:
                first = carry if carry is not None else await self._queue.get()
                carry = None
                batch = [first]
                deadline = time.monotonic() + self._window_s

                while len(batch) < self._max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        candidate = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        break
                    if candidate.model_name != first.model_name:
                        carry = candidate
                        break
                    batch.append(candidate)

                try:
                    results = await loop.run_in_executor(
                        self._executor,
                        _infer_named_batch,
                        self._get_model,
                        first.model_name,
                        [job.wav for job in batch],
                    )
                    if len(results) != len(batch):
                        raise RuntimeError(
                            "batched inference returned "
                            f"{len(results)} results for {len(batch)} inputs"
                        )
                except asyncio.CancelledError:
                    error = RuntimeError("batch worker stopped during inference")
                    for job in batch:
                        if not job.future.done():
                            job.future.set_exception(error)
                    raise
                except Exception as exc:
                    logger.exception("batch inference failed (size=%d)", len(batch))
                    for job in batch:
                        if not job.future.done():
                            job.future.set_exception(exc)
                else:
                    for job, result in zip(batch, results):
                        if not job.future.done():
                            job.future.set_result(result)
                finally:
                    for _job in batch:
                        self._queue.task_done()
        finally:
            if carry is not None:
                if not carry.future.done():
                    carry.future.set_exception(RuntimeError("batch worker stopped"))
                self._queue.task_done()


def _infer_batch(model, wavs: List[np.ndarray]) -> List[Any]:
    if len(wavs) == 1:
        return [model.recognize(wavs[0])]
    result = model.recognize(wavs)
    if isinstance(result, list):
        return result
    if isinstance(result, tuple):
        return list(result)
    try:
        return list(result)
    except TypeError as exc:
        raise RuntimeError("model returned a non-iterable batched result") from exc


def build_worker(get_model_fn):
    if BATCHED:
        logger.info("Using BatchWorker (cross-request micro-batching)")
        return BatchWorker(get_model_fn)
    logger.info("Using InferencePool (parallel single-item inference)")
    return InferencePool(get_model_fn)
