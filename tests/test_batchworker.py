from __future__ import annotations

import asyncio

import numpy as np
import pytest

from parakeet_service.batchworker import BatchWorker, InferencePool


class _Model:
    def recognize(self, value):
        if isinstance(value, list):
            return [float(item.sum()) for item in value]
        return float(value.sum())


@pytest.mark.asyncio
async def test_inference_pool_runs_and_rejects_after_stop():
    pool = InferencePool(lambda _name: _Model(), workers=2)
    results = await pool.submit_many(
        [np.ones(3, dtype=np.float32), np.ones(5, dtype=np.float32)], "model"
    )
    assert results == [3.0, 5.0]
    await pool.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await pool.submit(np.ones(1, dtype=np.float32), "model")


@pytest.mark.asyncio
async def test_batch_worker_batches_same_model():
    worker = BatchWorker(lambda _name: _Model(), max_batch=4, window_ms=20)
    await worker.start()
    try:
        results = await worker.submit_many(
            [np.ones(2, dtype=np.float32), np.ones(4, dtype=np.float32)], "a"
        )
        assert results == [2.0, 4.0]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_batch_worker_preserves_mixed_model_jobs():
    worker = BatchWorker(lambda _name: _Model(), max_batch=4, window_ms=20)
    await worker.start()
    try:
        first, second = await asyncio.gather(
            worker.submit(np.ones(2, dtype=np.float32), "a"),
            worker.submit(np.ones(3, dtype=np.float32), "b"),
        )
        assert (first, second) == (2.0, 3.0)
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_batch_worker_rejects_submission_before_start():
    worker = BatchWorker(lambda _name: _Model(), max_batch=2, window_ms=0)
    with pytest.raises(RuntimeError, match="not running"):
        await worker.submit(np.ones(1, dtype=np.float32), "a")
    await worker.stop()
