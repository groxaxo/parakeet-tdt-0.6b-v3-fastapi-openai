from __future__ import annotations

import numpy as np

from parakeet_service import chunker


def _assert_valid(ranges, total, maximum):
    previous_end = -1
    for start, end in ranges:
        assert 0 <= start < end <= total
        assert end - start <= maximum
        assert start >= previous_end
        previous_end = end


def test_empty_audio_has_no_chunks():
    assert chunker.auto_chunk(np.empty(0, dtype=np.float32)) == []


def test_short_audio_bypasses_vad(monkeypatch):
    monkeypatch.setattr(
        chunker,
        "_silero_speech_segments",
        lambda _wav: (_ for _ in ()).throw(AssertionError("VAD should not run")),
    )
    waveform = np.zeros(int(chunker.CHUNK_MAX_SEC * chunker.TARGET_SR) - 1)
    assert chunker.auto_chunk(waveform) == [(0, waveform.size)]


def test_long_silence_skips_inference(monkeypatch):
    monkeypatch.setattr(chunker, "_silero_speech_segments", lambda _wav: [])
    waveform = np.zeros(int((chunker.CHUNK_MAX_SEC + 10) * chunker.TARGET_SR))
    assert chunker.auto_chunk(waveform) == []


def test_long_uninterrupted_speech_has_no_phantom_tail(monkeypatch):
    total = int((chunker.CHUNK_MAX_SEC * 2.5) * chunker.TARGET_SR)
    monkeypatch.setattr(
        chunker, "_silero_speech_segments", lambda _wav: [(0, total)]
    )
    ranges = chunker.auto_chunk(np.ones(total, dtype=np.float32))
    maximum = int(chunker.CHUNK_MAX_SEC * chunker.TARGET_SR)
    _assert_valid(ranges, total, maximum)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == total
    assert all(end - start > 1 for start, end in ranges)


def test_slice_chunks_returns_views():
    waveform = np.arange(20, dtype=np.float32)
    pieces = chunker.slice_chunks(waveform, [(2, 8), (8, 12)])
    assert len(pieces) == 2
    assert np.shares_memory(waveform, pieces[0])
    assert pieces[0].flags.c_contiguous
