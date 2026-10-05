"""Real corpus and boundary parity using the repository's unchanged chunker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from parakeet_service.model import get_model, runtime_status
from parakeet_service.chunker import auto_chunk, slice_chunks
from parakeet_service.routes import _PreparedAudio, _stitch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in pq.read_table(args.corpus).to_pylist():
        audio, sr = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
        assert sr == 16000
        if len(audio) / sr <= 15:
            cases.append((row['id'], audio))
        if len(cases) == 50:
            break
    sample = cases[3][1]
    tiled = np.tile(sample, int(120 * 16000 / len(sample)) + 1)
    cases.extend([
        ('silence-10ms', np.zeros(160, np.float32)),
        ('silence-1s', np.zeros(16000, np.float32)),
        ('silence-20s', np.zeros(320000, np.float32)),
        ('short-50ms', sample[:800]),
        ('short-250ms', sample[:4000]),
        ('boundary-14.99s', tiled[:int(14.99 * 16000)]),
        ('boundary-15s', tiled[:15 * 16000]),
        ('boundary-15.01s', tiled[:int(15.01 * 16000)]),
        ('long-120s', tiled[:120 * 16000]),
    ])
    prepared = []
    for name, audio in cases:
        ranges = auto_chunk(audio)
        prepared.append((name, _PreparedAudio(audio, ranges, slice_chunks(audio, ranges), len(audio) / 16000)))
    model = get_model()
    def recognize(case):
        name, audio = case
        start = time.perf_counter()
        results = [model.recognize(piece, sample_rate=16000) for piece in audio.pieces]
        tokens = [dict(text=r.text, tokens=list(r.tokens), timestamps=list(r.timestamps),
                       token_indices=[round(t / .08) for t in r.timestamps]) for r in results]
        text, segments, words = _stitch(audio, results)
        return {'id': name, 'seconds': time.perf_counter() - start,
                'audio_seconds': audio.duration, 'ranges': audio.ranges,
                'text': text, 'segments': segments, 'words': words, 'chunks': tokens}
    model.asr.gpu_decoder_state = False
    baseline = [recognize(case) for case in prepared]
    (args.out / 'host.json').write_text(json.dumps(baseline, indent=2))
    model.asr.gpu_decoder_state = True
    summaries = []
    for workers in (1, 2, 4):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            candidate = list(pool.map(recognize, prepared))
        mismatches = []
        for host, gpu in zip(baseline, candidate):
            for key in ('text', 'segments', 'words', 'chunks'):
                if host[key] != gpu[key]:
                    mismatches.append({'id': host['id'], 'field': key})
        (args.out / f'gpu-workers-{workers}.json').write_text(json.dumps(candidate, indent=2))
        summary = {'workers': workers, 'cases': len(candidate), 'mismatches': mismatches,
                   'passed': not mismatches}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    report = {'passed': all(r['passed'] for r in summaries), 'runtime': runtime_status(),
              'baseline_cases': len(baseline), 'corpus_cases': 50, 'boundary_cases': 9,
              'results': summaries}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
