# Plan: Live Transcription WebSocket (OpenAI Realtime-Compatible)

**Generated**: 2026-03-04
**Estimated Complexity**: High

## Overview
Implement low-latency live transcription in `parakeet-tdt-0.6b-v3-fastapi-openai` using a browser microphone client and a Realtime-style WebSocket protocol compatible with OpenAI event semantics. The system will emit:
- `partial` transcript updates quickly (draft quality)
- `final` transcript updates on end-of-utterance (punctuated and context-corrected second pass)

The implementation will keep local/trusted operation (no auth/rate limit), exclude diarization, and include a simple web demo showing live textbox updates.
Compatibility target: **OpenAI Realtime-compatible subset with local extensions**, prioritized for shipping speed over strict parity.

## Prerequisites
- Python 3.10+
- Existing ONNX ASR model loading path in `app.py`
- Add ASGI runtime to support WebSocket (FastAPI/Uvicorn)
- Browser support for `MediaStream` + `AudioWorklet` or `ScriptProcessor` fallback
- FFmpeg already available for current batch endpoint

## Sprint 1: ASGI Foundation + Backward Compatibility
**Goal**: Introduce ASGI app structure that preserves existing REST behavior and enables WebSocket endpoints.

**Demo/Validation**:
- Start service with Uvicorn
- Confirm existing `POST /v1/audio/transcriptions` still works
- Confirm health/docs endpoints respond

### Task 1.1: Split server entrypoints (WSGI legacy vs ASGI new)
- **Location**: `app.py`, `asgi_app.py`, `server.py` (new)
- **Description**: Refactor route/model logic into shared service functions; expose ASGI-first app for websocket support while retaining REST compatibility.
- **Dependencies**: None
- **Acceptance Criteria**:
  - ASGI app boots and serves existing transcription endpoint
  - Legacy behavior for REST output formats preserved (`json`, `text`, `srt`, `vtt`, `verbose_json`)
- **Validation**:
  - `curl` transcription request returns same schema as before
  - Basic smoke tests pass

### Task 1.2: Add runtime and dependency updates
- **Location**: `requirements.txt`, `DOCKER.md`, `docker-compose.yml`, `README.md`
- **Description**: Add `fastapi`, `uvicorn`, and websocket-compatible server startup docs; de-emphasize Waitress for realtime mode.
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - New realtime run command documented
  - Docker command supports ASGI runtime
- **Validation**:
  - `uvicorn ...` boot succeeds
  - Container startup succeeds in realtime profile

### Task 1.3: Add protocol contract module
- **Location**: `realtime/protocol.py` (new)
- **Description**: Define typed event models and validation for incoming/outgoing Realtime-style events.
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Event parser accepts required event envelope format
  - Invalid events return deterministic error events
- **Validation**:
  - Unit tests for event parsing and validation

## Sprint 2: Realtime Session + Audio Ingestion Pipeline
**Goal**: Implement WebSocket session lifecycle and low-latency audio ingestion compatible with Realtime-style event flows.

**Demo/Validation**:
- Browser can connect to WS
- Audio chunks sent continuously
- Server emits partial text within ~1-3 seconds of speech

### Task 2.1: Implement `/v1/realtime` websocket endpoint
- **Location**: `realtime/routes.py` (new), `asgi_app.py`
- **Description**: Add WS endpoint with per-connection session state, event loop, disconnect handling, heartbeat/ping, and graceful close behavior.
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Connection accepted and session initialized
  - Session teardown cleans temporary buffers/resources
- **Validation**:
  - Integration test with test websocket client

### Task 2.2: Implement incoming event handling (OpenAI-style subset)
- **Location**: `realtime/handlers.py` (new), `realtime/protocol.py`
- **Description**: Support core event types for transcription flow (e.g. session config update, audio append, commit, response trigger) using an OpenAI-compatible subset plus explicit local extension fields/events.
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Events are accepted in expected order
  - Out-of-order/invalid events return protocol error events without crashing session
- **Validation**:
  - Contract tests for normal and invalid event sequences

### Task 2.3: Implement streaming audio ring buffer + chunk assembler
- **Location**: `realtime/audio_buffer.py` (new)
- **Description**: Decode base64 PCM16 chunks, append to bounded buffer, frame into model-ready windows, and produce utterance chunks for first-pass decode.
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Buffer limits enforced (no unbounded memory growth)
  - Commit/end-of-utterance produces deterministic chunk boundaries
- **Validation**:
  - Unit tests for append/commit/overflow behavior

### Task 2.4: Reuse or port VAD logic for turn segmentation
- **Location**: `realtime/vad.py` (new), optional reuse from v2 patterns
- **Description**: Add VAD-driven end-of-utterance detection tuned for low latency and stable finalization.
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - Speech bursts become utterance units
  - Silence threshold finalizes utterance promptly
- **Validation**:
  - Fixture audio tests for short pauses vs long pauses

## Sprint 3: Dual-Phase Transcript Engine (Partial + Final)
**Goal**: Emit fast draft transcript first, then corrected/punctuated final transcript when utterance completes.

**Demo/Validation**:
- Live text appears quickly while speaking
- Finalized sentence replaces/upgrades draft and includes punctuation

### Task 3.1: First-pass incremental decoding worker
- **Location**: `realtime/decoder.py` (new)
- **Description**: Create low-latency decoding path for incremental partial output using smaller chunk windows and minimal post-processing.
- **Dependencies**: Sprint 2 complete
- **Acceptance Criteria**:
  - Partial events produced before finalization
  - End-to-end median first-token latency meets practical target (<3s, aiming ~300ms where feasible)
- **Validation**:
  - Latency benchmark script with speech fixtures

### Task 3.2: Second-pass punctuation/context correction
- **Location**: `realtime/postprocess.py` (new)
- **Description**: Apply deterministic rule-based final pass on utterance-complete text to improve punctuation/capitalization; no LLM dependency in initial release.
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Final text is measurably cleaner than partial text
  - Final event includes stable utterance id for UI replacement
- **Validation**:
  - Golden tests comparing partial vs final outputs

### Task 3.3: Server event emission model
- **Location**: `realtime/events.py` (new), `realtime/handlers.py`
- **Description**: Emit OpenAI-style subset event stream with explicit `partial` and `final` transcript events mapped to protocol-compatible names and local extension metadata.
- **Dependencies**: Tasks 3.1, 3.2
- **Acceptance Criteria**:
  - Client can distinguish draft vs finalized text and update UI deterministically
  - Events are ordered and idempotent by utterance id + revision
- **Validation**:
  - Integration test verifies event order and replacement behavior

## Sprint 4: Browser Demo UI (Mic + Live Textbox)
**Goal**: Provide a simple, production-like demo page using browser mic and websocket streaming protocol.

**Demo/Validation**:
- Start/Stop recording buttons
- Live partial text appears while speaking
- Finalized punctuation-correct sentence replaces recent partial span

### Task 4.1: Add dedicated realtime demo page
- **Location**: `templates/realtime.html` (new), route in `asgi_app.py`
- **Description**: Create isolated demo page with status indicators, connection controls, transcript textbox, and latency indicators.
- **Dependencies**: Sprint 3 complete
- **Acceptance Criteria**:
  - Page loads and connects to WS endpoint
  - Clear UX for connected/recording/error states
- **Validation**:
  - Manual browser QA in Chrome/Safari/Firefox baseline

### Task 4.2: Browser audio capture and chunk sender
- **Location**: `static/realtime.js` (new)
- **Description**: Capture mic audio, resample to 16k mono PCM16, chunk at configurable interval, and send protocol events (`append`, `commit`, `response trigger`).
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Streaming begins within 1 second after Start
  - Chunks sent continuously without blocking UI thread
- **Validation**:
  - Browser devtools verifies WS frame cadence and payload size

### Task 4.3: Transcript state reconciliation in UI
- **Location**: `static/realtime.js`, `templates/realtime.html`
- **Description**: Implement two-layer rendering model:
  - Draft layer for partial deltas
  - Final layer replacing/locking completed utterances
- **Dependencies**: Task 4.2
- **Acceptance Criteria**:
  - No duplicated finalized lines
  - Punctuation-correct final text visually distinct from draft
- **Validation**:
  - E2E test/playback script validates replacement rules

## Sprint 5: Observability, Tests, and Docs
**Goal**: Make feature maintainable and safe to iterate.

**Demo/Validation**:
- Test suite covers protocol + UI smoke path
- README documents realtime protocol and quickstart

### Task 5.1: Add realtime metrics and logging
- **Location**: `realtime/metrics.py` (new), `app.py`/`asgi_app.py`
- **Description**: Track connect count, audio buffered ms, first-partial latency, finalization latency, and error codes.
- **Dependencies**: Sprint 4 complete
- **Acceptance Criteria**:
  - Metrics visible via existing `/metrics` or new realtime metrics endpoint
- **Validation**:
  - Metrics change during live session

### Task 5.2: Automated tests
- **Location**: `tests/test_realtime_protocol.py` (new), `tests/test_realtime_integration.py` (new), existing test files
- **Description**: Add unit + integration tests for event sequencing, partial/final emissions, disconnect behavior, and buffer limits.
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Core realtime flow tested in CI-compatible mode
- **Validation**:
  - Test run passes locally and in container

### Task 5.3: README + protocol docs
- **Location**: `README.md`, `docs/realtime-protocol.md` (new)
- **Description**: Document websocket endpoint, event contract, browser demo setup, and differences/scope vs full OpenAI hosted Realtime behavior.
- **Dependencies**: Task 5.2
- **Acceptance Criteria**:
  - User can run demo from docs in <10 minutes
  - Protocol examples included (connect, append, partial, final)
- **Validation**:
  - Fresh-start doc test from clean environment

## Testing Strategy
- Unit tests for protocol validation, ring buffer behavior, and post-processing.
- Integration tests with websocket client to verify event order and lifecycle.
- Browser manual QA for mic permission, start/stop cycles, reconnect, and transcript replacement.
- Performance checks:
  - Time to first partial text
  - Time to final sentence after pause
  - Memory stability over 10+ minute session

## Potential Risks & Gotchas
- Protocol drift risk: OpenAI Realtime evolves; strict naming/payload compatibility must be version-pinned.
- Flask/Waitress mismatch: Waitress cannot serve native websocket realtime endpoint; ASGI runtime is required for this feature path.
- Audio frontend complexity: browser resampling/PCM conversion quality affects model output and latency.
- Final correction quality: punctuation/context second pass can over-correct proper nouns or acronyms.
- Latency variance: CPU load and chunk size tuning may push first partial above target in lower-end hardware.

## Confirmed Product Decisions
- Protocol: OpenAI-compatible subset with local extensions (not strict full parity).
- Finalization pass v1: deterministic rule-based punctuation/casing only.
- Context scope default: revise current utterance only.
- Optional setting: `context_utterances` to include N previous utterances for final-pass context (default `0`).

## Rollback Plan
- Keep existing batch transcription endpoint unchanged behind current route paths.
- Feature-flag realtime endpoints (`REALTIME_ENABLED=false` default until stabilized).
- If realtime instability is detected, disable realtime routes and keep REST-only service active.
- Preserve previous startup command as fallback in docs until ASGI migration is fully validated.
