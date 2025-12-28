from __future__ import annotations
import asyncio
import shutil
import tempfile
import datetime
import json
from pathlib import Path
from collections import defaultdict

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status, Request, Form
from fastapi.responses import Response

from .audio import ensure_mono_16k, schedule_cleanup
from .model import _to_builtin
from .schemas import TranscriptionResponse
from .config import logger, MODEL_NAMES

from parakeet_service.model import reset_fast_path, load_model_for_language
from parakeet_service.chunker import vad_chunk_lowmem, vad_chunk_streaming


router = APIRouter(tags=["speech"])


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,ms"""
    delta = datetime.timedelta(seconds=seconds)
    # Format as 0:00:05.123000
    s = str(delta)
    # Split seconds and microseconds
    if '.' in s:
        parts = s.split('.')
        integer_part = parts[0]
        fractional_part = parts[1][:3]  # Take first three digits for milliseconds
    else:
        integer_part = s
        fractional_part = "000"

    # Pad hour position
    if len(integer_part.split(':')) == 2:
        integer_part = "0:" + integer_part
    
    return f"{integer_part},{fractional_part}"


def segments_to_srt(segments: list) -> str:
    """Convert NeMo segment timestamps to SRT format string"""
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment['start'])
        end_time = format_srt_time(segment['end'])
        text = segment.get('segment', segment.get('text', '')).strip()
        
        if text:  # Only add content if not empty
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")  # Empty line separator
            
    return "\n".join(srt_content)


@router.get("/healthz", summary="Liveness/readiness probe")
def health():
    return {"status": "ok"}


@router.post(
    "/transcribe",
    summary="Transcribe an audio file",
)
@router.post(
    "/v1/audio/transcriptions",
    summary="Transcribe an audio file (OpenAI compatible)",
)
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., media_type="audio/*"),
    include_timestamps: bool = Form(
        False, description="Return char/word/segment offsets",
    ),
    should_chunk: bool = Form(True,
        description="If true (default), split long audio into "
                    "~60s VAD-aligned chunks for batching"),
    model: str = Form("parakeet", description="Model to use: 'parakeet' or 'parakeet_srt_words' for SRT with word timestamps"),
    prompt: str = Form("en", description="Language code (e.g., 'en' for English, 'ja' for Japanese)"),
    response_format: str = Form("json", description="Response format: 'json' or 'srt'"),
):
    # Create temp file with appropriate extension
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
    
    # Stream upload directly to processing with cancellation handling
    try:
        # Use FFmpeg for MP3 files to fix header issues
        # Create temp MP3 file if needed
        mp3_tmp_path = None
        if suffix.lower() == ".mp3":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp:
                mp3_tmp_path = Path(mp3_tmp.name)
            # Write entire MP3 to temporary file
            with open(mp3_tmp_path, "wb") as f:
                while True:
                    try:
                        chunk = await file.read(8192)
                    except asyncio.CancelledError:
                        logger.warning("File upload cancelled during MP3 saving")
                        raise
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Update FFmpeg command to read from file
            ffmpeg_cmd = [
                "ffmpeg", "-v", "error", "-nostdin", "-y",
                "-i", str(mp3_tmp_path),
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                "-f", "wav", str(tmp_path)
            ]
        else:
            ffmpeg_cmd = None
            # For non-MP3, stream directly to file
            with open(tmp_path, "wb") as f:
                while True:
                    try:
                        chunk = await file.read(8192)
                    except asyncio.CancelledError:
                        logger.warning("File upload cancelled during processing")
                        raise
                    if not chunk:
                        break
                    f.write(chunk)
        
        # Run FFmpeg if processing MP3
        if ffmpeg_cmd:
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.DEVNULL,  # We don't need stdout
                stderr=asyncio.subprocess.PIPE
            )
            
            # Read stderr in real-time
            stderr_lines = []
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                stderr_lines.append(line_str)
                logger.debug(f"FFmpeg: {line_str}")
            
            # Wait for process to finish
            return_code = await process.wait()
            stderr_str = "\n".join(stderr_lines)
            
            if return_code != 0:
                logger.error(f"FFmpeg failed with return code {return_code}")
                logger.error(f"FFmpeg error output: {stderr_str}")
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Invalid audio format: {stderr_str[:200]}"
                )
            else:
                logger.debug(f"FFmpeg completed successfully")
    except asyncio.CancelledError:
        # Clean up temporary files if processing was cancelled
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise
    except BrokenPipeError:
        logger.error("FFmpeg process terminated unexpectedly")
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio processing failed due to FFmpeg crash"
        )
    finally:
        await file.close()

    # Process audio to ensure mono 16kHz
    original, to_model = ensure_mono_16k(tmp_path)

    if should_chunk:
        # Use low-memory chunker for non-streaming requests
      chunk_paths = vad_chunk_lowmem(to_model) or [to_model]
    else:
        chunk_paths = [to_model]

    logger.info("transcribe(): sending %d chunks to ASR", len(chunk_paths))

    # Clean up all temporary files
    cleanup_files = [original, to_model] + chunk_paths
    if mp3_tmp_path:
        cleanup_files.append(mp3_tmp_path)
    schedule_cleanup(background_tasks, *cleanup_files)

    # 2 – run ASR with language-specific model
    # Load the appropriate model for the language
    asr_model = load_model_for_language(prompt)
    
    # Determine if we need timestamps (for SRT format or include_timestamps flag)
    need_timestamps = include_timestamps or response_format == "srt" or model == "parakeet_srt_words"

    try:
        outs = asr_model.transcribe(
            [str(p) for p in chunk_paths],
            batch_size=2,
            timestamps=need_timestamps,
        )
        if (
          not need_timestamps                     # switch back to model fast-path if timestamps turned off
          and getattr(asr_model.cfg.decoding, "compute_timestamps", False)
        ):
          reset_fast_path(asr_model)                    
    except RuntimeError as exc:
        logger.exception("ASR failed")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    if isinstance(outs, tuple):
      outs = outs[0]
    texts = []
    ts_agg = [] if need_timestamps else None
    merged = defaultdict(list)

    for h in outs:
        texts.append(getattr(h, "text", str(h)))
        if need_timestamps:
            for k, v in _to_builtin(getattr(h, "timestamp", {})).items():
                merged[k].extend(v)           # concat lists

    merged_text = " ".join(texts).strip()
    timestamps  = dict(merged) if need_timestamps else None

    # Return SRT format if requested
    if response_format == "srt" or model == "parakeet_srt_words":
        # Convert to SRT format
        segments = timestamps.get('segment', []) if timestamps else []
        srt_result = segments_to_srt(segments)
        
        # Add word-level timestamps if requested
        if model == "parakeet_srt_words" and timestamps:
            words = timestamps.get('word', [])
            json_str_list = [
                {"start": it['start'], "end": it['end'], "word": it.get('word', it.get('text', ''))} 
                for it in words
            ]
            srt_result += "----..----" + json.dumps(json_str_list)
        
        return Response(srt_result, media_type='text/plain')
    
    # Return JSON format (default)
    return TranscriptionResponse(text=merged_text, timestamps=timestamps)

@router.get("/debug/cfg")
def show_cfg(request: Request):
    from omegaconf import OmegaConf
    model = request.app.state.asr_model         
    yaml_str = OmegaConf.to_yaml(model.cfg, resolve=True) 
    return yaml_str