from __future__ import annotations

import importlib.util
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import (
    STT_BEAM_SIZE,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_ENGINE,
    STT_LANGUAGE,
    STT_MODEL,
    STT_MODEL_DIR,
)


class TranscriptionUnavailableError(RuntimeError):
    """Raised when the configured local STT runtime/model is unavailable."""


class TranscriptionError(RuntimeError):
    """Raised when uploaded audio cannot be decoded or transcribed."""


def dependency_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def model_cached() -> bool:
    if not STT_MODEL_DIR.exists():
        return False
    return any(item.is_file() for item in STT_MODEL_DIR.rglob("model.bin"))


@lru_cache(maxsize=1)
def get_model() -> Any:
    if not dependency_available():
        raise TranscriptionUnavailableError(
           "faster-whisper is not installed. "
            "Install SekAI dependencies with "
            "'.\\.venv\\Scripts\\python.exe -m pip "
            "install -r requirements.txt'."
        )

    from faster_whisper import WhisperModel

    try:
        return WhisperModel(
            STT_MODEL,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
            download_root=str(STT_MODEL_DIR),
            local_files_only=True,
        )
    except Exception as error:
        raise TranscriptionUnavailableError(
            "The local faster-whisper model is not downloaded "
            "or could not be loaded. Run "
            "'.\\.venv\\Scripts\\python.exe "
            ".\\scripts\\setup_models.py setup --only stt' "
            "once while online. "
            f"Underlying error: {error}"
        ) from error


def transcribe_file(path: str | Path) -> dict[str, Any]:
    """Transcribe one short GM recording using Whisper on CTranslate2."""
    cold_start = get_model.cache_info().currsize == 0
    load_started = time.perf_counter()
    model = get_model()
    model_load_seconds = time.perf_counter() - load_started

    inference_started = time.perf_counter()
    try:
        segments, info = model.transcribe(
            str(path),
            language=STT_LANGUAGE,
            task="transcribe",
            beam_size=STT_BEAM_SIZE,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )
        # faster-whisper returns a generator; consuming it performs inference.
        segment_list = list(segments)
    except Exception as error:
        raise TranscriptionError(
            f"faster-whisper could not decode or transcribe the audio: {error}"
        ) from error
    inference_seconds = time.perf_counter() - inference_started

    text = " ".join(
        segment.text.strip()
        for segment in segment_list
        if segment.text.strip()
    )
    if not text:
        raise TranscriptionError("faster-whisper returned an empty transcript.")

    audio_seconds = float(getattr(info, "duration", 0.0) or 0.0)
    if audio_seconds <= 0 and segment_list:
        audio_seconds = max(float(segment.end) for segment in segment_list)
    if audio_seconds <= 0:
        raise TranscriptionError("The uploaded recording has no measurable audio duration.")

    total_seconds = model_load_seconds + inference_seconds
    real_time_factor = inference_seconds / audio_seconds
    probability = getattr(info, "language_probability", None)

    return {
        "engine": STT_ENGINE,
        "model": STT_MODEL,
        "device": STT_DEVICE,
        "compute_type": STT_COMPUTE_TYPE,
        "text": text,
        "language": str(getattr(info, "language", STT_LANGUAGE) or STT_LANGUAGE),
        "language_probability": (
            round(float(probability), 4)
            if probability is not None
            else None
        ),
        "audio_seconds": round(audio_seconds, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "real_time_factor": (
            round(real_time_factor, 4)
            if math.isfinite(real_time_factor)
            else None
        ),
        "cold_start": cold_start,
    }
