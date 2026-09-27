from __future__ import annotations

from pathlib import Path

from backend import config, transcription_service
from backend.schemas import TranscriptionResponse


def test_stt_configuration_is_local_cpu_candidate():
    assert config.STT_MODEL == "base.en"
    assert config.STT_DEVICE == "cpu"
    assert config.STT_BEAM_SIZE == 5
    assert config.STT_MAX_RECORDING_SECONDS == 60


def test_model_directory_is_inside_project_by_default():
    assert "models" in config.STT_MODEL_DIR.parts
    assert "stt" in config.STT_MODEL_DIR.parts


def test_transcription_service_exposes_comparable_contract():
    assert callable(transcription_service.transcribe_file)
    assert callable(transcription_service.dependency_available)
    assert hasattr(transcription_service, "get_model")


def test_transcription_response_accepts_manual_scene_target():
    payload = {
        "transcription_log_id": 1,
        "engine": "test",
        "model": "base.en",
        "device": "cpu",
        "compute_type": "test",
        "target_field": "manual_scene",
        "text": "A manually dictated scene.",
        "audio_seconds": 2.0,
        "audio_bytes": 100,
        "model_load_seconds": 0.0,
        "inference_seconds": 0.5,
        "total_seconds": 0.5,
        "real_time_factor": 0.25,
        "cold_start": False,
    }
    assert TranscriptionResponse.model_validate(payload).target_field == "manual_scene"
