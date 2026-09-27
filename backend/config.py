from __future__ import annotations

import os
from pathlib import Path


PIPELINE_VERSION = "v9-stt-image-openvino"

EVALUATION_MODE = os.getenv(
    "SEKAI_EVALUATION_MODE",
    "0",
).strip().casefold() in {"1", "true", "yes", "on"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Local speech-to-text.
STT_ENGINE = "faster-whisper"
STT_MODEL = os.getenv("SEKAI_STT_MODEL", "base.en").strip() or "base.en"
STT_LANGUAGE = os.getenv("SEKAI_STT_LANGUAGE", "en").strip() or "en"
STT_DEVICE = os.getenv("SEKAI_STT_DEVICE", "cpu").strip() or "cpu"
STT_COMPUTE_TYPE = (
    os.getenv("SEKAI_STT_COMPUTE_TYPE", "int8").strip() or "int8"
)
STT_BEAM_SIZE = int(os.getenv("SEKAI_STT_BEAM_SIZE", "5"))
STT_MAX_UPLOAD_MB = int(os.getenv("SEKAI_STT_MAX_UPLOAD_MB", "15"))
STT_MAX_RECORDING_SECONDS = int(
    os.getenv("SEKAI_STT_MAX_RECORDING_SECONDS", "60")
)

STT_MODEL_DIR = Path(
    os.getenv(
        "SEKAI_STT_MODEL_DIR",
        str(PROJECT_ROOT / "models" / "stt" / STT_ENGINE),
    )
).expanduser()
STT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Local scene-image generation.
IMAGE_ENGINE = "openvino-genai"
IMAGE_DEFAULT_MODEL = (
    os.getenv("SEKAI_IMAGE_MODEL", "sd15-int8").strip() or "sd15-int8"
)
IMAGE_DEVICE = os.getenv("SEKAI_IMAGE_DEVICE", "AUTO").strip() or "AUTO"
IMAGE_WIDTH = int(os.getenv("SEKAI_IMAGE_WIDTH", "512"))
IMAGE_HEIGHT = int(os.getenv("SEKAI_IMAGE_HEIGHT", "512"))

IMAGE_MODEL_ROOT = Path(
    os.getenv(
        "SEKAI_IMAGE_MODEL_ROOT",
        str(PROJECT_ROOT / "models" / "image"),
    )
).expanduser()
IMAGE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_OUTPUT_DIR = Path(
    os.getenv(
        "SEKAI_IMAGE_OUTPUT_DIR",
        str(PROJECT_ROOT / "generated_images"),
    )
).expanduser()
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
