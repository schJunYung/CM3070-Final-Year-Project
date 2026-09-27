from __future__ import annotations

import importlib.util
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import (
    IMAGE_DEFAULT_MODEL,
    IMAGE_DEVICE,
    IMAGE_HEIGHT,
    IMAGE_MODEL_ROOT,
    IMAGE_OUTPUT_DIR,
    IMAGE_WIDTH,
)


@dataclass(frozen=True)
class ImageModelSpec:
    """Configuration for one supported local image-generation model."""

    key: str
    model_id: str
    local_dir_name: str
    precision: str
    steps: int
    guidance_scale: float
    supports_negative_prompt: bool


MODEL_SPECS: dict[str, ImageModelSpec] = {
    "sd15-int8": ImageModelSpec(
        key="sd15-int8",
        model_id="OpenVINO/stable-diffusion-v1-5-int8-ov",
        local_dir_name="sd15-int8-ov",
        precision="OpenVINO INT8_ASYM weight-compressed",
        steps=20,
        guidance_scale=7.5,
        supports_negative_prompt=True,
    ),
    # Experimental only; the normal Python 3.14 setup does not install it.
    "sd-turbo": ImageModelSpec(
        key="sd-turbo",
        model_id="stabilityai/sd-turbo",
        local_dir_name="sd-turbo-openvino",
        precision="OpenVINO default",
        steps=2,
        guidance_scale=0.0,
        supports_negative_prompt=False,
    ),
}

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low detail, watermark, logo, text overlay, unreadable text, "
    "duplicate characters, malformed anatomy, extra limbs"
)


class ImageGenerationUnavailableError(RuntimeError):
    """Raised when the local image runtime or selected model is unavailable."""


class ImageGenerationError(RuntimeError):
    """Raised when an installed image model fails during inference."""


_PIPELINES: dict[tuple[str, str], Any] = {}
_PIPELINE_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()


def dependency_available() -> bool:
    """Return whether all production image-runtime packages are installed."""
    return (
        importlib.util.find_spec("openvino") is not None
        and importlib.util.find_spec("openvino_genai") is not None
        and importlib.util.find_spec("PIL") is not None
    )


def available_devices() -> list[str]:
    """Return OpenVINO devices visible to the current process."""
    if not dependency_available():
        return []

    try:
        from openvino import Core

        return list(Core().available_devices)
    except Exception:
        return []


def resolve_device(requested: str = IMAGE_DEVICE) -> str:
    """Resolve AUTO to GPU when available, otherwise fall back to CPU."""
    devices = available_devices()
    if not devices:
        raise ImageGenerationUnavailableError(
            "OpenVINO is installed but no inference device was detected."
        )

    requested_key = requested.strip().upper()

    if requested_key == "AUTO":
        if any(device.upper().startswith("GPU") for device in devices):
            return "GPU"
        if any(device.upper().startswith("CPU") for device in devices):
            return "CPU"

    if any(device.upper().startswith(requested_key) for device in devices):
        return requested_key

    if any(device.upper().startswith("CPU") for device in devices):
        return "CPU"

    raise ImageGenerationUnavailableError(
        f"Requested OpenVINO device '{requested}' is unavailable. "
        f"Detected devices: {devices}"
    )


def get_model_spec(model_key: str) -> ImageModelSpec:
    """Return the configuration associated with a model key."""
    try:
        return MODEL_SPECS[model_key]
    except KeyError as error:
        supported = ", ".join(MODEL_SPECS)
        raise ValueError(
            f"Unsupported image model '{model_key}'. Choose one of: {supported}."
        ) from error


def model_path(model_key: str) -> Path:
    return IMAGE_MODEL_ROOT / get_model_spec(model_key).local_dir_name


def model_cached(model_key: str = IMAGE_DEFAULT_MODEL) -> bool:
    """Return whether the preconverted OpenVINO model exists locally."""
    path = model_path(model_key)
    return (
        path.is_dir()
        and (path / "model_index.json").is_file()
        and any(path.rglob("*.xml"))
    )


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _limited_words(value: str, limit: int) -> str:
    return " ".join(_collapse(value).split()[:limit])


def _scene_excerpt(story_text: str, max_words: int = 46) -> str:
    """Retain the beginning and ending of long scenes for prompt grounding."""
    clean = _collapse(story_text)
    if not clean:
        return ""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean)
        if sentence.strip()
    ]

    if len(sentences) <= 3:
        return _limited_words(clean, max_words)

    selected = [sentences[0], sentences[1], sentences[-1]]
    return _limited_words(" ".join(selected), max_words)


def build_scene_prompt(
    *,
    title: str,
    story_text: str,
    genre: str,
    tone: str,
    memory_snapshot: dict[str, Any],
    prompt_override: str = "",
) -> str:
    """Build a deterministic prompt from reviewed scene and child-state data."""
    if prompt_override.strip():
        return _limited_words(prompt_override, 72)

    location = str(memory_snapshot.get("location", "")).strip()
    characters = [
        str(value).strip()
        for value in memory_snapshot.get("active_characters", [])
        if str(value).strip()
    ][:4]
    threat = str(memory_snapshot.get("threat", "")).strip()
    excerpt = _scene_excerpt(story_text)

    parts = [
        (
            f"{genre} tabletop RPG scene illustration"
            if genre
            else "tabletop RPG scene illustration"
        ),
        f"{tone} atmosphere" if tone else "",
        f"scene: {title}",
        (
            f"location: {location}"
            if location and location.casefold() != "unknown"
            else ""
        ),
        f"characters: {', '.join(characters)}" if characters else "",
        f"threat level: {threat}" if threat else "",
        f"key events: {excerpt}",
        (
            "cinematic wide composition, environmental storytelling, "
            "coherent scene, no interface, no written text"
        ),
    ]

    return _limited_words(", ".join(part for part in parts if part), 72)


def _load_pipeline(
    model_key: str,
    device: str,
) -> tuple[Any, float, bool]:
    """Load and cache one OpenVINO GenAI text-to-image pipeline."""
    cache_key = (model_key, device)

    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key], 0.0, False

        if not dependency_available():
            raise ImageGenerationUnavailableError(
                "The OpenVINO GenAI image dependencies are not installed."
            )

        if not model_cached(model_key):
            raise ImageGenerationUnavailableError(
                f"The local image model '{model_key}' is not installed. "
                "Run '.\\.venv\\Scripts\\python.exe "
                ".\\scripts\\setup_models.py setup --only image' "
                "once while online."
            )

        try:
            import openvino_genai as ov_genai
        except ImportError as error:
            raise ImageGenerationUnavailableError(
                "openvino-genai is not installed."
            ) from error

        started = time.perf_counter()

        try:
            pipeline = ov_genai.Text2ImagePipeline(
                str(model_path(model_key)),
                device,
            )
        except Exception as error:
            raise ImageGenerationUnavailableError(
                f"Could not load '{model_key}' on {device}: {error}"
            ) from error

        load_seconds = time.perf_counter() - started
        _PIPELINES[cache_key] = pipeline
        return pipeline, load_seconds, True


def generate_scene_image(
    *,
    project_id: str,
    source_kind: str,
    model_key: str,
    title: str,
    story_text: str,
    genre: str,
    tone: str,
    memory_snapshot: dict[str, Any],
    prompt_override: str = "",
    seed: int = 42,
) -> dict[str, Any]:
    """Generate, save, and return metadata for one scene illustration."""
    spec = get_model_spec(model_key)
    device = resolve_device()

    prompt = build_scene_prompt(
        title=title,
        story_text=story_text,
        genre=genre,
        tone=tone,
        memory_snapshot=memory_snapshot,
        prompt_override=prompt_override,
    )

    negative_prompt = (
        DEFAULT_NEGATIVE_PROMPT if spec.supports_negative_prompt else ""
    )

    with _GENERATION_LOCK:
        pipeline, load_seconds, cold_start = _load_pipeline(model_key, device)

        generation_kwargs: dict[str, Any] = {
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "num_images_per_prompt": 1,
            "num_inference_steps": spec.steps,
            "guidance_scale": spec.guidance_scale,
            "rng_seed": seed,
        }
        if negative_prompt:
            generation_kwargs["negative_prompt"] = negative_prompt

        try:
            inference_started = time.perf_counter()
            image_tensor = pipeline.generate(prompt, **generation_kwargs)
            inference_seconds = time.perf_counter() - inference_started

            if image_tensor.data.shape[0] < 1:
                raise ImageGenerationError(
                    "The image pipeline returned no image."
                )

            from PIL import Image

            image = Image.fromarray(image_tensor.data[0])
        except ImageGenerationError:
            raise
        except Exception as error:
            raise ImageGenerationError(
                f"{model_key} failed during OpenVINO GenAI inference: {error}"
            ) from error

    image_id = f"image_{uuid4().hex}"
    project_dir = IMAGE_OUTPUT_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    output_path = project_dir / f"{image_id}.png"
    image.save(output_path, format="PNG")

    total_seconds = load_seconds + inference_seconds

    return {
        "image_id": image_id,
        "source_kind": source_kind,
        "model_key": spec.key,
        "model_id": spec.model_id,
        "device": device,
        "precision": spec.precision,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "steps": spec.steps,
        "guidance_scale": spec.guidance_scale,
        "model_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "cold_start": cold_start,
        "file_path": str(output_path),
        "image_url": f"/generated-images/{project_id}/{output_path.name}",
    }
