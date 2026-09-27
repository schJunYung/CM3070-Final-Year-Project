from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import ai_service, db, image_service, services, transcription_service
from .config import (
    EVALUATION_MODE,
    PIPELINE_VERSION,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_ENGINE,
    STT_MAX_RECORDING_SECONDS,
    STT_MAX_UPLOAD_MB,
    STT_MODEL,
    IMAGE_DEFAULT_MODEL,
    IMAGE_DEVICE,
    IMAGE_OUTPUT_DIR,
    IMAGE_ENGINE,
)
from .continuity_service import MODEL_ID as NLI_MODEL_ID, MODEL_PATH as NLI_MODEL_PATH
from .similarity import EMBEDDING_MODEL
from .state import apply_state_delta

from .schemas import (
    DraftValidationRequest,
    DraftValidationResponse,
    GenerateStoryRequest,
    GenerateStoryResponse,
    SceneImageGenerationRequest,
    SceneImageGenerationResponse,
    ProjectCreateRequest,
    ProjectUpdateRequest,
    SaveNodeRequest,
    StoryElementWriteRequest,
    StoryMemory,
    TranscriptionResponse,
    TranscriptionReviewRequest,
    UpdateNodeRequest,
    NodeSceneImageRequest,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="SekAI Story Workspace API",
    version="0.4.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    installed_models: list[str] = []
    ollama_available = False

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{ai_service.OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            installed_models = [
                model["name"]
                for model in response.json().get("models", [])
                if model.get("name")
            ]
            ollama_available = True
    except (httpx.HTTPError, ValueError):
        pass

    embedding_available = EMBEDDING_MODEL in installed_models

    nli_available = (
        NLI_MODEL_PATH.is_dir()
        and any(
            item.is_file() and not item.name.startswith(".")
            for item in NLI_MODEL_PATH.iterdir()
        )
    )

    return {
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "ollama_url": ai_service.OLLAMA_BASE_URL,
        "ollama_available": ollama_available,
        "installed_models": installed_models,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_available": embedding_available,
        "continuity_model": NLI_MODEL_ID,
        "continuity_available": nli_available,
        "specialists_ready": embedding_available and nli_available,
        "stt_engine": STT_ENGINE,
        "stt_model": STT_MODEL,
        "stt_device": STT_DEVICE,
        "stt_compute_type": STT_COMPUTE_TYPE,
        "stt_dependency_available": transcription_service.dependency_available(),
        "stt_model_cached": transcription_service.model_cached(),
        "stt_max_recording_seconds": STT_MAX_RECORDING_SECONDS,
        "image_engine": IMAGE_ENGINE,
        "image_default_model": IMAGE_DEFAULT_MODEL,
        "image_device_requested": IMAGE_DEVICE,
        "image_dependency_available": image_service.dependency_available(),
        "image_available_devices": image_service.available_devices(),
        "image_default_model_cached": image_service.model_cached(
            IMAGE_DEFAULT_MODEL
        ),
        "database": str(db.DB_PATH),
    }



AUDIO_SUFFIXES = {
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
}


def _audio_suffix(content_type: str | None) -> str:
    base = (content_type or "").split(";", 1)[0].strip().casefold()
    return AUDIO_SUFFIXES.get(base, ".webm")


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile,
    project_id: str | None = Form(default=None),
    target_field: str = Form(default="player_input"),
) -> TranscriptionResponse:
    """Transcribe a short microphone recording locally and log evaluation metrics."""
    if target_field not in {"manual_scene", "player_input", "gm_instruction"}:
        raise HTTPException(
            status_code=422,
            detail=("target_field must be 'manual_scene', 'player_input' " "or 'gm_instruction'."),
        )
    if project_id is not None and not db.get_project(project_id):
        raise HTTPException(status_code=404, detail="Story project not found.")

    content_type = file.content_type or "application/octet-stream"
    base_content_type = content_type.split(";", 1)[0].casefold()
    if not (
        base_content_type.startswith("audio/")
        or base_content_type == "video/webm"
        or base_content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=415, detail="Upload an audio recording.")

    max_bytes = STT_MAX_UPLOAD_MB * 1024 * 1024
    audio_bytes = await file.read(max_bytes + 1)
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="The recording is empty.")
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Recording exceeds the {STT_MAX_UPLOAD_MB} MB limit.",
        )

    suffix = _audio_suffix(content_type)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(audio_bytes)
            temporary_path = handle.name

        result = await asyncio.to_thread(
            transcription_service.transcribe_file,
            temporary_path,
        )
    except transcription_service.TranscriptionUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except transcription_service.TranscriptionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)

    if result["audio_seconds"] > STT_MAX_RECORDING_SECONDS + 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Recording is {result['audio_seconds']:.1f}s; the SekAI voice "
                f"input limit is {STT_MAX_RECORDING_SECONDS}s."
            ),
        )

    log_id = db.insert_transcription_log(
        project_id=project_id,
        engine=result["engine"],
        model=result["model"],
        device=result["device"],
        compute_type=result["compute_type"],
        target_field=target_field,
        audio_content_type=content_type,
        audio_bytes=len(audio_bytes),
        audio_seconds=result["audio_seconds"],
        model_load_seconds=result["model_load_seconds"],
        inference_seconds=result["inference_seconds"],
        total_seconds=result["total_seconds"],
        real_time_factor=result["real_time_factor"],
        cold_start=result["cold_start"],
        language=result["language"],
        language_probability=result["language_probability"],
        transcript=result["text"],
    )

    return TranscriptionResponse(
        transcription_log_id=log_id,
        target_field=target_field,
        audio_bytes=len(audio_bytes),
        **result,
    )


@app.patch("/api/transcriptions/{log_id}")
def review_transcription(
    log_id: int,
    request: TranscriptionReviewRequest,
) -> dict[str, Any]:
    """Record the GM-edited text used after STT, without changing the transcript."""
    if not db.review_transcription_log(log_id, request.reviewed_text):
        raise HTTPException(status_code=404, detail="Transcription log not found.")
    return {"reviewed": True}


@app.get("/api/projects/{project_id}/transcriptions")
def transcription_logs(project_id: str) -> list[dict[str, Any]]:
    """Expose STT measurements for controlled project evaluation."""
    if not db.get_project(project_id):
        raise HTTPException(status_code=404, detail="Story project not found.")
    return db.get_transcription_logs(project_id)

@app.post(
    "/api/nodes/{node_id}/scene-image",
)
async def generate_node_scene_image(
    node_id: str,
    request: NodeSceneImageRequest,
) -> dict[str, Any]:
    """
    Generate or replace the illustration for an existing story node.
    """
    node = db.get_node(node_id)

    if not node:
        raise HTTPException(
            status_code=404,
            detail="Story node not found.",
        )

    project = db.get_project(node["project_id"])

    if not project:
        raise HTTPException(
            status_code=404,
            detail="Story project not found.",
        )

    try:
        result = await asyncio.to_thread(
            image_service.generate_scene_image,
            project_id=node["project_id"],
            source_kind=node["authoring_mode"],
            model_key=request.model_key,
            title=node["title"],
            story_text=node["story_text"],
            genre=project["genre"],
            tone=project["tone"],
            memory_snapshot=node["memory_snapshot"],
            prompt_override=request.prompt_override,
            seed=request.seed,
        )

    except image_service.ImageGenerationUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except (
        image_service.ImageGenerationError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    log_id = db.insert_image_generation_log(
        project_id=node["project_id"],
        **result,
    )

    updated_node = (
        db.replace_node_scene_image(
            node_id,
            log_id,
        )
    )

    return {
        "image_generation_log_id": log_id,
        "image": db.get_image_generation_log(log_id),
        "node": updated_node,
        "state": db.get_state(node["project_id"]),
    }


@app.post(
    "/api/generate-scene-image",
    response_model=SceneImageGenerationResponse,
)
async def generate_scene_image(
    request: SceneImageGenerationRequest,
) -> SceneImageGenerationResponse:
    """Generate and log one local illustration for a proposed next scene."""
    project = db.get_project(request.project_id)
    branch = db.get_branch(request.branch_id)
    parent = db.get_node(request.parent_node_id)

    if not project:
        raise HTTPException(status_code=404, detail="Story project not found.")

    if not branch or branch["project_id"] != request.project_id:
        raise HTTPException(status_code=400, detail="Selected branch is invalid.")

    if not parent or parent["project_id"] != request.project_id:
        raise HTTPException(
            status_code=400,
            detail="Selected parent node is invalid.",
        )

    branch_path_ids = {
        node["id"]
        for node in db.get_branch_path(branch["id"])
    }
    if parent["id"] not in branch_path_ids:
        raise HTTPException(
            status_code=400,
            detail="The selected parent node is not on the selected branch path.",
        )

    parent_memory = StoryMemory.model_validate(parent["memory_snapshot"])
    proposed_memory = apply_state_delta(
        parent_memory,
        request.applied_delta,
    )

    try:
        result = await asyncio.to_thread(
            image_service.generate_scene_image,
            project_id=request.project_id,
            source_kind=request.source_kind,
            model_key=request.model_key,
            title=request.title,
            story_text=request.story_text,
            genre=project["genre"],
            tone=project["tone"],
            memory_snapshot=proposed_memory.model_dump(),
            prompt_override=request.prompt_override,
            seed=request.seed,
        )
    except image_service.ImageGenerationUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (image_service.ImageGenerationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    log_id = db.insert_image_generation_log(
        project_id=request.project_id,
        **result,
    )

    return SceneImageGenerationResponse(
        image_generation_log_id=log_id,
        **result,
    )


@app.get("/api/projects/{project_id}/image-generations")
def image_generation_logs(
    project_id: str,
) -> list[dict[str, Any]]:
    """Expose image-model timing and selection evidence for evaluation."""
    if not db.get_project(project_id):
        raise HTTPException(
            status_code=404,
            detail="Story project not found.",
        )
    return db.get_image_generation_logs(project_id)


@app.get("/api/story-elements")
def list_story_elements() -> list[dict[str, Any]]:
    """Return the active editable Story Element Library from SQLite."""
    return db.list_story_elements()


@app.post("/api/story-elements")
def create_story_element(
    request: StoryElementWriteRequest,
) -> dict[str, Any]:
    """Create one reusable user-defined Story Element."""
    try:
        return db.create_story_element(request)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post("/api/story-elements/reset-defaults")
def reset_story_element_defaults() -> dict[str, Any]:
    """Restore every bundled Story Element to its original factory values."""
    try:
        return {
            "reset": True,
            "elements": db.reset_story_element_defaults(),
        }
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post("/api/story-elements/{element_id}/reset")
def reset_story_element(
    element_id: str,
) -> dict[str, Any]:
    """Restore one bundled Story Element to its factory value."""
    try:
        elements = db.reset_story_element_defaults(
            element_id
        )
        return {
            "reset": True,
            "element": (
                elements[0]
                if elements
                else None
            ),
        }
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.patch("/api/story-elements/{element_id}")
def update_story_element(
    element_id: str,
    request: StoryElementWriteRequest,
) -> dict[str, Any]:
    """Edit a built-in or user-created Story Element."""
    try:
        return db.update_story_element(
            element_id,
            request,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@app.delete("/api/story-elements/{element_id}")
def delete_story_element(
    element_id: str,
) -> dict[str, Any]:
    """
    Hide a Story Element from the active library.

    Built-in elements can later be restored with reset; user-created elements
    remain soft-deleted so historical project references are not corrupted.
    """
    deleted = db.delete_story_element(
        element_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Story element not found.",
        )

    return {"deleted": True}


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    return db.list_projects()


@app.post("/api/projects")
def create_project(request: ProjectCreateRequest) -> dict[str, Any]:
    return db.create_project(request)


@app.get("/api/projects/{project_id}")
def get_project_state(project_id: str) -> dict[str, Any]:
    try:
        return db.get_state(project_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.patch("/api/projects/{project_id}")
def update_project(project_id: str, request: ProjectUpdateRequest) -> dict[str, Any]:
    if not db.get_project(project_id):
        raise HTTPException(status_code=404, detail="Story project not found.")
    try:
        return db.update_project(project_id, request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, Any]:
    deleted = db.delete_project(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Story project not found.")
    return {"deleted": True}


@app.post(
    "/api/validate-ai-draft",
    response_model=DraftValidationResponse,
)
def validate_ai_draft(
    request: DraftValidationRequest,
) -> DraftValidationResponse:
    """Validate the current edited AI draft without saving it."""
    try:
        return services.validate_ai_draft(request)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post("/api/generate-story", response_model=GenerateStoryResponse)
async def generate_story(request: GenerateStoryRequest) -> GenerateStoryResponse:
    return await ai_service.generate_story(request)


@app.post("/api/nodes")
def save_node(request: SaveNodeRequest) -> dict[str, Any]:
    try:
        return services.save_story_node(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.patch("/api/nodes/{node_id}")
def update_node(node_id: str, request: UpdateNodeRequest) -> dict[str, Any]:
    if not db.get_node(node_id):
        raise HTTPException(status_code=404, detail="Story node not found.")
    try:
        db.update_node_content(
            node_id,
            request.title,
            request.story_text,
            [choice.model_dump() for choice in request.choices],
            (
                request.applied_delta.model_dump()
                if request.applied_delta is not None
                else None
            ),
        )
        node = db.get_node(node_id)

        if node is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Story node was not found "
                    "after the update."
                ),
            )

        return {
            "node": node,
            "state": db.get_state(
                node["project_id"]
            ),
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/nodes/{node_id}")
def delete_node(node_id: str) -> dict[str, Any]:
    try:
        result = db.delete_leaf_node(node_id)
        node = result["node"]
        return {
            "deleted": node,
            "batch_id": result["batch_id"],
            "branch_was_trashed": result["branch_was_trashed"],
            "state": db.get_state(node["project_id"]),
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/branches/{branch_id}")
def delete_branch(branch_id: str) -> dict[str, Any]:
    try:
        result = db.delete_branch(branch_id)
        return {
            **result,
            "state": db.get_state(result["branch"]["project_id"]),
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/projects/{project_id}/trash")
def list_trash(project_id: str) -> list[dict[str, Any]]:
    if not db.get_project(project_id):
        raise HTTPException(status_code=404, detail="Story project not found.")
    return db.list_deleted_items(project_id)


@app.post("/api/trash/{batch_id}/restore")
def restore_trash(batch_id: str) -> dict[str, Any]:
    try:
        state = db.restore_deletion_batch(batch_id)
        return {"restored": True, "state": state}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/nodes/{node_id}/path")
def full_story_path(node_id: str) -> dict[str, Any]:
    node = db.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Story node not found.")
    return {
        "project": db.get_project(node["project_id"]),
        "selected_node": node,
        "path": db.get_node_path(node_id),
    }


@app.get("/api/compare")
def compare_branches(
    left_branch_id: str = Query(...),
    right_branch_id: str = Query(...),
) -> dict[str, Any]:
    try:
        return services.compare_branches(left_branch_id, right_branch_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/projects/{project_id}/objectives")
def objectives(project_id: str) -> dict[str, Any]:
    try:
        return services.objective_evidence(project_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/about", response_class=PlainTextResponse)
def about() -> str:
    return (
        "SekAI Story Workspace: multi-story dashboard, clickable narrative map, manual writing, "
        "local-AI DO/SAY/ASK/CONTINUE generation, node-level memory snapshots, and selected-node forking."
    )


app.mount(
    "/generated-images",
    StaticFiles(directory=IMAGE_OUTPUT_DIR),
    name="generated-images",
)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
