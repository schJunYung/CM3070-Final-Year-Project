from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from backend.config import (
    IMAGE_DEFAULT_MODEL,
    IMAGE_MODEL_ROOT,
    STT_MODEL,
    STT_MODEL_DIR,
)
from backend.continuity_service import (
    MODEL_ID as NLI_MODEL_ID,
    MODEL_PATH as NLI_MODEL_DIR,
)
from backend.image_service import get_model_spec
from backend.similarity import EMBEDDING_MODEL


PRIMARY_WRITER_MODEL = "qwen2.5:1.5b"
FALLBACK_WRITER_MODEL = "qwen3:1.7b"

OLLAMA_MODELS = (
    PRIMARY_WRITER_MODEL,
    FALLBACK_WRITER_MODEL,
    EMBEDDING_MODEL,
)

IMAGE_SPEC = get_model_spec(IMAGE_DEFAULT_MODEL)

IMAGE_MODEL_ID = IMAGE_SPEC.model_id

IMAGE_MODEL_DIR = (IMAGE_MODEL_ROOT / IMAGE_SPEC.local_dir_name)

COMPONENTS = (
    "ollama",
    "nli",
    "stt",
    "image",
)


@dataclass(frozen=True)
class ComponentStatus:
    name: str
    ready: bool
    detail: str


def _run(command: list[str]) -> None:
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _ollama_executable() -> str | None:
    return shutil.which("ollama")


def _installed_ollama_models() -> set[str]:
    executable = _ollama_executable()
    if not executable:
        return set()

    result = subprocess.run(
        [executable, "list"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return set()

    models: set[str] = set()

    for line in result.stdout.splitlines()[1:]:
        columns = line.split()
        if columns:
            models.add(columns[0].strip())

    return models


def _nli_cached() -> bool:
    if not NLI_MODEL_DIR.is_dir():
        return False

    has_config = (NLI_MODEL_DIR / "config.json").is_file()
    has_weights = any(
        (NLI_MODEL_DIR / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_weights


def _stt_cached() -> bool:
    return (
        STT_MODEL_DIR.is_dir()
        and any(STT_MODEL_DIR.rglob("model.bin"))
    )


def _image_cached() -> bool:
    return (
        IMAGE_MODEL_DIR.is_dir()
        and (IMAGE_MODEL_DIR / "model_index.json").is_file()
        and any(IMAGE_MODEL_DIR.rglob("*.xml"))
    )


def status_ollama() -> ComponentStatus:
    executable = _ollama_executable()

    if not executable:
        return ComponentStatus(
            "ollama",
            False,
            "Ollama executable is not on PATH.",
        )

    installed = _installed_ollama_models()
    missing = [
        model
        for model in OLLAMA_MODELS
        if model not in installed
    ]

    if missing:
        return ComponentStatus(
            "ollama",
            False,
            "Missing: " + ", ".join(missing),
        )

    return ComponentStatus(
        "ollama",
        True,
        "Required writer/fallback/embedding models are installed.",
    )


def status_nli() -> ComponentStatus:
    if not _module_available("sentence_transformers"):
        return ComponentStatus(
            "nli",
            False,
            "sentence-transformers is not installed.",
        )

    return ComponentStatus(
        "nli",
        _nli_cached(),
        (
            f"Cached at {NLI_MODEL_DIR}"
            if _nli_cached()
            else f"Model not cached at {NLI_MODEL_DIR}"
        ),
    )


def status_stt() -> ComponentStatus:
    runtime_ready = _module_available("faster_whisper")

    model_ready = _stt_cached()

    ready = (runtime_ready and model_ready)

    detail = (
        f"runtime={'ready' if runtime_ready else 'missing'}; "
        f"model={'cached' if model_ready else 'missing'}"
    )

    if model_ready:
        detail += (f" at {STT_MODEL_DIR}")

    return ComponentStatus(
        "stt",
        ready,
        detail,
    )


def status_image() -> ComponentStatus:
    required_modules = ("openvino", "openvino_genai", "PIL")
    missing = [
        name
        for name in required_modules
        if not _module_available(name)
    ]

    if missing:
        return ComponentStatus(
            "image",
            False,
            "Missing Python package(s): " + ", ".join(missing),
        )

    return ComponentStatus(
        "image",
        _image_cached(),
        (
            f"Cached at {IMAGE_MODEL_DIR}"
            if _image_cached()
            else f"Model not cached at {IMAGE_MODEL_DIR}"
        ),
    )


STATUS_FUNCTIONS: dict[str, Callable[[], ComponentStatus]] = {
    "ollama": status_ollama,
    "nli": status_nli,
    "stt": status_stt,
    "image": status_image,
}


def install_ollama() -> None:
    executable = _ollama_executable()

    if not executable:
        raise RuntimeError(
            "Ollama is not installed or is not on PATH. "
            "Install Ollama, restart PowerShell, then rerun this script."
        )

    installed = _installed_ollama_models()

    for model in OLLAMA_MODELS:
        if model in installed:
            print(f"[READY] {model}")
            continue

        _run([executable, "pull", model])


def install_nli() -> None:
    if not _module_available("sentence_transformers"):
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install requirements.txt first."
        )

    if _nli_cached():
        print(f"[READY] NLI model: {NLI_MODEL_DIR}")
        return

    from sentence_transformers import CrossEncoder

    NLI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] {NLI_MODEL_ID}")
    model = CrossEncoder(NLI_MODEL_ID, device="cpu")
    model.save_pretrained(
        str(NLI_MODEL_DIR),
        model_name="SekAI DeBERTa NLI continuity reviewer",
        create_model_card=False,
        safe_serialization=True,
    )


def install_stt() -> None:
    if not _module_available("faster_whisper"):
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Install requirements.txt first."
        )

    if _stt_cached():
        print(f"[READY] STT model: {STT_MODEL_DIR}")
        return

    from faster_whisper import download_model

    STT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] faster-whisper {STT_MODEL}")
    path = download_model(
        STT_MODEL,
        cache_dir=str(STT_MODEL_DIR),
        local_files_only=False,
    )
    print(f"[READY] STT model cached at {path}")


def install_image() -> None:
    required_modules = (
        "openvino",
        "openvino_genai",
        "PIL",
        "huggingface_hub",
    )

    missing = [
        module
        for module in required_modules
        if not _module_available(
            module
        )
    ]

    if missing:
        raise RuntimeError(
            "Image-generation dependencies are missing: "
            f"{', '.join(missing)}. "
            "Install requirements.txt first."
        )

    if _image_cached():
        print(
            f"[READY] Image model: "
            f"{IMAGE_MODEL_DIR}"
        )
        return

    from huggingface_hub import (snapshot_download)

    IMAGE_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[DOWNLOAD] {IMAGE_MODEL_ID}")

    snapshot_download(
        repo_id=IMAGE_MODEL_ID,
        local_dir=str(IMAGE_MODEL_DIR),
    )


INSTALL_FUNCTIONS: dict[str, Callable[[], None]] = {
    "ollama": install_ollama,
    "nli": install_nli,
    "stt": install_stt,
    "image": install_image,
}


def print_status(components: list[str]) -> bool:
    all_ready = True

    print("\nSekAI model status")
    print("-" * 72)

    for component in components:
        status = STATUS_FUNCTIONS[component]()
        marker = "READY" if status.ready else "MISSING"
        print(f"[{marker:7}] {component:7} {status.detail}")
        all_ready = all_ready and status.ready

    print("-" * 72)
    return all_ready


def setup_components(components: list[str]) -> None:
    for component in components:
        status = STATUS_FUNCTIONS[component]()

        if status.ready:
            print(f"[READY] {component}: {status.detail}")
            continue

        print(f"\n[SETUP] {component}")
        INSTALL_FUNCTIONS[component]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install or check SekAI's required local models. "
            "The script is idempotent: cached models are skipped."
        )
    )

    parser.add_argument(
        "action",
        nargs="?",
        choices=("setup", "check"),
        default="setup",
        help="setup missing models or only check status (default: setup)",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        choices=COMPONENTS,
        default=list(COMPONENTS),
        metavar="COMPONENT",
        help=(
            "Limit the operation to selected components: "
            "ollama, nli, stt, image."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    components = list(dict.fromkeys(args.only))

    if args.action == "check":
        return 0 if print_status(components) else 1

    print("=" * 72)
    print("SekAI local model setup")
    print("=" * 72)

    try:
        setup_components(components)
    except (
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        print_status(components)
        return 1

    ready = print_status(components)

    if ready:
        print("\n[READY] Model setup complete.")
        print(
            r"Start SekAI with: "
            r".\.venv\Scripts\python.exe .\run_app.py"
        )
        return 0

    print(
        "\n[ERROR] One or more components are still unavailable.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
