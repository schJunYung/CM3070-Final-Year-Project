from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
OLLAMA_URL = "http://127.0.0.1:11434/api/tags"
APP_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{APP_URL}/api/health"


def available(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except (urllib.error.URLError, TimeoutError):
        return False


def wait_for(url: str, name: str, timeout: int = 40) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if available(url):
            print(f"[READY] {name}")
            return True
        time.sleep(0.5)
    print(f"[ERROR] {name} did not start within {timeout} seconds.")
    return False


def locate_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found

    if os.name == "nt":
        candidate = (
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs"
            / "Ollama"
            / "ollama.exe"
        )
        if candidate.exists():
            return str(candidate)

    return None


def stop_process(
    process: subprocess.Popen | None,
    name: str,
) -> None:
    if not process or process.poll() is not None:
        return

    print(f"[STOPPING] {name}")
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    python_exe = (
        VENV_PYTHON
        if VENV_PYTHON.exists()
        else Path(sys.executable)
    )

    if not VENV_PYTHON.exists():
        print("[NOTICE] No .venv detected; using the current Python interpreter.")
        print("For a repeatable setup, run: python -m venv .venv")
        print(
            r"Then: .\.venv\Scripts\python.exe "
            r"-m pip install -r requirements-dev.txt"
        )

    # Prevent a second launcher from starting another Uvicorn process on the
    # same port. The previous implementation could mistake an old server for
    # the newly launched process during its readiness check.
    if available(HEALTH_URL):
        print("[READY] SekAI is already running.")
        print(f"Application: {APP_URL}")
        webbrowser.open(APP_URL)
        return 0

    if available(APP_URL):
        print(
            "[ERROR] Port 8000 is already in use by another service. "
            "Stop that service before starting SekAI."
        )
        return 1

    ollama_process: subprocess.Popen | None = None
    app_process: subprocess.Popen | None = None

    try:
        if available(OLLAMA_URL):
            print("[READY] Ollama is already running.")
        else:
            ollama_exe = locate_ollama()
            if not ollama_exe:
                print(
                    "[WARNING] Ollama is not on PATH and the local API "
                    "is unavailable."
                )
                print(
                    "SekAI will still open for manual writing, but AI "
                    "generation will not work."
                )
            else:
                print(f"[STARTING] Ollama from {ollama_exe}")
                ollama_process = subprocess.Popen(
                    [ollama_exe, "serve"],
                    cwd=ROOT,
                )
                if not wait_for(OLLAMA_URL, "Ollama", timeout=30):
                    print(
                        "[WARNING] Continuing without local AI. "
                        "Manual writing remains available."
                    )

        print("[STARTING] SekAI FastAPI application")
        app_process = subprocess.Popen(
            [
                str(python_exe),
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=ROOT,
        )

        if not wait_for(HEALTH_URL, "SekAI", timeout=40):
            if app_process.poll() is not None:
                print(
                    "[ERROR] Uvicorn exited with code "
                    f"{app_process.returncode}."
                )
            return 1

        print("\n" + "=" * 62)
        print("SekAI Story Workspace is running")
        print(f"Application: {APP_URL}")
        print(f"API docs:   {APP_URL}/docs")
        print("Press Ctrl+C to stop the application.")
        print("=" * 62)
        webbrowser.open(APP_URL)

        while True:
            if app_process.poll() is not None:
                return app_process.returncode or 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutdown requested.")
        return 0

    finally:
        stop_process(app_process, "SekAI")
        stop_process(ollama_process, "Ollama")
        print("[STOPPED] Services closed.")


if __name__ == "__main__":
    raise SystemExit(main())
