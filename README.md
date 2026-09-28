# SekAI Story Workspace

SekAI is a local-first branching narrative workspace for tabletop RPG Game Masters. It combines manual authoring with local AI generation, branch-specific structured memory, semantic retrieval, NLI continuity checking, speech-to-text input, and optional local scene-image generation.

After the Python dependencies and models have been downloaded once, the main application is intended to run locally.

## Quick start

### First setup

Prerequisites:

- Python with the Windows `py` launcher
- Ollama
- Windows PowerShell
- Internet access for the initial package/model downloads

From the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1 -Dev
```

`-Dev` installs the runtime dependencies plus pytest. Omit it for a runtime-only installation:

```powershell
.\setup.ps1
```

The setup process:

1. creates `.venv` when missing;
2. upgrades pip;
3. installs the Python dependencies;
4. runs `pip check`;
5. installs only missing Ollama models;
6. caches the DeBERTa NLI model;
7. caches the faster-whisper model;
8. caches the preconverted OpenVINO Stable Diffusion model.

### Normal startup

```powershell
.\.venv\Scripts\python.exe .\run_app.py
```

Then open:

- Application: `http://127.0.0.1:8000`
- API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/api/health`

Stop with `Ctrl+C`.

---

## Dependency files

### `requirements.txt`

Contains every Python package required by the application at runtime:

- FastAPI / Uvicorn
- HTTPX
- Pydantic
- multipart upload support
- Sentence Transformers for semantic/NLI roles
- faster-whisper
- OpenVINO GenAI
- Pillow
- Hugging Face Hub

Install manually with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

### `requirements-dev.txt`

Includes `requirements.txt` plus pytest:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-dev.txt
```

The old `requirements-nli.txt` is redundant because `sentence-transformers` is now in the main runtime requirements.

The old production `requirements-image.txt` should also be retired because it installs the Optimum-Intel/Diffusers conversion stack. The current production image path downloads an already-converted OpenVINO IR model and runs it with `openvino-genai`.

---

## Model management

All normal model setup is handled by:

```text
scripts/setup_models.py
```

### Install only missing models

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_models.py
```

Equivalent explicit form:

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_models.py setup
```

### Check model status without downloading

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_models.py check
```

### Prepare only selected components

Examples:

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_models.py setup --only nli
```

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_models.py setup --only stt image
```

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_models.py check --only ollama image
```

### Required local models

| Role | Model |
| --- | --- |
| Primary writer | `qwen2.5:1.5b` |
| Writer fallback | `qwen3:1.7b` |
| Semantic embedding | `embeddinggemma:latest` |
| Continuity NLI | `cross-encoder/nli-deberta-v3-small` |
| Speech-to-text | faster-whisper `base.en` |
| Scene image | `OpenVINO/stable-diffusion-v1-5-int8-ov` |

The setup script is idempotent: already-cached models are skipped.

---

## Manual installation equivalent

If PowerShell scripts are not desired:

```powershell
py -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install -r .\requirements-dev.txt

.\.venv\Scripts\python.exe -m pip check

.\.venv\Scripts\python.exe .\scripts\setup_models.py setup

.\.venv\Scripts\python.exe .\run_app.py
```

---

## Verify the installation

Check Python packages:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Check model caches:

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_models.py check
```

Check Python syntax:

```powershell
.\.venv\Scripts\python.exe -m compileall `
    .\backend `
    .\scripts `
    .\tests `
    .\run_app.py
```

Optional JavaScript syntax check:

```powershell
node --check .\frontend\app.js
```

Run automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Check the running application:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health |
    ConvertTo-Json -Depth 6
```

---

## Main local AI workflow

```text
GM instruction / microphone
        ↓
faster-whisper (optional)
        ↓
editable instruction
        ↓
Qwen writer
        ↓
scene + three choices + StateDelta
        ↓
deterministic validation
        ↓
EmbeddingGemma retrieval/repetition
        ↓
DeBERTa NLI continuity review
        ↓
human review
        ↓
saved branch node + memory snapshot
        ↓
optional Stable Diffusion illustration
```

The Structured Story Element Library supplies optional project-selected reference content. Explicit GM instructions and branch memory remain authoritative.

---

## Evaluation scripts

Installation and evaluation are deliberately separate.

### STT benchmark

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_stt.py `
    .\evaluation\stt_dataset `
    --output .\evaluation\stt_benchmark.csv
```

The benchmark records latency, real-time factor and WER when matching reference `.txt` files are available.

### Image benchmark

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_image_models.py `
    .\evaluation\image_scenes.json `
    --output .\evaluation\image_model_benchmark.csv
```

The current production default is `sd15-int8`.

### Image smoke test

```powershell
.\.venv\Scripts\python.exe .\scripts\test_image_model.py
```

This is optional because image generation may take significant time on CPU/integrated graphics.

---



## Project structure

```text
SekAI/
├── backend/
├── content/
├── frontend/
├── models/
├── scripts/
│   ├── setup_models.py
│   ├── benchmark_image_models.py
│   ├── benchmark_stt.py
│   ├── test_image_model.py
│   └── export_image_models.py      # experimental only
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── setup.ps1
└── run_app.py
```

---

## Files replaced by `setup_models.py`

After testing the consolidated setup script successfully, the following old setup-only files can be removed:

```text
requirements-nli.txt
requirements-image.txt
scripts/download_nli_model.py
scripts/download_stt_model.py
scripts/download_image_model.py
scripts/check_nli_model.py
```

Keep:

```text
scripts/benchmark_stt.py
scripts/benchmark_image_models.py
scripts/test_image_model.py
```

Keep `scripts/export_image_models.py` only as an experimental/research artifact.

Before deleting the old download scripts, update backend error messages that still tell users to run them. See `docs/MIGRATION.md`.

---

## Common problems

### Ollama is missing

```powershell
ollama --version
ollama list
```

If Ollama is installed but PowerShell cannot find it, restart the terminal.

### One model is missing

Check:

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_models.py check
```

Install only that component, for example:

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_models.py setup --only image
```

### Package conflict

```powershell
.\.venv\Scripts\python.exe -m pip check
```

### Port 8000 is already in use

Stop the previous SekAI/Uvicorn process with `Ctrl+C`. Do not run direct Uvicorn and `run_app.py` simultaneously.

---

## Everyday commands

Normal run:

```powershell
.\.venv\Scripts\python.exe .\run_app.py
```

Quick development verification:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe .\scripts\setup_models.py check
.\.venv\Scripts\python.exe -m pytest -q
```

For reproducible project evaluation, record the model name, configuration, hardware, cold/warm state, and timing metrics used for each experiment.
