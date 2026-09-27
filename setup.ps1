param(
    [switch]$Dev,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "SekAI setup" -ForegroundColor Cyan

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[SETUP] Creating .venv"
    py -m venv .venv
}

$Python = ".\.venv\Scripts\python.exe"
$Requirements = if ($Dev) {
    ".\requirements-dev.txt"
} else {
    ".\requirements.txt"
}

Write-Host "[SETUP] Upgrading pip"
& $Python -m pip install --upgrade pip

Write-Host "[SETUP] Installing $Requirements"
& $Python -m pip install -r $Requirements

Write-Host "[CHECK] Python dependencies"
& $Python -m pip check

if (-not $SkipModels) {
    Write-Host "[SETUP] Preparing missing local models"
    & $Python .\scripts\setup_models.py setup
}

Write-Host ""
Write-Host "[READY] SekAI setup complete." -ForegroundColor Green
Write-Host "Run:"
Write-Host "  .\.venv\Scripts\python.exe .\run_app.py"
