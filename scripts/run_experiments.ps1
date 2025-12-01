<#
  Jalankan hanya eksperimen utama (preproc + FE + SMOTE + stacking)
  sesuai konfigurasi di experiments\configs.yaml
#>

param(
  [string]$ConfigPath = "experiments\configs.yaml"
)

$ErrorActionPreference = "Stop"

# Pindah ke root project
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Tentukan interpreter Python
$PythonExe = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
  Write-Warning "Venv python tidak ditemukan, fallback ke 'python' di PATH."
  $PythonExe = "python"
}

# Aktifkan venv jika ada
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
  . .\.venv\Scripts\Activate.ps1
}

# Jalankan runner eksperimen
$runnerPath = Join-Path $PWD "experiments\run_experiment.py"
if (-not (Test-Path $runnerPath)) {
  throw "experiments\run_experiment.py tidak ditemukan di root project."
}

Write-Host "== Run experiment pipeline ==" -ForegroundColor Cyan
& $PythonExe $runnerPath --config $ConfigPath