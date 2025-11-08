<# -------------
Run-all pipeline (Windows PowerShell)
Steps:
  0) Activate venv
  1) Preprocessing         (raw -> processed/*)
  2) EDA                   (figures & tables)
  3) Feature Engineering   (+OHE, +distance-to-coast jika shapefile ada)
  4) SMOTE pipeline        (smote / smote_tomek / smoteenn)
  5) Stacking              (train + threshold tuning, opsi ablation)
  6) Experiments           (experiments/configs.yaml)
  7) Compare runs          (compare_runs + SMOTE effects)
  8) Tests (pytest)        [opsional]
  9) Interface (FastAPI)   [opsional]
-------------#>

param(
  [string[]]$Datasets = @("tectonic","volcanic"),
  [string]$CoastPath  = "data/coastline/ne_10m_coastline.shp",
  [ValidateSet("auto","yes","no")] [string]$UseSmote = "auto",
  [int]$CV = 3,
  [switch]$Ablation,
  [switch]$StartUI,
  [switch]$RunTests,
  [switch]$WithMLP,
  [switch]$Fast,
  [int]$FastN = 5000,
  [ValidateSet("none","pearson","rfe")] [string]$FeatureSelect = "none",
  [int]$TopN = 30
)

$ErrorActionPreference = "Stop"

# -------------------------------------------------------------------
# Helper untuk mencetak durasi setiap step
# -------------------------------------------------------------------
function Invoke-Step([string]$Title, [scriptblock]$Action) {
  Write-Host "`n=== $Title ===" -ForegroundColor Cyan
  $t = Get-Date
  & $Action
  $dt = (Get-Date) - $t
  Write-Host "=== Completed: $Title ($([int]$dt.TotalSeconds) sec)" -ForegroundColor DarkCyan
}

# Helper cek modul python (misal pytest)
function Test-PyModule([string]$Name) {
  try {
    & python -c "import importlib, sys; sys.exit(0 if importlib.util.find_spec('$Name') else 1)"
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

# -------------------------------------------------------------------
# Pindah ke root project (folder yang berisi .venv, src, data, dll.)
# -------------------------------------------------------------------
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $here "..")

# 0) venv
Invoke-Step "0) Activate venv" {
  if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
  } else {
    Write-Warning "Virtual env .venv tidak ditemukan (lewati aktivasi)."
  }
}

# 1) Preprocessing (raw -> processed/*)
Invoke-Step "1) Preprocessing (clean CSV)" {
  python -m tsunami_prediction.preprocessing
}

# 2) EDA (gambar & tabel ringkas)
Invoke-Step "2) EDA (figures & tables)" {
  # default: tectonic + volcanic; bisa override lewat param --datasets
  python -m tsunami_prediction.eda --datasets @Datasets
}

# 3) Feature Engineering (tambahkan distance-to-coast jika shapefile ada)
Invoke-Step "3) Feature Engineering (+OHE, +coastline bila ada)" {
  $feArgs = @("--overwrite","--materialize-ohe")
  if (Test-Path $CoastPath) {
    Write-Host "  > Coastline found: $CoastPath"
    $feArgs += @("--coast", $CoastPath)
  } else {
    Write-Warning "  > Coastline shapefile tidak ditemukan: $CoastPath (lewati penambahan jarak pantai)."
  }
  python -m tsunami_prediction.feature_engineering @feArgs
}

# 4) SMOTE splits (smote / smote_tomek / smoteenn)
Invoke-Step "4) SMOTE pipeline (smote / tomek / enn)" {
  $baseArgs = @("--datasets") + $Datasets + @("--overwrite")

  python -m tsunami_prediction.smote_pipeline @baseArgs --variant smote
  python -m tsunami_prediction.smote_pipeline @baseArgs --variant smote_tomek
  python -m tsunami_prediction.smote_pipeline @baseArgs --variant smoteenn
}

# 5) Stacking (train + threshold tuning)
Invoke-Step "5) Stacking (train + threshold tuning)" {
  $stackArgs = @(
    "--datasets") + $Datasets + @(
    "--cv", $CV,
    "--random-state", 42,
    "--feature-select", $FeatureSelect,
    "--top-n", $TopN,
    "--use-smote", $UseSmote
  )

  # meta-grid ON (XGBoost sudah tidak dipakai)
  $stackArgs += @("--meta-grid")

  if ($WithMLP) { $stackArgs += "--with-mlp" }
  if ($Fast)    { $stackArgs += @("--fast","--fast-n",$FastN) }
  if ($Ablation){ $stackArgs += "--ablation" }

  python -m tsunami_prediction.stacking_pipeline @stackArgs
}

# 6) Experiments (configs.yaml) – opsional
Invoke-Step "6) Experiments (configs.yaml)" {
  if (Test-Path "experiments\configs.yaml") {
    python experiments/run_experiment.py --config experiments/configs.yaml
  } else {
    Write-Warning "experiments/configs.yaml tidak ditemukan; lewati langkah experiment."
  }
}

# 7) Compare runs (compare_runs + SMOTE effects)
Invoke-Step "7) Compare runs + SMOTE effects" {
  try {
    # compare_runs sekarang otomatis hitung efek SMOTE (tanpa --effects)
    python -m tsunami_prediction.compare_runs
  } catch {
    Write-Warning "compare_runs gagal: $_"
  }
}

# 8) Tests (opsional, butuh pytest)
if ($RunTests) {
  Invoke-Step "8) Unit tests (pytest)" {
    if (Test-PyModule "pytest") {
      python -m pytest -q tests/tests_pipeline.py
    } else {
      Write-Warning "pytest belum terpasang; lewati tests."
    }
  }
} else {
  Write-Host "`nLewati tests (aktifkan dengan -RunTests)." -ForegroundColor DarkYellow
}

# 9) Interface (opsional)
Invoke-Step "9) Interface (FastAPI UI)" {
  if ($StartUI) {
    if (Test-Path "interface\app.py") {
      Write-Host "Menjalankan FastAPI (Ctrl+C untuk stop)..." -ForegroundColor Cyan
      python interface/app.py
    } else {
      Write-Warning "interface/app.py tidak ada; lewati UI."
    }
  } else {
    Write-Host "Lewati UI (aktifkan dengan -StartUI)." -ForegroundColor DarkYellow
    if (Test-Path "interface\app.py") {
      Write-Host "Manual start:  python interface/app.py"
    }
  }
}

Write-Host "`nSelesai. Artefak utama:" -ForegroundColor Green
Write-Host " - artifacts\*.joblib        (model + decision_threshold)"
Write-Host " - reports\figures\*         (EDA, CM/ROC/PR, threshold sweep, stacking_architecture, dsb.)"
Write-Host " - reports\tables\*          (metrics/preds/ablation, model_hyperparams, runtime_summary, EDA)"
Write-Host " - data\processed\*           (train/test + SMOTE splits)"