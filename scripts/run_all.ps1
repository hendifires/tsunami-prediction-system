<# -------------
Run-all pipeline (Windows PowerShell)
Steps: Preprocessing → EDA → FE(+coast) → SMOTE → Stacking → Experiments → CompareRuns → Tests → (opt) UI
---------------- #>

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

function Invoke-Step([string]$Title, [scriptblock]$Action) {
  Write-Host "`n=== $Title ==="
  $t = Get-Date
  & $Action
  $dt = (Get-Date) - $t
  Write-Host "=== Completed: $Title ($([int]$dt.TotalSeconds) sec)"
}

# ✅ pakai approved verb 'Test'
function Test-PyModule([string]$Name) {
  try {
    & python -c "import importlib, sys; sys.exit(0 if importlib.util.find_spec('$Name') else 1)"
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

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
  python -m tsunami_prediction.eda
}

# 3) Feature Engineering (tambahkan distance-to-coast jika shapefile ada)
Invoke-Step "3) Feature Engineering (+OHE, +coastline bila ada)" {
  $feArgs = @("--overwrite","--materialize-ohe")
  if (Test-Path $CoastPath) { $feArgs += @("--coast", $CoastPath) }
  python -m tsunami_prediction.feature_engineering @feArgs
}

# 4) SMOTE splits
Invoke-Task "4) SMOTE splits" {
  $Datasets = @("tectonic","volcanic")   # pastikan ada variabel ini di atas
  $smArgs = @("--datasets") + $Datasets + @("--overwrite")
  python -m tsunami_prediction.smote_pipeline @smArgs
}

# 5) Stacking (train + threshold tuning)
Invoke-Step "5) Stacking (train + threshold tuning)" {
  $stackArgs = @(
    "--datasets") + $Datasets + @(
    "--cv",$CV,
    "--with-xgb","--meta-grid",
    "--feature-select",$FeatureSelect,
    "--top-n",$TopN,
    "--use-smote",$UseSmote
  )
  if ($WithMLP) { $stackArgs += "--with-mlp" }
  if ($Fast)    { $stackArgs += @("--fast","--fast-n",$FastN) }
  if ($Ablation){ $stackArgs += "--ablation" }

  python -m tsunami_prediction.stacking_pipeline @stackArgs
}

# 6) Experiments (configs.yaml)
Invoke-Step "6) Experiments (configs.yaml)" {
  if (Test-Path "experiments\configs.yaml") {
    python experiments/run_experiment.py --config experiments/configs.yaml
  } else {
    Write-Warning "experiments/configs.yaml tidak ditemukan; lewati langkah experiment."
  }
}

# 7) Compare runs (opsional; jika script ada)
Invoke-Step "7) Compare runs (optional)" {
  try { python -m tsunami_prediction.compare_runs } catch { Write-Host "skip compare_runs." }
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
  Write-Host "`nLewati tests (aktifkan dengan -RunTests)."
}

# 9) Interface (opsional)
Invoke-Step "9) Interface (UI)" {
  if ($StartUI) {
    if (Test-Path "interface\app.py") {
      python interface/app.py
    } else {
      Write-Warning "interface/app.py tidak ada; lewati UI."
    }
  } else {
    Write-Host "Lewati UI (aktifkan dengan -StartUI)."
    if (Test-Path "interface\app.py") {
      Write-Host "Manual:  python interface/app.py"
    }
  }
}

Write-Host "`nSelesai. Artefak utama:"
Write-Host " - artifacts\*.joblib  (model + decision_threshold)"
Write-Host " - reports\figures\*   (EDA, CM/ROC/PR/threshold sweep)"
Write-Host " - reports\tables\*    (metrics/preds/ablation, ringkasan EDA)"