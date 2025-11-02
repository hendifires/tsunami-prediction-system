<# -------------
Run-all pipeline (Windows PowerShell)
Steps: Preprocessing → EDA → FE(+coast) → SMOTE → Stacking → Experiments → CompareRuns → Tests → (opt) UI

Contoh:
  ./scripts/run_all.ps1
  ./scripts/run_all.ps1 -UseSmote yes -CV 5 -StartUI
  ./scripts/run_all.ps1 -CoastPath "data/coastline/ne_10m_coastline.shp" -Ablation
  ./scripts/run_all.ps1 -Fast -FastN 4000 -FeatureSelect none
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

function Run-Step([string]$title, [scriptblock]$block) {
  Write-Host "`n=== $title ==="
  $t = Get-Date
  & $block
  $dt = (Get-Date) - $t
  Write-Host "=== selesai: $title ($([int]$dt.TotalSeconds) sec)"
}

function Has-Module([string]$mod) {
  try { python - <<EOF
import importlib, sys
sys.exit(0 if importlib.util.find_spec("$mod") else 1)
EOF
    return $LASTEXITCODE -eq 0
  } catch { return $false }
}

# 0) venv
Run-Step "0) Activate venv" {
  if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
  } else {
    Write-Warning "Virtual env .venv tidak ditemukan (lewati aktivasi)."
  }
}

# 1) Preprocessing (raw -> processed/*)
Run-Step "1) Preprocessing (clean CSV)" {
  python -m tsunami_prediction.preprocessing
}

# 2) EDA (gambar & tabel ringkas)
Run-Step "2) EDA (figures & tables)" {
  python -m tsunami_prediction.eda
}

# 3) Feature Engineering (tambahkan distance-to-coast jika shapefile ada)
Run-Step "3) Feature Engineering (+OHE, +coastline bila ada)" {
  $feArgs = @("--overwrite","--materialize-ohe")
  if (Test-Path $CoastPath) { $feArgs += @("--coast", $CoastPath) }
  python -m tsunami_prediction.feature_engineering @feArgs
}

# 4) SMOTE split (coba yang lengkap → fallback minimal)
Run-Step "4) SMOTE splits" {
  try {
    python -m tsunami_prediction.smote_pipeline --datasets @Datasets --all
  } catch {
    Write-Warning "smote_pipeline --all gagal/tdk tersedia; fallback tanpa argumen…"
    python -m tsunami_prediction.smote_pipeline
  }
}

# 5) Stacking (train + threshold tuning)
Run-Step "5) Stacking (train + threshold tuning)" {
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
Run-Step "6) Experiments (configs.yaml)" {
  if (Test-Path "experiments\configs.yaml") {
    python experiments/run_experiment.py --config experiments/configs.yaml
  } else {
    Write-Warning "experiments/configs.yaml tidak ditemukan; lewati langkah experiment."
  }
}

# 7) Compare runs (opsional; jika script ada)
Run-Step "7) Compare runs (optional)" {
  try { python -m tsunami_prediction.compare_runs } catch { Write-Host "skip compare_runs." }
}

# 8) Tests (opsional, butuh pytest)
if ($RunTests) {
  Run-Step "8) Unit tests (pytest)" {
    if (Has-Module "pytest") {
      python -m pytest -q tests/tests_pipeline.py
    } else {
      Write-Warning "pytest belum terpasang; lewati tests."
    }
  }
} else {
  Write-Host "`nLewati tests (aktifkan dengan -RunTests)."
}

# 9) Interface (opsional)
Run-Step "9) Interface (UI)" {
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