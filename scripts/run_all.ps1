<# -------------
Run-all pipeline (Windows PowerShell)
Usage examples:
  ./scripts/run_all.ps1
  ./scripts/run_all.ps1 -UseSmote yes -CV 5 -Ablation
  ./scripts/run_all.ps1 -CoastPath "data/coastline/ne_10m_coastline.shp"
---------------- #>

param(
  [string]$CoastPath = "data/coastline/ne_10m_coastline.shp",
  [ValidateSet("auto","yes","no")] [string]$UseSmote = "auto",
  [int]$CV = 3,
  [switch]$Ablation
)

$ErrorActionPreference = "Stop"

Write-Host "=== 0) Activate venv ==="
if (Test-Path .\.venv\Scripts\Activate.ps1) {
  . .\.venv\Scripts\Activate.ps1
} else {
  Write-Warning "Virtual env not found: .venv. (Lewati aktivasi)"
}

Write-Host "`n=== 1) Feature Engineering ==="
$feArgs = @("--overwrite","--materialize-ohe")
if (Test-Path $CoastPath) { $feArgs += @("--coast", $CoastPath) }
python -m tsunami_prediction.feature_engineering @feArgs

Write-Host "`n=== 2) SMOTE splits (try default; fallback minimal) ==="
# Coba jalankan varian lengkap (silakan biarkan saja; kalau script punya default, tidak masalah)
$LASTEXITCODE = 0
try {
  python -m tsunami_prediction.smote_pipeline --datasets tectonic volcanic --all
} catch {
  Write-Warning "smote_pipeline --all gagal/tdk ada. Coba fallback tanpa argumen…"
  python -m tsunami_prediction.smote_pipeline
}

Write-Host "`n=== 3) Stacking (train + threshold tuning) ==="
$stackArgs = @(
  "--datasets","tectonic","volcanic",
  "--cv",$CV,
  "--with-xgb","--meta-grid",
  "--feature-select","none",
  "--use-smote",$UseSmote
)
if ($Ablation) { $stackArgs += "--ablation" }
python -m tsunami_prediction.stacking_pipeline @stackArgs

Write-Host "`n=== 4) Experiments (optional; configs.yaml) ==="
if (Test-Path "experiments\configs.yaml") {
  python experiments/run_experiment.py --config experiments/configs.yaml
} else {
  Write-Warning "experiments/configs.yaml tidak ditemukan; lewati langkah experiment."
}

Write-Host "`n=== 5) (Opsional) Jalankan UI ==="
if (Test-Path "interface\app.py") {
  Write-Host "Untuk menjalankan UI:  python interface/app.py"
} else {
  Write-Warning "interface/app.py tidak ada; lewati."
}

Write-Host "`nSelesai. Artefak:"
Write-Host " - artifacts\*.joblib  (model + threshold)"
Write-Host " - reports\figures\*   (CM/ROC/PR/threshold sweep)"
Write-Host " - reports\tables\*    (metrics/preds/ablation)"