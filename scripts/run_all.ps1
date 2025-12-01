<# -------------
Run-all pipeline (Windows PowerShell) — versi sederhana untuk tesis

Langkah utama:
  0) Activate venv
  1) Feature Engineering          (tectonic + volcanic -> events_fe.csv)
  2) EDA                          (tectonic & volcanic)
  3) Window 1900–2024:
       - Preprocessing (year_min=1900)
       - SMOTE (events_train -> events_train_smote)
       - Stacking (nosmote + smote)
       - Rename metrics -> *_y1900_2024_metrics.csv
  4) Window 2000–2024:
       - Preprocessing (year_min=2000)
       - SMOTE (events_train -> events_train_smote)
       - Stacking (nosmote + smote)
       - Rename metrics -> *_y2000_2024_metrics.csv
  5) Compare stacking runs        (multi-metric, multi-year)
  6) Analyze stacking result      (classification report & CM)
  7) Tests (pytest)               [opsional]
  8) Interface (FastAPI)          [opsional]
-------------#>

param(
  [switch]$RunTests,
  [switch]$StartUI
)

$ErrorActionPreference = "Stop"

# -------------------------------------------------------------------
# Helper: cetak durasi setiap step
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

# Direktori penting (relatif terhadap root project)
$TabDir = Join-Path (Get-Location) "reports\tables"
$FigDir = Join-Path (Get-Location) "reports\figures"

# 0) venv
Invoke-Step "0) Activate venv" {
  if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
  } else {
    Write-Warning "Virtual env .venv tidak ditemukan (lewati aktivasi)."
  }
}

# 1) Feature Engineering (bangun tectonic_fe, volcanic_fe, events_fe)
Invoke-Step "1) Feature Engineering (build events_fe.csv)" {
  # Versi FE ringan: OHE + gabung events
  python -m tsunami_prediction.feature_engineering --overwrite --materialize-ohe
}

# 2) EDA (gambar & tabel ringkas untuk tectonic & volcanic)
Invoke-Step "2) EDA (figures & tables)" {
  python -m tsunami_prediction.eda --datasets tectonic volcanic
}

# ---------------------------------------------------------------
# Helper untuk 1 window tahun (year_min tertentu)
# ---------------------------------------------------------------
function Run-Window(
  [int]$YearMin,
  [string]$YearTag  # contoh: "y1900_2024" atau "y2000_2024"
) {

  Write-Host "`n--- RUN WINDOW: year_min=$YearMin (tag=$YearTag) ---" -ForegroundColor Magenta

  # 3.x.1) Preprocessing untuk year_min tertentu
  Invoke-Step "Preprocessing events (year_min=$YearMin)" {
    python -m tsunami_prediction.preprocessing `
      --input "data/processed/events_fe.csv" `
      --dataset-name "events" `
      --year-min $YearMin `
      --test-size 0.2 `
      --random-state 42
  }

  # 3.x.2) SMOTE (events_train -> events_train_smote)
  Invoke-Step "SMOTE pipeline (events, year_min=$YearMin)" {
    python -m tsunami_prediction.smote_pipeline `
      --dataset events `
      --overwrite `
      --random-state 42 `
      --sampling-strategy "not majority" `
      --k-neighbors 5
  }

  # 3.x.3) Stacking (nosmote + smote) untuk window ini
  Invoke-Step "Stacking (nosmote + smote, year_min=$YearMin)" {
    python -m tsunami_prediction.stacking_pipeline `
      --variant both `
      --random-state 42
  }

  # 3.x.4) Rename metrics → tambahkan tag tahun (YearTag)
  Invoke-Step "Rename metrics for tag $YearTag" {

    # Sumber file generic (hasil run stacking terakhir)
    $nosmoteSrc = Join-Path $TabDir "events_nosmote_metrics.csv"
    $smoteSrc   = Join-Path $TabDir "events_smote_metrics.csv"

    # Nama file tujuan (hanya nama file, bukan full path)
    $nosmoteDstName = "events_nosmote_{0}_metrics.csv" -f $YearTag
    $smoteDstName   = "events_smote_{0}_metrics.csv"   -f $YearTag

    # ========= NOSMOTE =========
    if (Test-Path $nosmoteSrc) {
      $nosmoteDst = Join-Path $TabDir $nosmoteDstName

      # Jika sudah ada, hapus dulu supaya tidak error
      if (Test-Path $nosmoteDst) {
        Remove-Item $nosmoteDst -Force
      }

      Rename-Item -LiteralPath $nosmoteSrc -NewName $nosmoteDstName
      Write-Host "  > Renamed: events_nosmote_metrics.csv -> $nosmoteDstName"
    } else {
      Write-Warning "  > Source metrics not found (nosmote): $nosmoteSrc"
    }

    # ========= SMOTE =========
    if (Test-Path $smoteSrc) {
      $smoteDst = Join-Path $TabDir $smoteDstName

      if (Test-Path $smoteDst) {
        Remove-Item $smoteDst -Force
      }

      Rename-Item -LiteralPath $smoteSrc -NewName $smoteDstName
      Write-Host "  > Renamed: events_smote_metrics.csv -> $smoteDstName"
    } else {
      Write-Warning "  > Source metrics not found (smote): $smoteSrc"
    }
  }
}

# 3) Window 1900–2024
Run-Window -YearMin 1900 -YearTag "y1900_2024"

# 4) Window 2000–2024
Run-Window -YearMin 2000 -YearTag "y2000_2024"

# 5) Compare stacking runs (multi-metric, multi-year)
Invoke-Step "5) Compare stacking runs (multi-metric, multi-year)" {
  python -m tsunami_prediction.compare_stacking_runs
}

# 6) Analyze stacking result (classification report & confusion matrix)
Invoke-Step "6) Analyze stacking result (best SMOTE stacking)" {
  python -m tsunami_prediction.analyze_stacking_results
}

# 7) Tests (opsional, butuh pytest)
if ($RunTests) {
  Invoke-Step "7) Unit tests (pytest)" {
    if (Test-PyModule "pytest") {
      python -m pytest -q tests/tests_pipeline.py
    } else {
      Write-Warning "pytest belum terpasang; lewati tests."
    }
  }
} else {
  Write-Host "`nLewati tests (aktifkan dengan -RunTests)." -ForegroundColor DarkYellow
}

# 8) Interface (opsional)
Invoke-Step "8) Interface (FastAPI UI)" {
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
Write-Host " - artifacts\*.joblib        (model Stacking + base learners)"
Write-Host " - reports\figures\*         (EDA, CM/ROC, multi-metric multi-year)"
Write-Host " - reports\tables\*          (metrics/preds + stacking_experiments_all_metrics.csv)"
Write-Host " - data\processed\*          (events_fe, events_train/test, events_train_smote)"