# scripts/setup.ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -e .
if (Test-Path "requirements.txt") { pip install -r requirements.txt }