@echo off
echo ==========================================
echo Shakespeare LSTM - Training
echo ==========================================

python --version
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 or 3.12 and try again.
    pause
    exit /b 1
)

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Starting training...
python -m src.train

echo.
echo Training finished.
pause
