@echo off
REM Setup script for Complaint Auto-Routing System (Windows)

echo ==========================================
echo Complaint Auto-Routing System - Setup
echo ==========================================

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Download models
echo.
echo Downloading models...
python scripts\download_models.py

REM Generate training data
echo.
echo Generating training data...
python scripts\generate_data.py

REM Train models
echo.
echo Training models...
python scripts\train_pipeline.py

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To use the system:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Run CLI: python app\cli.py --text "Your complaint"
echo   3. Run web app: python app\web_app.py
echo.
pause
