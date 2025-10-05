@echo off
REM Enhanced Multimessenger AI Observatory Launcher
REM This script sets up and runs the enhanced web application

echo ========================================
echo   Multimessenger AI Observatory
echo   Enhanced Version Launcher
echo ========================================
echo.

REM Check Python version
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Installing/updating required packages...
pip install streamlit pandas numpy plotly scikit-learn seaborn scipy requests joblib

REM Create necessary directories
if not exist "alerts" mkdir alerts
if not exist "results" mkdir results
if not exist "uploads" mkdir uploads

echo.
echo Starting Enhanced Multimessenger AI Observatory...
echo Open your browser to: http://localhost:8507
echo.

REM Start the enhanced application
python -m streamlit run enhanced_app_v2.py --server.port 8507

pause
