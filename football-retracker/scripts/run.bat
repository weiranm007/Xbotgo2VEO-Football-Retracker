@echo off
echo ==========================================
echo   FootTrack -- AI Football Retracker
echo ==========================================

where python >nul 2>&1 || (echo Python not found. Install Python 3.9+ && pause && exit /b 1)
where ffmpeg >nul 2>&1 || (
    echo ffmpeg not found.
    echo Download from: https://ffmpeg.org/download.html
    pause && exit /b 1
)

:: Resolve root folder reliably (parent of scripts\)
pushd "%~dp0.."
set ROOT_DIR=%CD%
popd

set VENV_DIR=%ROOT_DIR%\.venv
set BACKEND_DIR=%ROOT_DIR%\backend

echo Root:    %ROOT_DIR%
echo Backend: %BACKEND_DIR%
echo.

:: Clean up misplaced .venv if it ended up in scripts\
if exist "%~dp0.venv" (
    echo Removing misplaced .venv from scripts folder...
    rmdir /s /q "%~dp0.venv"
)

if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

echo Installing dependencies...
python -m pip install --upgrade pip -q
python -m pip install -r "%BACKEND_DIR%\requirements.txt" -q

echo.
echo Backend starting on http://localhost:5000
echo Open frontend\index.html in your browser
echo.

cd /d "%BACKEND_DIR%"
python app.py
pause
