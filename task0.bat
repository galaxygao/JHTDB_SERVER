@echo off
setlocal
cd /d "%~dp0"
python task0.py %*
if errorlevel 1 (
    echo.
    echo Task 0 failed with exit code %errorlevel%.
    exit /b %errorlevel%
)
endlocal

