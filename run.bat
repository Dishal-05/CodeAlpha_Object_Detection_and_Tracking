@echo off
title AI Object Detection and Tracking
cd /d "%~dp0"
echo ========================================================
echo   Starting AI Object Detection & Tracking System...
echo ========================================================
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)

