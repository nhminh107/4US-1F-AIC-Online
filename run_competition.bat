@echo off
chcp 65001 >nul
title AIC ONLINE COMPETITION RUNNER (PIPELINE V2)

echo ======================================================================
echo           AIC ONLINE COMPETITION 1-CLICK LAUNCHER (V2)
echo ======================================================================
echo.

:: Try activating conda AIC environment if conda is present
where conda >nul 2>nul
if %errorlevel% equ 0 (
    call conda activate AIC 2>nul
)

:: Run the interactive Python runner
python run_competition.py

if %errorlevel% neq 0 (
    echo.
    echo [!] Program exited with error code %errorlevel%
    pause
)
