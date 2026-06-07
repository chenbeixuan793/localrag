@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ================================
echo   localrag - offline chat
echo ================================
echo.
echo Loading... (this takes ~20 seconds)
echo.
set PYTHONUNBUFFERED=1
"%~dp0python\python.exe" -u "%~dp0run.py"
pause
