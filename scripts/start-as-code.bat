@echo off
title AS Code - Local AI

cd /d "%~dp0\.."

call venv\Scripts\activate

echo ============================================
echo AS Code - Local AI Runtime
echo ============================================

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONPATH=%CD%

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

pause