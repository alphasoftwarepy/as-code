@echo off
title AS Code - Local Edge AI

cd /d C:\as-code

call venv\Scripts\activate

echo ============================================
echo AS Code - Local Edge AI Runtime
echo ============================================

uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

pause