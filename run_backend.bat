@echo off
cd /d "%~dp0"
echo Backend:  http://127.0.0.1:8000
echo API docs: http://127.0.0.1:8000/docs
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
