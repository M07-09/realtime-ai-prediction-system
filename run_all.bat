@echo off
cd /d "%~dp0"

echo.
echo  [1/3] Checking the trained model ...
if not exist "models\lstm_model.pt" (
    echo  No trained model found. Downloading data and training now ...
    python train_model.py
)

echo.
echo  [2/3] Starting the FastAPI backend on http://127.0.0.1:8000 ...
start "Backend" cmd /k python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo        Waiting for the backend to come up ...
timeout /t 8 /nobreak >nul

echo.
echo  [3/3] Starting the Streamlit dashboard on http://localhost:8501 ...
streamlit run streamlit_app.py

pause
