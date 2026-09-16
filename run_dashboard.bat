@echo off
cd /d "%~dp0"
echo Dashboard: http://localhost:8501
python -m streamlit run streamlit_app.py
pause
