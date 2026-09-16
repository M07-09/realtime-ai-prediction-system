@echo off
cd /d "%~dp0"
echo Downloading data and training the LSTM ...
python train_model.py
pause
