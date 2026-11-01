@echo off
setlocal
set PYTHONUTF8=1
set PYTHONPATH=%~dp0..\vendor
cd /d "%~dp0.."
python run.py
