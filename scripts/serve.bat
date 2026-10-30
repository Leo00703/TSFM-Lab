@echo off
rem Starts the TSFM-Lab server in this terminal. Ctrl+C stops it (answer Y or S if Windows asks whether to terminate the batch job).
rem It listens on this computer only and opens the browser. Any options are passed on, for example:
rem     scripts\serve.bat --port 9000 --no-browser
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo The environment is missing: run  python scripts\bootstrap.py  first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m tsfm_lab serve %*
if errorlevel 1 pause
