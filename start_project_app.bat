@echo off
setlocal
cd /d "%~dp0"
start "3DGS Project Library" http://127.0.0.1:8766/
py -3 "%~dp0project_server.py"
if errorlevel 1 (
  echo.
  echo Python 3 was not found. Install Python 3 and make sure 'py -3' works.
  pause
)
