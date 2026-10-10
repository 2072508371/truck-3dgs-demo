@echo off
setlocal
cd /d "%~dp0"
start "3DGS Project Page" http://127.0.0.1:8766/
py -3 -m http.server 8766 --bind 127.0.0.1
if errorlevel 1 (
  echo.
  echo Python launcher 'py' was not found. Install Python 3 or use start_project_app.bat.
  pause
)
