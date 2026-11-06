@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Preparando el entorno por primera vez...
  call INSTALAR.bat
  if errorlevel 1 exit /b 1
)
echo Iniciando DroneEnjambreAI...
start "DroneEnjambreAI" cmd /k ""%CD%\.venv\Scripts\python.exe" -m streamlit run "%CD%\app.py" --server.address=localhost --server.port=8501 --browser.gatherUsageStats=false"
timeout /t 4 /nobreak >nul
start "" "http://localhost:8501"
endlocal
