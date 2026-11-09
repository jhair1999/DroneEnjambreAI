@echo off
setlocal
cd /d "%~dp0"
title DroneEnjambreAI - DJI Tello

if not exist ".venv\Scripts\python.exe" (
  echo Falta el entorno virtual. Ejecuta primero INSTALAR.bat
  pause
  exit /b 1
)

echo.
echo 1. Enciende el Tello y espera la luz amarilla intermitente.
echo 2. Conecta esta laptop al Wi-Fi TELLO-XXXXXX.
echo 3. Coloca el dron en una superficie plana y despejada.
echo 4. El navegador abrira el panel local. Usa la pestana DJI Tello fisico.
echo.
echo IMPORTANTE: el panel publico de Streamlit no puede controlar el Wi-Fi local del dron.
echo.

"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
pause
