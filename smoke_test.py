from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
command = [
    sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
    "--server.headless=true", "--server.address=127.0.0.1", "--server.port=8510",
    "--browser.gatherUsageStats=false",
]
process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    deadline = time.time() + 30
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"Streamlit terminó antes de iniciar:\n{output}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8510/_stcore/health", timeout=2) as response:
                if response.status == 200 and response.read().decode().strip() == "ok":
                    print("OK: la aplicación Streamlit inició y respondió en el puerto 8510.")
                    break
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    else:
        raise RuntimeError(f"La aplicación no respondió a tiempo: {last_error}")
finally:
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()

