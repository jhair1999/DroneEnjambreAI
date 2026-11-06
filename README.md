# DroneEnjambreAI

Prototipo funcional de coordinación multiagente para tres drones virtuales. Incluye simulación 2D, movimiento visible, asignación explicable por reglas, fallos y reasignación automática, persistencia, historial y detección real de personas en video con YOLO/OpenCV.

La aplicación está preparada para ejecutarse localmente y para desplegarse en Streamlit Community Cloud. `packages.txt` instala FFmpeg para generar videos H.264 compatibles con navegadores.

## Inicio rápido

1. Haz doble clic en `INICIAR.bat`.
2. Espera a que se abra `http://localhost:8501`.
3. Pulsa **Demo** para cargar un escenario listo para presentar.

También puedes abrir la carpeta en Visual Studio Code y ejecutar:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe smoke_test.py
.\.venv\Scripts\python.exe verify_yolo.py
```

## Demostración recomendada

1. Pulsa **Demo** y muestra el movimiento de los tres agentes.
2. Enseña la explicación de la asignación: batería, distancia, prioridad y puntuación.
3. Selecciona un dron en misión y pulsa **Simular fallo**.
4. Muestra la reasignación en el mapa y el registro del historial.
5. Carga un video con personas en **Visión IA** y pulsa **Procesar con YOLO**.
6. Explica que las misiones normales son simuladas y las alertas con origen `REAL - YOLO` proceden de inferencia sobre video.

## Archivos esenciales

- `app.py`: dashboard Streamlit.
- `drone_ai/core.py`: agentes, coordinador, movimiento, fallos y persistencia.
- `drone_ai/vision.py`: procesamiento YOLO/OpenCV.
- `config.json`: configuración base persistente.
- `data/runtime_state.json`: último estado de ejecución.
- `tests/`: pruebas automatizadas.

La simulación no controla drones físicos. Las detecciones se presentan como apoyo supervisado y no toman decisiones coercitivas sobre personas.
