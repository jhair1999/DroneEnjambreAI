# DroneEnjambreAI

Prototipo funcional de coordinación multiagente para tres drones virtuales y control supervisado de un dron físico DJI/Ryze Tello. Incluye simulación 2D, asignación explicable, fallos y reasignación, telemetría Tello, cámara en vivo, seguimiento ByteTrack y detección multiclase con YOLO/OpenCV.

La aplicación está preparada para ejecutarse localmente y para desplegarse en Streamlit Community Cloud. `packages.txt` instala FFmpeg para generar videos H.264 compatibles con navegadores.

## Inicio rápido

1. Haz doble clic en `INICIAR.bat`.
2. Espera a que se abra `http://localhost:8501`.
3. Pulsa **Demo** para cargar un escenario listo para presentar.

También puedes abrir la carpeta en Visual Studio Code y ejecutar:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Uso con DJI/Ryze Tello

El Tello no ejecuta Python o YOLO dentro de su pequeño controlador. La aplicación se ejecuta en la laptop, recibe la cámara del dron y le envía órdenes mediante el SDK oficial por Wi-Fi.

1. Ejecuta `INSTALAR.bat` una vez.
2. Enciende el Tello y espera su red `TELLO-XXXXXX`.
3. Conecta Windows a esa red Wi-Fi.
4. Coloca protectores de hélices y deja un área libre de al menos 3 metros.
5. Ejecuta `INICIAR_TELLO.bat` y abre **DJI Tello físico**.
6. Pulsa **Conectar Tello** y luego **Iniciar cámara**. Esto no despega el dron.
7. Comprueba telemetría y video. Para volar, marca la confirmación de seguridad y usa **DESPEGAR**.
8. Usa **ATERRIZAR** para terminar. La parada de emergencia corta motores y solo debe utilizarse ante peligro inmediato.

La URL pública de Streamlit sirve para la simulación, pero no puede acceder al Tello conectado a la red local. El control físico debe abrirse localmente con `INICIAR_TELLO.bat`.

## Prototipo con celular (sin comprar el dron)

1. Abre la URL pública de la aplicación desde el teléfono.
2. Entra en **Prototipo celular** y permite el acceso a la cámara.
3. Ajusta batería, altura y coordenadas simuladas.
4. Toma una fotografía y pulsa **Analizar fotografía con YOLO**.
5. Pulsa **Crear misión desde esta detección** para integrarla al coordinador.

La cámara y el reconocimiento son reales. La batería, altura, ubicación y vuelo permanecen claramente identificados como datos simulados. El celular sustituye solamente la cámara del dron y permite demostrar el flujo completo sin afirmar que hubo un vuelo físico.

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
- `drone_ai/tello.py`: conexión, telemetría, cámara y control protegido del DJI/Ryze Tello.
- `config.json`: configuración base persistente.
- `data/runtime_state.json`: último estado de ejecución.
- `tests/`: pruebas automatizadas.

El vuelo físico es supervisado: el sistema nunca despega automáticamente y bloquea el despegue con menos de 30 % de batería. Las detecciones se presentan como apoyo y no toman decisiones coercitivas sobre personas.
