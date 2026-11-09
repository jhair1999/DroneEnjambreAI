from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from drone_ai.core import SimulationEngine
from drone_ai.follow import compute_follow_command, select_person_target
from drone_ai.tello import (
    MIN_TAKEOFF_BATTERY,
    TelloController,
    dependency_status as tello_dependency_status,
)
from drone_ai.vision import LiveVisionProcessor, dependency_status, process_video


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"
STATE = ROOT / "data" / "runtime_state.json"
UPLOADS = ROOT / "data" / "uploads"
VISION_OUTPUTS = ROOT / "outputs" / "vision"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / "ultralytics"))

st.set_page_config(page_title="DroneEnjambreAI", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg,#07111f 0%,#0b1b2e 55%,#10243a 100%); color:#eaf2ff;}
    [data-testid="stSidebar"] {background:#081522; border-right:1px solid #193650;}
    .hero {padding:1.1rem 1.35rem; border:1px solid #214b6b; border-radius:18px;
      background:linear-gradient(110deg,rgba(19,64,93,.8),rgba(11,29,48,.92)); margin-bottom:1rem;}
    .hero h1 {margin:0;color:#e9f6ff;font-size:2rem}.hero p{color:#9fc5df;margin:.3rem 0 0}
    .badge-real {display:inline-block;padding:.2rem .55rem;border-radius:999px;background:#7f1d1d;color:#fecaca;font-weight:700}
    .badge-sim {display:inline-block;padding:.2rem .55rem;border-radius:999px;background:#164e63;color:#cffafe;font-weight:700}
    div[data-testid="stMetric"] {background:rgba(11,31,50,.85);border:1px solid #1d4967;padding:.65rem;border-radius:14px}
    .stButton>button {border-radius:10px;font-weight:700}
    </style>
    <div class="hero"><h1>🛰️ DroneEnjambreAI</h1>
    <p>Coordinación multiagente, recuperación ante fallos y detección real de personas con YOLO</p></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine() -> SimulationEngine:
    return SimulationEngine(CONFIG, STATE)


engine = get_engine()


@st.cache_resource
def get_tello_controller() -> TelloController:
    return TelloController()


@st.cache_resource
def get_live_vision_processor() -> LiveVisionProcessor:
    return LiveVisionProcessor(str(ROOT / "yolo11n.pt"))


tello_controller = get_tello_controller()


def drone_rows():
    return [
        {
            "Dron": d.drone_id,
            "Estado": d.state,
            "Batería": round(d.battery, 1),
            "X": round(d.x, 1),
            "Y": round(d.y, 1),
            "Misión": d.mission_id or "-",
        }
        for d in engine.drones.values()
    ]


def mission_rows():
    return [
        {
            "Misión": m.mission_id,
            "Nombre": m.name,
            "Prioridad": m.priority,
            "Origen": "REAL - YOLO" if m.source == "REAL_YOLO" else "SIMULADA",
            "Estado": m.state,
            "Dron": m.assigned_drone or "-",
            "Destino": f"({m.x:.0f}, {m.y:.0f})",
        }
        for m in sorted(engine.missions.values(), key=lambda x: x.mission_id, reverse=True)
    ]


def event_rows(limit: int = 80):
    return [
        {
            "Hora": e["time"][11:19],
            "Tipo": e["type"],
            "Nivel": e["level"],
            "Evento": e["message"],
        }
        for e in reversed(engine.events[-limit:])
    ]


def map_figure() -> go.Figure:
    fig = go.Figure()
    state_colors = {"DISPONIBLE": "#22c55e", "EN_MISION": "#38bdf8", "FALLA": "#ef4444", "BATERIA_BAJA": "#f59e0b"}
    for mission in engine.missions.values():
        color = "#fb7185" if mission.source == "REAL_YOLO" else "#fbbf24"
        fig.add_trace(
            go.Scatter(
                x=[mission.x], y=[mission.y], mode="markers+text",
                marker=dict(size=15, symbol="star", color=color, line=dict(width=1, color="white")),
                text=[mission.mission_id], textposition="top center",
                name=f"{mission.mission_id} ({mission.state})",
                hovertext=f"{mission.name}<br>Prioridad {mission.priority}<br>Origen {mission.source}",
                hoverinfo="text",
            )
        )
    for drone in engine.drones.values():
        if drone.mission_id and drone.mission_id in engine.missions:
            target = engine.missions[drone.mission_id]
            fig.add_trace(
                go.Scatter(
                    x=[drone.x, target.x], y=[drone.y, target.y], mode="lines",
                    line=dict(color="#38bdf8", width=2, dash="dot"), showlegend=False, hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[drone.x], y=[drone.y], mode="markers+text",
                marker=dict(
                    size=24,
                    symbol="x" if drone.state == "FALLA" else "triangle-up",
                    color=state_colors.get(drone.state, "#94a3b8"),
                    line=dict(width=2, color="#e2e8f0"),
                ),
                text=[drone.drone_id], textposition="bottom center", name=drone.drone_id,
                hovertext=f"{drone.state}<br>Batería {drone.battery:.1f}%<br>Misión {drone.mission_id or '-'}",
                hoverinfo="text",
            )
        )
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=40, b=10), title="Mapa operacional 2D",
        paper_bgcolor="#0b1b2e", plot_bgcolor="#0d2539", font=dict(color="#e2e8f0"),
        xaxis=dict(range=[0, 100], title="Coordenada X", gridcolor="#1d4059"),
        yaxis=dict(range=[0, 100], title="Coordenada Y", gridcolor="#1d4059", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


with st.sidebar:
    st.header("Centro de control")
    c1, c2 = st.columns(2)
    if c1.button("▶ Iniciar", width="stretch"):
        engine.start(); st.rerun()
    if c2.button("⏸ Pausar", width="stretch"):
        engine.pause(); st.rerun()
    c3, c4 = st.columns(2)
    if c3.button("↻ Reiniciar", width="stretch"):
        engine.reset(); st.rerun()
    if c4.button("🎬 Demo", width="stretch", help="Carga un escenario listo para grabar"):
        engine.load_demo(); st.rerun()
    st.caption("Estado: " + ("🟢 EN EJECUCIÓN" if engine.running else "🟡 PAUSADA"))
    st.divider()
    st.subheader("Nueva misión simulada")
    with st.form("mission_form", clear_on_submit=True):
        name = st.text_input("Nombre", "Inspección de incidente")
        x = st.slider("Destino X", 0, 100, 70)
        y = st.slider("Destino Y", 0, 100, 65)
        priority = st.select_slider("Prioridad", options=[1, 2, 3, 4, 5], value=4)
        if st.form_submit_button("Crear y asignar", width="stretch"):
            engine.create_mission(name, x, y, priority, "SIMULADA"); st.rerun()
    active = [d.drone_id for d in engine.drones.values() if d.state == "EN_MISION"]
    st.divider()
    st.subheader("Prueba de recuperación")
    if not active:
        st.info("No hay una misión activa. Prepara una prueba para asignar un dron antes de simular el fallo.")
        if st.button("🧪 Preparar prueba de fallo", width="stretch"):
            engine.create_mission("Prueba de continuidad ante fallo", 95, 95, 5, "SIMULADA")
            engine.pause()
            st.rerun()
    failed_target = st.selectbox("Dron en misión", active, disabled=not active, placeholder="No hay drones en misión")
    if st.button("⚠ Simular fallo", type="primary", width="stretch", disabled=not active):
        engine.fail_drone(failed_target); st.rerun()


tab_operation, tab_vision, tab_mobile, tab_tello, tab_history, tab_help = st.tabs(
    [
        "📡 Operación",
        "👁️ Visión IA",
        "📱 Prototipo celular",
        "🚁 DJI Tello físico",
        "📜 Historial",
        "🎥 Guion de demostración",
    ]
)


with tab_operation:
    def live_operation():
        if engine.running:
            engine.tick(0.6)
        completed = sum(m.state == "COMPLETADA" for m in engine.missions.values())
        active_count = sum(m.state == "ASIGNADA" for m in engine.missions.values())
        failures = sum(d.state == "FALLA" for d in engine.drones.values())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Drones", len(engine.drones))
        m2.metric("Misiones activas", active_count)
        m3.metric("Completadas", completed)
        m4.metric("Fallos detectados", failures)
        left, right = st.columns([1.7, 1])
        with left:
            st.plotly_chart(map_figure(), width="stretch", key="operational-map")
        with right:
            st.subheader("Estados de los agentes")
            st.dataframe(pd.DataFrame(drone_rows()), hide_index=True, width="stretch")
            st.subheader("Última decisión explicada")
            decisions = [e for e in reversed(engine.events) if e["type"] in {"ASIGNACION", "REASIGNACION", "FALLO", "BATERIA_BAJA"}]
            latest_decision = decisions[0]["message"] if decisions else "Crea una misión para observar la decisión del coordinador."
            st.info(latest_decision)
        st.subheader("Misiones")
        rows = mission_rows()
        if len(rows) > 0:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("No hay misiones.")
        if engine.running:
            time.sleep(0.6)
            st.rerun()

    live_operation()


with tab_vision:
    st.markdown('<span class="badge-real">DETECCIÓN REAL EN VIDEO</span>', unsafe_allow_html=True)
    st.write("Este módulo ejecuta inferencia YOLO sobre el video cargado. No confunde estas detecciones con eventos simulados.")
    available, status_text = dependency_status()
    (st.success if available else st.error)(status_text)
    uploaded = st.file_uploader("Cargar video aéreo o de prueba", type=["mp4", "avi", "mov", "mkv"])
    v1, v2, v3 = st.columns(3)
    confidence = v1.slider("Confianza mínima", 0.10, 0.90, 0.35, 0.05)
    alert_x = v2.number_input("Coordenada X de la alerta", 0, 100, 50)
    alert_y = v3.number_input("Coordenada Y de la alerta", 0, 100, 50)
    create_alert = st.checkbox("Crear misión real si se detectan personas", value=True)
    if uploaded and available and st.button("Procesar con YOLO", type="primary"):
        UPLOADS.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(uploaded.getvalue()).hexdigest()[:10]
        input_path = UPLOADS / f"{digest}_{uploaded.name}"
        output_path = VISION_OUTPUTS / f"detectado_{digest}.mp4"
        input_path.write_bytes(uploaded.getvalue())
        progress_bar = st.progress(0.0, "Ejecutando inferencia YOLO...")
        try:
            result = process_video(input_path, output_path, confidence, progress=lambda p: progress_bar.progress(p))
            engine.register_real_detection(result.person_detections, result.frames_with_people, uploaded.name)
            if result.person_detections and create_alert:
                engine.create_mission(
                    f"Verificar detección real en {uploaded.name}", alert_x, alert_y, 5, "REAL_YOLO",
                    {"detections": result.person_detections, "frames_with_people": result.frames_with_people},
                )
            st.success(
                f"Procesamiento real completado: {result.person_detections} detecciones en "
                f"{result.frames_with_people} fotogramas; máximo {result.max_people_in_frame} personas por fotograma."
            )
            compatible_video = Path(result.output_path).read_bytes()
            st.video(compatible_video, format="video/mp4")
            st.download_button(
                "⬇ Descargar video detectado",
                data=compatible_video,
                file_name=Path(result.output_path).name,
                mime="video/mp4",
            )
        except Exception as exc:
            st.exception(exc)
    elif uploaded and not available:
        st.warning("Ejecuta INSTALAR.bat para habilitar OpenCV y YOLO.")


with tab_mobile:
    st.markdown('<span class="badge-sim">PROTOTIPO CON CÁMARA DE CELULAR</span>', unsafe_allow_html=True)
    st.write(
        "Este modo reemplaza temporalmente la cámara del dron por la cámara del teléfono. "
        "La imagen y la detección YOLO son reales; batería, altura y movimiento son simulados."
    )
    st.info(
        "Abre la URL pública de DroneEnjambreAI desde el celular, entra a esta pestaña y permite el acceso "
        "a la cámara. No necesitas tener un DJI Tello para esta demostración."
    )

    mobile_left, mobile_right = st.columns([1, 1])
    with mobile_left:
        st.subheader("Telemetría simulada del prototipo")
        mobile_battery = st.slider("Batería simulada (%)", 0, 100, 82, key="mobile_battery")
        mobile_altitude = st.slider("Altura simulada (m)", 0, 30, 8, key="mobile_altitude")
        mobile_x = st.slider("Coordenada X simulada", 0, 100, 50, key="mobile_x")
        mobile_y = st.slider("Coordenada Y simulada", 0, 100, 50, key="mobile_y")
        mc1, mc2 = st.columns(2)
        mc1.metric("Batería", f"{mobile_battery}%", help="Dato simulado porque el celular no entrega batería del dron.")
        mc2.metric("Altura", f"{mobile_altitude} m", help="Dato simulado para representar el vuelo del prototipo.")
        if mobile_battery < MIN_TAKEOFF_BATTERY:
            st.error("Misión bloqueada: batería simulada inferior al 30 %.")
    with mobile_right:
        st.subheader("Captura real desde la cámara")
        mobile_photo = st.camera_input(
            "Tomar fotografía para reconocimiento",
            help="En el teléfono abrirá la cámara. La fotografía se procesa con YOLO.",
        )
        mobile_confidence = st.slider(
            "Confianza mínima", 0.10, 0.90, 0.35, 0.05, key="mobile_confidence"
        )

    if mobile_photo is not None:
        import cv2
        import numpy as np

        photo_bytes = mobile_photo.getvalue()
        photo_hash = hashlib.sha1(photo_bytes).hexdigest()
        if st.button("🔎 Analizar fotografía con YOLO", type="primary", width="stretch"):
            decoded = cv2.imdecode(np.frombuffer(photo_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                st.error("No se pudo leer la fotografía del celular.")
            else:
                result = get_live_vision_processor().process(decoded, mobile_confidence)
                st.session_state["mobile_detection"] = {
                    "hash": photo_hash,
                    "frame": result.annotated_frame,
                    "counts": result.counts,
                    "total": result.total_detections,
                    "people": result.counts.get("person", 0),
                }

    mobile_detection = st.session_state.get("mobile_detection")
    if mobile_detection and mobile_photo is not None and mobile_detection["hash"] == hashlib.sha1(mobile_photo.getvalue()).hexdigest():
        st.image(mobile_detection["frame"], channels="BGR", width="stretch")
        detected_text = ", ".join(
            f"{name}: {count}" for name, count in mobile_detection["counts"].items()
        )
        st.success(
            f"Detección real completada: {mobile_detection['total']} objetos. "
            f"Clases reconocidas: {detected_text or 'ninguna'}."
        )
        mission_disabled = mobile_detection["total"] <= 0 or mobile_battery < MIN_TAKEOFF_BATTERY
        if st.button(
            "📍 Crear misión desde esta detección",
            type="primary",
            disabled=mission_disabled,
            width="stretch",
        ):
            engine.register_real_detection(
                mobile_detection["people"], 1, "Fotografía real desde celular"
            )
            mission = engine.create_mission(
                "Inspeccionar objetos detectados por cámara celular",
                mobile_x,
                mobile_y,
                5,
                "REAL_YOLO",
                {
                    "camera": "CELULAR",
                    "objects": mobile_detection["counts"],
                    "simulated_altitude_m": mobile_altitude,
                    "simulated_battery": mobile_battery,
                },
            )
            st.success(
                f"Misión {mission.mission_id} creada y enviada al coordinador. "
                "La detección es real; posición, batería y altura son simuladas."
            )

    with st.expander("¿Qué representa cada opción del modo celular?"):
        st.markdown(
            """
            - **Batería simulada:** imita el porcentaje que reportaría el Tello; bajo 30 % bloquea la misión.
            - **Altura simulada:** representa la altura de vuelo porque el teléfono permanece en tierra.
            - **Coordenadas X/Y:** lugar del mapa donde se generará la alerta.
            - **Tomar fotografía:** abre la cámara real del celular.
            - **Confianza mínima:** elimina detecciones poco seguras; 0.35 es un valor equilibrado.
            - **Analizar con YOLO:** reconoce personas, vehículos y otros objetos reales en la fotografía.
            - **Crear misión:** registra la alerta y deja que el coordinador seleccione un dron virtual.
            """
        )


with tab_tello:
    st.markdown('<span class="badge-real">CONTROL FÍSICO DJI/RYZE TELLO</span>', unsafe_allow_html=True)
    st.warning(
        "Este módulo solo controla el dron desde la aplicación LOCAL abierta con INICIAR_TELLO.bat. "
        "No funciona desde la web pública porque el Tello crea su propia red Wi-Fi."
    )
    tello_available, tello_status_text = tello_dependency_status()
    (st.success if tello_available else st.error)(tello_status_text)

    if "tello_notice" in st.session_state:
        notice_type, notice_text = st.session_state.pop("tello_notice")
        getattr(st, notice_type)(notice_text)

    status = tello_controller.snapshot
    connect_col, refresh_col, camera_col, disconnect_col = st.columns(4)
    if connect_col.button(
        "🔌 Conectar Tello", width="stretch", disabled=not tello_available or status.connected
    ):
        try:
            tello_controller.connect()
            st.session_state["tello_notice"] = ("success", "Tello conectado. No se ha ordenado ningún despegue.")
        except Exception as exc:
            st.session_state["tello_notice"] = ("error", str(exc))
        st.rerun()
    if refresh_col.button("↻ Telemetría", width="stretch", disabled=not status.connected):
        try:
            tello_controller.refresh_telemetry()
            st.session_state["tello_notice"] = ("success", "Telemetría actualizada.")
        except Exception as exc:
            st.session_state["tello_notice"] = ("error", str(exc))
        st.rerun()
    if camera_col.button(
        "📷 Iniciar cámara" if not status.streaming else "⏹ Detener cámara",
        width="stretch",
        disabled=not status.connected,
    ):
        try:
            if status.streaming:
                tello_controller.stop_stream()
            else:
                tello_controller.start_stream()
            st.session_state["tello_notice"] = ("success", "Estado de cámara actualizado.")
        except Exception as exc:
            st.session_state["tello_notice"] = ("error", str(exc))
        st.rerun()
    if disconnect_col.button("Desconectar", width="stretch", disabled=not status.connected or status.flying):
        try:
            tello_controller.disconnect()
            st.session_state["tello_notice"] = ("success", "Tello desconectado correctamente.")
        except Exception as exc:
            st.session_state["tello_notice"] = ("error", str(exc))
        st.rerun()

    status = tello_controller.snapshot
    tm1, tm2, tm3, tm4, tm5 = st.columns(5)
    tm1.metric("Enlace", "CONECTADO" if status.connected else "SIN CONEXIÓN")
    tm2.metric("Batería", f"{status.battery}%" if status.battery is not None else "—")
    tm3.metric("Vuelo", "EN EL AIRE" if status.flying else "EN TIERRA")
    tm4.metric("Altura", f"{status.height_cm} cm" if status.height_cm is not None else "—")
    tm5.metric("Temperatura", f"{status.temperature_c:.0f} °C" if status.temperature_c is not None else "—")
    st.caption(f"Último comando: {status.last_command}")
    if status.last_error:
        st.error(status.last_error)

    st.subheader("Control de vuelo protegido")
    st.write(
        "Antes de armar: instala protectores de hélices, despeja al menos 3 metros, coloca el Tello en "
        "una superficie plana y mantén disponible el botón Aterrizar."
    )
    armed = st.checkbox(
        "Confirmo que el área está despejada y deseo habilitar órdenes físicas de vuelo",
        key="tello_armed",
    )
    battery_ok = status.battery is not None and status.battery >= MIN_TAKEOFF_BATTERY
    takeoff_col, land_col = st.columns(2)
    if takeoff_col.button(
        "🚀 DESPEGAR",
        type="primary",
        width="stretch",
        disabled=not (status.connected and armed and battery_ok and not status.flying),
    ):
        try:
            tello_controller.takeoff(authorized=armed)
            st.session_state["tello_notice"] = ("success", "Despegue confirmado por el Tello.")
        except Exception as exc:
            st.session_state["tello_notice"] = ("error", str(exc))
        st.rerun()
    if land_col.button(
        "🛬 ATERRIZAR", width="stretch", disabled=not (status.connected and status.flying)
    ):
        try:
            tello_controller.land()
            st.session_state["tello_notice"] = ("success", "Aterrizaje confirmado.")
        except Exception as exc:
            st.session_state["tello_notice"] = ("error", str(exc))
        st.rerun()
    if status.connected and not battery_ok:
        st.error(f"Despegue bloqueado: se requiere al menos {MIN_TAKEOFF_BATTERY}% de batería.")

    distance = st.slider("Distancia por orden (cm)", 20, 100, 30, 10, disabled=not status.connected)
    turn_degrees = st.slider("Giro por orden (grados)", 15, 180, 45, 15, disabled=not status.connected)
    move_enabled = status.connected and status.flying and armed
    move_columns = st.columns(4)
    movement_actions = [
        ("⬆ Adelante", "forward"),
        ("⬇ Atrás", "back"),
        ("⬅ Izquierda", "left"),
        ("➡ Derecha", "right"),
    ]
    for column, (label, direction) in zip(move_columns, movement_actions):
        if column.button(label, width="stretch", disabled=not move_enabled, key=f"move_{direction}"):
            try:
                tello_controller.move(direction, distance, authorized=armed)
                st.session_state["tello_notice"] = ("success", f"Orden ejecutada: {label} {distance} cm.")
            except Exception as exc:
                st.session_state["tello_notice"] = ("error", str(exc))
            st.rerun()
    vertical_columns = st.columns(4)
    extra_actions = [
        ("🔼 Subir", "up", None),
        ("🔽 Bajar", "down", None),
        ("↻ Giro horario", None, True),
        ("↺ Giro antihorario", None, False),
    ]
    for column, (label, direction, clockwise) in zip(vertical_columns, extra_actions):
        if column.button(label, width="stretch", disabled=not move_enabled, key=f"physical_{label}"):
            try:
                if direction:
                    tello_controller.move(direction, distance, authorized=armed)
                else:
                    tello_controller.rotate(bool(clockwise), turn_degrees, authorized=armed)
                st.session_state["tello_notice"] = ("success", f"Orden ejecutada: {label}.")
            except Exception as exc:
                st.session_state["tello_notice"] = ("error", str(exc))
            st.rerun()

    with st.expander("⚠ Parada de emergencia (el dron caerá)"):
        emergency_confirm = st.checkbox(
            "Confirmo que existe peligro inmediato y debo cortar los motores",
            key="tello_emergency_confirm",
        )
        if st.button(
            "CORTAR MOTORES",
            type="primary",
            disabled=not (status.connected and status.flying and emergency_confirm),
        ):
            try:
                tello_controller.emergency(authorized=emergency_confirm)
                st.session_state["tello_notice"] = ("error", "Se ejecutó la parada de emergencia.")
            except Exception as exc:
                st.session_state["tello_notice"] = ("error", str(exc))
            st.rerun()

    st.subheader("Cámara en vivo y reconocimiento")
    if status.streaming:
        live_confidence = st.slider("Confianza YOLO en vivo", 0.10, 0.90, 0.35, 0.05)
        analyze_live = st.checkbox("Activar YOLO + seguimiento ByteTrack", value=True)
        follow_safety = st.checkbox(
            "Autorizo seguimiento físico supervisado, con protectores y área despejada",
            key="tello_follow_safety",
            disabled=not (status.connected and status.flying and armed and analyze_live),
        )
        follow_speed = st.slider(
            "Velocidad máxima de seguimiento",
            8,
            20,
            12,
            disabled=not follow_safety,
            help="Limita el avance y el giro automático. Para la primera prueba usa 8.",
        )
        follow_enabled = st.checkbox(
            "🎯 Iniciar misión: detectar y seguir una persona",
            key="tello_follow_enabled",
            disabled=not follow_safety,
        )
        auto_refresh = st.checkbox("Actualizar video continuamente", value=True)

        was_following = st.session_state.get("tello_follow_active", False)
        if was_following and not follow_enabled:
            try:
                tello_controller.hover()
                physical_mission_id = st.session_state.pop("tello_follow_mission_id", None)
                if physical_mission_id:
                    engine.finish_physical_mission(physical_mission_id, "detenida por el operador")
            except Exception as exc:
                st.warning(f"No se pudo confirmar la detención: {exc}")
            st.session_state.pop("tello_target_id", None)
            st.session_state["tello_target_missed"] = 0
        st.session_state["tello_follow_active"] = follow_enabled

        try:
            live_frame = tello_controller.get_frame()
            if analyze_live:
                live_result = get_live_vision_processor().process(live_frame, live_confidence)
                st.image(live_result.annotated_frame, channels="BGR", width="stretch")
                classes_text = ", ".join(f"{name}: {count}" for name, count in live_result.counts.items())
                st.info(
                    f"Objetos visibles: {live_result.total_detections}. "
                    f"Clases: {classes_text or 'ninguna'}. IDs seguidos: {live_result.track_ids or 'ninguno'}."
                )
                st.session_state["last_tello_detection"] = {
                    "total": live_result.total_detections,
                    "people": live_result.counts.get("person", 0),
                }
                if follow_enabled:
                    preferred_id = st.session_state.get("tello_target_id")
                    target = select_person_target(live_result.person_targets, preferred_id)
                    command = compute_follow_command(
                        target,
                        frame_width=live_frame.shape[1],
                        frame_height=live_frame.shape[0],
                        max_speed=follow_speed,
                    )
                    if target is not None:
                        st.session_state["tello_target_id"] = target.track_id
                        st.session_state["tello_target_missed"] = 0
                        if not st.session_state.get("tello_follow_mission_id"):
                            physical_mission = engine.create_physical_mission(
                                "Seguimiento supervisado de persona detectada",
                                50,
                                50,
                                metadata={
                                    "camera": "DJI_TELLO_LIVE",
                                    "track_id": target.track_id,
                                    "control": "RC_SUPERVISADO",
                                },
                            )
                            st.session_state["tello_follow_mission_id"] = physical_mission.mission_id
                            engine.register_real_detection(1, 1, "Cámara DJI Tello en vivo")
                        tello_controller.send_follow_control(
                            command.forward_back,
                            command.yaw,
                            authorized=follow_safety and armed,
                        )
                        st.success(
                            f"MISIÓN ACTIVA · Persona ID {target.track_id or 'sin ID'} · {command.reason} · "
                            f"avance {command.forward_back}, giro {command.yaw}."
                        )
                    else:
                        tello_controller.hover()
                        missed = st.session_state.get("tello_target_missed", 0) + 1
                        st.session_state["tello_target_missed"] = missed
                        st.warning("Objetivo no visible: movimiento detenido y posición mantenida.")
                        if missed >= 12:
                            physical_mission_id = st.session_state.pop("tello_follow_mission_id", None)
                            if physical_mission_id:
                                engine.finish_physical_mission(
                                    physical_mission_id, "persona fuera de cámara durante 3 segundos"
                                )
                            st.session_state.pop("tello_target_id", None)
            else:
                st.image(live_frame, channels="BGR", width="stretch")
        except Exception as exc:
            if follow_enabled:
                try:
                    tello_controller.hover()
                except Exception:
                    pass
            st.warning(f"Esperando video del Tello: {exc}")

        detection = st.session_state.get("last_tello_detection", {"total": 0, "people": 0})
        if st.button("Registrar detección real en el coordinador", disabled=detection["total"] <= 0):
            engine.register_real_detection(detection["people"], 1, "Cámara DJI Tello en vivo")
            if detection["people"] > 0:
                engine.create_mission(
                    "Verificar persona detectada por DJI Tello", 50, 50, 5, "REAL_YOLO",
                    {"physical_drone": "DJI Tello", "objects": detection["total"]},
                )
            st.success("Detección incorporada al historial y al coordinador.")
        if auto_refresh:
            time.sleep(0.25)
            st.rerun()
    else:
        st.info("Conecta el Tello y presiona Iniciar cámara para reconocer personas y otros objetos en vivo.")


with tab_history:
    st.subheader("Registro auditable de asignaciones, fallos y alertas")
    rows = event_rows()
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=520)
    if STATE.exists():
        st.download_button("Descargar estado JSON", STATE.read_bytes(), "runtime_state.json", "application/json")


with tab_help:
    st.markdown(
        """
        ### Escenario corto para grabar
        1. Presiona **Demo**. Verás tres drones moviéndose y dos misiones asignadas por reglas.
        2. Abre la decisión explicada y muestra batería, distancia, prioridad y puntuación.
        3. En la barra lateral elige un dron activo y presiona **Simular fallo**.
        4. Observa cómo el dron cambia a rojo, la misión se libera y otro agente la recibe automáticamente.
        5. Abre **Historial** para mostrar los eventos `FALLO`, `REASIGNACION` y `ASIGNACION`.
        6. En **Visión IA**, carga un video con personas, procésalo y muestra los recuadros YOLO.

        **Azul/amarillo:** misión simulada. **Rojo y etiqueta REAL - YOLO:** evento originado por inferencia real en video.
        """
    )

