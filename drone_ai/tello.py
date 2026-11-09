from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable


MIN_TAKEOFF_BATTERY = 30
VALID_MOVEMENTS = {"forward", "back", "left", "right", "up", "down"}


@dataclass
class TelloSnapshot:
    connected: bool = False
    streaming: bool = False
    flying: bool = False
    battery: int | None = None
    height_cm: int | None = None
    flight_time_s: int | None = None
    temperature_c: float | None = None
    last_command: str = "Sin conexión"
    last_error: str = ""


def dependency_status() -> tuple[bool, str]:
    try:
        from djitellopy import Tello  # noqa: F401
    except Exception as exc:
        return False, f"DJITelloPy no está disponible: {exc}"
    return True, "Controlador DJITelloPy disponible."


class TelloController:
    """Adaptador seguro para un Ryze/DJI Tello conectado por Wi-Fi.

    La clase no despega durante la conexión. Las acciones de vuelo requieren una
    autorización explícita y se validan de nuevo dentro del adaptador.
    """

    def __init__(self, tello_factory: Callable[[], Any] | None = None):
        self._factory = tello_factory
        self._tello: Any | None = None
        self._frame_reader: Any | None = None
        self._lock = threading.RLock()
        self._snapshot = TelloSnapshot()

    @property
    def snapshot(self) -> TelloSnapshot:
        with self._lock:
            return TelloSnapshot(**asdict(self._snapshot))

    def _set_error(self, exc: Exception) -> None:
        self._snapshot.last_error = str(exc)

    def _require_connected(self) -> Any:
        if not self._snapshot.connected or self._tello is None:
            raise RuntimeError("El Tello no está conectado.")
        return self._tello

    def connect(self) -> TelloSnapshot:
        with self._lock:
            if self._snapshot.connected:
                return self.refresh_telemetry()
            try:
                if self._factory is None:
                    from djitellopy import Tello

                    self._tello = Tello()
                else:
                    self._tello = self._factory()
                self._tello.connect()
                self._snapshot.connected = True
                self._snapshot.last_command = "Conectado en modo SDK"
                self._snapshot.last_error = ""
                return self.refresh_telemetry()
            except Exception as exc:
                self._snapshot.connected = False
                self._set_error(exc)
                self._tello = None
                raise RuntimeError(
                    "No se pudo conectar. Conecta Windows a la red Wi-Fi TELLO-XXXXXX y vuelve a intentar."
                ) from exc

    def refresh_telemetry(self) -> TelloSnapshot:
        with self._lock:
            tello = self._require_connected()
            try:
                state = tello.get_current_state() or {}
                battery = state.get("bat")
                if battery is None:
                    battery = tello.get_battery()
                self._snapshot.battery = int(battery)
                self._snapshot.height_cm = int(state["h"]) if state.get("h") is not None else None
                self._snapshot.flight_time_s = int(state["time"]) if state.get("time") is not None else None
                low = state.get("templ")
                high = state.get("temph")
                if low is not None and high is not None:
                    self._snapshot.temperature_c = (float(low) + float(high)) / 2.0
                self._snapshot.flying = bool(getattr(tello, "is_flying", self._snapshot.flying))
                self._snapshot.last_error = ""
            except Exception as exc:
                self._set_error(exc)
                raise RuntimeError(f"No se pudo leer la telemetría del Tello: {exc}") from exc
            return self.snapshot

    def start_stream(self) -> TelloSnapshot:
        with self._lock:
            tello = self._require_connected()
            if not self._snapshot.streaming:
                try:
                    tello.streamon()
                    self._frame_reader = tello.get_frame_read()
                    self._snapshot.streaming = True
                    self._snapshot.last_command = "Cámara iniciada"
                    self._snapshot.last_error = ""
                except Exception as exc:
                    self._set_error(exc)
                    raise RuntimeError(f"No se pudo iniciar la cámara del Tello: {exc}") from exc
            return self.snapshot

    def stop_stream(self) -> TelloSnapshot:
        with self._lock:
            tello = self._require_connected()
            if self._snapshot.streaming:
                try:
                    tello.streamoff()
                finally:
                    self._frame_reader = None
                    self._snapshot.streaming = False
                    self._snapshot.last_command = "Cámara detenida"
            return self.snapshot

    def get_frame(self):
        with self._lock:
            self._require_connected()
            if not self._snapshot.streaming or self._frame_reader is None:
                raise RuntimeError("La cámara del Tello no está iniciada.")
            frame = getattr(self._frame_reader, "frame", None)
            if frame is None:
                raise RuntimeError("Aún no llegó un fotograma desde el Tello.")
            return frame.copy()

    def takeoff(self, authorized: bool = False) -> TelloSnapshot:
        with self._lock:
            tello = self._require_connected()
            self.refresh_telemetry()
            if not authorized:
                raise PermissionError("El despegue requiere armar y confirmar el vuelo.")
            if self._snapshot.flying:
                raise RuntimeError("El Tello ya está volando.")
            if self._snapshot.battery is None or self._snapshot.battery < MIN_TAKEOFF_BATTERY:
                raise RuntimeError(f"Se requiere al menos {MIN_TAKEOFF_BATTERY}% de batería para despegar.")
            tello.takeoff()
            self._snapshot.flying = True
            self._snapshot.last_command = "Despegue"
            return self.refresh_telemetry()

    def land(self) -> TelloSnapshot:
        with self._lock:
            tello = self._require_connected()
            tello.land()
            self._snapshot.flying = False
            self._snapshot.last_command = "Aterrizaje"
            return self.snapshot

    def move(self, direction: str, distance_cm: int, authorized: bool = False) -> TelloSnapshot:
        with self._lock:
            tello = self._require_connected()
            direction = direction.lower().strip()
            distance_cm = int(distance_cm)
            if not authorized:
                raise PermissionError("El movimiento requiere mantener armado el control físico.")
            if not self._snapshot.flying:
                raise RuntimeError("El Tello debe estar volando para desplazarse.")
            if direction not in VALID_MOVEMENTS:
                raise ValueError(f"Dirección no válida: {direction}")
            if not 20 <= distance_cm <= 500:
                raise ValueError("La distancia permitida por el SDK es de 20 a 500 cm.")
            getattr(tello, f"move_{direction}")(distance_cm)
            self._snapshot.last_command = f"{direction} {distance_cm} cm"
            return self.refresh_telemetry()

    def rotate(self, clockwise: bool, degrees: int, authorized: bool = False) -> TelloSnapshot:
        with self._lock:
            tello = self._require_connected()
            degrees = int(degrees)
            if not authorized:
                raise PermissionError("La rotación requiere mantener armado el control físico.")
            if not self._snapshot.flying:
                raise RuntimeError("El Tello debe estar volando para rotar.")
            if not 1 <= degrees <= 360:
                raise ValueError("El giro debe estar entre 1 y 360 grados.")
            method = tello.rotate_clockwise if clockwise else tello.rotate_counter_clockwise
            method(degrees)
            self._snapshot.last_command = f"Giro {'horario' if clockwise else 'antihorario'} {degrees}°"
            return self.refresh_telemetry()

    def send_follow_control(
        self,
        forward_back: int,
        yaw: int,
        authorized: bool = False,
    ) -> TelloSnapshot:
        """Envía una orden RC limitada para seguimiento supervisado."""
        with self._lock:
            tello = self._require_connected()
            if not authorized:
                raise PermissionError("El seguimiento físico requiere confirmación explícita.")
            if not self._snapshot.flying:
                raise RuntimeError("El Tello debe estar volando para activar el seguimiento.")
            forward_back = max(-20, min(20, int(forward_back)))
            yaw = max(-20, min(20, int(yaw)))
            tello.send_rc_control(0, forward_back, 0, yaw)
            self._snapshot.last_command = f"Seguimiento RC: avance {forward_back}, giro {yaw}"
            return self.snapshot

    def hover(self) -> TelloSnapshot:
        """Detiene todas las velocidades sin cortar motores."""
        with self._lock:
            tello = self._require_connected()
            if self._snapshot.flying:
                tello.send_rc_control(0, 0, 0, 0)
                self._snapshot.last_command = "Mantener posición"
            return self.snapshot

    def emergency(self, authorized: bool = False) -> TelloSnapshot:
        with self._lock:
            tello = self._require_connected()
            if not authorized:
                raise PermissionError("La parada de emergencia requiere confirmación explícita.")
            tello.emergency()
            self._snapshot.flying = False
            self._snapshot.last_command = "PARADA DE EMERGENCIA"
            return self.snapshot

    def disconnect(self) -> TelloSnapshot:
        with self._lock:
            tello = self._require_connected()
            if self._snapshot.flying:
                raise RuntimeError("No se puede desconectar durante el vuelo. Aterriza primero.")
            try:
                if self._snapshot.streaming:
                    tello.streamoff()
                tello.end()
            finally:
                self._tello = None
                self._frame_reader = None
                self._snapshot = TelloSnapshot(last_command="Desconectado")
            return self.snapshot
