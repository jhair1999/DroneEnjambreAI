from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Drone:
    drone_id: str
    x: float
    y: float
    battery: float
    speed: float = 8.0
    state: str = "DISPONIBLE"
    mission_id: str | None = None
    home_x: float = 0.0
    home_y: float = 0.0
    distance_travelled: float = 0.0

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class Mission:
    mission_id: str
    name: str
    x: float
    y: float
    priority: int
    source: str = "SIMULADA"
    state: str = "PENDIENTE"
    assigned_drone: str | None = None
    created_at: str = ""
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulationEngine:
    """Motor determinista para agentes, asignación, movimiento y recuperación."""

    def __init__(self, config_path: str | Path, state_path: str | Path | None = None):
        self.config_path = Path(config_path)
        self.state_path = Path(state_path) if state_path else self.config_path.parent / "data" / "runtime_state.json"
        self.lock = threading.RLock()
        self.running = False
        self.simulation_time = 0.0
        self.drones: dict[str, Drone] = {}
        self.missions: dict[str, Mission] = {}
        self.events: list[dict[str, Any]] = []
        self.next_mission_number = 1
        self.settings: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                self.load_state()
                return
            except (OSError, ValueError, TypeError, KeyError):
                pass
        self.reset(persist=False)

    def _now(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _event(self, event_type: str, message: str, level: str = "INFO", **details: Any) -> None:
        self.events.append(
            {
                "time": self._now(),
                "sim_s": round(self.simulation_time, 1),
                "type": event_type,
                "level": level,
                "message": message,
                "details": details,
            }
        )
        max_events = int(self.settings.get("max_events", 500))
        self.events = self.events[-max_events:]

    def reset(self, persist: bool = True) -> None:
        with self.lock:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.settings = config.get("settings", {})
            self.drones = {}
            for item in config["drones"]:
                drone = Drone(**item)
                drone.home_x = item.get("home_x", drone.x)
                drone.home_y = item.get("home_y", drone.y)
                self.drones[drone.drone_id] = drone
            self.missions = {}
            self.events = []
            self.running = False
            self.simulation_time = 0.0
            self.next_mission_number = 1
            self._event("SISTEMA", "Simulación reiniciada con la configuración base.")
            if persist:
                self.save_state()

    def start(self) -> None:
        self.running = True
        self._event("CONTROL", "Simulación iniciada.")
        self.save_state()

    def pause(self) -> None:
        self.running = False
        self._event("CONTROL", "Simulación pausada.")
        self.save_state()

    def create_mission(
        self,
        name: str,
        x: float,
        y: float,
        priority: int,
        source: str = "SIMULADA",
        metadata: dict[str, Any] | None = None,
    ) -> Mission:
        with self.lock:
            mission_id = f"M-{self.next_mission_number:03d}"
            self.next_mission_number += 1
            mission = Mission(
                mission_id=mission_id,
                name=name.strip() or f"Misión {mission_id}",
                x=float(x),
                y=float(y),
                priority=max(1, min(5, int(priority))),
                source=source,
                created_at=self._now(),
                metadata=metadata or {},
            )
            self.missions[mission_id] = mission
            self._event(
                "MISION_REAL" if source == "REAL_YOLO" else "MISION_SIMULADA",
                f"Creada {mission_id}: {mission.name} en ({mission.x:.1f}, {mission.y:.1f}), prioridad {mission.priority}.",
                source=source,
            )
            self.assign_pending_missions()
            self.save_state()
            return mission

    def _candidate_scores(self, mission: Mission, excluded: set[str] | None = None) -> list[dict[str, Any]]:
        excluded = excluded or set()
        reserve = float(self.settings.get("battery_reserve", 12.0))
        energy_per_unit = float(self.settings.get("energy_per_unit", 0.08))
        candidates: list[dict[str, Any]] = []
        for drone in self.drones.values():
            distance = drone.distance_to(mission.x, mission.y)
            estimated_battery = distance * energy_per_unit
            eligible = (
                drone.drone_id not in excluded
                and drone.state == "DISPONIBLE"
                and drone.mission_id is None
                and drone.battery - estimated_battery >= reserve
            )
            if not eligible:
                continue
            score = 100.0 + mission.priority * 4.0 + drone.battery * 0.55 - distance * 1.8
            candidates.append(
                {
                    "drone": drone,
                    "distance": distance,
                    "estimated_battery": estimated_battery,
                    "score": score,
                }
            )
        return sorted(candidates, key=lambda item: (-item["score"], item["drone"].drone_id))

    def assign_pending_missions(self, excluded: set[str] | None = None) -> None:
        with self.lock:
            pending = sorted(
                (m for m in self.missions.values() if m.state == "PENDIENTE"),
                key=lambda m: (-m.priority, m.created_at, m.mission_id),
            )
            for mission in pending:
                candidates = self._candidate_scores(mission, excluded)
                if not candidates:
                    self._event(
                        "ASIGNACION_PENDIENTE",
                        f"{mission.mission_id} continúa pendiente: no hay drones disponibles con batería suficiente.",
                        level="WARN",
                    )
                    continue
                winner = candidates[0]
                drone: Drone = winner["drone"]
                drone.state = "EN_MISION"
                drone.mission_id = mission.mission_id
                mission.state = "ASIGNADA"
                mission.assigned_drone = drone.drone_id
                explanation = (
                    f"Seleccionado {drone.drone_id} para {mission.mission_id}: disponible, batería {drone.battery:.1f}%, "
                    f"distancia {winner['distance']:.1f} u, prioridad {mission.priority}/5 y puntuación {winner['score']:.1f}."
                )
                self._event(
                    "ASIGNACION",
                    explanation,
                    mission_id=mission.mission_id,
                    drone_id=drone.drone_id,
                    score=round(winner["score"], 2),
                )

    def fail_drone(self, drone_id: str) -> str | None:
        with self.lock:
            if drone_id not in self.drones:
                raise KeyError(f"Dron inexistente: {drone_id}")
            drone = self.drones[drone_id]
            old_mission_id = drone.mission_id
            drone.state = "FALLA"
            drone.mission_id = None
            self._event("FALLO", f"{drone_id} dejó de responder. El coordinador bloqueó nuevas asignaciones.", level="ERROR")
            if old_mission_id and old_mission_id in self.missions:
                mission = self.missions[old_mission_id]
                mission.state = "PENDIENTE"
                mission.assigned_drone = None
                self._event(
                    "REASIGNACION",
                    f"{old_mission_id} fue liberada tras el fallo de {drone_id}; se busca un agente sustituto.",
                    level="WARN",
                )
                self.assign_pending_missions(excluded={drone_id})
            self.save_state()
            return old_mission_id

    def tick(self, seconds: float = 0.6) -> None:
        with self.lock:
            if not self.running:
                return
            seconds = max(0.05, min(float(seconds), 3.0))
            self.simulation_time += seconds
            energy_per_unit = float(self.settings.get("energy_per_unit", 0.08))
            reserve = float(self.settings.get("battery_reserve", 12.0))
            for drone in self.drones.values():
                if drone.state != "EN_MISION" or not drone.mission_id:
                    continue
                mission = self.missions.get(drone.mission_id)
                if not mission or mission.state != "ASIGNADA":
                    drone.state = "DISPONIBLE"
                    drone.mission_id = None
                    continue
                distance = drone.distance_to(mission.x, mission.y)
                if distance <= 0.001:
                    self._complete_mission(drone, mission)
                    continue
                step = min(distance, drone.speed * seconds)
                drone.x += (mission.x - drone.x) / distance * step
                drone.y += (mission.y - drone.y) / distance * step
                drone.distance_travelled += step
                drone.battery = max(0.0, drone.battery - step * energy_per_unit)
                if step >= distance - 1e-9:
                    self._complete_mission(drone, mission)
                elif drone.battery <= reserve:
                    abandoned_id = drone.mission_id
                    drone.state = "BATERIA_BAJA"
                    drone.mission_id = None
                    mission.state = "PENDIENTE"
                    mission.assigned_drone = None
                    self._event(
                        "BATERIA_BAJA",
                        f"{drone.drone_id} liberó {abandoned_id} al alcanzar la reserva de batería ({drone.battery:.1f}%).",
                        level="WARN",
                    )
                    self.assign_pending_missions(excluded={drone.drone_id})
            self.save_state()

    def _complete_mission(self, drone: Drone, mission: Mission) -> None:
        drone.x, drone.y = mission.x, mission.y
        drone.state = "DISPONIBLE"
        drone.mission_id = None
        mission.state = "COMPLETADA"
        mission.completed_at = self._now()
        self._event(
            "MISION_COMPLETADA",
            f"{drone.drone_id} completó {mission.mission_id}: {mission.name}.",
            mission_id=mission.mission_id,
            drone_id=drone.drone_id,
        )
        self.assign_pending_missions()

    def register_real_detection(self, people: int, frames: int, video_name: str) -> None:
        self._event(
            "VISION_REAL",
            f"YOLO detectó {people} apariciones de personas en {frames} fotogramas del video {video_name}.",
            level="ALERT" if people else "INFO",
            people=people,
            frames=frames,
            video_name=video_name,
        )
        self.save_state()

    def load_demo(self) -> None:
        self.reset(persist=False)
        self.create_mission("Inspección del sector norte", 82, 76, 4, "SIMULADA")
        self.create_mission("Verificación de punto crítico", 20, 86, 5, "SIMULADA")
        self.start()

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "simulation_time": self.simulation_time,
            "next_mission_number": self.next_mission_number,
            "settings": self.settings,
            "drones": [asdict(d) for d in self.drones.values()],
            "missions": [asdict(m) for m in self.missions.values()],
            "events": self.events,
        }

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.parent / f"runtime_state.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            for attempt in range(6):
                try:
                    os.replace(temporary, self.state_path)
                    return
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.03 * (attempt + 1))
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    def load_state(self) -> None:
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.running = False
        self.simulation_time = float(data.get("simulation_time", 0.0))
        self.next_mission_number = int(data.get("next_mission_number", 1))
        self.settings = data.get("settings", {})
        self.drones = {item["drone_id"]: Drone(**item) for item in data["drones"]}
        self.missions = {item["mission_id"]: Mission(**item) for item in data.get("missions", [])}
        self.events = data.get("events", [])

