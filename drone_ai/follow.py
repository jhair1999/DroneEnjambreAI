from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonTarget:
    track_id: int | None
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class FollowCommand:
    forward_back: int
    yaw: int
    reason: str


def select_person_target(targets: list[PersonTarget], preferred_id: int | None = None) -> PersonTarget | None:
    """Conserva el ID elegido; si no existe, selecciona la persona más visible."""
    if preferred_id is not None:
        for target in targets:
            if target.track_id == preferred_id:
                return target
    return max(targets, key=lambda target: target.area, default=None)


def compute_follow_command(
    target: PersonTarget | None,
    frame_width: int,
    frame_height: int,
    max_speed: int = 15,
) -> FollowCommand:
    """Control conservador: gira, avanza o se queda inmóvil.

    No modifica altura y nunca retrocede automáticamente. Si no hay objetivo,
    devuelve velocidad cero para que el Tello mantenga posición.
    """
    max_speed = max(8, min(20, int(max_speed)))
    if target is None or frame_width <= 0 or frame_height <= 0:
        return FollowCommand(0, 0, "Objetivo perdido: mantener posición")

    center_x = (target.x1 + target.x2) / 2.0
    horizontal_error = (center_x - frame_width / 2.0) / (frame_width / 2.0)
    area_ratio = target.area / float(frame_width * frame_height)
    dead_zone = 0.10

    if abs(horizontal_error) > dead_zone:
        yaw = int(max(-max_speed, min(max_speed, horizontal_error * max_speed * 1.8)))
        if 0 < abs(yaw) < 8:
            yaw = 8 if yaw > 0 else -8
        return FollowCommand(0, yaw, "Girar para centrar a la persona")

    if area_ratio < 0.09:
        return FollowCommand(max_speed, 0, "Persona centrada y distante: avanzar lentamente")
    if area_ratio > 0.28:
        return FollowCommand(0, 0, "Persona demasiado cerca: mantener posición")
    return FollowCommand(0, 0, "Distancia de seguimiento alcanzada: mantener posición")
