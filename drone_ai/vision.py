from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Callable


@dataclass
class VisionResult:
    output_path: str
    processed_frames: int
    frames_with_people: int
    person_detections: int
    max_people_in_frame: int


@dataclass
class LiveVisionResult:
    annotated_frame: object
    counts: dict[str, int]
    total_detections: int
    track_ids: list[int]


class LiveVisionProcessor:
    """Detección y seguimiento persistente para la cámara en vivo del Tello."""

    def __init__(self, weights: str = "yolo11n.pt"):
        from ultralytics import YOLO

        self.model = YOLO(weights)

    def process(self, frame, confidence: float = 0.35) -> LiveVisionResult:
        prediction = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=float(confidence),
            verbose=False,
        )[0]
        counts: Counter[str] = Counter()
        track_ids: list[int] = []
        if prediction.boxes is not None:
            classes = prediction.boxes.cls.int().cpu().tolist()
            for class_id in classes:
                counts[str(prediction.names[int(class_id)])] += 1
            if prediction.boxes.id is not None:
                track_ids = [int(value) for value in prediction.boxes.id.int().cpu().tolist()]
        return LiveVisionResult(
            annotated_frame=prediction.plot(),
            counts=dict(sorted(counts.items())),
            total_detections=sum(counts.values()),
            track_ids=sorted(set(track_ids)),
        )


def dependency_status() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
    except Exception as exc:
        return False, f"Dependencias de visión no disponibles: {exc}"
    return True, "OpenCV y Ultralytics YOLO disponibles."


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    confidence: float = 0.35,
    weights: str = "yolo11n.pt",
    max_frames: int = 1800,
    progress: Callable[[float], None] | None = None,
) -> VisionResult:
    """Ejecuta inferencia YOLO real, conserva solo la clase persona y anota el video."""
    import cv2
    from ultralytics import YOLO

    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_path.with_name(f"{output_path.stem}_opencv.mp4")
    model = YOLO(weights)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("No se pudo abrir el video cargado.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    limit = min(total, max_frames)
    writer = cv2.VideoWriter(str(raw_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("No se pudo crear el video de salida.")

    processed = frames_with_people = person_detections = max_people = 0
    try:
        while processed < limit:
            ok, frame = capture.read()
            if not ok:
                break
            prediction = model.predict(frame, conf=float(confidence), classes=[0], verbose=False)[0]
            count = 0
            if prediction.boxes is not None:
                for box in prediction.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    score = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 220, 120), 2)
                    cv2.putText(
                        frame,
                        f"Persona {score:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (36, 220, 120),
                        2,
                    )
                    count += 1
            cv2.putText(
                frame,
                f"Deteccion REAL YOLO | Personas: {count}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (20, 20, 240),
                2,
            )
            writer.write(frame)
            processed += 1
            person_detections += count
            max_people = max(max_people, count)
            frames_with_people += int(count > 0)
            if progress and processed % 5 == 0:
                progress(processed / max(1, limit))
    finally:
        capture.release()
        writer.release()
    if progress:
        progress(1.0)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        conversion = subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_output_path),
                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if conversion.returncode != 0:
            raise RuntimeError(f"FFmpeg no pudo generar el video compatible: {conversion.stderr.strip()}")
        raw_output_path.unlink(missing_ok=True)
    else:
        raw_output_path.replace(output_path)
    return VisionResult(str(output_path), processed, frames_with_people, person_detections, max_people)

