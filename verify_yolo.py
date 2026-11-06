from __future__ import annotations

import json
import os
from pathlib import Path

import cv2

from drone_ai.vision import process_video


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / "ultralytics"))
image_path = ROOT / "data" / "bus.jpg"
video_path = ROOT / "data" / "video_prueba_personas.mp4"
output_path = ROOT / "outputs" / "vision" / "video_prueba_detectado.mp4"
report_path = ROOT / "outputs" / "vision" / "verificacion_yolo.json"

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError("Falta data/bus.jpg; descarga una imagen de prueba con personas.")
height, width = image.shape[:2]
video_path.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
if not writer.isOpened():
    raise RuntimeError("No se pudo crear el video de prueba.")
for _ in range(12):
    writer.write(image)
writer.release()

result = process_video(video_path, output_path, confidence=0.25, max_frames=12)
report = {
    "status": "OK" if result.person_detections > 0 else "SIN_DETECCIONES",
    "input": str(video_path),
    "output": result.output_path,
    "processed_frames": result.processed_frames,
    "frames_with_people": result.frames_with_people,
    "person_detections": result.person_detections,
    "max_people_in_frame": result.max_people_in_frame,
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
if result.person_detections <= 0:
    raise SystemExit("YOLO no detectó personas en el video de prueba.")
