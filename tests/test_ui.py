import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DashboardContractTests(unittest.TestCase):
    def test_required_controls_are_present(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        required_labels = [
            "▶ Iniciar",
            "⏸ Pausar",
            "↻ Reiniciar",
            "🎬 Demo",
            "Crear y asignar",
            "🧪 Preparar prueba de fallo",
            "⚠ Simular fallo",
            "Procesar con YOLO",
            "🔌 Conectar Tello",
            "🚀 DESPEGAR",
            "🛬 ATERRIZAR",
            "Activar YOLO + seguimiento ByteTrack",
            "Analizar fotografía con YOLO",
            "Crear misión desde esta detección",
        ]
        for label in required_labels:
            with self.subTest(label=label):
                self.assertIn(label, source)

    def test_dashboard_exposes_all_required_sections(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        for section in [
            "Operación",
            "Visión IA",
            "Prototipo celular",
            "DJI Tello físico",
            "Historial",
            "Guion de demostración",
        ]:
            with self.subTest(section=section):
                self.assertIn(section, source)


if __name__ == "__main__":
    unittest.main()
