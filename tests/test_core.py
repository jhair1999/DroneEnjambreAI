import json
import tempfile
import unittest
from pathlib import Path

from drone_ai.core import SimulationEngine


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        root = Path(self.temp.name)
        self.config = root / "config.json"
        self.state = root / "state.json"
        self.config.write_text(
            json.dumps(
                {
                    "settings": {"battery_reserve": 10, "energy_per_unit": 0.05, "max_events": 200},
                    "drones": [
                        {"drone_id": "D1", "x": 0, "y": 0, "battery": 90, "speed": 10},
                        {"drone_id": "D2", "x": 20, "y": 0, "battery": 90, "speed": 10},
                        {"drone_id": "D3", "x": 50, "y": 50, "battery": 80, "speed": 10},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.engine = SimulationEngine(self.config, self.state)

    def tearDown(self):
        self.temp.cleanup()

    def test_initializes_three_independent_agents(self):
        self.assertEqual(3, len(self.engine.drones))
        self.assertTrue(all(d.state == "DISPONIBLE" for d in self.engine.drones.values()))

    def test_assignment_uses_distance_and_battery_rules(self):
        mission = self.engine.create_mission("Inspección", 2, 0, 4)
        self.assertEqual("D1", mission.assigned_drone)
        self.assertEqual("EN_MISION", self.engine.drones["D1"].state)
        self.assertTrue(any(e["type"] == "ASIGNACION" and "distancia" in e["message"] for e in self.engine.events))

    def test_movement_is_visible_and_mission_completes(self):
        mission = self.engine.create_mission("Corta", 5, 0, 3)
        self.engine.start()
        before = self.engine.drones[mission.assigned_drone].x
        self.engine.tick(0.2)
        after = self.engine.drones[mission.assigned_drone].x
        self.assertGreater(after, before)
        self.engine.tick(1.0)
        self.assertEqual("COMPLETADA", mission.state)

    def test_failure_reassigns_active_mission(self):
        mission = self.engine.create_mission("Rescate", 8, 0, 5)
        first = mission.assigned_drone
        self.engine.fail_drone(first)
        self.assertNotEqual(first, mission.assigned_drone)
        self.assertEqual("ASIGNADA", mission.state)
        self.assertEqual("FALLA", self.engine.drones[first].state)
        self.assertTrue(any(e["type"] == "REASIGNACION" for e in self.engine.events))

    def test_runtime_state_survives_restart(self):
        mission = self.engine.create_mission("Persistente", 15, 0, 2)
        reloaded = SimulationEngine(self.config, self.state)
        self.assertIn(mission.mission_id, reloaded.missions)
        self.assertFalse(reloaded.running)
        self.assertGreaterEqual(len(reloaded.events), 2)


if __name__ == "__main__":
    unittest.main()
