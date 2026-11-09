import unittest

import numpy as np

from drone_ai.tello import MIN_TAKEOFF_BATTERY, TelloController


class FakeFrameReader:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)


class FakeTello:
    def __init__(self, battery=80):
        self.battery = battery
        self.is_flying = False
        self.commands = []

    def connect(self):
        self.commands.append("connect")

    def get_current_state(self):
        return {"bat": self.battery, "h": 40 if self.is_flying else 0, "time": 3, "templ": 50, "temph": 54}

    def get_battery(self):
        return self.battery

    def streamon(self):
        self.commands.append("streamon")

    def streamoff(self):
        self.commands.append("streamoff")

    def get_frame_read(self):
        return FakeFrameReader()

    def takeoff(self):
        self.commands.append("takeoff")
        self.is_flying = True

    def land(self):
        self.commands.append("land")
        self.is_flying = False

    def move_forward(self, distance):
        self.commands.append(("forward", distance))

    def rotate_clockwise(self, degrees):
        self.commands.append(("cw", degrees))

    def rotate_counter_clockwise(self, degrees):
        self.commands.append(("ccw", degrees))

    def emergency(self):
        self.commands.append("emergency")
        self.is_flying = False

    def end(self):
        self.commands.append("end")


class TelloControllerTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeTello()
        self.controller = TelloController(lambda: self.fake)

    def test_connect_reads_telemetry_without_takeoff(self):
        status = self.controller.connect()
        self.assertTrue(status.connected)
        self.assertEqual(80, status.battery)
        self.assertNotIn("takeoff", self.fake.commands)

    def test_camera_returns_frame(self):
        self.controller.connect()
        self.controller.start_stream()
        frame = self.controller.get_frame()
        self.assertEqual((240, 320, 3), frame.shape)

    def test_takeoff_requires_explicit_authorization(self):
        self.controller.connect()
        with self.assertRaises(PermissionError):
            self.controller.takeoff(authorized=False)
        self.controller.takeoff(authorized=True)
        self.assertTrue(self.controller.snapshot.flying)

    def test_low_battery_blocks_takeoff(self):
        low = FakeTello(MIN_TAKEOFF_BATTERY - 1)
        controller = TelloController(lambda: low)
        controller.connect()
        with self.assertRaises(RuntimeError):
            controller.takeoff(authorized=True)

    def test_movement_requires_flight_and_sdk_distance(self):
        self.controller.connect()
        with self.assertRaises(RuntimeError):
            self.controller.move("forward", 30, authorized=True)
        self.controller.takeoff(authorized=True)
        with self.assertRaises(ValueError):
            self.controller.move("forward", 10, authorized=True)
        self.controller.move("forward", 30, authorized=True)
        self.assertIn(("forward", 30), self.fake.commands)

    def test_cannot_disconnect_while_flying(self):
        self.controller.connect()
        self.controller.takeoff(authorized=True)
        with self.assertRaises(RuntimeError):
            self.controller.disconnect()
        self.controller.land()
        self.controller.disconnect()
        self.assertFalse(self.controller.snapshot.connected)


if __name__ == "__main__":
    unittest.main()
