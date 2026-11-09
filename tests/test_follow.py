import unittest

from drone_ai.follow import PersonTarget, compute_follow_command, select_person_target


class FollowLogicTests(unittest.TestCase):
    def test_target_selection_keeps_requested_tracking_id(self):
        targets = [
            PersonTarget(1, 0.9, 0, 0, 100, 100),
            PersonTarget(2, 0.8, 0, 0, 200, 200),
        ]
        self.assertEqual(1, select_person_target(targets, preferred_id=1).track_id)

    def test_missing_target_stops_all_motion(self):
        command = compute_follow_command(None, 640, 480)
        self.assertEqual(0, command.forward_back)
        self.assertEqual(0, command.yaw)

    def test_off_center_target_rotates_without_advancing(self):
        target = PersonTarget(1, 0.9, 500, 100, 620, 400)
        command = compute_follow_command(target, 640, 480)
        self.assertGreater(command.yaw, 0)
        self.assertEqual(0, command.forward_back)

    def test_centered_distant_target_advances_slowly(self):
        target = PersonTarget(1, 0.9, 290, 170, 350, 310)
        command = compute_follow_command(target, 640, 480, max_speed=15)
        self.assertEqual(15, command.forward_back)
        self.assertEqual(0, command.yaw)

    def test_close_target_never_moves_backward(self):
        target = PersonTarget(1, 0.9, 100, 20, 540, 460)
        command = compute_follow_command(target, 640, 480)
        self.assertEqual(0, command.forward_back)
        self.assertEqual(0, command.yaw)


if __name__ == "__main__":
    unittest.main()
