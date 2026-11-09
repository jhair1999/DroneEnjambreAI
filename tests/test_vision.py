import unittest

from drone_ai.vision import LiveVisionResult, VisionResult


class VisionContractTests(unittest.TestCase):
    def test_result_distinguishes_real_video_metrics(self):
        result = VisionResult("salida.mp4", 100, 12, 18, 3)
        self.assertEqual(18, result.person_detections)
        self.assertGreater(result.frames_with_people, 0)

    def test_live_result_keeps_classes_and_tracking_ids(self):
        result = LiveVisionResult(object(), {"person": 2, "car": 1}, 3, [4, 8], [])
        self.assertEqual(2, result.counts["person"])
        self.assertEqual(3, result.total_detections)
        self.assertEqual([4, 8], result.track_ids)


if __name__ == "__main__":
    unittest.main()
