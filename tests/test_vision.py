import unittest

from drone_ai.vision import VisionResult


class VisionContractTests(unittest.TestCase):
    def test_result_distinguishes_real_video_metrics(self):
        result = VisionResult("salida.mp4", 100, 12, 18, 3)
        self.assertEqual(18, result.person_detections)
        self.assertGreater(result.frames_with_people, 0)


if __name__ == "__main__":
    unittest.main()
