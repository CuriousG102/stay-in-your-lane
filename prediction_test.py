import unittest

import prediction

class FakeTelemetry:
    def __init__(self, fake_speed, fake_steering):
        self.speed = fake_speed
        self.steering = fake_steering

class TelemetryAfterDeltaTimeTestMixin:
    pos = None  # (x, z)
    rot_y = None # degrees
    s_angle = None  # degrees
    delta_time = None  # degrees
    speed = None  # units / s

    expected_pos_prediction = None  # (x, z)
    expected_pos_prediction_places = 7
    expected_rot_y_prediction = None  # degrees
    expected_rot_y_prediction_places = 5

    def test_works(self):
        t = FakeTelemetry(self.speed, self.s_angle)
        pos, rot_y = prediction.telemetry_after_delta_time(
            t, self.pos, self.rot_y, self.delta_time)
        self.assertAlmostEqual(
            pos[0], self.expected_pos_prediction[0],
            places=self.expected_pos_prediction_places)
        self.assertAlmostEqual(
            pos[1], self.expected_pos_prediction[1],
            places=self.expected_pos_prediction_places)
        self.assertAlmostEqual(
            rot_y, self.expected_rot_y_prediction,
            places=self.expected_rot_y_prediction_places)

class TestVerticalTravel(unittest.TestCase, TelemetryAfterDeltaTimeTestMixin):
    pos = (0, 0)
    rot_y = 0
    s_angle = 0
    delta_time = 1
    speed = 5

    expected_pos_prediction = (0, 5)
    expected_rot_y_prediction = 0

if __name__ == '__main__':
    unittest.main()
