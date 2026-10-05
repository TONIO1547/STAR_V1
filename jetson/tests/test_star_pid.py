import os
import sys
import tempfile
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The realtime arbitration classes do not require OpenCV, while the lightweight
# CI image used for these unit tests does not ship the Jetson cv2 package.
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from star_predictive_pid import (  # noqa: E402
    AlphaBetaEstimator,
    PIDConfigError,
    TrackingController,
)
from star_realtime import PerformanceMode, PriorityInferenceGate  # noqa: E402


def controller(config_path=None):
    return TrackingController(90, 120, 0, 180, 90, 150, config_path=config_path)


class TrackingControllerTests(unittest.TestCase):
    def test_starts_disarmed_and_centered(self):
        tracking = controller()
        snapshot = tracking.snapshot()
        self.assertFalse(snapshot["config"]["tracking_enabled"])
        self.assertEqual(snapshot["telemetry"]["pan_deg"], 90.0)
        self.assertEqual(snapshot["telemetry"]["tilt_deg"], 120.0)

    def test_target_to_right_moves_pan_with_existing_mount_direction(self):
        tracking = controller()
        tracking.update_config({
            "tracking_enabled": True,
            "pan": {"kp": 0.1, "ki": 0.0, "kd": 0.0},
            "tilt": {"kp": 0.0, "ki": 0.0, "kd": 0.0},
        })
        tracking.update_target(420, 240, 640, 480, 0.9, now=10.0)
        tracking.update_target(420, 240, 640, 480, 0.9, now=10.1)
        self.assertLess(tracking.snapshot()["telemetry"]["pan_deg"], 90.0)

    def test_rate_controller_is_nearly_independent_from_detection_fps(self):
        def run(detection_step):
            tracking = controller()
            tracking.update_config({
                "tracking_enabled": True,
                "smoothing_alpha": 1.0,
                "pan": {
                    "kp": 0.1,
                    "ki": 0.0,
                    "kd": 0.0,
                    "max_rate_dps": 90.0,
                    "max_accel_dps2": 1000.0,
                    "deadband_px": 0.0,
                },
            })
            now = 100.0
            next_detection = now
            while now <= 101.0001:
                if now + 0.00001 >= next_detection:
                    tracking.observe_target(
                        420, 240, 640, 480, 0.9,
                        frame_time=now, now=now,
                    )
                    next_detection += detection_step
                tracking.control_step(now)
                now += 0.02
            return tracking.snapshot()["telemetry"]["pan_deg"]

        self.assertAlmostEqual(run(0.1), run(0.05), delta=0.6)

    def test_center_disarms_and_returns_command(self):
        tracking = controller()
        tracking.set_enabled(True)
        pan, tilt, snapshot = tracking.center()
        self.assertEqual((pan, tilt), (90, 120))
        self.assertFalse(snapshot["config"]["tracking_enabled"])

    def test_manual_position_disarms_clamps_and_updates_telemetry(self):
        tracking = controller()
        tracking.set_enabled(True)
        pan, tilt, snapshot = tracking.manual_position(999, 0)
        self.assertEqual((pan, tilt), (180, 90))
        self.assertFalse(snapshot["config"]["tracking_enabled"])
        self.assertEqual(snapshot["telemetry"]["mode"], "manual")
        self.assertEqual(snapshot["telemetry"]["pan_deg"], 180.0)
        self.assertEqual(snapshot["telemetry"]["tilt_deg"], 90.0)

    def test_persisted_values_are_reloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "pid.json")
            first = controller(path)
            first.update_config({"pan": {"kp": 0.123}}, persist=True)
            second = controller(path)
            self.assertEqual(second.snapshot()["config"]["pan"]["kp"], 0.123)

    def test_unknown_or_dangerous_values_are_rejected(self):
        tracking = controller()
        with self.assertRaises(PIDConfigError):
            tracking.update_config({"pan": {"kp": 99}})
        with self.assertRaises(PIDConfigError):
            tracking.update_config({"unexpected": 1})

    def test_alpha_beta_estimates_velocity_and_predicts_forward(self):
        estimator = AlphaBetaEstimator(1.0, 1.0)
        estimator.update(0.0, 10.0)
        _, velocity = estimator.update(10.0, 10.1)
        self.assertGreater(velocity, 0.0)
        self.assertGreater(estimator.predict(10.15), 10.0)

    def test_state_machine_coasts_then_reacquires(self):
        tracking = controller()
        tracking.set_enabled(True)
        tracking.observe_target(400, 240, 640, 480, 0.9, frame_time=10.0, now=10.0)
        tracking.control_step(10.10)
        self.assertEqual(tracking.snapshot()["telemetry"]["state"], "LOST")
        tracking.control_step(10.60)
        self.assertEqual(tracking.snapshot()["telemetry"]["state"], "REACQUIRE")
        self.assertTrue(tracking.needs_handoff())

    def test_servo_command_keeps_sub_degree_precision(self):
        tracking = controller()
        tracking.update_config({
            "tracking_enabled": True,
            "command_rate_hz": 60.0,
            "pan": {
                "kp": 0.1,
                "ki": 0.0,
                "kd": 0.0,
                "feedforward_gain": 0.0,
                "max_accel_dps2": 1000.0,
                "deadband_px": 0.0,
            },
        })
        tracking.observe_target(370, 240, 640, 480, 0.9, frame_time=20.0, now=20.0)
        command = tracking.control_step(20.02)
        self.assertIsNotNone(command)
        self.assertNotEqual(command[0], round(command[0]))

    def test_prediction_settings_are_validated_logically(self):
        tracking = controller()
        with self.assertRaises(PIDConfigError):
            tracking.update_config({"lost_grace_s": 0.3, "coast_time_s": 0.2})


class RealtimeBuildingBlockTests(unittest.TestCase):
    def test_canon_waiting_blocks_new_guet_inference(self):
        gate = PriorityInferenceGate()
        self.assertTrue(gate.begin_guet())
        # An active inference is enough to make the opportunistic path skip.
        self.assertFalse(gate.begin_guet())
        gate.end()
        self.assertTrue(gate.begin_canon())
        gate.end()

    def test_performance_mode_has_explicit_stream_consequences(self):
        mode = PerformanceMode()
        self.assertTrue(mode.snapshot()["stream_enabled"])
        maximum = mode.set("maximum")
        self.assertFalse(maximum["stream_enabled"])
        self.assertEqual(maximum["stream_fps"], 0.0)


if __name__ == "__main__":
    unittest.main()
