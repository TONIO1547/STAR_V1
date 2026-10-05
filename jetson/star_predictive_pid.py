"""Predictive, thread-safe visual controller for S.T.A.R.

The controller receives timestamped detections, estimates target position and
velocity with a light alpha-beta filter, predicts the present/future error and
drives a rate PID at a cadence independent from the inference FPS.

Compatible with Python 3.6 (JetPack 4.6.6).
"""

from __future__ import print_function

import copy
import json
import math
import os
import threading
import time


class PIDConfigError(ValueError):
    pass


CONFIG_LIMITS = {
    "smoothing_alpha": {"minimum": 0.05, "maximum": 1.0, "step": 0.05},
    "estimator_beta": {"minimum": 0.0, "maximum": 1.0, "step": 0.01},
    "prediction_time_s": {"minimum": 0.0, "maximum": 0.25, "step": 0.005},
    "max_prediction_px": {"minimum": 0.0, "maximum": 300.0, "step": 5.0},
    "lost_grace_s": {"minimum": 0.02, "maximum": 0.50, "step": 0.01},
    "coast_time_s": {"minimum": 0.05, "maximum": 0.50, "step": 0.01},
    "target_timeout_s": {"minimum": 0.10, "maximum": 2.0, "step": 0.05},
    "command_rate_hz": {"minimum": 5.0, "maximum": 60.0, "step": 1.0},
    "axis": {
        "kp": {"minimum": 0.0, "maximum": 0.50, "step": 0.005},
        "ki": {"minimum": 0.0, "maximum": 0.05, "step": 0.0005},
        "kd": {"minimum": 0.0, "maximum": 0.10, "step": 0.001},
        "feedforward_gain": {"minimum": 0.0, "maximum": 0.10, "step": 0.001},
        "max_rate_dps": {"minimum": 2.0, "maximum": 180.0, "step": 1.0},
        "max_accel_dps2": {"minimum": 20.0, "maximum": 1000.0, "step": 10.0},
        "deadband_px": {"minimum": 0.0, "maximum": 100.0, "step": 1.0},
    },
}


DEFAULT_CONFIG = {
    "tracking_enabled": False,
    "smoothing_alpha": 0.70,
    "estimator_beta": 0.15,
    "prediction_time_s": 0.040,
    "max_prediction_px": 80.0,
    "lost_grace_s": 0.080,
    "coast_time_s": 0.200,
    "target_timeout_s": 0.550,
    "command_rate_hz": 30.0,
    "pan": {
        "kp": 0.080,
        "ki": 0.0020,
        "kd": 0.004,
        "feedforward_gain": 0.015,
        "max_rate_dps": 35.0,
        "max_accel_dps2": 250.0,
        "deadband_px": 8.0,
        "invert": True,
    },
    "tilt": {
        "kp": 0.070,
        "ki": 0.0020,
        "kd": 0.004,
        "feedforward_gain": 0.012,
        "max_rate_dps": 25.0,
        "max_accel_dps2": 180.0,
        "deadband_px": 8.0,
        "invert": True,
    },
}


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _finite_number(value, label, minimum, maximum):
    if isinstance(value, bool):
        raise PIDConfigError("%s must be a number" % label)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise PIDConfigError("%s must be a number" % label)
    if not math.isfinite(parsed):
        raise PIDConfigError("%s must be finite" % label)
    if parsed < minimum or parsed > maximum:
        raise PIDConfigError(
            "%s must be between %s and %s" % (label, minimum, maximum)
        )
    return parsed


def _merge_config(current, patch):
    if not isinstance(patch, dict):
        raise PIDConfigError("config must be an object")
    global_names = {
        "tracking_enabled",
        "smoothing_alpha",
        "estimator_beta",
        "prediction_time_s",
        "max_prediction_px",
        "lost_grace_s",
        "coast_time_s",
        "target_timeout_s",
        "command_rate_hz",
        "pan",
        "tilt",
    }
    unknown = sorted(set(patch.keys()) - global_names)
    if unknown:
        raise PIDConfigError("unknown setting: %s" % unknown[0])
    result = copy.deepcopy(current)
    if "tracking_enabled" in patch:
        if not isinstance(patch["tracking_enabled"], bool):
            raise PIDConfigError("tracking_enabled must be a boolean")
        result["tracking_enabled"] = patch["tracking_enabled"]
    for name in (
        "smoothing_alpha",
        "estimator_beta",
        "prediction_time_s",
        "max_prediction_px",
        "lost_grace_s",
        "coast_time_s",
        "target_timeout_s",
        "command_rate_hz",
    ):
        if name in patch:
            limit = CONFIG_LIMITS[name]
            result[name] = _finite_number(
                patch[name], name, limit["minimum"], limit["maximum"]
            )
    if result["lost_grace_s"] > result["coast_time_s"]:
        raise PIDConfigError("lost_grace_s must be <= coast_time_s")
    if result["coast_time_s"] > result["target_timeout_s"]:
        raise PIDConfigError("coast_time_s must be <= target_timeout_s")
    for axis_name in ("pan", "tilt"):
        if axis_name not in patch:
            continue
        axis_patch = patch[axis_name]
        if not isinstance(axis_patch, dict):
            raise PIDConfigError("%s must be an object" % axis_name)
        allowed_axis = set(CONFIG_LIMITS["axis"].keys()) | {"invert"}
        unknown_axis = sorted(set(axis_patch.keys()) - allowed_axis)
        if unknown_axis:
            raise PIDConfigError(
                "unknown %s setting: %s" % (axis_name, unknown_axis[0])
            )
        for name, value in axis_patch.items():
            label = "%s.%s" % (axis_name, name)
            if name == "invert":
                if not isinstance(value, bool):
                    raise PIDConfigError("%s must be a boolean" % label)
                result[axis_name][name] = value
                continue
            limit = CONFIG_LIMITS["axis"][name]
            result[axis_name][name] = _finite_number(
                value, label, limit["minimum"], limit["maximum"]
            )
    return result


class AlphaBetaEstimator(object):
    def __init__(self, alpha, beta):
        self.configure(alpha, beta)
        self.reset()

    def configure(self, alpha, beta):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def reset(self):
        self.position = None
        self.velocity = 0.0
        self.timestamp = None

    def update(self, measurement, timestamp):
        measurement = float(measurement)
        timestamp = float(timestamp)
        if self.position is None or self.timestamp is None:
            self.position = measurement
            self.velocity = 0.0
            self.timestamp = timestamp
            return self.position, self.velocity
        dt = timestamp - self.timestamp
        if dt <= 0.0:
            return self.position, self.velocity
        dt = _clamp(dt, 0.005, 0.50)
        predicted = self.position + self.velocity * dt
        residual = measurement - predicted
        self.position = predicted + self.alpha * residual
        self.velocity = self.velocity + (self.beta / dt) * residual
        self.velocity = _clamp(self.velocity, -4000.0, 4000.0)
        self.timestamp = timestamp
        return self.position, self.velocity

    def predict(self, timestamp):
        if self.position is None or self.timestamp is None:
            return None
        dt = _clamp(float(timestamp) - self.timestamp, 0.0, 0.75)
        return self.position + self.velocity * dt


class VelocityPID(object):
    def __init__(self, config):
        self.integral = 0.0
        self.previous_error = None
        self.last_time = None
        self.configure(config)

    def configure(self, config):
        self.kp = float(config["kp"])
        self.ki = float(config["ki"])
        self.kd = float(config["kd"])
        self.max_rate = float(config["max_rate_dps"])
        self.deadband = float(config["deadband_px"])

    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        self.last_time = None

    def compute(self, error, now, feedforward=0.0):
        effective_error = 0.0 if abs(error) <= self.deadband else float(error)
        dt = 0.0 if self.last_time is None else _clamp(now - self.last_time, 0.001, 0.25)
        derivative = 0.0
        if dt > 0.0 and self.previous_error is not None:
            derivative = (effective_error - self.previous_error) / dt
        if dt > 0.0:
            if effective_error == 0.0:
                candidate_integral = self.integral * max(0.0, 1.0 - 3.0 * dt)
            else:
                candidate_integral = _clamp(
                    self.integral + effective_error * dt, -2000.0, 2000.0
                )
        else:
            candidate_integral = self.integral
        raw_output = (
            self.kp * effective_error
            + self.ki * candidate_integral
            + self.kd * derivative
            + float(feedforward)
        )
        output = _clamp(raw_output, -self.max_rate, self.max_rate)
        if not (raw_output != output and effective_error * raw_output > 0.0):
            self.integral = candidate_integral
        self.previous_error = effective_error
        self.last_time = now
        return output, dt, effective_error


STATE_TO_MODE = {
    "IDLE": "idle",
    "SEARCH": "acquiring",
    "ACQUIRE": "acquiring",
    "TRACK": "tracking",
    "LOST": "lost",
    "REACQUIRE": "reacquiring",
    "MANUAL": "manual",
    "CENTERED": "centered",
    "DETECTED_DISARMED": "detected_disarmed",
}


class TrackingController(object):
    def __init__(self, pan_center, tilt_center, pan_min, pan_max, tilt_min,
                 tilt_max, config_path=None):
        self.lock = threading.RLock()
        self.pan_center = float(pan_center)
        self.tilt_center = float(tilt_center)
        self.pan_min = float(pan_min)
        self.pan_max = float(pan_max)
        self.tilt_min = float(tilt_min)
        self.tilt_max = float(tilt_max)
        self.config_path = config_path
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self._load_config()
        self.pan_pid = VelocityPID(self.config["pan"])
        self.tilt_pid = VelocityPID(self.config["tilt"])
        self.x_estimator = AlphaBetaEstimator(
            self.config["smoothing_alpha"], self.config["estimator_beta"]
        )
        self.y_estimator = AlphaBetaEstimator(
            self.config["smoothing_alpha"], self.config["estimator_beta"]
        )
        self.pan_position = self.pan_center
        self.tilt_position = self.tilt_center
        self.state = "IDLE"
        self.last_target_time = None
        self.last_measurement_time = None
        self.acquire_started = None
        self.last_command_time = 0.0
        self.last_control_time = None
        self.last_sent_pan = None
        self.last_sent_tilt = None
        self.previous_pan_rate = 0.0
        self.previous_tilt_rate = 0.0
        self.revision = 1
        self.telemetry = {
            "mode": "idle", "state": "IDLE", "target_detected": False,
            "control_active": False, "confidence": None,
            "raw_error_x_px": None, "raw_error_y_px": None,
            "error_x_px": None, "error_y_px": None,
            "target_velocity_x_px_s": 0.0, "target_velocity_y_px_s": 0.0,
            "prediction_x_px": 0.0, "prediction_y_px": 0.0,
            "frame_age_ms": None, "pan_rate_dps": 0.0,
            "tilt_rate_dps": 0.0,
        }

    def _load_config(self):
        if not self.config_path:
            return
        try:
            with open(self.config_path, "r") as handle:
                self.config = _merge_config(self.config, json.load(handle))
        except (IOError, OSError, ValueError, PIDConfigError):
            self.config = copy.deepcopy(DEFAULT_CONFIG)
        # A persisted tuning file may contain an old armed state. Reboots are
        # deliberately safe: only an authenticated runtime action can re-arm.
        self.config["tracking_enabled"] = False

    def _save_config_locked(self):
        if not self.config_path:
            raise PIDConfigError("PID persistence path is not configured")
        directory = os.path.dirname(self.config_path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        temporary = self.config_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(self.config, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.rename(temporary, self.config_path)

    def _set_state_locked(self, state):
        self.state = state
        self.telemetry["state"] = state
        self.telemetry["mode"] = STATE_TO_MODE.get(state, state.lower())

    def _reset_pid_locked(self):
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.previous_pan_rate = 0.0
        self.previous_tilt_rate = 0.0
        self.last_control_time = None
        self.telemetry["pan_rate_dps"] = 0.0
        self.telemetry["tilt_rate_dps"] = 0.0

    def _reset_target_locked(self):
        self.x_estimator.reset()
        self.y_estimator.reset()
        self.last_target_time = None
        self.last_measurement_time = None
        self.acquire_started = None
        self.telemetry["target_detected"] = False
        self.telemetry["confidence"] = None
        self._reset_pid_locked()

    def is_enabled(self):
        with self.lock:
            return bool(self.config["tracking_enabled"])

    def update_config(self, patch, persist=False):
        with self.lock:
            was_enabled = self.config["tracking_enabled"]
            self.config = _merge_config(self.config, patch)
            self.pan_pid.configure(self.config["pan"])
            self.tilt_pid.configure(self.config["tilt"])
            self.x_estimator.configure(self.config["smoothing_alpha"], self.config["estimator_beta"])
            self.y_estimator.configure(self.config["smoothing_alpha"], self.config["estimator_beta"])
            self._reset_pid_locked()
            self.revision += 1
            if not self.config["tracking_enabled"]:
                self._set_state_locked("IDLE")
                self.telemetry["control_active"] = False
            elif not was_enabled:
                self._set_state_locked("SEARCH")
            if persist:
                self._save_config_locked()
            return self._snapshot_locked()

    def set_enabled(self, enabled, persist=False):
        if not isinstance(enabled, bool):
            raise PIDConfigError("enabled must be a boolean")
        return self.update_config({"tracking_enabled": enabled}, persist=persist)

    def reset(self):
        with self.lock:
            self._reset_target_locked()
            self._set_state_locked("SEARCH" if self.config["tracking_enabled"] else "IDLE")
            self.revision += 1
            return self._snapshot_locked()

    def center(self, persist=False):
        with self.lock:
            self.config["tracking_enabled"] = False
            self.pan_position = self.pan_center
            self.tilt_position = self.tilt_center
            self.last_sent_pan = self.pan_position
            self.last_sent_tilt = self.tilt_position
            self.last_command_time = time.monotonic()
            self._reset_target_locked()
            self._set_state_locked("CENTERED")
            self.telemetry["control_active"] = False
            self.revision += 1
            if persist:
                self._save_config_locked()
            return self.pan_position, self.tilt_position, self._snapshot_locked()

    def manual_position(self, pan, tilt):
        with self.lock:
            self.config["tracking_enabled"] = False
            self.pan_position = _clamp(float(pan), self.pan_min, self.pan_max)
            self.tilt_position = _clamp(float(tilt), self.tilt_min, self.tilt_max)
            self.last_sent_pan = self.pan_position
            self.last_sent_tilt = self.tilt_position
            self.last_command_time = time.monotonic()
            self._reset_target_locked()
            self._set_state_locked("MANUAL")
            self.telemetry["control_active"] = False
            self.revision += 1
            return self.pan_position, self.tilt_position, self._snapshot_locked()

    def needs_handoff(self):
        with self.lock:
            return bool(self.config["tracking_enabled"] and self.state in ("SEARCH", "REACQUIRE"))

    def handoff(self, pan, tilt):
        with self.lock:
            if not self.config["tracking_enabled"] or self.state not in ("SEARCH", "REACQUIRE", "ACQUIRE"):
                return None
            self.pan_position = _clamp(float(pan), self.pan_min, self.pan_max)
            self.tilt_position = _clamp(float(tilt), self.tilt_min, self.tilt_max)
            # A Guet handoff must not reuse the stale Canon velocity/error that
            # triggered REACQUIRE. The next Canon box starts a clean estimate.
            self._reset_target_locked()
            self._set_state_locked("ACQUIRE")
            self.acquire_started = time.monotonic()
            self.telemetry["control_active"] = False
            self.last_command_time = time.monotonic()
            self.last_sent_pan = self.pan_position
            self.last_sent_tilt = self.tilt_position
            return self.pan_position, self.tilt_position

    def observe_target(self, center_x, center_y, width, height, confidence,
                       frame_time=None, now=None):
        now = time.monotonic() if now is None else float(now)
        frame_time = now if frame_time is None else float(frame_time)
        with self.lock:
            if self.last_measurement_time is not None and frame_time <= self.last_measurement_time:
                return False
            raw_x = float(center_x) - float(width) / 2.0
            raw_y = float(center_y) - float(height) / 2.0
            filtered_x, velocity_x = self.x_estimator.update(raw_x, frame_time)
            filtered_y, velocity_y = self.y_estimator.update(raw_y, frame_time)
            self.last_target_time = now
            self.last_measurement_time = frame_time
            self.telemetry.update({
                "target_detected": True,
                "confidence": round(float(confidence), 4),
                "raw_error_x_px": round(raw_x, 2), "raw_error_y_px": round(raw_y, 2),
                "error_x_px": round(filtered_x, 2), "error_y_px": round(filtered_y, 2),
                "target_velocity_x_px_s": round(velocity_x, 2),
                "target_velocity_y_px_s": round(velocity_y, 2),
                "frame_age_ms": round(max(0.0, (now - frame_time) * 1000.0), 2),
            })
            if self.config["tracking_enabled"]:
                self._set_state_locked("TRACK")
                self.acquire_started = None
            else:
                self._set_state_locked("DETECTED_DISARMED")
                self.telemetry["control_active"] = False
            return True

    def _bounded_prediction(self, estimator, timestamp):
        predicted = estimator.predict(timestamp)
        if predicted is None:
            return None
        maximum = self.config["max_prediction_px"]
        return _clamp(predicted, estimator.position - maximum, estimator.position + maximum)

    def control_step(self, now=None):
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            if not self.config["tracking_enabled"]:
                self.telemetry["control_active"] = False
                return None
            if self.last_target_time is None or self.last_measurement_time is None:
                if (
                    self.state == "ACQUIRE"
                    and self.acquire_started is not None
                    and now - self.acquire_started < 0.40
                ):
                    self.telemetry["control_active"] = False
                    return None
                self._set_state_locked("SEARCH")
                self.telemetry["control_active"] = False
                return None
            target_age = max(0.0, now - self.last_target_time)
            if target_age > self.config["target_timeout_s"]:
                if self.state != "REACQUIRE":
                    self._reset_target_locked()
                self._set_state_locked("REACQUIRE")
                self.telemetry["target_detected"] = False
                self.telemetry["control_active"] = False
                return None
            if target_age > self.config["coast_time_s"]:
                self._set_state_locked("LOST")
                self.telemetry["target_detected"] = False
                self.telemetry["control_active"] = False
                self.previous_pan_rate = 0.0
                self.previous_tilt_rate = 0.0
                return None
            self._set_state_locked("LOST" if target_age > self.config["lost_grace_s"] else "TRACK")
            predict_at = now + self.config["prediction_time_s"]
            error_x = self._bounded_prediction(self.x_estimator, predict_at)
            error_y = self._bounded_prediction(self.y_estimator, predict_at)
            if error_x is None or error_y is None: return None
            pan_ff = self.x_estimator.velocity * self.config["pan"]["feedforward_gain"]
            tilt_ff = self.y_estimator.velocity * self.config["tilt"]["feedforward_gain"]
            pan_rate, pan_dt, _ = self.pan_pid.compute(error_x, now, pan_ff)
            tilt_rate, tilt_dt, _ = self.tilt_pid.compute(error_y, now, tilt_ff)
            control_dt = max(pan_dt, tilt_dt)
            if control_dt <= 0.0: control_dt = 1.0 / self.config["command_rate_hz"]
            pan_direction = -1.0 if self.config["pan"]["invert"] else 1.0
            tilt_direction = -1.0 if self.config["tilt"]["invert"] else 1.0
            requested_pan = pan_direction * pan_rate
            requested_tilt = tilt_direction * tilt_rate
            pan_delta = self.config["pan"]["max_accel_dps2"] * control_dt
            tilt_delta = self.config["tilt"]["max_accel_dps2"] * control_dt
            actual_pan = _clamp(requested_pan, self.previous_pan_rate - pan_delta, self.previous_pan_rate + pan_delta)
            actual_tilt = _clamp(requested_tilt, self.previous_tilt_rate - tilt_delta, self.previous_tilt_rate + tilt_delta)
            self.previous_pan_rate, self.previous_tilt_rate = actual_pan, actual_tilt
            self.last_control_time = now
            self.pan_position = _clamp(self.pan_position + actual_pan * control_dt, self.pan_min, self.pan_max)
            self.tilt_position = _clamp(self.tilt_position + actual_tilt * control_dt, self.tilt_min, self.tilt_max)
            self.telemetry.update({
                "control_active": True, "error_x_px": round(error_x, 2),
                "error_y_px": round(error_y, 2),
                "prediction_x_px": round(error_x - self.x_estimator.position, 2),
                "prediction_y_px": round(error_y - self.y_estimator.position, 2),
                "pan_rate_dps": round(actual_pan, 2), "tilt_rate_dps": round(actual_tilt, 2),
                "frame_age_ms": round(max(0.0, (now - self.last_measurement_time) * 1000.0), 2),
            })
            interval = 1.0 / self.config["command_rate_hz"]
            due = now - self.last_command_time >= interval
            changed = self.last_sent_pan is None or abs(self.pan_position - self.last_sent_pan) >= 0.02 or abs(self.tilt_position - self.last_sent_tilt) >= 0.02
            if not due or not changed: return None
            self.last_command_time = now
            self.last_sent_pan, self.last_sent_tilt = self.pan_position, self.tilt_position
            return self.pan_position, self.tilt_position

    def update_target(self, center_x, center_y, width, height, confidence,
                      now=None, frame_time=None):
        now = time.monotonic() if now is None else float(now)
        self.observe_target(center_x, center_y, width, height, confidence,
                            frame_time=now if frame_time is None else frame_time,
                            now=now)
        return self.control_step(now)

    def mark_target_missed(self, now=None):
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            if self.last_target_time is None: return False
            age = max(0.0, now - self.last_target_time)
            previous = self.state
            self.telemetry["target_detected"] = False
            if not self.config["tracking_enabled"]:
                self._set_state_locked("IDLE")
            elif age > self.config["target_timeout_s"]:
                if self.state != "REACQUIRE":
                    self._reset_target_locked()
                self._set_state_locked("REACQUIRE")
                self.telemetry["control_active"] = False
            elif age > self.config["lost_grace_s"]:
                self._set_state_locked("LOST")
            return self.state != previous

    def snapshot(self):
        with self.lock: return self._snapshot_locked()

    def _snapshot_locked(self):
        now = time.monotonic()
        age = None if self.last_target_time is None else max(0, int((now - self.last_target_time) * 1000.0))
        telemetry = copy.deepcopy(self.telemetry)
        telemetry.update({"pan_deg": round(self.pan_position, 3),
                          "tilt_deg": round(self.tilt_position, 3),
                          "target_age_ms": age})
        return {"ok": True, "revision": self.revision,
                "config": copy.deepcopy(self.config),
                "defaults": copy.deepcopy(DEFAULT_CONFIG),
                "limits": copy.deepcopy(CONFIG_LIMITS),
                "telemetry": telemetry}
