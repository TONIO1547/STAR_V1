"""Thread-safe visual PID controller for the S.T.A.R. Jetson service.

The controller converts image error (pixels) into a bounded servo angular
velocity (degrees/second).  This makes the behaviour independent from the
inference FPS, unlike adding a fixed angle on every detection frame.

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
    "target_timeout_s": {"minimum": 0.10, "maximum": 2.0, "step": 0.05},
    "command_rate_hz": {"minimum": 2.0, "maximum": 30.0, "step": 1.0},
    "axis": {
        "kp": {"minimum": 0.0, "maximum": 0.50, "step": 0.005},
        "ki": {"minimum": 0.0, "maximum": 0.05, "step": 0.0005},
        "kd": {"minimum": 0.0, "maximum": 0.10, "step": 0.001},
        "max_rate_dps": {"minimum": 2.0, "maximum": 90.0, "step": 1.0},
        "deadband_px": {"minimum": 0.0, "maximum": 100.0, "step": 1.0},
    },
}


DEFAULT_CONFIG = {
    # Deliberately disabled at boot.  The operator must explicitly arm motion.
    "tracking_enabled": False,
    "smoothing_alpha": 0.30,
    "target_timeout_s": 0.45,
    "command_rate_hz": 15.0,
    "pan": {
        "kp": 0.080,
        "ki": 0.0020,
        "kd": 0.004,
        "max_rate_dps": 35.0,
        "deadband_px": 8.0,
        "invert": True,
    },
    "tilt": {
        "kp": 0.070,
        "ki": 0.0020,
        "kd": 0.004,
        "max_rate_dps": 25.0,
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

    allowed = {
        "tracking_enabled",
        "smoothing_alpha",
        "target_timeout_s",
        "command_rate_hz",
        "pan",
        "tilt",
    }
    unknown = sorted(set(patch.keys()) - allowed)
    if unknown:
        raise PIDConfigError("unknown setting: %s" % unknown[0])

    result = copy.deepcopy(current)

    if "tracking_enabled" in patch:
        if not isinstance(patch["tracking_enabled"], bool):
            raise PIDConfigError("tracking_enabled must be a boolean")
        result["tracking_enabled"] = patch["tracking_enabled"]

    for name in ("smoothing_alpha", "target_timeout_s", "command_rate_hz"):
        if name in patch:
            limit = CONFIG_LIMITS[name]
            result[name] = _finite_number(
                patch[name], name, limit["minimum"], limit["maximum"]
            )

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


class VelocityPID(object):
    """PID whose output is a bounded angular velocity in degrees/second."""

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

    def compute(self, error, now):
        effective_error = 0.0 if abs(error) <= self.deadband else float(error)

        if self.last_time is None:
            dt = 0.0
        else:
            # Avoid derivative spikes after a stall and ignore impossible dt.
            dt = _clamp(float(now - self.last_time), 0.001, 0.25)

        derivative = 0.0
        if dt > 0.0 and self.previous_error is not None:
            derivative = (effective_error - self.previous_error) / dt

        if dt > 0.0:
            if effective_error == 0.0:
                # Slowly remove residual integral inside the deadband.
                candidate_integral = self.integral * max(0.0, 1.0 - 3.0 * dt)
            else:
                candidate_integral = _clamp(
                    self.integral + effective_error * dt,
                    -2000.0,
                    2000.0,
                )
        else:
            candidate_integral = self.integral

        raw_output = (
            self.kp * effective_error
            + self.ki * candidate_integral
            + self.kd * derivative
        )
        output = _clamp(raw_output, -self.max_rate, self.max_rate)

        # Conditional integration prevents wind-up when the rate saturates.
        saturated_same_direction = (
            raw_output != output and effective_error * raw_output > 0.0
        )
        if not saturated_same_direction:
            self.integral = candidate_integral

        self.previous_error = effective_error
        self.last_time = now
        return output, dt, effective_error


class TrackingController(object):
    def __init__(
        self,
        pan_center,
        tilt_center,
        pan_min,
        pan_max,
        tilt_min,
        tilt_max,
        config_path=None,
    ):
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

        self.pan_position = self.pan_center
        self.tilt_position = self.tilt_center
        self.filtered_error_x = None
        self.filtered_error_y = None
        self.last_target_time = None
        self.last_command_time = 0.0
        self.last_sent_pan = None
        self.last_sent_tilt = None
        self.revision = 1
        self.telemetry = {
            "mode": "idle",
            "target_detected": False,
            "control_active": False,
            "confidence": None,
            "raw_error_x_px": None,
            "raw_error_y_px": None,
            "error_x_px": None,
            "error_y_px": None,
            "pan_rate_dps": 0.0,
            "tilt_rate_dps": 0.0,
        }

    def _load_config(self):
        if not self.config_path:
            return
        try:
            with open(self.config_path, "r") as handle:
                saved = json.load(handle)
            self.config = _merge_config(self.config, saved)
        except (IOError, OSError, ValueError, PIDConfigError):
            # A malformed optional file must never prevent the vision service
            # from starting; defaults remain deliberately disarmed.
            self.config = copy.deepcopy(DEFAULT_CONFIG)

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

    def _reset_control_locked(self):
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.filtered_error_x = None
        self.filtered_error_y = None
        self.telemetry["pan_rate_dps"] = 0.0
        self.telemetry["tilt_rate_dps"] = 0.0

    def is_enabled(self):
        with self.lock:
            return bool(self.config["tracking_enabled"])

    def update_config(self, patch, persist=False):
        with self.lock:
            previous_enabled = self.config["tracking_enabled"]
            self.config = _merge_config(self.config, patch)
            self.pan_pid.configure(self.config["pan"])
            self.tilt_pid.configure(self.config["tilt"])
            self._reset_control_locked()
            self.revision += 1
            if not self.config["tracking_enabled"]:
                self.telemetry["mode"] = "idle"
                self.telemetry["control_active"] = False
            elif not previous_enabled:
                self.telemetry["mode"] = "acquiring"
            if persist:
                self._save_config_locked()
            return self._snapshot_locked()

    def set_enabled(self, enabled, persist=False):
        if not isinstance(enabled, bool):
            raise PIDConfigError("enabled must be a boolean")
        return self.update_config({"tracking_enabled": enabled}, persist=persist)

    def reset(self):
        with self.lock:
            self._reset_control_locked()
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
            self._reset_control_locked()
            self.telemetry["mode"] = "centered"
            self.telemetry["control_active"] = False
            self.revision += 1
            if persist:
                self._save_config_locked()
            return (
                int(round(self.pan_position)),
                int(round(self.tilt_position)),
                self._snapshot_locked(),
            )

    def manual_position(self, pan, tilt):
        """Take manual control and keep internal position in sync with hardware."""
        with self.lock:
            self.config["tracking_enabled"] = False
            self.pan_position = _clamp(float(pan), self.pan_min, self.pan_max)
            self.tilt_position = _clamp(float(tilt), self.tilt_min, self.tilt_max)
            self.last_sent_pan = self.pan_position
            self.last_sent_tilt = self.tilt_position
            self.last_command_time = time.monotonic()
            self._reset_control_locked()
            self.telemetry["mode"] = "manual"
            self.telemetry["control_active"] = False
            self.revision += 1
            return (
                int(round(self.pan_position)),
                int(round(self.tilt_position)),
                self._snapshot_locked(),
            )

    def handoff(self, pan, tilt):
        """Point the tracking camera at the search camera's current pose."""
        with self.lock:
            if not self.config["tracking_enabled"]:
                return None
            self.pan_position = _clamp(float(pan), self.pan_min, self.pan_max)
            self.tilt_position = _clamp(float(tilt), self.tilt_min, self.tilt_max)
            self._reset_control_locked()
            self.telemetry["mode"] = "acquiring"
            self.telemetry["control_active"] = False
            self.last_command_time = time.monotonic()
            self.last_sent_pan = self.pan_position
            self.last_sent_tilt = self.tilt_position
            return int(round(self.pan_position)), int(round(self.tilt_position))

    def update_target(self, center_x, center_y, width, height, confidence, now=None):
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            raw_x = float(center_x) - float(width) / 2.0
            raw_y = float(center_y) - float(height) / 2.0
            alpha = float(self.config["smoothing_alpha"])
            if self.filtered_error_x is None:
                self.filtered_error_x = raw_x
                self.filtered_error_y = raw_y
            else:
                self.filtered_error_x = (
                    alpha * raw_x + (1.0 - alpha) * self.filtered_error_x
                )
                self.filtered_error_y = (
                    alpha * raw_y + (1.0 - alpha) * self.filtered_error_y
                )

            self.last_target_time = now
            self.telemetry.update({
                "target_detected": True,
                "confidence": round(float(confidence), 4),
                "raw_error_x_px": round(raw_x, 2),
                "raw_error_y_px": round(raw_y, 2),
                "error_x_px": round(self.filtered_error_x, 2),
                "error_y_px": round(self.filtered_error_y, 2),
            })

            if not self.config["tracking_enabled"]:
                self.telemetry["mode"] = "detected_disarmed"
                self.telemetry["control_active"] = False
                return None

            pan_rate, pan_dt, _ = self.pan_pid.compute(
                self.filtered_error_x, now
            )
            tilt_rate, tilt_dt, _ = self.tilt_pid.compute(
                self.filtered_error_y, now
            )
            dt = max(pan_dt, tilt_dt)
            pan_direction = -1.0 if self.config["pan"]["invert"] else 1.0
            tilt_direction = -1.0 if self.config["tilt"]["invert"] else 1.0

            if dt > 0.0:
                self.pan_position = _clamp(
                    self.pan_position + pan_direction * pan_rate * dt,
                    self.pan_min,
                    self.pan_max,
                )
                self.tilt_position = _clamp(
                    self.tilt_position + tilt_direction * tilt_rate * dt,
                    self.tilt_min,
                    self.tilt_max,
                )

            self.telemetry["mode"] = "tracking"
            self.telemetry["control_active"] = True
            self.telemetry["pan_rate_dps"] = round(pan_direction * pan_rate, 2)
            self.telemetry["tilt_rate_dps"] = round(tilt_direction * tilt_rate, 2)

            command_interval = 1.0 / float(self.config["command_rate_hz"])
            due = now - self.last_command_time >= command_interval
            changed = (
                self.last_sent_pan is None
                or abs(self.pan_position - self.last_sent_pan) >= 0.25
                or abs(self.tilt_position - self.last_sent_tilt) >= 0.25
            )
            if not due or not changed:
                return None

            self.last_command_time = now
            self.last_sent_pan = self.pan_position
            self.last_sent_tilt = self.tilt_position
            return int(round(self.pan_position)), int(round(self.tilt_position))

    def mark_target_missed(self, now=None):
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            if self.last_target_time is None:
                return False
            if now - self.last_target_time <= self.config["target_timeout_s"]:
                return False
            transitioned = bool(self.telemetry["target_detected"])
            self.telemetry["target_detected"] = False
            self.telemetry["control_active"] = False
            self.telemetry["mode"] = (
                "acquiring" if self.config["tracking_enabled"] else "idle"
            )
            self.telemetry["confidence"] = None
            self._reset_control_locked()
            return transitioned

    def snapshot(self):
        with self.lock:
            return self._snapshot_locked()

    def _snapshot_locked(self):
        age_ms = None
        if self.last_target_time is not None:
            age_ms = max(0, int((time.monotonic() - self.last_target_time) * 1000.0))
        telemetry = copy.deepcopy(self.telemetry)
        telemetry.update({
            "pan_deg": round(self.pan_position, 2),
            "tilt_deg": round(self.tilt_position, 2),
            "target_age_ms": age_ms,
        })
        return {
            "ok": True,
            "revision": self.revision,
            "config": copy.deepcopy(self.config),
            "defaults": copy.deepcopy(DEFAULT_CONFIG),
            "limits": copy.deepcopy(CONFIG_LIMITS),
            "telemetry": telemetry,
        }
