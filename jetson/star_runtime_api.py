"""Runtime telemetry and AI-threshold API for the S.T.A.R. Jetson server.

Compatible with Python 3.6 and Flask 2.0.x.
"""

import glob
import os
import threading

from flask import jsonify, request


_CONFIDENCE_MIN = 0.20
_CONFIDENCE_MAX = 0.90
_confidence = 0.25
_confidence_lock = threading.Lock()


def get_confidence():
    """Return the confidence threshold currently used by inference."""
    with _confidence_lock:
        return _confidence


def _set_confidence(value):
    global _confidence
    with _confidence_lock:
        _confidence = value


def _jetson_temperatures():
    """Read the Linux thermal zones exposed by Jetson Linux."""
    readings = []

    for zone_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(os.path.join(zone_path, "temp"), "r") as handle:
                raw_value = float(handle.read().strip())

            # Jetson normally reports millidegrees Celsius. Keep compatibility
            # with kernels that expose degrees directly.
            temperature_c = raw_value / 1000.0 if abs(raw_value) > 1000 else raw_value
            if not -40.0 <= temperature_c <= 150.0:
                continue

            type_path = os.path.join(zone_path, "type")
            try:
                with open(type_path, "r") as handle:
                    sensor = handle.read().strip()
            except (IOError, OSError):
                sensor = os.path.basename(zone_path)

            readings.append({
                "sensor": sensor or os.path.basename(zone_path),
                "temperature_c": round(temperature_c, 1),
            })
        except (IOError, OSError, TypeError, ValueError):
            continue

    return readings


def install_runtime_api(app):
    """Register protected telemetry routes on the existing Flask app."""

    @app.route("/api/confidence", methods=["GET", "POST"])
    def confidence_api():
        if request.method == "GET":
            return jsonify({
                "ok": True,
                "confidence": get_confidence(),
                "minimum": _CONFIDENCE_MIN,
                "maximum": _CONFIDENCE_MAX,
            })

        payload = request.get_json(silent=True) or {}
        try:
            confidence = float(payload.get("confidence"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid confidence"}), 400

        if not _CONFIDENCE_MIN <= confidence <= _CONFIDENCE_MAX:
            return jsonify({
                "ok": False,
                "error": "confidence must be between 0.20 and 0.90",
            }), 400

        _set_confidence(confidence)
        return jsonify({"ok": True, "confidence": get_confidence()})

    @app.route("/api/jetson-temperature", methods=["GET"])
    def jetson_temperature_api():
        readings = _jetson_temperatures()
        if not readings:
            return jsonify({"ok": False, "error": "no thermal sensor"}), 503

        hottest = max(readings, key=lambda reading: reading["temperature_c"])
        return jsonify({
            "ok": True,
            "temperature_c": hottest["temperature_c"],
            "sensor": hottest["sensor"],
            "sensors": readings,
        })

