#!/usr/bin/env python3
"""S.T.A.R. detection history for the Jetson (Python 3.6 compatible).

The module records one event when a new object appears, saves an annotated
snapshot in ``report/``, exposes the authenticated Flask API used by
star-ai.fr, and removes events and images once they are 24 hours old.

It can also patch the current ``stream_drone_pid.py`` safely and idempotently:

    python3 star_detection_report.py --patch-main stream_drone_pid.py
"""

from __future__ import print_function

import argparse
import datetime
import json
import math
import os
import re
import shutil
import threading
import time

import cv2
from flask import jsonify, send_from_directory


REPORT_RETENTION_SECONDS = 24 * 60 * 60
TRACK_ABSENCE_SECONDS = 2.5
PATCH_VERSION = "2026-08-08-v2"
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
EVENTS_FILE = os.path.join(REPORT_DIR, "events.jsonl")
SAFE_IMAGE_RE = re.compile(r"^[A-Za-z0-9_.-]+\.jpg$")

_lock = threading.RLock()
_active_tracks = {}
_cleanup_started = False


def _ensure_report_dir():
    if not os.path.isdir(REPORT_DIR):
        os.makedirs(REPORT_DIR)


def _read_events_locked():
    events = []
    try:
        with open(EVENTS_FILE, "r") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        events.append(event)
                except (TypeError, ValueError):
                    continue
    except IOError:
        pass
    return events


def _write_events_locked(events):
    _ensure_report_dir()
    temporary = EVENTS_FILE + ".tmp"
    with open(temporary, "w") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.rename(temporary, EVENTS_FILE)


def _purge_locked(now=None):
    _ensure_report_dir()
    now = time.time() if now is None else float(now)
    cutoff_ms = int((now - REPORT_RETENTION_SECONDS) * 1000)
    kept = []
    referenced = set()

    for event in _read_events_locked():
        timestamp_ms = event.get("timestamp_ms", 0)
        image_name = event.get("image", "")
        if isinstance(timestamp_ms, (int, float)) and timestamp_ms >= cutoff_ms:
            if isinstance(image_name, str) and SAFE_IMAGE_RE.match(image_name):
                referenced.add(image_name)
            kept.append(event)
        elif isinstance(image_name, str) and SAFE_IMAGE_RE.match(image_name):
            try:
                os.remove(os.path.join(REPORT_DIR, image_name))
            except OSError:
                pass

    for name in os.listdir(REPORT_DIR):
        if not SAFE_IMAGE_RE.match(name) or name in referenced:
            continue
        path = os.path.join(REPORT_DIR, name)
        try:
            if os.path.getmtime(path) < now - REPORT_RETENTION_SECONDS:
                os.remove(path)
        except OSError:
            pass

    _write_events_locked(kept)


def _cleanup_loop():
    while True:
        time.sleep(10 * 60)
        with _lock:
            _purge_locked()


def _iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(first, second):
    first_x = (first[0] + first[2]) / 2.0
    first_y = (first[1] + first[3]) / 2.0
    second_x = (second[0] + second[2]) / 2.0
    second_y = (second[1] + second[3]) / 2.0
    return math.hypot(first_x - second_x, first_y - second_y)


def _same_object(first, second):
    if _iou(first, second) >= 0.18:
        return True
    first_size = max(first[2] - first[0], first[3] - first[1], 1.0)
    second_size = max(second[2] - second[0], second[3] - second[1], 1.0)
    return _center_distance(first, second) <= max(first_size, second_size) * 0.85


def _clean_text(value, fallback):
    text = str(value or fallback).strip()
    text = re.sub(r"[^A-Za-z0-9_. -]+", "", text)
    return text[:48] or fallback


def _save_event_locked(camera, label, confidence, frame, box, now):
    _ensure_report_dir()
    camera = _clean_text(camera, "CAM")
    label = _clean_text(label, "Objet")
    timestamp = datetime.datetime.fromtimestamp(now)
    stamp = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    filename = "%s_%s_%s.jpg" % (
        stamp,
        camera.replace(" ", "-"),
        label.replace(" ", "-"),
    )
    image_path = os.path.join(REPORT_DIR, filename)

    snapshot = frame.copy()
    height, width = snapshot.shape[:2]
    x1 = max(0, min(width - 1, int(box[0])))
    y1 = max(0, min(height - 1, int(box[1])))
    x2 = max(0, min(width - 1, int(box[2])))
    y2 = max(0, min(height - 1, int(box[3])))
    cv2.rectangle(snapshot, (x1, y1), (x2, y2), (223, 255, 82), 2)
    caption = "%s %.1f%%" % (label, confidence * 100.0)
    cv2.putText(
        snapshot,
        caption,
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (223, 255, 82),
        2,
    )
    if snapshot.shape[1] > 960:
        scale = 960.0 / snapshot.shape[1]
        snapshot = cv2.resize(
            snapshot,
            (960, max(1, int(snapshot.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if not cv2.imwrite(image_path, snapshot, [int(cv2.IMWRITE_JPEG_QUALITY), 82]):
        return

    event = {
        "id": filename[:-4],
        "timestamp_ms": int(now * 1000),
        "detected_at": timestamp.isoformat(),
        "label": label,
        "camera": camera,
        "confidence": round(float(confidence), 4),
        "image": filename,
    }
    with open(EVENTS_FILE, "a") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def update_detection_report(camera, frame, boxes, scores, label="Drone"):
    """Update object tracks and save each newly appearing detected object.

    ``boxes`` are expected in the model's 640x640 xyxy coordinate space, as
    returned by the existing S.T.A.R. inference code after NMS.
    """
    if frame is None:
        return
    now = time.time()
    camera = _clean_text(camera, "CAM")
    height, width = frame.shape[:2]
    detections = []

    for raw_box, raw_score in zip(list(boxes), list(scores)):
        try:
            confidence = float(raw_score)
            box = (
                float(raw_box[0]) * width / 640.0,
                float(raw_box[1]) * height / 640.0,
                float(raw_box[2]) * width / 640.0,
                float(raw_box[3]) * height / 640.0,
            )
        except (IndexError, TypeError, ValueError):
            continue
        if confidence < 0.0 or confidence > 1.0 or box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append((box, confidence))

    detections.sort(key=lambda item: item[1], reverse=True)

    with _lock:
        tracks = [
            track
            for track in _active_tracks.get(camera, [])
            if now - track["last_seen"] <= TRACK_ABSENCE_SECONDS
        ]
        used_tracks = set()

        for box, confidence in detections:
            matched_index = None
            for index, track in enumerate(tracks):
                if index in used_tracks:
                    continue
                if _same_object(box, track["box"]):
                    matched_index = index
                    break

            if matched_index is None:
                tracks.append({"box": box, "last_seen": now})
                used_tracks.add(len(tracks) - 1)
                _save_event_locked(camera, label, confidence, frame, box, now)
            else:
                tracks[matched_index]["box"] = box
                tracks[matched_index]["last_seen"] = now
                used_tracks.add(matched_index)

        _active_tracks[camera] = tracks


def install_detection_report(app):
    """Register protected report routes on the existing Flask application."""
    global _cleanup_started
    with _lock:
        _ensure_report_dir()
        _purge_locked()
        if not _cleanup_started:
            thread = threading.Thread(target=_cleanup_loop, daemon=True)
            thread.start()
            _cleanup_started = True

    @app.route("/api/detections", methods=["GET"])
    def star_detection_events():
        with _lock:
            _purge_locked()
            events = _read_events_locked()
        events.sort(key=lambda item: item.get("timestamp_ms", 0), reverse=True)
        return jsonify({
            "ok": True,
            "retention_hours": 24,
            "count": len(events),
            "events": events,
        })

    @app.route("/api/detections/images/<filename>", methods=["GET"])
    def star_detection_image(filename):
        if not SAFE_IMAGE_RE.match(filename):
            return jsonify({"ok": False, "error": "Invalid filename"}), 404
        return send_from_directory(REPORT_DIR, filename, conditional=True)


IMPORT_LINE = (
    "from star_detection_report import install_detection_report, "
    "update_detection_report"
)


def _patch_capture_segment(segment, camera):
    """Insert report hooks while preserving the loop's original indentation."""
    empty_pattern = re.compile(
        r"update_detection_report\(\s*['\"]%s['\"]\s*,\s*frame\s*,\s*\[\]\s*,\s*\[\]\s*\)"
        % re.escape(camera)
    )
    live_pattern = re.compile(
        r"update_detection_report\(\s*['\"]%s['\"]\s*,\s*frame\s*,\s*boxes\s*,\s*scores\s*\)"
        % re.escape(camera)
    )

    if not empty_pattern.search(segment):
        inference_pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]+)pred\s*=\s*run_inference\(\s*frame\s*\)[ \t]*$"
        )
        matches = list(inference_pattern.finditer(segment))
        if len(matches) != 1:
            raise RuntimeError("Unable to locate inference call for %s" % camera)
        match = matches[0]
        call = '%supdate_detection_report("%s", frame, [], [])\n' % (
            match.group("indent"),
            camera,
        )
        segment = segment[:match.end()] + "\n" + call + segment[match.end():]

    if not live_pattern.search(segment):
        scores_pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]+)scores\s*=\s*scores\s*"
            r"\[\s*keep\s*\]\.cpu\(\)\.numpy\(\)[ \t]*$"
        )
        matches = list(scores_pattern.finditer(segment))
        if len(matches) != 1:
            raise RuntimeError("Unable to locate NMS scores for %s" % camera)
        match = matches[0]
        call = '%supdate_detection_report("%s", frame, boxes, scores)\n' % (
            match.group("indent"),
            camera,
        )
        segment = segment[:match.end()] + "\n" + call + segment[match.end():]
    return segment


def _find_camera_segments(source):
    """Find the two inference loops by their contents, not their function names."""
    functions = list(re.finditer(
        r"(?m)^def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^\n]*\)\s*:",
        source,
    ))
    candidates = []
    for index, match in enumerate(functions):
        end = functions[index + 1].start() if index + 1 < len(functions) else len(source)
        segment = source[match.start():end]
        if (
            re.search(r"pred\s*=\s*run_inference\(\s*frame\s*\)", segment)
            and re.search(
                r"scores\s*=\s*scores\s*\[\s*keep\s*\]\.cpu\(\)\.numpy\(\)",
                segment,
            )
        ):
            candidates.append((match.start(), end, match.group("name")))
    return candidates


def patch_main_program(path):
    print("S.T.A.R. detection report patch %s" % PATCH_VERSION)
    path = os.path.abspath(path)
    with open(path, "r") as handle:
        source = handle.read()
    original = source

    if IMPORT_LINE not in source:
        import_candidates = [
            "from star_runtime_api import install_runtime_api, get_confidence\n",
            "from star_auth import install_star_auth\n",
            "from flask import Flask, Response, render_template_string, request, jsonify\n",
        ]
        for candidate in import_candidates:
            if candidate in source:
                source = source.replace(candidate, candidate + IMPORT_LINE + "\n", 1)
                break
        else:
            raise RuntimeError("Unable to locate the import section")

    install_line = "install_detection_report(app)"
    if install_line not in source:
        install_candidates = [
            "install_runtime_api(app)\n",
            "install_star_auth(app)\n",
            "app = Flask(__name__)\n",
        ]
        for candidate in install_candidates:
            if candidate in source:
                source = source.replace(candidate, candidate + install_line + "\n", 1)
                break
        else:
            raise RuntimeError("Unable to locate Flask application setup")

    camera_segments = _find_camera_segments(source)
    if len(camera_segments) != 2:
        names = ", ".join(item[2] for item in camera_segments) or "none"
        raise RuntimeError(
            "Expected two camera inference loops; found %d (%s)"
            % (len(camera_segments), names)
        )

    # Work backwards so replacing the first loop cannot invalidate later offsets.
    for (start, end, _name), camera in reversed(list(zip(
        camera_segments,
        ("CAM-01", "CAM-02"),
    ))):
        patched = _patch_capture_segment(source[start:end], camera)
        source = source[:start] + patched + source[end:]

    if source == original:
        print("Detection report is already installed; no change required.")
        return

    backup = path + ".before-detection-report"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    temporary = path + ".detection-report.tmp"
    with open(temporary, "w") as handle:
        handle.write(source)
    os.rename(temporary, path)
    print("Detection report installed in %s" % path)
    print("Backup: %s" % backup)


def main():
    parser = argparse.ArgumentParser(description="S.T.A.R. detection report")
    parser.add_argument("--version", action="version", version=PATCH_VERSION)
    parser.add_argument("--patch-main", metavar="PATH", help="patch stream_drone_pid.py")
    args = parser.parse_args()
    if not args.patch_main:
        parser.error("use --patch-main PATH")
    patch_main_program(args.patch_main)


if __name__ == "__main__":
    main()
