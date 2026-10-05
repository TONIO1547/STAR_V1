"""S.T.A.R. low-latency dual-camera tracking service.

CAM01 / Canon owns the priority TensorRT path. CAM02 / Guet runs at a bounded
low rate and never blocks a waiting Canon inference. Capture, inference,
control, UDP output and JPEG encoding use independent latest-data-only paths.

Python 3.6 / JetPack 4.6.6 compatible.
"""

from __future__ import print_function

import glob
import json
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np
import serial
import torch
import torchvision
from flask import Flask, Response, jsonify, request

sys.path.append('/mnt/sdcard/lib/python3.6/site-packages')

from device_auth import install_device_auth
from models.common import DetectMultiBackend
from star_predictive_pid import PIDConfigError, TrackingController
from star_realtime import (
    LatestFrameCamera,
    LatestServoSender,
    PerformanceMode,
    PriorityInferenceGate,
    RealtimeMetrics,
    StreamHub,
)
from utils.torch_utils import select_device


def env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def env_float(name, default, minimum, maximum):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, min(maximum, value))


def env_int(name, default, minimum, maximum):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, min(maximum, value))


# ============================================================================
# PARAMETRES TEMPS REEL — point unique de reglage Jetson
# ============================================================================
MODEL_PATH = os.environ.get('STAR_MODEL_PATH', 'new_best_v1.engine')
CANON_CAMERA = os.environ.get(
    'STAR_CANON_CAMERA',
    '/dev/v4l/by-path/platform-70090000.xusb-usb-0:2.1:1.0-video-index0',
)
GUET_CAMERA = os.environ.get(
    'STAR_GUET_CAMERA',
    '/dev/v4l/by-path/platform-70090000.xusb-usb-0:2.2:1.0-video-index0',
)

# Keep 640 for the current fixed-shape TensorRT engine and small distant drones.
CANON_AI_SIZE = env_int('STAR_CANON_AI_SIZE', 640, 320, 960)
GUET_AI_SIZE = env_int('STAR_GUET_AI_SIZE', 640, 320, 960)
USE_FP16 = env_bool('STAR_TENSORRT_FP16', True)
CANON_INFERENCE_HZ = env_float('STAR_CANON_INFERENCE_HZ', 0.0, 0.0, 60.0)
GUET_INFERENCE_HZ = env_float('STAR_GUET_INFERENCE_HZ', 2.0, 0.2, 5.0)
CONTROL_LOOP_HZ = env_float('STAR_CONTROL_LOOP_HZ', 50.0, 20.0, 100.0)
GUET_SERVO_HZ = env_float('STAR_GUET_SERVO_HZ', 20.0, 5.0, 50.0)
GUET_PAN_SCAN_DPS = env_float('STAR_GUET_PAN_SCAN_DPS', 12.0, 1.0, 60.0)
GUET_TILT_SCAN_DPS = env_float('STAR_GUET_TILT_SCAN_DPS', 5.0, 0.5, 30.0)
HANDOFF_MIN_INTERVAL_S = env_float(
    'STAR_HANDOFF_MIN_INTERVAL_S', 0.15, 0.05, 1.0
)
HANDOFF_MIN_CONFIDENCE = env_float(
    'STAR_HANDOFF_MIN_CONFIDENCE', 0.40, 0.20, 0.90
)
METRICS_PRINT_INTERVAL_S = env_float(
    'STAR_METRICS_PRINT_INTERVAL_S', 2.0, 1.0, 10.0
)

STREAM_MAX_FPS = env_float('STAR_STREAM_MAX_FPS', 6.0, 1.0, 15.0)
STREAM_JPEG_QUALITY = env_int('STAR_STREAM_JPEG_QUALITY', 55, 40, 80)
STREAM_MAX_WIDTH = env_int('STAR_STREAM_MAX_WIDTH', 512, 320, 960)
STREAM_MAX_HEIGHT = env_int('STAR_STREAM_MAX_HEIGHT', 384, 240, 540)

# Physical limits are duplicated and enforced in the ESP32 firmware as a final
# hardware-side safety barrier. Do not widen one side without validating both.
PAN_MIN, PAN_MAX = 0.0, 180.0
TILT_MIN, TILT_MAX = 90.0, 150.0
PAN_CENTER, TILT_CENTER = 90.0, 120.0

THERMAL_PORT = os.environ.get('STAR_THERMAL_PORT', '/dev/ttyTHS1')
THERMAL_FRAME_LEN = 1544
FUSION_WIDTH, FUSION_HEIGHT = 640, 480
FUSION_VISIBLE_ALPHA = env_float('STAR_FUSION_VISIBLE_ALPHA', 0.25, 0.0, 1.0)
FUSION_THERMAL_ALPHA = 1.0 - FUSION_VISIBLE_ALPHA
FUSION_SCALE_X = env_float('STAR_FUSION_SCALE_X', 1.0, 0.5, 2.0)
FUSION_SCALE_Y = env_float('STAR_FUSION_SCALE_Y', 1.0, 0.5, 2.0)
FUSION_OFFSET_X = env_int('STAR_FUSION_OFFSET_X', 0, -640, 640)
FUSION_OFFSET_Y = env_int('STAR_FUSION_OFFSET_Y', 0, -480, 480)


app = Flask(__name__)
install_device_auth(app)

metrics = RealtimeMetrics(window_s=2.0)
performance_mode = PerformanceMode({
    'stream_fps': STREAM_MAX_FPS,
    'stream_width': STREAM_MAX_WIDTH,
    'stream_height': STREAM_MAX_HEIGHT,
    'jpeg_quality': STREAM_JPEG_QUALITY,
})

CONFIDENCE_MIN, CONFIDENCE_MAX = 0.20, 0.90
confidence_threshold = env_float('STAR_CONFIDENCE', 0.25, CONFIDENCE_MIN, CONFIDENCE_MAX)
confidence_lock = threading.Lock()


def get_confidence():
    with confidence_lock:
        return confidence_threshold


def set_confidence(value):
    global confidence_threshold
    with confidence_lock:
        confidence_threshold = float(value)


def read_jetson_temperatures():
    readings = []
    for zone_path in sorted(glob.glob('/sys/class/thermal/thermal_zone*')):
        try:
            with open(os.path.join(zone_path, 'temp'), 'r') as handle:
                raw_value = float(handle.read().strip())
            temperature_c = raw_value / 1000.0 if abs(raw_value) > 1000 else raw_value
            if not -40.0 <= temperature_c <= 150.0:
                continue
            try:
                with open(os.path.join(zone_path, 'type'), 'r') as handle:
                    sensor = handle.read().strip()
            except (IOError, OSError):
                sensor = os.path.basename(zone_path)
            readings.append({
                'sensor': sensor or os.path.basename(zone_path),
                'temperature_c': round(temperature_c, 1),
            })
        except (IOError, OSError, TypeError, ValueError):
            continue
    return readings


# ============================================================================
# UDP latest-command-only path
# ============================================================================
ESP32_IP = os.environ.get('STAR_ESP32_IP', '10.51.90.92').strip()
ESP32_PORT = env_int('STAR_ESP32_PORT', 4210, 1, 65535)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_lock = threading.Lock()


def send_udp(message):
    try:
        with udp_lock:
            udp_socket.sendto(message.encode('ascii'), (ESP32_IP, ESP32_PORT))
        return True
    except (IOError, OSError, socket.error) as error:
        print('Erreur UDP ESP32 :', error)
        return False


servo_sender = LatestServoSender(send_udp, metrics)


tracking_controller = TrackingController(
    PAN_CENTER,
    TILT_CENTER,
    PAN_MIN,
    PAN_MAX,
    TILT_MIN,
    TILT_MAX,
    config_path=os.environ.get(
        'STAR_PID_CONFIG_PATH', '/mnt/sdcard/yolov5/star_pid_config.json'
    ),
)

servo_state_lock = threading.RLock()
pan0_pos, tilt0_pos = PAN_CENTER, TILT_CENTER
pan0_dir, tilt0_dir = 1.0, 1.0
guet_manual_mode = False
guet_detection_enabled = True


def parse_servo_angle(value, minimum, maximum, label):
    if isinstance(value, bool):
        raise ValueError('%s must be a number' % label)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be a number' % label)
    if not minimum <= number <= maximum:
        raise ValueError('%s must be between %.1f and %.1f' % (label, minimum, maximum))
    return number


def servo_snapshot():
    tracking = tracking_controller.snapshot()
    with servo_state_lock:
        guet = {
            'pan': round(pan0_pos, 3),
            'tilt': round(tilt0_pos, 3),
            'manual': bool(guet_manual_mode),
            'detection_enabled': bool(guet_detection_enabled),
        }
    return {
        'ok': True,
        'tracking_enabled': bool(tracking['config']['tracking_enabled']),
        'mode': 'manual' if tracking['telemetry']['mode'] == 'manual' else 'automatic',
        'turrets': {
            'guet': guet,
            'canon': {
                'pan': tracking['telemetry']['pan_deg'],
                'tilt': tracking['telemetry']['tilt_deg'],
                'manual': tracking['telemetry']['mode'] == 'manual',
            },
        },
        'limits': {
            'pan': {'minimum': PAN_MIN, 'maximum': PAN_MAX},
            'tilt': {'minimum': TILT_MIN, 'maximum': TILT_MAX},
        },
        'centers': {
            'guet': {'pan': PAN_CENTER, 'tilt': TILT_CENTER},
            'canon': {'pan': PAN_CENTER, 'tilt': TILT_CENTER},
        },
        'laser': 'off',
    }


# ============================================================================
# TensorRT — Canon priority, Guet opportunistic
# ============================================================================
device = select_device('0')
try:
    model = DetectMultiBackend(MODEL_PATH, device=device, fp16=USE_FP16)
except TypeError:
    model = DetectMultiBackend(MODEL_PATH, device=device)
inference_gate = PriorityInferenceGate()
input_buffers = {}


def sync_cuda():
    if getattr(device, 'type', str(device)) != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()


def warmup_model():
    try:
        model.warmup(imgsz=(1, 3, CANON_AI_SIZE, CANON_AI_SIZE))
        print('TensorRT warm-up termine (%dx%d, FP16=%s)' % (
            CANON_AI_SIZE, CANON_AI_SIZE, USE_FP16
        ))
    except Exception as error:
        print('Warm-up TensorRT non disponible :', error)


def xywh2xyxy(boxes):
    converted = boxes.clone()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def run_inference(frame, frame_time, pipeline, image_size):
    """Return (ran, detection). Detection is None when no box is retained."""
    gate_started = time.monotonic()
    if pipeline == 'canon':
        inference_gate.begin_canon()
        metrics.observe('canon', 'gpu_wait_ms', (time.monotonic() - gate_started) * 1000.0)
    elif not inference_gate.begin_guet():
        metrics.event('guet', 'inference_skipped_priority')
        return False, None

    started = time.monotonic()
    try:
        preprocess_start = time.monotonic()
        image = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        tensor_array = image[:, :, ::-1].transpose(2, 0, 1)
        tensor_array = np.ascontiguousarray(tensor_array)
        dtype = torch.float16 if getattr(model, 'fp16', False) else torch.float32
        buffer_key = (int(image_size), str(dtype))
        tensor = input_buffers.get(buffer_key)
        if tensor is None:
            tensor = torch.empty(
                (1, 3, image_size, image_size), device=device, dtype=dtype
            )
            input_buffers[buffer_key] = tensor
        tensor.copy_(torch.from_numpy(tensor_array).unsqueeze(0), non_blocking=False)
        tensor.div_(255.0)
        preprocess_ms = (time.monotonic() - preprocess_start) * 1000.0

        inference_start = time.monotonic()
        sync_cuda()
        with torch.no_grad():
            prediction = model(tensor)
        sync_cuda()
        inference_ms = (time.monotonic() - inference_start) * 1000.0

        post_start = time.monotonic()
        prediction = prediction[0].T
        scores = prediction[:, 4]
        prediction = prediction[scores > get_confidence()]
        detection = None
        if len(prediction):
            # torchvision NMS is not implemented for FP16 on every JetPack 4
            # build. Convert on GPU; only the final best box crosses to CPU.
            boxes = xywh2xyxy(prediction[:, :4]).float()
            scores = prediction[:, 4].float()
            kept = torchvision.ops.nms(boxes, scores, 0.45)
            if len(kept):
                kept_scores = scores[kept]
                best_index = kept[torch.argmax(kept_scores)]
                selected = torch.cat((boxes[best_index], scores[best_index:best_index + 1]))
                selected = selected.detach().float().cpu().numpy()
                height, width = frame.shape[:2]
                detection = {
                    'x1': max(0, min(width - 1, int(selected[0] * width / float(image_size)))),
                    'y1': max(0, min(height - 1, int(selected[1] * height / float(image_size)))),
                    'x2': max(0, min(width - 1, int(selected[2] * width / float(image_size)))),
                    'y2': max(0, min(height - 1, int(selected[3] * height / float(image_size)))),
                    'confidence': float(selected[4]),
                }
        nms_ms = (time.monotonic() - post_start) * 1000.0
        completed = time.monotonic()
        metrics.event(pipeline, 'inference', completed)
        metrics.observe(pipeline, 'preprocess_ms', preprocess_ms, completed)
        metrics.observe(pipeline, 'inference_ms', inference_ms, completed)
        metrics.observe(pipeline, 'nms_ms', nms_ms, completed)
        metrics.observe(pipeline, 'frame_age_ms', (completed - frame_time) * 1000.0, completed)
        metrics.observe(pipeline, 'pipeline_ms', (completed - started) * 1000.0, completed)
        return True, detection
    finally:
        inference_gate.end()


class OverlayState(object):
    def __init__(self):
        self.lock = threading.RLock()
        self.box = None
        self.message = ''
        self.updated_at = 0.0

    def update(self, box=None, message=''):
        with self.lock:
            self.box = None if box is None else dict(box)
            self.message = message
            self.updated_at = time.monotonic()

    def snapshot(self):
        with self.lock:
            box = None if self.box is None else dict(self.box)
            return box, self.message, self.updated_at


canon_overlay = OverlayState()
guet_overlay = OverlayState()

canon_hub = StreamHub('canon', metrics, performance_mode)
guet_hub = StreamHub('guet', metrics, performance_mode)
thermal_hub = StreamHub('thermal', metrics, performance_mode)

canon_camera = None
guet_camera = None
thermal_stats_lock = threading.Lock()
thermal_stats = None
thermal_frame_count = 0


def make_offline_frame(label):
    frame = np.zeros((384, 512, 3), dtype=np.uint8)
    cv2.putText(frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, 'CAMERA DECONNECTEE', (120, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)
    cv2.putText(frame, 'reconnexion automatique... %s' % time.strftime('%H:%M:%S'),
                (95, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def camera_stream_loop(camera, hub, overlay, label):
    last_id = -1
    last_offline_publish = 0.0
    while True:
        if not hub.has_viewers() or not performance_mode.snapshot()['stream_enabled']:
            time.sleep(0.10)
            continue
        last_id, captured_at, frame = camera.read_latest(last_id, timeout=0.5)
        if frame is None:
            # Keep the MJPEG connection alive with an explicit status frame
            # instead of leaving the site on the last frozen image.
            now = time.monotonic()
            if not camera.snapshot()['ok'] and now - last_offline_publish >= 1.0:
                hub.publish(make_offline_frame(label))
                last_offline_publish = now
            continue
        box, message, updated_at = overlay.snapshot()
        if box is not None and time.monotonic() - updated_at < 0.75:
            cv2.rectangle(
                frame, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2
            )
            cv2.putText(
                frame, 'drone %.2f' % box['confidence'],
                (box['x1'], max(18, box['y1'] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 2
            )
        height, width = frame.shape[:2]
        cv2.drawMarker(frame, (width // 2, height // 2), (255, 0, 0), cv2.MARKER_CROSS, 36, 1)
        cv2.putText(frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if message:
            cv2.putText(frame, message, (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        hub.publish(frame)


def canon_inference_loop():
    last_id = -1
    next_inference = 0.0
    while True:
        last_id, frame_time, frame = canon_camera.read_latest(last_id, timeout=0.5)
        if frame is None:
            continue
        now = time.monotonic()
        if CANON_INFERENCE_HZ > 0.0 and now < next_inference:
            continue
        ran, detection = run_inference(frame, frame_time, 'canon', CANON_AI_SIZE)
        completed = time.monotonic()
        if CANON_INFERENCE_HZ > 0.0:
            next_inference = completed + 1.0 / CANON_INFERENCE_HZ
        if not ran:
            continue
        if detection is None:
            tracking_controller.mark_target_missed(completed)
            canon_overlay.update(None, tracking_controller.snapshot()['telemetry']['state'])
            continue
        center_x = (detection['x1'] + detection['x2']) / 2.0
        center_y = (detection['y1'] + detection['y2']) / 2.0
        height, width = frame.shape[:2]
        tracking_start = time.monotonic()
        tracking_controller.observe_target(
            center_x, center_y, width, height, detection['confidence'],
            frame_time=frame_time, now=tracking_start,
        )
        tracking_ms = (time.monotonic() - tracking_start) * 1000.0
        metrics.observe('canon', 'tracking_ms', tracking_ms)
        metrics.observe('canon', 'detection_to_state_ms', (time.monotonic() - frame_time) * 1000.0)
        state = tracking_controller.snapshot()['telemetry']['state']
        canon_overlay.update(detection, state)


def canon_control_loop():
    interval = 1.0 / CONTROL_LOOP_HZ
    next_tick = time.monotonic()
    while True:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.005, next_tick - now))
            continue
        started = time.monotonic()
        command = tracking_controller.control_step(started)
        metrics.event('canon', 'control_tick', started)
        metrics.observe('canon', 'control_ms', (time.monotonic() - started) * 1000.0)
        if command is not None:
            servo_sender.submit('canon', command[0], command[1])
            telemetry = tracking_controller.snapshot()['telemetry']
            if telemetry['frame_age_ms'] is not None:
                metrics.observe('canon', 'frame_to_command_ms', telemetry['frame_age_ms'])
        next_tick += interval
        if next_tick < time.monotonic() - interval:
            next_tick = time.monotonic() + interval


def guet_inference_loop():
    global pan0_pos, tilt0_pos
    last_id = -1
    next_inference = 0.0
    last_handoff = 0.0
    while True:
        last_id, frame_time, frame = guet_camera.read_latest(last_id, timeout=0.5)
        if frame is None:
            continue
        with servo_state_lock:
            enabled = guet_detection_enabled
        now = time.monotonic()
        if not enabled or now < next_inference:
            continue
        ran, detection = run_inference(frame, frame_time, 'guet', GUET_AI_SIZE)
        if not ran:
            time.sleep(0.005)
            continue
        next_inference = time.monotonic() + 1.0 / GUET_INFERENCE_HZ
        if detection is None:
            guet_overlay.update(None, 'SEARCH')
            continue
        guet_overlay.update(detection, 'DETECTION GUET')
        now = time.monotonic()
        if (
            detection['confidence'] >= HANDOFF_MIN_CONFIDENCE
            and tracking_controller.needs_handoff()
            and now - last_handoff >= HANDOFF_MIN_INTERVAL_S
        ):
            with servo_state_lock:
                pose = (pan0_pos, tilt0_pos)
            command = tracking_controller.handoff(pose[0], pose[1])
            if command is not None:
                servo_sender.submit('canon', command[0], command[1])
                metrics.event('guet', 'handoff', now)
                last_handoff = now


def guet_scan_loop():
    global pan0_pos, tilt0_pos, pan0_dir, tilt0_dir
    interval = 1.0 / GUET_SERVO_HZ
    last_tick = time.monotonic()
    while True:
        now = time.monotonic()
        dt = max(0.0, min(0.10, now - last_tick))
        if dt < interval:
            time.sleep(min(0.005, interval - dt))
            continue
        last_tick = now
        with servo_state_lock:
            if guet_manual_mode or not guet_detection_enabled:
                continue
            pan0_pos += GUET_PAN_SCAN_DPS * dt * pan0_dir
            tilt0_pos += GUET_TILT_SCAN_DPS * dt * tilt0_dir
            if pan0_pos >= PAN_MAX:
                pan0_pos, pan0_dir = PAN_MAX, -1.0
            elif pan0_pos <= PAN_MIN:
                pan0_pos, pan0_dir = PAN_MIN, 1.0
            if tilt0_pos >= TILT_MAX:
                tilt0_pos, tilt0_dir = TILT_MAX, -1.0
            elif tilt0_pos <= TILT_MIN:
                tilt0_pos, tilt0_dir = TILT_MIN, 1.0
            command = (pan0_pos, tilt0_pos)
        servo_sender.submit('guet', command[0], command[1])


def align_thermal_to_visible(image):
    center_x, center_y = FUSION_WIDTH / 2.0, FUSION_HEIGHT / 2.0
    matrix = np.float32([
        [FUSION_SCALE_X, 0.0, FUSION_OFFSET_X + center_x * (1.0 - FUSION_SCALE_X)],
        [0.0, FUSION_SCALE_Y, FUSION_OFFSET_Y + center_y * (1.0 - FUSION_SCALE_Y)],
    ])
    return cv2.warpAffine(image, matrix, (FUSION_WIDTH, FUSION_HEIGHT),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def thermal_loop():
    """Read all thermal packets, but render only while the fusion has viewers."""
    global thermal_stats, thermal_frame_count
    header = b'\x5A\x5A\x02'
    while True:
        port = None
        try:
            print('Ouverture camera thermique sur', THERMAL_PORT)
            port = serial.Serial(
                THERMAL_PORT, 115200, timeout=0.2, write_timeout=1,
                rtscts=False, dsrdtr=False, xonxoff=False,
            )
            port.reset_input_buffer()
            port.write(bytes([0xA5, 0x25, 0x01, 0xCB]))
            port.flush()
            time.sleep(0.2)
            port.write(bytes([0xA5, 0x35, 0x02, 0xDC]))
            port.flush()
            buffer = bytearray()
            while True:
                block = port.read(port.in_waiting or 1)
                if not block:
                    continue
                buffer.extend(block)
                while True:
                    start = buffer.find(header)
                    if start < 0:
                        if len(buffer) > 2:
                            del buffer[:-2]
                        break
                    if start > 0:
                        del buffer[:start]
                    if len(buffer) < THERMAL_FRAME_LEN:
                        break
                    packet = bytes(buffer[:THERMAL_FRAME_LEN])
                    del buffer[:THERMAL_FRAME_LEN]
                    values = np.frombuffer(packet[4:1540], dtype='<i2').astype(np.float32) / 100.0
                    if values.size != 768:
                        continue
                    temperatures = values.reshape((24, 32))
                    ambient = float(np.frombuffer(packet[1540:1542], dtype='<i2')[0]) / 100.0
                    t_min = float(np.min(temperatures))
                    t_max = float(np.max(temperatures))
                    t_center = float(temperatures[12, 16])
                    with thermal_stats_lock:
                        thermal_frame_count += 1
                        thermal_stats = {
                            'minimum_c': round(t_min, 1),
                            'maximum_c': round(t_max, 1),
                            'center_c': round(t_center, 1),
                            'ambient_c': round(ambient, 1),
                        }
                    metrics.event('thermal', 'capture')

                    if (
                        not thermal_hub.has_viewers()
                        or not performance_mode.snapshot()['stream_enabled']
                    ):
                        continue

                    render_started = time.monotonic()
                    low = float(np.percentile(temperatures, 2))
                    high = float(np.percentile(temperatures, 98))
                    if high - low < 2.0:
                        middle = (high + low) / 2.0
                        low, high = middle - 1.0, middle + 1.0
                    gray = np.clip(
                        (temperatures - low) * 255.0 / (high - low), 0, 255
                    ).astype(np.uint8)
                    color_map = getattr(cv2, 'COLORMAP_INFERNO', cv2.COLORMAP_JET)
                    thermal_image = cv2.applyColorMap(gray, color_map)
                    thermal_image = cv2.flip(thermal_image, 1)
                    thermal_image = cv2.rotate(thermal_image, cv2.ROTATE_90_CLOCKWISE)
                    thermal_image = cv2.resize(
                        thermal_image, (FUSION_WIDTH, FUSION_HEIGHT),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    thermal_image = align_thermal_to_visible(thermal_image)
                    visible = None
                    if canon_camera is not None:
                        _, _, visible = canon_camera.latest()
                    if visible is None:
                        image = thermal_image
                        fusion_state = 'CAM-01 Canon en attente'
                    else:
                        visible = cv2.resize(
                            visible, (FUSION_WIDTH, FUSION_HEIGHT),
                            interpolation=cv2.INTER_AREA,
                        )
                        image = cv2.addWeighted(
                            visible, FUSION_VISIBLE_ALPHA,
                            thermal_image, FUSION_THERMAL_ALPHA, 0.0,
                        )
                        fusion_state = 'Fusion CAM-01 Canon + thermique'
                    cv2.drawMarker(
                        image, (FUSION_WIDTH // 2, FUSION_HEIGHT // 2),
                        (255, 255, 255), cv2.MARKER_CROSS, 20, 1,
                    )
                    cv2.putText(
                        image, 'Min %.1f C  Max %.1f C' % (t_min, t_max),
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2,
                    )
                    cv2.putText(
                        image, 'Centre %.1f C  Ambiante %.1f C' % (t_center, ambient),
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2,
                    )
                    cv2.putText(
                        image, fusion_state, (10, 78), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2,
                    )
                    thermal_hub.publish(image)
                    metrics.observe(
                        'thermal', 'render_ms',
                        (time.monotonic() - render_started) * 1000.0,
                    )
        except Exception as error:
            print('Erreur camera thermique :', error)
            if port is not None:
                try:
                    port.close()
                except Exception:
                    pass
            time.sleep(2.0)


def make_mjpeg_chunk(jpeg):
    return (
        b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
        + str(len(jpeg)).encode('ascii') + b'\r\n\r\n' + jpeg + b'\r\n'
    )


def mjpeg_response(hub):
    response = Response(
        hub.generate(make_mjpeg_chunk),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/video0')
def video0():
    """CAM02 Guet; kept on the historical URL expected by the site."""
    return mjpeg_response(guet_hub)


@app.route('/video1')
def video1():
    """CAM01 Canon, primary camera and sole precise tracking source."""
    return mjpeg_response(canon_hub)


@app.route('/thermal')
@app.route('/fusion')
def thermal():
    return mjpeg_response(thermal_hub)


@app.route('/api/confidence', methods=['GET', 'POST'])
def confidence_api():
    if request.method == 'GET':
        return jsonify({
            'ok': True, 'confidence': get_confidence(),
            'minimum': CONFIDENCE_MIN, 'maximum': CONFIDENCE_MAX,
        })
    payload = request.get_json(silent=True) or {}
    try:
        confidence = float(payload.get('confidence'))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'invalid confidence'}), 400
    if not CONFIDENCE_MIN <= confidence <= CONFIDENCE_MAX:
        return jsonify({
            'ok': False,
            'error': 'confidence must be between 0.20 and 0.90',
        }), 400
    set_confidence(confidence)
    return jsonify({'ok': True, 'confidence': get_confidence()})


@app.route('/api/jetson-temperature')
def jetson_temperature_api():
    readings = read_jetson_temperatures()
    if not readings:
        return jsonify({'ok': False, 'error': 'no thermal sensor'}), 503
    hottest = max(readings, key=lambda reading: reading['temperature_c'])
    return jsonify({
        'ok': True, 'temperature_c': hottest['temperature_c'],
        'sensor': hottest['sensor'], 'sensors': readings,
    })


@app.route('/api/pid', methods=['GET', 'POST'])
def pid_api():
    global guet_manual_mode
    if request.method == 'GET':
        return jsonify(tracking_controller.snapshot())
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'ok': False, 'error': 'JSON object required'}), 400
    action = payload.get('action', 'update')
    persist = payload.get('persist', False)
    if not isinstance(persist, bool):
        return jsonify({'ok': False, 'error': 'persist must be a boolean'}), 400
    try:
        if action == 'update':
            snapshot = tracking_controller.update_config(
                payload.get('config', {}), persist=persist,
            )
            if not snapshot['config']['tracking_enabled']:
                send_udp('laser_off')
        elif action == 'enable':
            if not isinstance(payload.get('enabled'), bool):
                raise PIDConfigError('enabled must be a boolean')
            snapshot = tracking_controller.set_enabled(
                payload['enabled'], persist=persist,
            )
            send_udp('laser_off')
        elif action == 'reset':
            snapshot = tracking_controller.reset()
        elif action == 'center':
            pan, tilt, snapshot = tracking_controller.center(persist=persist)
            send_udp('laser_off')
            servo_sender.submit('canon', pan, tilt)
        else:
            return jsonify({'ok': False, 'error': 'unknown action'}), 400
    except PIDConfigError as error:
        return jsonify({'ok': False, 'error': str(error)}), 400
    except (IOError, OSError) as error:
        print('Erreur sauvegarde PID :', error)
        return jsonify({'ok': False, 'error': 'PID settings could not be saved'}), 500
    return jsonify(snapshot)


@app.route('/api/servos', methods=['GET', 'POST'])
def servos_api():
    global pan0_pos, tilt0_pos, guet_manual_mode, guet_detection_enabled
    if request.method == 'GET':
        return jsonify(servo_snapshot())
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'ok': False, 'error': 'JSON object required'}), 400
    action = payload.get('action')
    turret = payload.get('turret')
    try:
        if action == 'move':
            if turret not in ('guet', 'canon'):
                raise ValueError('turret must be guet or canon')
            pan = parse_servo_angle(payload.get('pan'), PAN_MIN, PAN_MAX, 'pan')
            tilt = parse_servo_angle(payload.get('tilt'), TILT_MIN, TILT_MAX, 'tilt')
            if turret == 'guet':
                with servo_state_lock:
                    guet_manual_mode = True
                    pan0_pos, tilt0_pos = pan, tilt
                servo_sender.submit('guet', pan, tilt)
            else:
                tracking_controller.set_enabled(False, persist=False)
                send_udp('laser_off')
                pan, tilt, _ = tracking_controller.manual_position(pan, tilt)
                servo_sender.submit('canon', pan, tilt)
        elif action == 'center':
            if turret not in ('guet', 'canon', 'all'):
                raise ValueError('turret must be guet, canon or all')
            if turret in ('guet', 'all'):
                with servo_state_lock:
                    guet_manual_mode = True
                    pan0_pos, tilt0_pos = PAN_CENTER, TILT_CENTER
                servo_sender.submit('guet', PAN_CENTER, TILT_CENTER)
            if turret in ('canon', 'all'):
                pan, tilt, _ = tracking_controller.center(persist=False)
                send_udp('laser_off')
                servo_sender.submit('canon', pan, tilt)
        elif action == 'scan':
            if turret != 'guet':
                raise ValueError('scan is only available for guet')
            with servo_state_lock:
                guet_manual_mode = False
                guet_detection_enabled = True
        elif action == 'guet_detection':
            if not isinstance(payload.get('enabled'), bool):
                raise ValueError('enabled must be a boolean')
            with servo_state_lock:
                guet_detection_enabled = payload['enabled']
                guet_manual_mode = not payload['enabled']
            guet_overlay.update(
                None, 'GUET ACTIVE' if payload['enabled'] else 'GUET IGNOREE'
            )
            send_udp('laser_off')
        elif action == 'stop':
            tracking_controller.set_enabled(False, persist=False)
            with servo_state_lock:
                guet_manual_mode = True
            send_udp('stop')
        else:
            raise ValueError('unknown action')
    except (ValueError, PIDConfigError) as error:
        return jsonify({'ok': False, 'error': str(error)}), 400
    return jsonify(servo_snapshot())


@app.route('/api/performance', methods=['GET', 'POST'])
def performance_api():
    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        try:
            performance_mode.set(str(payload.get('mode', '')))
        except ValueError as error:
            return jsonify({'ok': False, 'error': str(error)}), 400
    return jsonify({
        'ok': True,
        'active': performance_mode.snapshot(),
        'modes': performance_mode.catalog(),
        'metrics': metrics.snapshot(),
        'streams': {
            'canon': canon_hub.snapshot(),
            'guet': guet_hub.snapshot(),
            'thermal': thermal_hub.snapshot(),
        },
    })


@app.route('/status')
def status():
    with thermal_stats_lock:
        current_thermal_stats = None if thermal_stats is None else dict(thermal_stats)
        current_thermal_count = thermal_frame_count
    return jsonify({
        'ok': True,
        'camera0': (
            {'ok': False, 'frames': 0, 'role': 'guet'}
            if guet_camera is None else dict(guet_camera.snapshot(), role='guet')
        ),
        'camera1': (
            {'ok': False, 'frames': 0, 'role': 'canon_primary'}
            if canon_camera is None else dict(canon_camera.snapshot(), role='canon_primary')
        ),
        'thermal': {
            'ok': current_thermal_stats is not None,
            'frames': current_thermal_count,
            'port': THERMAL_PORT,
            'mode': 'fusion_cam01_canon_thermal',
            'temperatures': current_thermal_stats,
        },
        'confidence': get_confidence(),
        'tracking': tracking_controller.snapshot(),
        'performance': performance_mode.snapshot(),
        'metrics': metrics.snapshot(),
    })


@app.route('/motor', methods=['POST'])
def motor():
    payload = request.get_json(silent=True) or {}
    try:
        left = int(payload.get('gauche', 0))
        right = int(payload.get('droite', 0))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'invalid motor command'}), 400
    left = max(-255, min(255, left))
    right = max(-255, min(255, right))
    send_udp('motor:%d,%d' % (left, right))
    return jsonify({'status': 'ok', 'gauche': left, 'droite': right})


@app.route('/')
def index():
    return jsonify({
        'ok': True,
        'service': 'S.T.A.R. realtime tracking',
        'canon': 'CAM01 / video1 / primary',
        'guet': 'CAM02 / video0 / acquisition only',
        'performance': performance_mode.snapshot(),
    })


def metrics_printer_loop():
    while True:
        time.sleep(METRICS_PRINT_INTERVAL_S)
        print('STAR_METRICS ' + json.dumps({
            'performance': performance_mode.snapshot(),
            'pipelines': metrics.snapshot(),
            'tracking': tracking_controller.snapshot()['telemetry'],
        }, sort_keys=True))


def start_background_threads():
    global canon_camera, guet_camera
    warmup_model()
    canon_camera = LatestFrameCamera(
        CANON_CAMERA, 'canon', metrics, rotate_code=cv2.ROTATE_180
    )
    guet_camera = LatestFrameCamera(
        GUET_CAMERA, 'guet', metrics, rotate_code=cv2.ROTATE_180
    )
    targets = (
        (camera_stream_loop, (canon_camera, canon_hub, canon_overlay, 'CAM-01 CANON · PRIORITAIRE')),
        (camera_stream_loop, (guet_camera, guet_hub, guet_overlay, 'CAM-02 GUET · ACQUISITION')),
        (canon_inference_loop, ()),
        (canon_control_loop, ()),
        (guet_inference_loop, ()),
        (guet_scan_loop, ()),
        (thermal_loop, ()),
        (metrics_printer_loop, ()),
    )
    for target, arguments in targets:
        threading.Thread(target=target, args=arguments, daemon=True).start()


if __name__ == '__main__':
    # Safe boot: motors stopped, laser forced off, visual tracking disarmed
    # until an authenticated operator explicitly arms it from the website.
    send_udp('laser_off')
    send_udp('motor:0,0')
    start_background_threads()
    app.run(host='127.0.0.1', port=5000, threaded=True, use_reloader=False)
