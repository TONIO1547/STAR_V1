"""Low-latency building blocks for the S.T.A.R. Jetson service.

The module is deliberately compatible with Python 3.6 / JetPack 4.6.6.
It implements latest-data-only capture, priority arbitration for the single
TensorRT engine, on-demand MJPEG encoding and a latest-command-only UDP path.
"""

from __future__ import print_function

from collections import defaultdict, deque
import os
import threading
import time

import cv2
import numpy as np


class RealtimeMetrics(object):
    """Small rolling metrics registry with a bounded two-second history."""

    def __init__(self, window_s=2.0):
        self.window_s = float(window_s)
        self.lock = threading.RLock()
        self.events = defaultdict(lambda: defaultdict(lambda: deque(maxlen=600)))
        self.values = defaultdict(lambda: defaultdict(lambda: deque(maxlen=240)))

    def event(self, pipeline, name, timestamp=None):
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        with self.lock:
            self.events[pipeline][name].append(stamp)

    def observe(self, pipeline, name, value, timestamp=None):
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.values[pipeline][name].append((stamp, number))

    def snapshot(self):
        now = time.monotonic()
        cutoff = now - self.window_s
        result = {}
        with self.lock:
            pipelines = set(self.events.keys()) | set(self.values.keys())
            for pipeline in pipelines:
                data = {}
                for name, samples in self.events[pipeline].items():
                    recent = [stamp for stamp in samples if stamp >= cutoff]
                    data[name + '_hz'] = round(len(recent) / self.window_s, 2)
                for name, samples in self.values[pipeline].items():
                    recent = [value for stamp, value in samples if stamp >= cutoff]
                    if recent:
                        data[name] = round(sum(recent) / len(recent), 2)
                        data[name + '_max'] = round(max(recent), 2)
                result[pipeline] = data
        return result


class LatestFrameCamera(object):
    """Continuously captures and exposes only the newest frame and timestamp.

    A supervisor thread acts as a per-camera watchdog: when no frame arrives
    for STALL_TIMEOUT_S (USB unplug, frozen driver, cap.read() that never
    returns), the capture worker is abandoned, the stale frame is dropped,
    and the camera is reopened as soon as its device path exists again.
    """

    STALL_TIMEOUT_S = 2.0
    READ_FAILURE_TIMEOUT_S = 1.0
    REOPEN_INTERVAL_S = 0.5

    def __init__(self, source, name, metrics, rotate_code=None):
        self.source = source
        self.name = name
        self.metrics = metrics
        self.rotate_code = rotate_code
        self.condition = threading.Condition()
        self.frame = None
        self.frame_id = 0
        self.capture_time = None
        self.generation = 0
        self.worker = None
        self.worker_started = 0.0
        self.connected = False
        self.reconnects = 0
        self.running = True
        self.thread = threading.Thread(target=self._supervisor_loop, daemon=True)
        self.thread.start()

    def _device_present(self):
        # Integer indexes cannot be checked; by-path/by-id sources can.
        if isinstance(self.source, str) and self.source.startswith('/dev/'):
            return os.path.exists(self.source)
        return True

    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _drop_worker(self, reason):
        """Abandon the current worker; it releases its own capture if it wakes."""
        with self.condition:
            was_connected = self.connected
            self.generation += 1
            self.worker = None
            self.connected = False
            self.frame = None
            self.capture_time = None
            self.condition.notify_all()
        if was_connected:
            print('Camera %s perdue (%s), attente de reconnexion' % (self.name, reason))
        self.metrics.event(self.name, 'capture_error')

    def _supervisor_loop(self):
        while self.running:
            with self.condition:
                worker = self.worker
                started = self.worker_started
                last_frame = self.capture_time
            now = time.monotonic()
            if worker is not None:
                if not worker.is_alive():
                    self._drop_worker('lecture en echec')
                elif now - (last_frame or started) > self.STALL_TIMEOUT_S:
                    self._drop_worker('aucune image depuis %.0f s' % self.STALL_TIMEOUT_S)
                else:
                    time.sleep(0.25)
                    continue
            if not self._device_present():
                self.metrics.event(self.name, 'capture_error')
                time.sleep(self.REOPEN_INTERVAL_S)
                continue
            cap = self._open()
            if cap is None:
                self.metrics.event(self.name, 'capture_error')
                time.sleep(self.REOPEN_INTERVAL_S)
                continue
            with self.condition:
                self.generation += 1
                generation = self.generation
                self.worker = threading.Thread(
                    target=self._capture_worker, args=(cap, generation), daemon=True
                )
                self.worker_started = time.monotonic()
                self.worker.start()

    def _capture_worker(self, cap, generation):
        failure_started = None
        try:
            while self.running and generation == self.generation:
                ret, frame = cap.read()
                captured_at = time.monotonic()
                if not ret:
                    if failure_started is None:
                        failure_started = captured_at
                    elif captured_at - failure_started >= self.READ_FAILURE_TIMEOUT_S:
                        return
                    self.metrics.event(self.name, 'capture_error', captured_at)
                    time.sleep(0.01)
                    continue
                failure_started = None
                if self.rotate_code is not None:
                    frame = cv2.rotate(frame, self.rotate_code)
                with self.condition:
                    if generation != self.generation:
                        return
                    if not self.connected:
                        self.connected = True
                        if self.frame_id > 0:
                            self.reconnects += 1
                            self.metrics.event(self.name, 'reconnect', captured_at)
                        print('Camera %s connectee sur %s' % (self.name, self.source))
                    self.frame = frame
                    self.capture_time = captured_at
                    self.frame_id += 1
                    self.condition.notify_all()
                self.metrics.event(self.name, 'capture', captured_at)
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def read_latest(self, last_frame_id, timeout=0.5):
        with self.condition:
            if self.frame is None or self.frame_id == last_frame_id:
                self.condition.wait(timeout)
            if self.frame is None or self.frame_id == last_frame_id:
                return last_frame_id, None, None
            return self.frame_id, self.capture_time, self.frame.copy()

    def latest(self):
        with self.condition:
            if self.frame is None:
                return self.frame_id, None, None
            return self.frame_id, self.capture_time, self.frame.copy()

    def snapshot(self):
        with self.condition:
            age_ms = None
            if self.capture_time is not None:
                age_ms = max(0.0, (time.monotonic() - self.capture_time) * 1000.0)
            return {
                'ok': self.frame is not None and age_ms is not None and age_ms < 2000.0,
                'connected': self.connected,
                'frames': self.frame_id,
                'reconnects': self.reconnects,
                'age_ms': None if age_ms is None else round(age_ms, 1),
                'source': self.source,
            }


class PriorityInferenceGate(object):
    """Non-preemptive priority gate: Canon waits; Guet skips when Canon waits."""

    def __init__(self):
        self.condition = threading.Condition()
        self.active = None
        self.canon_waiting = 0

    def begin_canon(self):
        with self.condition:
            self.canon_waiting += 1
            try:
                while self.active is not None:
                    self.condition.wait()
                self.active = 'canon'
            finally:
                self.canon_waiting -= 1
        return True

    def begin_guet(self):
        with self.condition:
            if self.active is not None or self.canon_waiting > 0:
                return False
            self.active = 'guet'
            return True

    def end(self):
        with self.condition:
            self.active = None
            self.condition.notify_all()


class PerformanceMode(object):
    MODES = {
        'normal': {
            'stream_enabled': True,
            'stream_fps': 6.0,
            'stream_width': 512,
            'stream_height': 384,
            'jpeg_quality': 55,
            'description': 'Flux 6 FPS, tracking prioritaire',
        },
        'performance': {
            'stream_enabled': True,
            'stream_fps': 3.0,
            'stream_width': 384,
            'stream_height': 288,
            'jpeg_quality': 45,
            'description': 'Flux reduit a 3 FPS, charge JPEG minimale',
        },
        'maximum': {
            'stream_enabled': False,
            'stream_fps': 0.0,
            'stream_width': 384,
            'stream_height': 288,
            'jpeg_quality': 40,
            'description': 'Flux coupes, priorite absolue au tracking',
        },
    }

    def __init__(self, normal_policy=None):
        self.lock = threading.RLock()
        self.mode = 'normal'
        self.policies = dict((name, dict(value)) for name, value in self.MODES.items())
        if normal_policy:
            self.policies['normal'].update(normal_policy)

    def set(self, mode):
        if mode not in self.policies:
            raise ValueError('mode must be normal, performance or maximum')
        with self.lock:
            self.mode = mode
            return self.snapshot()

    def snapshot(self):
        with self.lock:
            result = dict(self.policies[self.mode])
            result['mode'] = self.mode
            return result

    def catalog(self):
        with self.lock:
            return dict((name, dict(value)) for name, value in self.policies.items())


class StreamHub(object):
    """Encodes at most one latest frame and only while clients are watching."""

    def __init__(self, name, metrics, performance_mode):
        self.name = name
        self.metrics = metrics
        self.performance_mode = performance_mode
        self.condition = threading.Condition()
        self.frame = None
        self.frame_id = 0
        self.jpeg = None
        self.jpeg_id = 0
        self.viewers = 0
        placeholder = np.zeros((288, 512, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            'MODE PERFORMANCE MAXIMUM - FLUX COUPE',
            (18, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )
        ok, jpeg = cv2.imencode('.jpg', placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
        self.placeholder = jpeg.tobytes() if ok else None
        threading.Thread(target=self._encoder_loop, daemon=True).start()

    def publish(self, frame):
        with self.condition:
            self.frame = frame
            self.frame_id += 1
            self.condition.notify_all()

    def has_viewers(self):
        with self.condition:
            return self.viewers > 0

    def snapshot(self):
        with self.condition:
            return {
                'viewers': self.viewers,
                'frames_published': self.frame_id,
                'frames_encoded': self.jpeg_id,
            }

    def _resize(self, frame, width_limit, height_limit):
        height, width = frame.shape[:2]
        scale = min(1.0, width_limit / float(width), height_limit / float(height))
        if scale >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _encoder_loop(self):
        last_frame_id = -1
        next_encode = 0.0
        while True:
            with self.condition:
                policy = self.performance_mode.snapshot()
                if self.viewers <= 0 or self.frame is None or self.frame_id == last_frame_id:
                    self.condition.wait(0.25)
                    continue
                if not policy['stream_enabled']:
                    if self.placeholder is not None and self.jpeg != self.placeholder:
                        self.jpeg = self.placeholder
                        self.jpeg_id += 1
                        self.condition.notify_all()
                    self.condition.wait(0.25)
                    continue
                frame_id = self.frame_id
                frame = self.frame

            now = time.monotonic()
            interval = 1.0 / max(0.1, float(policy['stream_fps']))
            if now < next_encode:
                time.sleep(min(0.02, next_encode - now))
                continue
            started = time.monotonic()
            resized = self._resize(frame, policy['stream_width'], policy['stream_height'])
            ok, encoded = cv2.imencode(
                '.jpg',
                resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(policy['jpeg_quality'])],
            )
            elapsed_ms = (time.monotonic() - started) * 1000.0
            if ok:
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.jpeg_id += 1
                    last_frame_id = frame_id
                    self.condition.notify_all()
                self.metrics.event(self.name, 'stream')
                self.metrics.observe(self.name, 'jpeg_ms', elapsed_ms)
            next_encode = time.monotonic() + interval

    def generate(self, make_chunk):
        with self.condition:
            self.viewers += 1
            self.metrics.observe(self.name, 'viewers', self.viewers)
            self.condition.notify_all()
        last_jpeg_id = -1
        try:
            while True:
                with self.condition:
                    if self.jpeg is None or self.jpeg_id == last_jpeg_id:
                        self.condition.wait(1.0)
                    if self.jpeg is None or self.jpeg_id == last_jpeg_id:
                        continue
                    jpeg = self.jpeg
                    last_jpeg_id = self.jpeg_id
                yield make_chunk(jpeg)
        finally:
            with self.condition:
                self.viewers = max(0, self.viewers - 1)
                self.metrics.observe(self.name, 'viewers', self.viewers)
                self.condition.notify_all()


class LatestServoSender(object):
    """A size-one command slot per turret; newer targets replace older ones."""

    def __init__(self, send_callback, metrics):
        self.send_callback = send_callback
        self.metrics = metrics
        self.condition = threading.Condition()
        self.pending = {}
        self.sequence = {'canon': 0, 'guet': 0}
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, turret, pan, tilt):
        if turret not in ('canon', 'guet'):
            raise ValueError('unknown turret')
        with self.condition:
            self.sequence[turret] = (self.sequence[turret] + 1) & 0x7fffffff
            self.pending[turret] = (
                self.sequence[turret],
                float(pan),
                float(tilt),
                time.monotonic(),
            )
            self.condition.notify_all()

    def _loop(self):
        while True:
            with self.condition:
                while not self.pending:
                    self.condition.wait()
                commands = self.pending
                self.pending = {}
            # Canon first if both commands arrive during the same scheduling slice.
            for turret in ('canon', 'guet'):
                if turret not in commands:
                    continue
                sequence, pan, tilt, submitted = commands[turret]
                message = '%s2:%d,%.3f,%.3f' % (
                    turret,
                    sequence,
                    pan,
                    tilt,
                )
                if self.send_callback(message):
                    sent_at = time.monotonic()
                    self.metrics.event(turret, 'servo_command', sent_at)
                    self.metrics.observe(
                        turret,
                        'servo_queue_age_ms',
                        (sent_at - submitted) * 1000.0,
                        sent_at,
                    )
