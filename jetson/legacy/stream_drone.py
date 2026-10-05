import cv2
import torch
import numpy as np
import torchvision
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from flask import Flask, Response, render_template_string
import threading

app = Flask(__name__)

device = select_device('0')
model = DetectMultiBackend('new_best_v1.engine', device=device)
print("✅ Modèle chargé !")

frame0_global = None
frame1_global = None
lock0 = threading.Lock()
lock1 = threading.Lock()
model_lock = threading.Lock()

def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def detect(frame):
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    h0, w0 = frame.shape[:2]

    img = cv2.resize(frame, (640, 640))
    img_tensor = img[:, :, ::-1].transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with model_lock:
        pred = model(img_tensor)

    pred = pred[0].T
    scores = pred[:, 4]
    mask = scores > 0.25
    pred = pred[mask]

    if len(pred):
        boxes = xywh2xyxy(pred[:, :4])
        scores = pred[:, 4]
        keep = torchvision.ops.nms(boxes, scores, 0.45)
        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()

        for box, score in zip(boxes, scores):
            x1 = int(box[0] * w0 / 640)
            y1 = int(box[1] * h0 / 640)
            x2 = int(box[2] * w0 / 640)
            y2 = int(box[3] * h0 / 640)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'drone {score:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def capture_loop0():
    global frame0_global
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, f = cap.read()
        if ret:
            f = detect(f)
            with lock0:
                frame0_global = f

def capture_loop1():
    global frame1_global
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, f = cap.read()
        if ret:
            f = detect(f)
            with lock1:
                frame1_global = f

def generate(cam):
    while True:
        if cam == 0:
            with lock0:
                frame = frame0_global
        else:
            with lock1:
                frame = frame1_global
        if frame is None:
            continue
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Drone Detection</title>
    <style>
        body { background: #111; color: white; text-align: center; font-family: Arial; }
        h1 { color: #0f0; }
        img { margin: 10px; border: 2px solid #0f0; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>🚁 Drone Detection - Live Stream</h1>
    <div>
        <img src="/video0" width="640"><br>
        <b>Camera 0</b>
    </div>
    <div>
        <img src="/video1" width="640"><br>
        <b>Camera 1</b>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video0')
def video0():
    return Response(generate(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video1')
def video1():
    return Response(generate(1), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t0 = threading.Thread(target=capture_loop0, daemon=True)
    t1 = threading.Thread(target=capture_loop1, daemon=True)
    t0.start()
    t1.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
