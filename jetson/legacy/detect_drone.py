import cv2
import torch
import numpy as np
import torchvision
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

device = select_device('0')
model = DetectMultiBackend('new_best_v1.engine', device=device)
print("✅ Modèle chargé !")

cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)

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

    pred = model(img_tensor)
    pred = pred[0].T

    conf_thres = 0.25
    iou_thres = 0.45

    scores = pred[:, 4]
    mask = scores > conf_thres
    pred = pred[mask]

    if len(pred):
        boxes = xywh2xyxy(pred[:, :4])
        scores = pred[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
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

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if ret0:
        frame0 = detect(frame0)
        cv2.imshow('Camera 0', frame0)

    if ret1:
        frame1 = detect(frame1)
        cv2.imshow('Camera 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
