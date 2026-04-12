import cv2
import numpy as np

from .config import INPUT_HEIGHT, INPUT_WIDTH
from .postprocess import get_class_name


def preprocess_frame(frame):
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    batched = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(batched), h, w


def draw_detections(frame, boxes, scores, labels):
    drawn = 0
    for box, score, label in zip(boxes, scores, labels):
        class_name = get_class_name(label)
        if class_name is None:
            continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {float(score):.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        drawn += 1
    return drawn
