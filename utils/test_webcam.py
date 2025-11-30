import cv2
import time
import numpy as np
from ultralytics import YOLO

# --------------------------------
# CONFIG
# --------------------------------
# MODEL_PATH = r"C:\Users\devka\crowdhuman\runs\detect\train5\weights\best.pt"

from configs.config import YOLO_MODEL_PATH

CONF = 0.35
IOU = 0.55
MAX_DET = 30
CLS = [0]  # only person

SMOOTH_FACTOR = 0.5
MIN_IOU_MERGE = 0.50
EDGE_SUPPRESS = 5

prev_boxes = None   # history for smoothing


# --------------------------------
# Helper: Merge boxes
# --------------------------------
def safe_merge(boxes, threshold=0.5):
    if len(boxes) <= 1:
        return boxes

    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        x1a, y1a, x2a, y2a = boxes[i]

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue

            x1b, y1b, x2b, y2b = boxes[j]

            # IoU
            xx1, yy1 = max(x1a, x1b), max(y1a, y1b)
            xx2, yy2 = min(x2a, x2b), min(y2a, y2b)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            if inter == 0:
                continue

            area1 = (x2a - x1a) * (y2a - y1a)
            area2 = (x2b - x1b) * (y2b - y1b)
            iou = inter / (area1 + area2 - inter + 1e-12)

            if iou > threshold:
                # average merge
                x1a = int((x1a + x1b) / 2)
                y1a = int((y1a + y1b) / 2)
                x2a = int((x2a + x2b) / 2)
                y2a = int((y2a + y2b) / 2)
                used[j] = True

        merged.append([x1a, y1a, x2a, y2a])
        used[i] = True

    return merged


# --------------------------------
# Helper: Temporal smoothing
# --------------------------------
def smooth_boxes(prev, curr, alpha=0.5):
    if prev is None or len(prev) != len(curr):
        return curr

    smoothed = []
    for p, c in zip(prev, curr):
        x1 = int((1 - alpha) * p[0] + alpha * c[0])
        y1 = int((1 - alpha) * p[1] + alpha * c[1])
        x2 = int((1 - alpha) * p[2] + alpha * c[2])
        y2 = int((1 - alpha) * p[3] + alpha * c[3])
        smoothed.append([x1, y1, x2, y2])

    return smoothed


# --------------------------------
# MAIN LOOP
# --------------------------------
model = YOLO(YOLO_MODEL_PATH)
cap = cv2.VideoCapture(0)
fps_avg = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        t0 = time.time()

        # YOLO detection
        results = model.predict(
            frame,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            classes=CLS,
            verbose=False
        )

        # Extract raw boxes
        raw_boxes = []
        if results and len(results[0].boxes):
            xyxy = results[0].boxes.xyxy
            try:
                arr = xyxy.cpu().numpy()
            except Exception:
                arr = np.asarray(xyxy)

            for b in arr:
                x1, y1, x2, y2 = map(int, b[:4])
                raw_boxes.append([x1, y1, x2, y2])

        # Merge duplicates
        merged = safe_merge(raw_boxes, MIN_IOU_MERGE)

        # Smooth boxes
        smoothed = smooth_boxes(prev_boxes, merged, SMOOTH_FACTOR)
        prev_boxes = smoothed

        # --------------------------------------
        # EXTRACT PERSON CROPS
        # --------------------------------------
        person_crops = []
        for (x1, y1, x2, y2) in smoothed:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                crop = frame[y1:y2, x1:x2].copy()
                person_crops.append(crop)

        # --------------------------------------
        # DRAW BOXES
        # --------------------------------------
        for (x1, y1, x2, y2) in smoothed:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, "person", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # Show first crop (debug)
        if len(person_crops) > 0:
            preview = cv2.resize(person_crops[0], (120, 160))
            frame[10:170, 10:130] = preview

        # FPS smoothing
        fps = 1.0 / (time.time() - t0 + 1e-12)
        fps_avg = fps_avg * 0.8 + fps * 0.2
        cv2.putText(frame, f"{fps_avg:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display
        cv2.imshow("Stable Person Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
