import cv2 # type: ignore
import time
import numpy as np # type: ignore
from ultralytics import YOLO #type: ignore
import torch # type: ignore
from ultralytics.nn.tasks import DetectionModel # type: ignore
import time

# Import your custom modules
from recognizer.identify import find_person
from configs.config import YOLO_MODEL_PATH, YOLO_PERSON_MODEL, YOLO_FACE_MODEL

# Safety check for model loading
torch.serialization.add_safe_globals([DetectionModel])

# --- CONFIGURATION ---
CONF_PERSON = 0.35
CONF_FACE = 0.50  # Confidence threshold for face detection
IOU = 0.55
MAX_DET = 30
CLS = [0]  # Person class

SMOOTH_FACTOR = 0.5
MIN_IOU_MERGE = 0.50

prev_boxes = None

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

            xx1 = max(x1a, x1b)
            yy1 = max(y1a, y1b)
            xx2 = min(x2a, x2b)
            yy2 = min(y2a, y2b)

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            if inter == 0:
                continue

            area1 = (x2a - x1a) * (y2a - y1a)
            area2 = (x2b - x1b) * (y2b - y1b)
            iou = inter / (area1 + area2 - inter + 1e-12)

            if iou > threshold:
                x1a = int((x1a + x1b) / 2)
                y1a = int((y1a + y1b) / 2)
                x2a = int((x2a + x2b) / 2)
                y2a = int((y2a + y2b) / 2)
                used[j] = True

        merged.append([x1a, y1a, x2a, y2a])
        used[i] = True

    return merged


def smooth_boxes(prev, curr, alpha=0.5):
    if prev is None or len(prev) != len(curr):
        return curr

    smoothed = []
    for p, c in zip(prev, curr):
        smoothed.append([
            int((1 - alpha) * p[0] + alpha * c[0]),
            int((1 - alpha) * p[1] + alpha * c[1]),
            int((1 - alpha) * p[2] + alpha * c[2]),
            int((1 - alpha) * p[3] + alpha * c[3]),
        ])
    return smoothed


print("Loading Person Model...")
person_model = YOLO(YOLO_PERSON_MODEL)

print("Loading Face Model...")
face_model = YOLO(YOLO_FACE_MODEL)

cap = cv2.VideoCapture(0)
fps_avg = 0.0

try:
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        t_read_end = time.time()
        h, w = frame.shape[:2]
        t0 = time.time()

        # 1. DETECT PEOPLE
        results = person_model.predict(
            frame,
            conf=CONF_PERSON,
            iou=IOU,
            max_det=MAX_DET,
            classes=CLS,
            verbose=False
        )

        t_person_det_end = time.time()

        raw_boxes = [] 
        if results and len(results[0].boxes):
            xyxy = results[0].boxes.xyxy
            arr = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
            for b in arr:
                raw_boxes.append(list(map(int, b[:4])))

        # 2. MERGE & SMOOTH PERSON BOXES
        merged = safe_merge(raw_boxes, MIN_IOU_MERGE)
        smoothed = smooth_boxes(prev_boxes, merged, SMOOTH_FACTOR)
        prev_boxes = smoothed

        t_process_boxes_end = time.time()

        # Timers for the loop
        time_spent_face_det = 0
        time_spent_recognition = 0
        people_count = 0

        # 3. PROCESS EACH PERSON
        for (px1, py1, px2, py2) in smoothed:
            # Safe crop coordinates for person
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w-1, px2), min(h-1, py2)
            
            # Skip if person box is too small
            if (px2 - px1) < 20 or (py2 - py1) < 20:
                continue

            # Crop the person out
            person_crop = frame[py1:py2, px1:px2].copy()
            name = "Unknown"

            try:
                name = find_person(person_crop)  
                if name is None or not isinstance(name, str) or name == "":
                    name = "Unknown"
            except Exception as e:
                # print(f"Recognition error: {e}")
                name = "Unknown"
            t_face_start = time.time()
            # # 4. DETECT FACES *INSIDE* THE PERSON CROP
            face_results = face_model.predict(person_crop, conf=CONF_FACE, verbose=False)

            time_spent_face_det += (time.time() - t_face_start)
            
            face_found = False
            
            if face_results and len(face_results[0].boxes) > 0:
                # Identify the largest face in the crop (in case of false positives)
                best_face = None
                max_area = 0

                for box in face_results[0].boxes:
                    fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                    area = (fx2 - fx1) * (fy2 - fy1)
                    if area > max_area:
                        max_area = area
                        best_face = (fx1, fy1, fx2, fy2)

                if best_face:
                    fx1, fy1, fx2, fy2 = best_face
                    
                    # Crop the FACE
                    face_img = person_crop[fy1:fy2, fx1:fx2].copy()
                    face_found = True

                    # Calculate Global Coordinates for drawing the face box
                    global_fx1 = px1 + fx1
                    global_fy1 = py1 + fy1
                    global_fx2 = px1 + fx2
                    global_fy2 = py1 + fy2

                    t_rec_start = time.time()
                    # 5. RECOGNIZE THE FACE
                    try:
                        # name = find_person(face_img)  
                        if name is None or not isinstance(name, str) or name == "":
                            name = "Unknown"
                    except Exception as e:
                        # print(f"Recognition error: {e}")
                        name = "Unknown"

                    time_spent_recognition += (time.time() - t_rec_start)

                    cv2.rectangle(frame, (global_fx1, global_fy1), (global_fx2, global_fy2), (255, 0, 0), 2)


            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 0), 2)

            label_color = (0, 255, 0) if name != "Unknown" else (50, 50, 255)
            cv2.putText(
                frame,
                name,
                (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                label_color,
                2
            )

        t_viz_end = time.time()

        dur_cap = t_read_end - t_start
        dur_person = t_person_det_end - t_read_end
        dur_box_proc = t_process_boxes_end - t_person_det_end
        dur_viz = t_viz_end - t_process_boxes_end - time_spent_face_det - time_spent_recognition # Subtract nested times
        dur_total = time.time() - t_start

        # --- PRINT DIAGNOSTICS ---
        # Using \r to overwrite line in console is cleaner, or just print block
        # print(f"--- FRAME STATS (Total: {dur_total:.4f}s) ---")
        # print(f"1. Capture:      {dur_cap:.4f}s")
        # print(f"2. Person Detect:{dur_person:.4f}s")
        # print(f"3. Box Merge:    {dur_box_proc:.4f}s")
        # print(f"4. Face Detect:  {time_spent_face_det:.4f}s [Count: {people_count}]")
        # print(f"5. Recognition:  {time_spent_recognition:.4f}s")
        # print(f"6. Viz/Draw:     {dur_viz:.4f}s")
        # print("-" * 30)

        fps = 1.0 / (time.time() - t0)
        fps_avg = 0.8 * fps_avg + 0.2 * fps
        cv2.putText(frame, f"{fps_avg:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Person -> Face -> Identify", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
