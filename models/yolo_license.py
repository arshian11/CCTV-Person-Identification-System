import cv2 # type: ignore
import numpy as np # type: ignore
from ultralytics import YOLO # type: ignore
from paddleocr import PaddleOCR # type: ignore


def center(box):
    x1, y1, x2, y2 = map(int, box)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(c1, c2):
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])


def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0

    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return interArea / float(areaA + areaB - interArea)


# class YOLOLicense:
#     """
#     Vehicle + License Plate + OCR
#     Using YOLO + PaddleOCR
#     """

#     def __init__(self, vehicle_model_path, plate_model_path):
#         print("[LPR] Loading YOLO vehicle model...")
#         self.vehicle_model = YOLO(vehicle_model_path)

#         print("[LPR] Loading YOLO plate model...")
#         self.plate_model = YOLO(plate_model_path)

#         print("[LPR] Loading PaddleOCR...")
#         # --- FIX: removed 'show_log=False' (invalid arg) ---
#         self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

#     def detect(self, frame):
#         """
#         Detect vehicles and plates, attach nearest plate to vehicle.
#         Returns list of:
#             {
#                 "vehicle_type": ...,
#                 "vehicle_box": [x1,y1,x2,y2],
#                 "plate_box":   [x1,y1,x2,y2],
#                 "plate_text":  "...",
#             }
#         """

#         # -------------------------
#         # 1) VEHICLE DETECTION
#         # -------------------------
#         det = self.vehicle_model(frame, imgsz=960)[0]

#         vehicles = []
#         labels = []

#         for box in det.boxes:
#             cls = int(box.cls[0])
#             label = self.vehicle_model.names[cls]
#             bb = box.xyxy[0].tolist()

#             if label in ["car", "motorbike", "motorcycle", "bus", "truck"]:
#                 vehicles.append(bb)
#                 labels.append(label)

#         # -------------------------
#         # 2) PLATE DETECTION
#         # -------------------------
#         plate_det = self.plate_model(frame, imgsz=1280)[0]
#         plate_boxes = [b.xyxy[0].tolist() for b in plate_det.boxes]

#         # OCR each plate
#         plates = []
#         for pb in plate_boxes:
#             x1, y1, x2, y2 = map(int, pb)
#             crop = frame[y1:y2, x1:x2]

#             text = ""
#             if crop.size != 0:
#                 ocr_res = self.ocr.ocr(crop, cls=True)
#                 # Extract text from OCR result
#                 if ocr_res and len(ocr_res[0]) > 0:
#                     text = ocr_res[0][0][1][0]

#             plates.append({
#                 "box": pb,
#                 "text": text
#             })

#         # -------------------------
#         # 3) VEHICLE ↔ PLATE matching
#         # -------------------------
#         final_pairs = []

#         for vbox, label in zip(vehicles, labels):

#             best_iou = -1
#             best_plate = None

#             # Try IoU first
#             for p in plates:
#                 score = iou(vbox, p["box"])
#                 if score > best_iou:
#                     best_iou = score
#                     best_plate = p

#             # Fallback: distance
#             if best_iou == 0:
#                 vc = center(vbox)
#                 best_dist = 999999

#                 for p in plates:
#                     pc = center(p["box"])
#                     d = distance(vc, pc)
#                     if d < best_dist:
#                         best_dist = d
#                         best_plate = p

#             final_pairs.append({
#                 "vehicle_type": label,
#                 "vehicle_box": vbox,
#                 "plate_box": best_plate["box"], # type: ignore
#                 "plate_text": best_plate["text"] # type: ignore
#             })

#         return final_pairs

class YOLOLicense:
    """
    Only License Plate Detection + OCR.
    Vehicle detection is removed.
    Input: vehicle crop from base YOLO model.
    """

    def __init__(self, plate_model_path):
        print("[LPR] Loading YOLO plate model...")
        self.plate_model = YOLO(plate_model_path)

        print("[LPR] Loading PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def detect_from_vehicle(self, vehicle_crop):
        h, w = vehicle_crop.shape[:2]

        det = self.plate_model(vehicle_crop, imgsz=640)[0]
        plate_boxes = [b.xyxy[0].tolist() for b in det.boxes]

        results = []

        for pb in plate_boxes:
            x1, y1, x2, y2 = map(int, pb)
            crop = vehicle_crop[y1:y2, x1:x2]

            text = ""
            if crop.size != 0:
                ocr_res = self.ocr.ocr(crop, cls=True)
                if ocr_res and len(ocr_res[0]) > 0:
                    text = ocr_res[0][0][1][0]

            results.append({
                "plate_box": pb,
                "plate_text": text
            })

        return results
