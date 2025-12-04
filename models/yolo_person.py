import numpy as np # type: ignore
from ultralytics import YOLO # type: ignore


class YOLOPerson:
    """
    Clean production-ready YOLO person detector.
    Only does:
    - detect persons
    - merge duplicate boxes
    - smooth temporally
    - return boxes + crops
    """

    def __init__(
        self,
        model_path,
        conf=0.35,
        iou=0.55,
        max_det=30,
        merge_iou=0.50,
        smooth_factor=0.5,
    ):
        self.model = YOLO(model_path)

        # config
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.target_class = [0]  # only person

        self.merge_iou = merge_iou
        self.smooth_factor = smooth_factor

        # tracking
        self.prev_boxes = None

    # ----------------------------------------------------
    # Helper: merge overlapping boxes
    # ----------------------------------------------------
    def _merge_boxes(self, boxes, threshold=0.5):
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
                xx1 = max(x1a, x1b)
                yy1 = max(y1a, y1b)
                xx2 = min(x2a, x2b)
                yy2 = min(y2a, y2b)
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                if inter <= 0:
                    continue

                area1 = (x2a - x1a) * (y2a - y1a)
                area2 = (x2b - x1b) * (y2b - y1b)
                iou = inter / (area1 + area2 - inter + 1e-12)

                if iou > threshold:
                    # average-merge
                    x1a = int((x1a + x1b) / 2)
                    y1a = int((y1a + y1b) / 2)
                    x2a = int((x2a + x2b) / 2)
                    y2a = int((y2a + y2b) / 2)
                    used[j] = True

            merged.append([x1a, y1a, x2a, y2a])
            used[i] = True

        return merged

    # ----------------------------------------------------
    # Helper: temporal smoothing
    # ----------------------------------------------------
    def _smooth_boxes(self, prev, curr, alpha=0.5):
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

    # ----------------------------------------------------
    # Main detection function
    # ----------------------------------------------------
    def detect(self, frame):
        """
        input: BGR frame
        return:
            - smoothed person boxes [x1,y1,x2,y2]
            - person_crops (array of images)
        """

        h, w = frame.shape[:2]

        # YOLO inference
        res = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            classes=self.target_class,
            verbose=False
        )

        boxes = []
        if res and len(res[0].boxes):
            xyxy = res[0].boxes.xyxy
            try:
                arr = xyxy.cpu().numpy()
            except:
                arr = np.asarray(xyxy)

            for b in arr:
                x1, y1, x2, y2 = map(int, b[:4])
                boxes.append([x1, y1, x2, y2])

        # merge duplicates
        merged = self._merge_boxes(boxes, self.merge_iou)

        # temporal smoothing
        smoothed = self._smooth_boxes(self.prev_boxes, merged, self.smooth_factor)
        self.prev_boxes = smoothed

        # extract crops
        crops = []
        for (x1, y1, x2, y2) in smoothed:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                crop = frame[y1:y2, x1:x2].copy()
                crops.append(crop)

        return smoothed, crops


# class YOLOPerson:
#     """
#     Extended YOLO detector:
#     - detect persons
#     - detect vehicles
#     - merge duplicate boxes
#     - smooth temporally
#     - return person/vehicle boxes + crops
#     """

#     def __init__(
#         self,
#         model_path,
#         conf=0.35,
#         iou=0.55,
#         max_det=30,
#         merge_iou=0.50,
#         smooth_factor=0.5,
#     ):
#         self.model = YOLO(model_path)

#         # config
#         self.conf = conf
#         self.iou = iou
#         self.max_det = max_det

#         # person + vehicle classes
#         self.person_class = [0]
#         self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
#         self.target_classes = self.person_class + self.vehicle_classes

#         self.merge_iou = merge_iou
#         self.smooth_factor = smooth_factor

#         self.prev_person = None
#         self.prev_vehicle = None

#     def _merge_boxes(self, boxes, threshold=0.5):
#         if len(boxes) <= 1:
#             return boxes

#         merged = []
#         used = [False] * len(boxes)

#         for i in range(len(boxes)):
#             if used[i]:
#                 continue

#             x1a, y1a, x2a, y2a = boxes[i]

#             for j in range(i + 1, len(boxes)):
#                 if used[j]:
#                     continue

#                 x1b, y1b, x2b, y2b = boxes[j]

#                 xx1 = max(x1a, x1b)
#                 yy1 = max(y1a, y1b)
#                 xx2 = min(x2a, x2b)
#                 yy2 = min(y2a, y2b)
#                 inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

#                 if inter <= 0:
#                     continue

#                 area1 = (x2a - x1a) * (y2a - y1a)
#                 area2 = (x2b - x1b) * (y2b - y1b)
#                 iou = inter / (area1 + area2 - inter + 1e-12)

#                 if iou > threshold:
#                     x1a = int((x1a + x1b) / 2)
#                     y1a = int((y1a + y1b) / 2)
#                     x2a = int((x2a + x2b) / 2)
#                     y2a = int((y2a + y2b) / 2)
#                     used[j] = True

#             merged.append([x1a, y1a, x2a, y2a])
#             used[i] = True

#         return merged

#     def _smooth_boxes(self, prev, curr, alpha=0.5):
#         if prev is None or len(prev) != len(curr):
#             return curr

#         smoothed = []
#         for p, c in zip(prev, curr):
#             x1 = int((1 - alpha) * p[0] + alpha * c[0])
#             y1 = int((1 - alpha) * p[1] + alpha * c[1])
#             x2 = int((1 - alpha) * p[2] + alpha * c[2])
#             y2 = int((1 - alpha) * p[3] + alpha * c[3])
#             smoothed.append([x1, y1, x2, y2])

#         return smoothed

#     def detect(self, frame):
#         h, w = frame.shape[:2]

#         res = self.model.predict(
#             frame,
#             conf=self.conf,
#             iou=self.iou,
#             max_det=self.max_det,
#             classes=self.target_classes,
#             verbose=False
#         )

#         person_boxes = []
#         vehicle_boxes = []

#         if res and len(res[0].boxes):
#             for box in res[0].boxes:
#                 cls = int(box.cls[0])
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

#                 if cls in self.person_class:
#                     person_boxes.append([x1, y1, x2, y2])
#                 elif cls in self.vehicle_classes:
#                     vehicle_boxes.append([x1, y1, x2, y2])

#         # merge
#         person_boxes = self._merge_boxes(person_boxes, self.merge_iou)
#         vehicle_boxes = self._merge_boxes(vehicle_boxes, self.merge_iou)

#         # smooth
#         person_boxes = self._smooth_boxes(self.prev_person, person_boxes, self.smooth_factor)
#         vehicle_boxes = self._smooth_boxes(self.prev_vehicle, vehicle_boxes, self.smooth_factor)

#         self.prev_person = person_boxes
#         self.prev_vehicle = vehicle_boxes

#         # crops
#         person_crops = []
#         vehicle_crops = []

#         for (x1, y1, x2, y2) in person_boxes:
#             crop = frame[y1:y2, x1:x2].copy()
#             if crop.size > 0:
#                 person_crops.append(crop)

#         for (x1, y1, x2, y2) in vehicle_boxes:
#             crop = frame[y1:y2, x1:x2].copy()
#             if crop.size > 0:
#                 vehicle_crops.append(crop)

#         return person_boxes, person_crops, vehicle_boxes, vehicle_crops
