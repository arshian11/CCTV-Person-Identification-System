import cv2 # type: ignore
import numpy as np # type: ignore
from ultralytics import YOLO # type: ignore
import onnxruntime as ort # type: ignore


class FaceDetector:
    def __init__(
        self,
        yolo_model_path,
        pfld_model_path,
        conf=0.25,
        input_size=112,
        providers=["CPUExecutionProvider"],
    ):
        # YOLO Face
        self.yolo = YOLO(yolo_model_path)
        self.conf = conf
        self.yolo.to('cuda')
        print("[YOLO-Face] Using device:", self.yolo.device)


        # PFLD Landmark Model
        self.sess = ort.InferenceSession(pfld_model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.INPUT_SIZE = input_size

        # Canonical 5 points for ArcFace
        self.ARC_IDX = [96, 97, 54, 76, 82]

        self.ARC_TEMPLATE = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    # ----------------------------------------------------
    # 5-point ArcFace alignment
    # ----------------------------------------------------
    def _align(self, img, pts5):
        M, _ = cv2.estimateAffinePartial2D(pts5.astype(np.float32),
                                           self.ARC_TEMPLATE,
                                           method=cv2.LMEDS)
        if M is None:
            return None

        aligned = cv2.warpAffine(
            img, M, (112, 112), flags=cv2.INTER_LINEAR)
        return aligned

    # ----------------------------------------------------
    # MAIN FUNCTION: detect faces + landmarks + aligned
    # ----------------------------------------------------
    def detect(self, frame):
        """
        Input: BGR frame
        Returns:
            - face_boxes : list of [x1,y1,x2,y2]
            - landmarks98 : list of arrays shape (98,2)
            - pts5_list : list of 5 landmark points
            - aligned_faces : list of 112x112 aligned faces
        """

        h, w = frame.shape[:2]

        # 1) YOLO FACE DETECTION
        results = self.yolo.predict(frame, conf=self.conf, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        face_boxes = []
        landmarks_list = []
        pts5_list = []
        aligned_list = []

        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # Clip
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # # 2) PFLD 98 LANDMARK
            # crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # crop_resized = cv2.resize(crop_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))

            # inp = crop_resized.astype(np.float32) / 255.0
            # inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]

            # out = self.sess.run(None, {self.input_name: inp})[0]
            # pts98 = out.reshape(98, 2)

            # # Scale back to original image coords
            # pts = pts98.copy()
            # pts[:, 0] = pts[:, 0] * (x2 - x1) + x1
            # pts[:, 1] = pts[:, 1] * (y2 - y1) + y1

            # # Extract 5 points
            # pts5 = pts[self.ARC_IDX]

            # # 3) ALIGN FACE
            # aligned = self._align(frame, pts5)

            # 2) PFLD 98 LANDMARK
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))

            inp = crop_resized.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]

            out = self.sess.run(None, {self.input_name: inp})[0]
            pts98 = out.reshape(98, 2)

            # Convert to crop coordinates (NOT full frame)
            pts98[:, 0] *= crop.shape[1]
            pts98[:, 1] *= crop.shape[0]

            # Extract 5 points
            pts5 = pts98[self.ARC_IDX]

            # 3) ALIGN FACE using crop, NOT full frame
            aligned = self._align(crop, pts5)


            # Append
            face_boxes.append([x1, y1, x2, y2])
            landmarks_list.append(pts5)
            pts5_list.append(pts5)
            aligned_list.append(aligned)

        return face_boxes, landmarks_list, pts5_list, aligned_list
