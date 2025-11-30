import cv2 # type: ignore
import numpy as np # type: ignore
import faiss # type: ignore
import json
from models.yolo_person import YOLOPerson
from models.yolo_face import FaceDetector
from models.arcface import ArcFaceRecognizer

from models.yolo_license import YOLOLicense

from configs.config import PERSON_MODEL, FACE_MODEL, PFLD_MODEL, ARCFACE_MODEL, VEHICLE_MODEL, PLATE_MODEL, MATCH_THRESHOLD

# Load FAISS index
faiss_index = faiss.read_index("dataset/faiss_index.bin")
labels = np.load("dataset/labels.npy")

with open("dataset/id_map.json", "r") as f:
    id_map = json.load(f)



print("[+] Loading Person Detector...")
person_det = YOLOPerson(PERSON_MODEL)

print("[+] Loading Face Detector...")
face_det = FaceDetector(FACE_MODEL, PFLD_MODEL)

print("[+] Loading ArcFace...")
arc = ArcFaceRecognizer(ARCFACE_MODEL)

# # Load embedding DB
# print("[+] Loading Embedding Database...")
# db_names, db_embs = arc.load_embeddings(EMB_DB_PATH)
# print(f"[+] Loaded identities: {db_names}")

print("[+] Loading Vehicle + License Plate System...")
lpr = YOLOLicense(VEHICLE_MODEL, PLATE_MODEL)


# -----------------------------
# CAMERA LOOP
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error.")
    exit()

print("\nPress Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    orig = frame.copy()

    # ---------------------------------
    # PERSON DETECTION (NO CHANGES)
    # ---------------------------------
    person_boxes, person_crops = person_det.detect(frame)

    for (px1, py1, px2, py2) in person_boxes:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 0), 2)

    # ---------------------------------
    # FACE RECOGNITION (NO CHANGES)
    # ---------------------------------
    for idx, (px1, py1, px2, py2) in enumerate(person_boxes):

        crop = orig[py1:py2, px1:px2]
        if crop.size == 0:
            continue

        boxes, pts98_list, pts5_list, aligned_list = face_det.detect(crop)

        if len(aligned_list) == 0:
            cv2.putText(frame, "NO FACE",
                        (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            continue

        aligned = aligned_list[0]
        if aligned is None:
            continue

        emb = arc.get_embeddings([aligned])[0]

        name, score = arc.find_match(
            query_emb=emb,
            faiss_index=faiss_index,
            labels=labels,
            id_map=id_map,
            threshold = MATCH_THRESHOLD
        )

        # name, score = matches[0]
        # if score < MATCH_THRESHOLD:
        #     name = "Unknown"

        label = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"
        cv2.putText(
            frame,
            label,
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )


    # -----------------------------------------------------
    # >>> LICENSE PLATE DETECTION (ADDED SAFELY BELOW) <<<
    # -----------------------------------------------------
    pairs = lpr.detect(orig)   # use original frame

    last_plate = ""

    for p in pairs:
        vx1, vy1, vx2, vy2 = map(int, p["vehicle_box"])
        px1, py1, px2, py2 = map(int, p["plate_box"])

        vehicle_type = p["vehicle_type"]
        plate_text   = p["plate_text"]

        # Draw vehicle box
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 150, 255), 2)
        cv2.putText(frame, vehicle_type, (vx1, vy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 150, 255), 2)

        # Draw plate box
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(frame, plate_text, (px1, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)

        last_plate = plate_text

    # Show last detected plate on top-left
    if last_plate != "":
        cv2.putText(frame, f"Plate: {last_plate}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)


    # ---------------------------------
    # SHOW OUTPUT (NO CHANGES)
    # ---------------------------------
    cv2.imshow("Face Recognition + LPR", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
