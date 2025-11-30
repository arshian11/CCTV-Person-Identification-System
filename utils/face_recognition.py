import cv2
import numpy as np
from models.yolo_person import YOLOPerson
from models.yolo_face import FaceDetector
from models.arcface import ArcFaceRecognizer


# -----------------------------
# CONFIG
# -----------------------------
PERSON_MODEL = r"C:\Users\devka\crowdhuman\runs\detect\train5\weights\best.pt"
FACE_MODEL = r"C:\Users\devka\crowdhuman\yolov8n-face-lindevs.pt"
PFLD_MODEL = r"C:\Users\devka\crowdhuman\PFLD_GhostOne_112_1_opt_sim.onnx"
ARCFACE_MODEL = r"C:\Users\devka\crowdhuman\glintr100.onnx"
EMB_DB_PATH = r"C:\Users\devka\face_project\embeddings_db.npz"

MATCH_THRESHOLD = 0.45   # cosine similarity threshold


# -----------------------------
# LOAD MODELS
# -----------------------------
print("[+] Loading Person Detector...")
person_det = YOLOPerson(PERSON_MODEL)

print("[+] Loading Face Detector...")
face_det = FaceDetector(FACE_MODEL, PFLD_MODEL)

print("[+] Loading ArcFace...")
arc = ArcFaceRecognizer(ARCFACE_MODEL)

# Load embedding DB
print("[+] Loading Embedding Database...")
db_names, db_embs = arc.load_embeddings(EMB_DB_PATH)
print(f"[+] Loaded identities: {db_names}")


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
    # PERSON DETECTION
    # ---------------------------------
    person_boxes, person_crops = person_det.detect(frame)

    # Draw person boxes first
    for (px1, py1, px2, py2) in person_boxes:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 0), 2)

    # ---------------------------------
    # LOOP OVER PERSONS
    # ---------------------------------
    for idx, (px1, py1, px2, py2) in enumerate(person_boxes):

        crop = orig[py1:py2, px1:px2]
        if crop.size == 0:
            continue

        # -----------------------------
        # FACE DETECTION inside PERSON
        # -----------------------------
        boxes, pts98_list, pts5_list, aligned_list = face_det.detect(crop)

        if len(aligned_list) == 0:
            # no face in this person box
            cv2.putText(frame, "NO FACE",
                        (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            continue

        aligned = aligned_list[0]  # use first face found
        if aligned is None:
            continue

        # -----------------------------
        # ArcFace EMBEDDING
        # -----------------------------
        emb = arc.get_embeddings([aligned])[0]

        # -----------------------------
        # MATCHING
        # -----------------------------
        matches = arc.find_match(
            query_emb=emb,
            db_names=db_names,
            db_embs=db_embs,
            topk=1,
            threshold=MATCH_THRESHOLD
        )

        name, score = matches[0]
        if score < MATCH_THRESHOLD:
            name = "Unknown"

        # -----------------------------
        # DRAW RESULT
        # -----------------------------
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

    # ---------------------------------
    # SHOW OUTPUT
    # ---------------------------------
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
