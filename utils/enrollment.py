import os
import cv2
import numpy as np
from models.yolo_face import FaceDetector
from models.arcface import ArcFaceRecognizer

# -----------------------------------------
# CONFIG
# -----------------------------------------

INPUT_ROOT = r"C:\Users\devka\face_project\input_images"   # <- your folder structure
OUTPUT_DB = r"C:\Users\devka\face_project\embeddings_db.npz"

YOLO_MODEL = r"C:\Users\devka\crowdhuman\yolov8n-face-lindevs.pt"
PFLD_MODEL = r"C:\Users\devka\crowdhuman\PFLD_GhostOne_112_1_opt_sim.onnx"
ARCFACE_MODEL = r"C:\Users\devka\crowdhuman\glintr100.onnx"   # change if needed

MIN_IMAGES_PER_PERSON = 1   # minimum images required to enroll
MAX_EMB_PER_PERSON = 20     # limit to prevent huge DB

# -----------------------------------------
# LOAD MODELS
# -----------------------------------------

print("[+] Loading FaceDetector ...")
face_detector = FaceDetector(
    yolo_model_path=YOLO_MODEL,
    pfld_model_path=PFLD_MODEL,
    conf=0.25,
)

print("[+] Loading ArcFaceRecognizer ...")
arc = ArcFaceRecognizer(
    model_path=ARCFACE_MODEL,
    input_size=112,
)

# -----------------------------------------
# ENROLLMENT PIPELINE
# -----------------------------------------

all_names = []
all_embs = []

people = sorted(os.listdir(INPUT_ROOT))

for person_name in people:
    person_path = os.path.join(INPUT_ROOT, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"\n[+] Enrolling: {person_name}")

    person_embeddings = []
    images = os.listdir(person_path)

    for img_name in images:
        img_path = os.path.join(person_path, img_name)
        if not img_path.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"    [x] Could not load: {img_path}")
            continue

        # -------------------------
        # Detect face + landmarks
        # -------------------------
        boxes, pts98_list, pts5_list, aligned_list = face_detector.detect(img)

        if len(aligned_list) == 0:
            print(f"    [x] No face detected in: {img_name}")
            continue

        # Use first detected face
        aligned = aligned_list[0]
        if aligned is None:
            print(f"    [x] Alignment failed: {img_name}")
            continue

        # -------------------------
        # ArcFace embedding
        # -------------------------
        emb = arc.get_embeddings([aligned])[0]  # (512,)
        person_embeddings.append(emb)

        print(f"    [+] Added embedding from {img_name}")

        if len(person_embeddings) >= MAX_EMB_PER_PERSON:
            print("    [!] Max embeddings reached for this person.")
            break

    if len(person_embeddings) < MIN_IMAGES_PER_PERSON:
        print(f"[!] Not enough valid images for {person_name}, skipping.")
        continue

    # Average embedding (optional)
    person_embeddings = np.vstack(person_embeddings)
    final_emb = np.mean(person_embeddings, axis=0)
    final_emb = final_emb / (np.linalg.norm(final_emb) + 1e-10)

    all_names.append(person_name)
    all_embs.append(final_emb)

# -----------------------------------------
# SAVE DATABASE
# -----------------------------------------

if len(all_names) == 0:
    print("\n[!] No embeddings created. Check your input folders!")
    exit()

all_embs = np.vstack(all_embs).astype(np.float32)

arc.save_embeddings(OUTPUT_DB, all_names, all_embs)

print("\n======================================")
print(" Enrollment Completed Successfully! ")
print(" Saved database:", OUTPUT_DB)
print(" Total Persons Enrolled:", len(all_names))
print("======================================")
