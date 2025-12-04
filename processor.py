# processor.py

import cv2
import numpy as np

def process_frame(frame, M):
    """
    Runs your entire multi-model inference on a single frame.
    M = dictionary of loaded models
    """

    person_det = M["person"]
    face_det   = M["face"]
    arc        = M["arc"]
    lpr        = M["lpr"]
    faiss_idx  = M["faiss"]
    labels     = M["labels"]
    id_map     = M["id_map"]
    unknown    = M["unknown"]

    orig = frame.copy()

    # --------------------------------------
    # PERSON DETECTION
    # --------------------------------------
    person_boxes, person_crops = person_det.detect(frame)

    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # --------------------------------------
    # FACE + ARCFACE + FAISS
    # --------------------------------------
    for idx, (px1, py1, px2, py2) in enumerate(person_boxes):

        crop = orig[py1:py2, px1:px2]
        if crop.size == 0:
            continue

        boxes, pts98_list, pts5_list, aligned_list = face_det.detect(crop)
        if len(aligned_list) == 0:
            continue

        aligned = aligned_list[0]
        emb = arc.get_embeddings([aligned])[0]

        name, score = arc.find_match(
            emb, faiss_idx, labels, id_map, threshold=0.5
        )

        if name == "Unknown" or name is None or score < 0.5:
            unknown_label, _ = unknown.match_unknown(emb)
            name = unknown_label

        cv2.putText(frame, f"{name} ({score:.2f})", (px1, py1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # --------------------------------------
    # VEHICLE + LICENSE PLATE DETECTION
    # --------------------------------------
    # pairs = lpr.detect(orig)   # vehicle + plate

    # for p in pairs:
    #     vx1, vy1, vx2, vy2 = map(int, p["vehicle_box"])
    #     px1, py1, px2, py2 = map(int, p["plate_box"])

    #     cv2.rectangle(frame, (vx1,vy1), (vx2,vy2), (0,150,255), 2)
    #     cv2.rectangle(frame, (px1,py1), (px2,py2), (255,0,0), 2)

    #     cv2.putText(frame, p["plate_text"], (px1,py1-5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # --------------------------------------
    # VEHICLE + LICENSE PLATE DETECTION (Optimized)
    # --------------------------------------

    # 1️⃣ First: run ONLY vehicle detector
    vehicle_boxes = lpr.detect_vehicles(orig)

    # If NO vehicles → skip the expensive OCR pipeline
    if len(vehicle_boxes) == 0:
        return frame

    # 2️⃣ Only now run the full pipeline (vehicle + plate + OCR)
    pairs = lpr.detect(orig)

    for p in pairs:
        vx1, vy1, vx2, vy2 = map(int, p["vehicle_box"])
        px1, py1, px2, py2 = map(int, p["plate_box"])

        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 150, 255), 2)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

        cv2.putText(frame, p["plate_text"], (px1, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


    # RETURN ANNOTATED FRAME
    return frame
