# server.py

import cv2
import numpy as np
import base64
import faiss
import json
from fastapi import FastAPI, WebSocket
from processor import process_frame

from models.yolo_person import YOLOPerson
from models.yolo_face import FaceDetector
from models.arcface import ArcFaceRecognizer
from models.yolo_license import YOLOLicense
from models.unknown_manager import UnknownManager
from configs.config import *

app = FastAPI()

print("[+] Loading models...")

faiss_index = faiss.read_index("dataset/faiss_index.bin")
labels = np.load("dataset/labels.npy")
with open("dataset/id_map.json") as f:
    id_map = json.load(f)

unknown_manager = UnknownManager(512, 0.45, 30)

print("[+] Loading Person Detector...")
person_det = YOLOPerson(PERSON_MODEL)
print("[+] Loading Face Detector...")
face_det   = FaceDetector(FACE_MODEL, PFLD_MODEL)
print("[+] Loading ArcFace...")
arc        = ArcFaceRecognizer(ARCFACE_MODEL)
print("[+] Loading Vehicle Detector...")
lpr        = YOLOLicense(PLATE_MODEL, VEHICLE_MODEL)

MODELS = {
    "person": person_det,
    "face": face_det,
    "arc": arc,
    "lpr": lpr,
    "faiss": faiss_index,
    "labels": labels,
    "id_map": id_map,
    "unknown": unknown_manager
}


@app.websocket("/ws")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("[+] Client connected.")

    while True:
        data = await websocket.receive_text()
        img_bytes = base64.b64decode(data)

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # PROCESS ONE FRAME
        annotated = process_frame(frame, MODELS)

        # ENCODE BACK
        _, buffer = cv2.imencode(".jpg", annotated)
        encoded = base64.b64encode(buffer).decode()

        await websocket.send_text(encoded)
