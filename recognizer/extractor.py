
import numpy as np # type: ignore
import cv2 # type: ignore
import insightface # type: ignore
from functools import lru_cache

 
ARC_MODEL = None

@lru_cache(maxsize=1)
def load_arcface():
    """Load ONNX ArcFace model (new InsightFace versions)."""
    global ARC_MODEL

    if ARC_MODEL is None:
        print("[ArcFace] Loading ONNX ArcFace model ...")

        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]  # GPUExecutionProvider
        )
        # app.prepare(ctx_id=0)
        app.prepare(ctx_id=0, det_size=(640, 640))

        ARC_MODEL = app.models["recognition"]

        print("[ArcFace] ONNX model loaded.")
    
    return ARC_MODEL


def get_arcface_embedding(face_bgr):
    if face_bgr is None:
        return None

    model = load_arcface()

    # face = cv2.resize(face_bgr, (112, 112))

    # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # face = np.transpose(face, (2, 0, 1))  # (3, 112, 112)

    # face = np.expand_dims(face, axis=0).astype("float32")
    if len(face_bgr.shape) == 2:  # grayscale
        img = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(face_bgr, (112, 112))  # ArcFace expected input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # emb = model.get_feat(face_bgr)
    # print("[ArcFace] face_crop shape:", img.shape)

    emb = model.get_feat(img)[0]  # model outputs (1, 512)
    
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype("float32")
