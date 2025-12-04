

EXTRACT_PATH = "dataset/raw"
AUGMENTED_PATH = "dataset/npy_augmented_embeddings"

DATASET_DIR = "dataset/augmented"
SAVE_FAISS_PATH = "dataset/faiss_index.bin"
SAVE_LABELS_PATH = "dataset/labels.npy"
SAVE_ID_MAP_PATH = "dataset/id_map.json"

COSINE_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD = 0.4

N_AUGMENTS = 10

MATCH_THRESHOLD = 0.45

PERSON_MODEL = "weights/best.pt"
FACE_MODEL = "weights/yolov8n-face-lindevs.pt"
PFLD_MODEL = "weights/PFLD_GhostOne_112_1_opt_sim.onnx"
ARCFACE_MODEL = "weights/glintr100.onnx"
# EMB_DB_PATH = "C:\Users\devka\face_project\embeddings_db.npz"

VEHICLE_MODEL = "weights/yolov8s.pt"
PLATE_MODEL   = "weights/license_plate_detection.pt"

YOLO_MODEL_PATH = "weights/best.pt"
YOLO_PERSON_MODEL = "weights/best.pt"
YOLO_FACE_MODEL = "weights/yolov8n-face-lindevs.pt"