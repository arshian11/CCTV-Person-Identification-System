import faiss # type: ignore
import numpy as np # type: ignore
import json
# import threading
from recognizer.extractor import get_arcface_embedding
from configs.config import CLASSIFICATION_THRESHOLD

# arcface_lock = threading.Lock()
# faiss_lock = threading.Lock()


# Load FAISS index
index = faiss.read_index("dataset/faiss_index.bin")
labels = np.load("dataset/labels.npy")

with open("dataset/id_map.json", "r") as f:
    id_map = json.load(f)


def find_person(face_crop):
    """Face embedding → FAISS lookup → return person name."""
    
    emb = get_arcface_embedding(face_crop)
    # with arcface_lock:
    #     emb = get_arcface_embedding(face_crop)
    if emb is None:
        return "Unknown"

    emb = emb.reshape(1, -1)
    # with faiss_lock:
    #distances, indices = index.search(emb.reshape(1, -1), 1)
    distances, indices = index.search(emb, 1)

    dist = float(distances[0][0])
    idx = int(indices[0][0])

    # monotonic_similarity = 1 / (1 + dist)
    # cosine_sim = 1 - (dist * dist) / 2.0
    cosine_sim = dist 

    # sigmoid_similarity = np.exp(-dist)    # smooth, range (0 to 1)



    # print(f"FAISS distance: {dist:.4f}")
    # print(f"Monotonic similarity: {monotonic_similarity:.4f}")
    # print(f"Cosine similarity: {cosine_sim:.4f}")
    # print(f"Sigmoid similarity: {sigmoid_similarity:.4f}")
    # print("Threshold:", CLASSIFICATION_THRESHOLD)

    if cosine_sim < CLASSIFICATION_THRESHOLD:
        return "Unknown"

    person_id = labels[idx]

    # print(" Person → predicted: ", id_map[str(person_id)])

    return id_map[str(person_id)]

