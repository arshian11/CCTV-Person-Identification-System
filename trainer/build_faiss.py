import os
import numpy as np # type: ignore
import faiss # type: ignore
import json

AUGMENTED_PATH = "dataset/augmented"

from configs.config import AUGMENTED_PATH, SAVE_FAISS_PATH, SAVE_LABELS_PATH, SAVE_ID_MAP_PATH

embeddings = []
labels = []

person_to_id = {}
id_to_person = {}

id_counter = 0

for person in sorted(os.listdir(AUGMENTED_PATH)):
    person_path = os.path.join(AUGMENTED_PATH, person)
    if not os.path.isdir(person_path):
        continue

    id_counter += 1
    person_to_id[person] = id_counter
    id_to_person[id_counter] = person

    for file in os.listdir(person_path):
        if file.endswith(".npy"):
            emb = np.load(os.path.join(person_path, file))
            embeddings.append(emb)
            labels.append(id_counter)

embeddings = np.vstack(embeddings).astype("float32")    # shape: (N, 512)
labels = np.array(labels).astype("int")

print(f"[INFO] Loaded {embeddings.shape[0]} embeddings for {len(person_to_id)} identities.")

# np.save("labels.npy", labels)


print("Loaded embeddings:", embeddings.shape)
print("Loaded labels:", labels.shape)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(512)# cos index

index.add(embeddings)   #type: ignore
print("[INFO] FAISS index size:", index.ntotal)

# print("FAISS index size:", index.ntotal)

# faiss.write_index(index, "faiss_index.bin")

# with open("id_map.json", "w") as f:
#     json.dump(id_to_person, f)

faiss.write_index(index, SAVE_FAISS_PATH)
np.save(SAVE_LABELS_PATH, labels)

with open(SAVE_ID_MAP_PATH, "w") as f:
    json.dump(id_to_person, f, indent=4)

print("FAISS index + id_map saved!")
