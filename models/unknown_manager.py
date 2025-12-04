import time
import faiss # type: ignore
import numpy as np # type: ignore
from datetime import datetime, timedelta


class UnknownManager:
    def __init__(self, emb_dim=512, threshold=0.45, expiry_minutes=30):
        self.threshold = threshold
        self.expiry = timedelta(minutes=expiry_minutes)

        self.index = faiss.IndexFlatIP(emb_dim)  # inner product index
        self.embeddings = []                    # raw embeddings
        self.ids = []                           # int id per embedding
        self.next_id = 1                        # Unknown_1, Unknown_2, ...
        self.id_to_label = {}                   # id → "Unknown_X"
        self.timestamps = {}                    # id → timestamp

    def _cleanup(self):
        """Remove expired unknown persons."""
        now = datetime.now()
        valid_ids = []
        valid_embs = []

        for emb, uid in zip(self.embeddings, self.ids):
            if now - self.timestamps[uid] < self.expiry:
                valid_ids.append(uid)
                valid_embs.append(emb)

        # rebuild index
        if len(valid_embs) > 0:
            valid_embs = np.vstack(valid_embs).astype("float32")
            self.index = faiss.IndexFlatIP(valid_embs.shape[1])
            self.index.add(valid_embs)
        else:
            self.index = faiss.IndexFlatIP(512)

        self.embeddings = valid_embs
        self.ids = valid_ids

    def match_unknown(self, emb):
        """Returns (label, id) for unknown person."""
        self._cleanup()

        if len(self.embeddings) == 0:
            return self._add_new_unknown(emb)

        emb = emb.reshape(1, -1).astype("float32")

        distances, indices = self.index.search(emb, 1)
        dist = float(distances[0][0])
        idx = int(indices[0][0])

        similarity = dist 

        if similarity < self.threshold:
            return self._add_new_unknown(emb)

        known_id = self.ids[idx]
        self.timestamps[known_id] = datetime.now()  # refresh timestamp

        return self.id_to_label[known_id], known_id

    def _add_new_unknown(self, emb):
        uid = self.next_id
        self.next_id += 1

        label = f"Unknown_{uid}"

        # store embedding
        emb = emb.reshape(1, -1).astype("float32")
        emb = emb/(np.linalg.norm(emb) + 1e-12)
        self.index.add(emb)

        self.embeddings = emb if len(self.embeddings) == 0 else np.vstack([self.embeddings, emb])
        self.ids.append(uid)
        self.timestamps[uid] = datetime.now()
        self.id_to_label[uid] = label

        return label, uid
