"""
arcface.py
Production-ready ArcFace recognizer wrapper.

Features:
- Supports PyTorch (.pt/.pth) and ONNX (.onnx) ArcFace (glintr100 or similar).
- Auto-detect backend based on model file extension.
- Batch embedding extraction from aligned faces (112x112).
- L2 normalization of embeddings.
- Save / load embeddings DB (npz).
- Fast cosine similarity search (top-k).
- Simple thresholding for recognition.

Usage (example):
    from models.arcface import ArcFaceRecognizer
    recog = ArcFaceRecognizer(r"C:/path/to/glintr100.onnx") # type: ignore

    # aligned_faces: list/ndarray of 112x112 BGR or RGB images
    embs = recog.get_embeddings(aligned_faces)   # shape: (N, D)

    # Save DB
    recog.save_embeddings("emb_db.npz", names_list, embs)

    # Load DB
    names, db_embs = recog.load_embeddings("emb_db.npz")

    # Find match
    matches = recog.find_match(emb, names, db_embs, topk=1, threshold=0.45)
"""

from typing import List, Tuple, Optional, Union
import os
import numpy as np # type: ignore
import cv2 # type: ignore

# Try optional imports
_torch_available = False
try:
    import torch # type: ignore
    _torch_available = True
except Exception:
    _torch_available = False

_onx_available = False
try:
    import onnxruntime as ort # type: ignore
    _onx_available = True
except Exception:
    _onx_available = False


# class ArcFaceRecognizer:
#     def __init__(
#         self,
#         model_path: str,
#         input_size: int = 112,
#         device: Optional[str] = None,
#         normalize: bool = True,
#     ):
#         """
#         Params:
#             model_path: path to .onnx or PyTorch .pt/.pth ArcFace model
#             input_size: expected face input (default 112)
#             device: 'cpu' or 'cuda' (if PyTorch available). If None, auto-select.
#             normalize: L2-normalize embeddings before returning
#         """
#         self.model_path = model_path
#         self.input_size = input_size
#         self.normalize = normalize

#         ext = os.path.splitext(model_path.lower())[1]
#         self.backend = None

#         # Choose device
#         if device is None:
#             device = "cuda" if (_torch_available and torch.cuda.is_available()) else "cpu"
#         self.device = device

#         # Load model
#         if ext in [".onnx"]:
#             if not _onx_available:
#                 raise RuntimeError("onnxruntime not available. Install onnxruntime to use ONNX models.")
#             self.backend = "onnx"
#             self._load_onnx(model_path)
#         elif ext in [".pt", ".pth"]:
#             if not _torch_available:
#                 raise RuntimeError("torch not available. Install PyTorch to use .pt models.")
#             self.backend = "torch"
#             self._load_torch(model_path)
#         else:
#             raise ValueError("Unsupported model extension. Use .onnx or .pt/.pth")

class ArcFaceRecognizer:
    def __init__(
        self,
        model_path: str = None, # type: ignore
        input_size: int = 112,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        NOTE:
        model_path is IGNORED.
        We always load InsightFace Buffalo_L ArcFace model
        to match your FAISS embeddings.
        """
        self.input_size = input_size
        self.normalize = normalize

        # select device
        if device is None:
            device = "cuda" if (_torch_available and torch.cuda.is_available()) else "cpu"
        self.device = device

        # FORCE buffalo_l model
        self.backend = "onnx"
        self._load_buffalo_l()

    def _load_buffalo_l(self):
        """
        Load InsightFace Buffalo_L ArcFace model.
        If missing, auto-download it (same behavior as InsightFace library).
        """
        import insightface # type: ignore
        from insightface.app import FaceAnalysis # type: ignore

        # Use InsightFace to handle download + model management
        print("[ArcFace] Loading Buffalo_L model via InsightFace...")

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(112, 112))

        # Get the path to the automatically downloaded backbone ONNX model
        model_root = app.models["recognition"].model_file
        buffalo_model = model_root

        print(f"[ArcFace] Buffalo_L model path: {buffalo_model}")

        # Load using ONNX Runtime (keep your pipeline unchanged)
        self.onnx_sess = ort.InferenceSession(
            buffalo_model,
            providers=["CPUExecutionProvider"]
        )

        # Embedding dimension
        out = self.onnx_sess.get_outputs()[0]
        self.emb_dim = out.shape[-1] if len(out.shape) >= 1 else 512


    # -----------------------
    # Backend loaders
    # -----------------------
    def _load_onnx(self, path: str):
        # Use CPU execution provider by default; if CUDA available use it
        providers = ["CPUExecutionProvider"]
        # if onnxruntime-gpu installed, it may expose CUDAExecutionProvider
        try:
            sess_options = ort.SessionOptions()
            self.onnx_sess = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
        except Exception as e:
            # try without options
            self.onnx_sess = ort.InferenceSession(path, providers=providers)

        # inspect output shape to detect embedding dim
        out0 = self.onnx_sess.get_outputs()[0]
        self.emb_dim = out0.shape[-1] if len(out0.shape) >= 1 else 512

    def _load_torch(self, path: str):
        # Load PyTorch model, set to eval
        self.torch_model = torch.jit.load(path, map_location=self.device) if path.endswith(".pt") else torch.load(path, map_location=self.device)
        # If it's a nn.Module, ensure eval
        if hasattr(self.torch_model, "eval"):
            self.torch_model.eval()
        # Try to infer output dim by a dummy forward if safe
        self.emb_dim = None

    # -----------------------
    # Preprocessing
    # -----------------------
    def _preprocess(self, imgs: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Input: list or ndarray of images (H,W,3) BGR or RGB.
        Output: float32 numpy array shaped (N,C,H,W) in model-expected order.
        Standard ArcFace preprocess: RGB, /255, (maybe mean/std not used for most ArcFace)
        """
        if isinstance(imgs, np.ndarray):
            arr = imgs
        else:
            arr = np.stack(imgs, axis=0)

        # Ensure shape (N,H,W,3)
        if arr.ndim != 4: # type: ignore
            raise ValueError("Input must be list of images or ndarray with shape (N,H,W,3)")

        # Resize if needed
        N, H, W, C = arr.shape # type: ignore
        if H != self.input_size or W != self.input_size:
            resized = []
            for i in range(N):
                resized.append(cv2.resize(arr[i], (self.input_size, self.input_size)))
            arr = np.stack(resized, axis=0)

        # Convert BGR->RGB (many aligned crops are BGR from cv2)
        # Heuristic: if mean R channel < mean B channel -> assume BGR
        mean_r = arr[..., 0].mean() # type: ignore
        mean_b = arr[..., 2].mean() # type: ignore
        if mean_b > mean_r + 1e-3:
            arr = arr[..., ::-1]  # type: ignore # BGR->RGB

        arr = arr.astype(np.float32) / 255.0 # type: ignore
        # transpose to N,C,H,W
        arr = np.transpose(arr, (0, 3, 1, 2))
        return arr

    # -----------------------
    # Embedding extraction
    # -----------------------
    def get_embeddings(self, aligned_faces: Union[List[np.ndarray], np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        aligned_faces: list or ndarray of (112,112,3) images (BGR/RGB)
        returns: embeddings (N, D) numpy float32, optionally L2-normalized
        """
        if len(aligned_faces) == 0:
            return np.zeros((0, getattr(self, "emb_dim", 512)), dtype=np.float32)

        arr = self._preprocess(aligned_faces)  # (N,C,H,W)
        N = arr.shape[0]

        embs = []
        if self.backend == "onnx":
            for i in range(0, N, batch_size):
                batch = arr[i : i + batch_size]
                # onnxruntime expects (N,C,H,W) as float32 contiguous
                ort_in = {self.onnx_sess.get_inputs()[0].name: batch.astype(np.float32)}
                out = self.onnx_sess.run(None, ort_in)[0]
                embs.append(out.astype(np.float32))
            embs = np.vstack(embs)
        else:  # torch backend
            # Convert to torch tensor
            tensor = torch.from_numpy(arr).to(self.device)
            with torch.no_grad():
                if hasattr(self.torch_model, "__call__"):
                    out = self.torch_model(tensor)
                else:
                    out = self.torch_model.forward(tensor)
                # if tensor, convert
                if isinstance(out, torch.Tensor):
                    out = out.cpu().numpy()
            embs = out.astype(np.float32)

        # infer emb_dim if unknown
        if not hasattr(self, "emb_dim") or self.emb_dim is None:
            self.emb_dim = embs.shape[1]

        # L2 normalize
        if self.normalize:
            embs = self._l2_normalize(embs)

        return embs

    # -----------------------
    # Utilities: normalize + similarity
    # -----------------------
    @staticmethod
    def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-10) -> np.ndarray:
        norms = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norms + eps)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between rows of a and rows of b.
        Returns matrix shape (len(a), len(b)).
        """
        # assume already normalized -> cos = dot
        return np.dot(a, b.T)

    # -----------------------
    # Embedding DB IO
    # -----------------------
    def save_embeddings(self, path: str, names: List[str], embeddings: np.ndarray):
        """
        Save embedding DB to .npz
        path: file path .npz
        names: list of strings (len == N)
        embeddings: (N,D)
        """
        names_arr = np.array(names, dtype=object)
        np.savez_compressed(path, names=names_arr, embeddings=embeddings)
        return path

    def load_embeddings(self, path: str) -> Tuple[List[str], np.ndarray]:
        """
        Load DB saved by save_embeddings.
        Returns names list and embeddings ndarray.
        """
        data = np.load(path, allow_pickle=True)
        names = data["names"].tolist()
        embeddings = data["embeddings"].astype(np.float32)
        # Ensure normalized if requested
        if self.normalize:
            embeddings = self._l2_normalize(embeddings)
        return names, embeddings

    # -----------------------
    # Recognition search
    # -----------------------
    # def find_match(
    #     self,
    #     query_emb: np.ndarray,
    #     db_names: List[str],
    #     db_embs: np.ndarray,
    #     topk: int = 1,
    #     threshold: float = 0.45,
    # ) -> List[Tuple[str, float]]:
    #     """
    #     Given a single query embedding (D,) or (1,D), find top-k from DB.
    #     Returns list of tuples: (name, score) sorted desc by score.
    #     Score is cosine similarity (1 = identical). If score < threshold, you can treat as unknown.
    #     """
    #     if query_emb.ndim == 2 and query_emb.shape[0] == 1:
    #         q = query_emb[0]
    #     elif query_emb.ndim == 1:
    #         q = query_emb
    #     else:
    #         raise ValueError("query_emb must be 1D or shape (1,D)")

    #     if self.normalize:
    #         q = q / (np.linalg.norm(q) + 1e-10)
    #     sims = np.dot(db_embs, q)  # (N,)
    #     idx = np.argsort(-sims)[:topk]
    #     results = []
    #     for i in idx:
    #         score = float(sims[i])
    #         name = db_names[i]
    #         results.append((name, score))
    #     return results

    def find_match(
        self,
        query_emb: np.ndarray,
        faiss_index,
        labels: np.ndarray,
        id_map: dict,
        threshold: float = 0.45
    ):
        """
        FAISS-based matching (same style as find_person()).

        Params:
            query_emb: (D,) embedding or (1,D)
            faiss_index: faiss.Index
            labels: numpy array mapping index → person_id
            id_map: dict mapping person_id → person_name
            threshold: cosine similarity threshold
        """

        if query_emb.ndim == 2:
            query_emb = query_emb[0]

        if self.normalize:
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        q = query_emb.reshape(1, -1).astype("float32")

        distances, indices = faiss_index.search(q, 1)
        dist = float(distances[0][0])
        idx = int(indices[0][0])

        monotonic_similarity = 1 / (1 + dist)
        # cosine_sim = 1 - (dist * dist) / 2.0
        sigmoid_similarity = np.exp(-dist)

        cosine_sim = dist
        
        # print(f"FAISS distance: {dist:.4f}")
        # print(f"Monotonic similarity: {monotonic_similarity:.4f}")
        # print(f"Cosine similarity: {cosine_sim:.4f}")
        # print(f"Sigmoid similarity: {sigmoid_similarity:.4f}")
        # print("Threshold:", threshold)

        if cosine_sim < threshold:
            return ("Unknown", cosine_sim)

        person_id = labels[idx]
        person_name = id_map[str(person_id)]

        # print(" Person → predicted: ", id_map[str(person_id)])


        return (person_name, cosine_sim)


    def batch_search(
        self,
        query_embs: np.ndarray,
        db_names: List[str],
        db_embs: np.ndarray,
        topk: int = 3,
    ) -> List[List[Tuple[str, float]]]:
        """
        For multiple query embeddings. Returns list (len = Q) of topk lists.
        """
        # Normalize queries if necessary
        if self.normalize:
            query_embs = self._l2_normalize(query_embs)
        sims = self.cosine_similarity(query_embs, db_embs)  # (Q,N)
        results = []
        for row in sims:
            idx = np.argsort(-row)[:topk]
            results.append([(db_names[i], float(row[i])) for i in idx])
        return results
