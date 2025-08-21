from typing import List, Tuple
import json
import numpy as np
from pathlib import Path
from .embeddings import embed_one

class SimpleIndex:
    def __init__(self, index_dir: str = "index"):
        p = Path(index_dir)
        data = np.load(p / "vectors.npz")
        self.vectors = data["vectors"].astype("float32")
        payload = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        self.texts: List[str] = payload["texts"]
        self.metas: List[dict] = payload["metas"]
        # pre-normalize for cosine
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = self.vectors / norms

    def search(self, query: str, top_k: int = 6) -> List[Tuple[str, dict, float]]:
        q = embed_one(query)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self.vectors @ q
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        results = []
        for i in idx:
            results.append((self.texts[int(i)], self.metas[int(i)], float(sims[int(i)])))
        return results
