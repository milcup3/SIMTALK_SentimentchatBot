from typing import List
import numpy as np
from openai import OpenAI
from .config import OPENAI_API_KEY, EMBED_MODEL, assert_env

assert_env()

_client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str], model: str = EMBED_MODEL, batch: int = 128) -> np.ndarray:
    """Return shape: (N, D) float32"""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = _client.embeddings.create(model=model, input=chunk)
        vecs = [np.array(e.embedding, dtype="float32") for e in resp.data]
        out.append(np.vstack(vecs))
    return np.vstack(out) if out else np.zeros((0, 0), dtype="float32")

def embed_one(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    resp = _client.embeddings.create(model=model, input=[text])
    return np.array(resp.data[0].embedding, dtype="float32")
