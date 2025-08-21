# app/minirag.py (신규)
import numpy as np
from typing import List
from openai import OpenAI
from .config import OPENAI_API_KEY, EMBED_MODEL, assert_env

assert_env()

_client = OpenAI(api_key=OPENAI_API_KEY)

class _Doc:
    def __init__(self, text: str): self.page_content = text

def _embed(texts: List[str]) -> np.ndarray:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return vecs

class MiniRAG:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.vecs = _embed(texts)

    def similarity_search(self, query: str, k: int = 1):
        q = _embed([query])[0]
        sims = self.vecs @ q
        top = sims.argsort()[-k:][::-1]
        return [_Doc(self.texts[i]) for i in top]
