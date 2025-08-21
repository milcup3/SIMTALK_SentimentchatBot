# app/indexer.py
import argparse, json, re
from typing import List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from .embeddings import embed_texts

PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF\uE000-\uFFFD]")

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = PRINTABLE_RE.sub(" ", s)          # 제어문자/비인쇄 제거
    s = re.sub(r"\s+", " ", s).strip()    # 공백 정규화
    return s

def _chunk_by_chars(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    n = len(text)
    if n == 0:
        return chunks
    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def _build_records(df: pd.DataFrame, text_cols: List[str], source_name: str) -> List[Tuple[str, dict]]:
    records: List[Tuple[str, dict]] = []
    for idx, row in df.iterrows():
        parts = []
        for c in text_cols:
            val = _clean_text(row.get(c, ""))
            if val:
                parts.append(f"{c}: {val}")
        merged = "\n".join(parts)
        meta = {"row": int(idx), "source": source_name, "cols": list(text_cols)}
        records.append((merged, meta))
    return records

def main(csv_path: str, text_cols: List[str], out_dir: str, chunk_size: int, chunk_overlap: int, encoding: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading CSV: {csv_path} (encoding={encoding})")
    df = pd.read_csv(csv_path, encoding=encoding)
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"지정한 컬럼 '{c}' 이(가) CSV에 없습니다. 실제 컬럼: {list(df.columns)}")

    source_name = Path(csv_path).name
    records = _build_records(df, text_cols, source_name)
    print(f"[INFO] Records: {len(records)} rows")

    texts, metas = [], []
    for text, meta in tqdm(records, desc="Chunking"):
        for ch in _chunk_by_chars(text, chunk_size, chunk_overlap):
            if ch:  # 빈 문자열 방지
                texts.append(ch)
                metas.append(meta)

    print(f"[INFO] Chunks: {len(texts)} (chunk_size={chunk_size}, overlap={chunk_overlap})")
    if len(texts) == 0:
        raise RuntimeError("생성된 청크가 0개입니다. 텍스트 컬럼이 비었거나 전처리에서 모두 제거됐을 수 있습니다.")

    # 임베딩
    try:
        vecs = embed_texts(texts)
    except Exception as e:
        raise RuntimeError(f"임베딩 호출 실패: {e}")

    if not isinstance(vecs, np.ndarray) or vecs.size == 0:
        raise RuntimeError("임베딩 결과가 비었습니다. API 키/네트워크/모델명을 확인하세요.")

    print(f"[INFO] Embeddings shape: {vecs.shape}")

    # 저장
    np.savez_compressed(out / "vectors.npz", vectors=vecs.astype("float32"))
    (out / "meta.json").write_text(
        json.dumps({"texts": texts, "metas": metas}, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"[OK] Saved index to: {out} (vectors.npz, meta.json)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV 파일 경로")
    ap.add_argument("--text-cols", nargs="+", required=True, help="검색 대상 컬럼명들")
    ap.add_argument("--out", default="index", help="인덱스 저장 폴더")
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--encoding", default="utf-8", help="CSV 파일 인코딩")
    args = ap.parse_args()
    main(args.csv, args.text_cols, args.out, args.chunk_size, args.chunk_overlap, args.encoding)
