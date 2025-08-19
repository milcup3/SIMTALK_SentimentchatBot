# mini_text_400.py
import re
from typing import List

PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF\uE000-\uFFFD]")
MULTISPACE_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[\.!?…｡。！？])\s+|[\n\r]+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = PRINTABLE_RE.sub(" ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+\.\S+", " ", s)
    s = re.sub(r"`{1,3}.+?`{1,3}", " ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def split_sentences(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return [p.strip() for p in SENT_SPLIT_RE.split(s) if p.strip()]

def mini_one(text: str, target_chars: int = 400, hard_cap: int = 600) -> str:
    """텍스트를 400자 근처로 압축, 최대 600자까지 허용"""
    t = clean_text(text)
    if len(t) <= target_chars:
        return t
    sents = split_sentences(t)
    kept, total = [], 0
    for sent in sents:
        if not sent:
            continue
        if len(sent) > target_chars // 2:  # 문장이 너무 길면 하드 컷
            sent = sent[: target_chars // 2].rstrip() + "…"
        if total + len(sent) + 1 > target_chars:
            break
        kept.append(sent)
        total += len(sent) + 1
    out = " ".join(kept).strip()
    if len(out) > hard_cap:
        out = out[:hard_cap].rstrip() + "…"
    if len(out) < min(120, target_chars // 3):
        out = (t[:target_chars].rstrip() + "…") if len(t) > target_chars else t
    return out

def mini_texts(texts: List[str], dedup: bool = True) -> List[str]:
    out, seen = [], set()
    for tx in texts:
        m = mini_one(tx, target_chars=400, hard_cap=600)
        if dedup:
            key = (m[:200], len(m))
            if key in seen:
                continue
            seen.add(key)
        out.append(m)
    return out
