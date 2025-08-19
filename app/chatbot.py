from typing import List, Tuple
from openai import OpenAI
from .config import OPENAI_API_KEY, GEN_MODEL, TOP_K, MAX_CONTEXT_TOKENS, MAX_ANSWER_TOKENS, assert_env

assert_env()
from .retriever import SimpleIndex
from .ranker import gpt_rerank
from .prompting import SYSTEM_PROMPT, ANSWER_TEMPLATE
import re
from .utils_lang import ko2en

_client = OpenAI(api_key=OPENAI_API_KEY)

RELEVANCE_FLOOR = 0.35

def _gate_hits(hits):
    if not hits: return []
    return hits if hits[0][2] >= RELEVANCE_FLOOR else []


def _bilingual_queries(question: str) -> list[str]:
    # 원문 + (한국어일 때) 번역 영문을 함께 검색
    q_en = ko2en(question)
    # 한국어/영문 동일하면 원문만
    if q_en.strip().lower() == question.strip().lower():
        return [question]
    # 둘 다 사용 (dedup)
    out, seen = [], set()
    for q in [question, q_en]:
        if q and q not in seen:
            out.append(q); seen.add(q)
    return out

def _truncate_ctx(snippets: List[str], token_cap: int = MAX_CONTEXT_TOKENS) -> List[str]:
    # 아주 단순한 문자 길이 기반 컷 (토큰 근사)
    budget = token_cap * 4  # rough bytes
    kept = []
    used = 0
    for s in snippets:
        l = len(s)
        if used + l > budget: break
        kept.append(s)
        used += l
    return kept

def _format_refs(selected: List[Tuple[str, dict, float]]) -> str:
    lines = []
    for i, (txt, meta, sim) in enumerate(selected, 1):
        snippet = (txt[:120] + "...") if len(txt) > 120 else txt
        src = meta.get("source") or meta.get("row", "")
        lines.append(f"- [{i}] {src} | sim={sim:.3f} | {snippet}")
    return "\n".join(lines) if lines else "- (관련 컨텍스트 없음)"

def answer_naive(index: SimpleIndex, question: str) -> str:
    # ✨ bilingual 검색
    qs = _bilingual_queries(question)
    all_hits = []
    for q in qs:
        all_hits.extend(index.search(q, top_k=TOP_K))
    # 유니크 + 유사도 정렬
    uniq, seen = [], set()
    for h in all_hits:
        key = h[0][:200]
        if key in seen: continue
        seen.add(key); uniq.append(h)
    hits = sorted(uniq, key=lambda x: -x[2])[:max(6, TOP_K)]
    # (선택) GPT 재랭킹
    hits = gpt_rerank(question, hits, top_k=min(6, len(hits)))
    # 관련도 가드
    hits = _gate_hits(hits)
    ctx = _truncate_ctx([h[0] for h in hits]) if hits else []
    refs = _format_refs(hits) if hits else "- (관련 컨텍스트 없음)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"질문: {question}\n\n컨텍스트:\n" + "\n---\n".join(ctx)},
    ]
    resp = _client.chat.completions.create(model=GEN_MODEL, messages=messages, temperature=0.2, max_tokens=MAX_ANSWER_TOKENS)
    return ANSWER_TEMPLATE.format(answer=resp.choices[0].message.content.strip(), refs=refs)

def _multiqueries(question: str, n: int = 4) -> List[str]:
    prompt = (
        "다음 질문을 다양한 관점에서 재질의 4가지를 만드세요. 각 줄에 하나씩:\n"
        f"원질문: {question}"
    )

    msg = [{"role": "user", "content": prompt}]
    resp = _client.chat.completions.create(model=GEN_MODEL, messages=msg, temperature=0.2)
    lines = [l.strip("-• ").strip() for l in resp.choices[0].message.content.splitlines() if l.strip()]
    out, seen = [], set()
    for l in lines:
        if l not in seen:
            out.append(l)
            seen.add(l)
        if len(out) >= n: break
    return out or [question]

def answer_multiquery(index: SimpleIndex, question: str, mq: int = 4) -> str:
    # 1) KO/EN 병렬 시드 생성
    seeds = _bilingual_queries(question)

    # 2) 각 시드별로 재질의 생성하여 qs 채우기 (빈 리스트 방지)
    qs: List[str] = []
    per_seed = max(1, mq // max(1, len(seeds)))  # 최소 1개
    for seed in seeds:
        qs.extend(_multiqueries(seed, n=per_seed))

    # 3) 검색 (시드·재질의 전부)
    all_hits: List[Tuple[str, dict, float]] = []
    for q in qs:
        all_hits.extend(index.search(q, top_k=max(2, TOP_K // 2)))

    # 4) 중복 제거 + 유사도 정렬
    uniq, seen = [], set()
    for h in all_hits:
        key = h[0][:200]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    hits = sorted(uniq, key=lambda x: -x[2])[:max(TOP_K, 6)]

    # 5) (선택) GPT 재랭킹
    if hits:
        hits = gpt_rerank(question, hits, top_k=min(6, len(hits)))

    # 6) 관련도 가드
    hits = _gate_hits(hits)

    # 7) 컨텍스트/근거
    ctx = _truncate_ctx([h[0] for h in hits]) if hits else []
    refs = _format_refs(hits)

    # 8) 생성
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"질문: {question}\n\n컨텍스트:\n" + "\n---\n".join(ctx)},
    ]
    resp = _client.chat.completions.create(
        model=GEN_MODEL, messages=messages, temperature=0.2, max_tokens=MAX_ANSWER_TOKENS
    )
    return ANSWER_TEMPLATE.format(answer=resp.choices[0].message.content.strip(), refs=refs)
    
def _hyde_passage(question: str) -> str:
    prompt = f"다음 질문에 답하는 6~8문장 요약 글을 쓰세요(근거 가정 가능):\n질문: {question}"
    msg = [{"role": "user", "content": prompt}]
    resp = _client.chat.completions.create(model=GEN_MODEL, messages=msg, temperature=0.3)
    return resp.choices[0].message.content.strip()

def answer_hyde(index: SimpleIndex, question: str) -> str:
    pseudo = _hyde_passage(question)
    hits = index.search(pseudo, top_k=max(TOP_K, 8))
    hits = gpt_rerank(question, hits, top_k=min(6, len(hits)))
    ctx = _truncate_ctx([h[0] for h in hits])
    refs = _format_refs(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"질문: {question}\n\n컨텍스트:\n" + "\n---\n".join(ctx)},
    ]
    resp = _client.chat.completions.create(model=GEN_MODEL, messages=messages, temperature=0.2, max_tokens=MAX_ANSWER_TOKENS)
    return ANSWER_TEMPLATE.format(answer=resp.choices[0].message.content.strip(), refs=refs)
