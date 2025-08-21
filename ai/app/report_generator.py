# report_generator.py (improved)
# - 끊김 방지: max_tokens 상향 + 이어쓰기(pass-2) 복구
# - 요약 공정성: 전체 5턴 균형 압축(mini_text_400) 유지 강화
# - 긍정 맥락 보존: 간단한 긍/부 텍스트 휴리스틱으로 프롬프트 보정
# - 마크다운 정리: 불필요한 치환(r"markdown" → "- ") 제거, 평문 유지
# - 섹션 완결성: 1)~8) 누락 시 보완 생성(pass-3)

from typing import List, Dict, Any
from collections import Counter
import re

from openai import OpenAI
from .config import REPORT_PROMPT_TEMPLATE, SYSTEM_ROLE, assert_env

assert_env()

# ---------- 텍스트 전처리 ----------

def strip_markdown(text: str) -> str:
    """UI가 마크다운을 지원하지 않을 수 있으므로, 가독성 해치지 않는 선에서 평문화.
    - 코드펜스/인라인코드/헤더 기호 제거
    - 목록 기호는 '- '만 표준화 유지
    - 링크 문법 제거
    """
    # 코드 블록, 인라인 코드 제거
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # 헤더 기호 제거('#'만 제거, 제목 텍스트는 남김)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # 굵게/기울임 제거
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    # 리스트 기호 표준화
    text = re.sub(r"^\s*[\-\*\+]\s+", "- ", text, flags=re.MULTILINE)
    # 링크 제거 [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # 여분 공백 정리
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# ---------- 요약 및 맥락 보정 ----------

def _build_summary_text(conversation_history):
    """
    🔄 변경점: 요약/샘플링/압축 없이,
    대화의 '모든' 질문/답변을 그대로 연결해 리포트에 넣습니다.
    """
    if not conversation_history:
        return ""
    return "\n".join(
        f"Q: {c.get('question','')}\nA: {c.get('answer','')}"
        for c in conversation_history
    )

_POS_TOKENS = ["긍정", "좋", "존경", "배우", "친절", "기쁨", "행복", "감사", "의욕", "열심"]
_NEG_TOKENS = ["실망", "우울", "불안", "힘들", "싫", "화", "분노", "두렵", "공포", "짜증"]

def _estimate_positive_ratio(conversation_history: List[Dict[str, Any]]) -> float:
    pos = neg = 0
    for c in conversation_history:
        a = (c.get("answer") or "").lower()
        # 한글 포함이므로 단순 포함 체크(대소문자 혼용 대비)
        pos += sum(tok in a for tok in _POS_TOKENS)
        neg += sum(tok in a for tok in _NEG_TOKENS)
        # KoELECTRA 라벨 사용 가능 시 보조 신호
        lbl = (c.get("emotion_label") or "").lower()
        if lbl in {"positive", "긍정"}: pos += 1
        if lbl in {"negative", "부정"}: neg += 1
    denom = max(1, pos + neg)
    return pos / denom

# ---------- 후편집(레이아웃/문체) ----------

def _refine_layout_markdown(client: OpenAI, model: str, text: str) -> str:
    prompt = f"""
아래 텍스트를 임상 보고서의 형식과 흐름을 유지하면서
- H2 헤딩(##) 유지, 섹션 간 공백 1줄,
- 문단당 2~3문장,
- 목록은 '- '로 정리,
- 중복/군더더기 제거
하도록 출력해 주세요.

텍스트:
{text}
"""
    res = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2, max_tokens=2000
    )
    return (res.choices[0].message.content or "").strip()

def _refine_style_coherence(client: OpenAI, model: str, text: str) -> str:
    prompt = f"""
아래 텍스트의 의미는 유지하고 문장 연결을 매끄럽게 교정해 주세요.
- 단정 대신 '보임/시사됨/가능성' 표현 사용
- 과한 공손체/장황함 줄이기
- 오탈자/반복 제거

텍스트:
{text}
"""
    res = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2, max_tokens=700
    )
    return (res.choices[0].message.content or "").strip()

# ---------- 섹션 완결성 보장 ----------

_SECTION_PATTERNS = [
    r"##\s*1\)\s*요약",
    r"##\s*2\)\s*관찰",
    r"##\s*3\)\s*유사 사례",
    r"##\s*4\)\s*개인적 강점/보호요인",
    r"##\s*5\)\s*임상적 공식화",
    r"##\s*6\)\s*권고",
    r"##\s*7\)\s*주의 신호 및 대응",
    r"##\s*8\)\s*참고 및 한계",
]

def _missing_sections(md_text: str) -> List[int]:
    missing = []
    for i, pat in enumerate(_SECTION_PATTERNS, start=1):
        if not re.search(pat, md_text):
            missing.append(i)
    return missing

def _complete_missing_sections(client: OpenAI, model: str, draft_md: str) -> str:
    missing = _missing_sections(draft_md)
    if not missing:
        return draft_md
    want = ", ".join(f"{i})" for i in missing)
    prompt = f"""
아래 보고서 초안에서 {want} 섹션이 누락되었습니다.
동일한 톤과 구조로 해당 섹션을 보완하여 전체 문서를 완결성 있게 다시 출력해 주세요.
모든 섹션 제목은 '## n) 제목' 형식을 유지하세요.

초안:
{draft_md}
"""
    res = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.3, max_tokens=900
    )
    return (res.choices[0].message.content or draft_md).strip()

# ---------- 메인 엔트리 ----------

def generate_final_report(
    conversation_history: List[Dict[str, Any]],
    user_info: Dict[str, str],
    analysis_rag_db,  # .similarity_search(text, k) 지원
    openai_api_key: str,
    gen_model: str
) -> Dict[str, Any]:
    client = OpenAI(api_key=openai_api_key)

    # 0) 균형 요약
    summary_text = _build_summary_text(conversation_history)

    # 1) 유사 사례 검색
    retrieved_docs = analysis_rag_db.similarity_search(summary_text, k=1)
    retrieved_case = retrieved_docs[0].page_content if retrieved_docs else "(유사 사례 없음)"

    # 2) 키워드/감정 수집
    all_keywords = []
    for conv in conversation_history:
        all_keywords.extend(conv.get('keywords', []))
    top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(5)]
    emotion_scores = [float(conv.get('emotion_score', 0.0)) for conv in conversation_history]

    # 3) 긍정 맥락 보정 힌트
    pos_ratio = _estimate_positive_ratio(conversation_history)
    context_hint = """
[맥락 힌트]
- 본 대화는 전반적으로 긍정/건설적 정서가 두드러집니다. 불필요한 병리적 해석을 피하고, 강점과 보호요인을 중심으로 기술하세요.
""" if pos_ratio >= 0.6 else ""

    # 4) 1차 생성 (넉넉한 토큰)
    prompt = REPORT_PROMPT_TEMPLATE.format(
        name=user_info.get('name',''),
        age=user_info.get('age',''),
        gender=user_info.get('gender',''),
        summary_text=summary_text,
        retrieved_case=retrieved_case
    )
    prompt = context_hint + "\n" + prompt if context_hint else prompt

    draft = client.chat.completions.create(
        model=gen_model,
        messages=[
            {"role":"system","content": SYSTEM_ROLE or "한국어 심리 상담 전문가"},
            {"role":"user","content": prompt},
        ],
        temperature=0.4,
        max_tokens=1800,   # 기존 1100 → 1800 상향
        presence_penalty=0.0,
        frequency_penalty=0.1,
    ).choices[0].message.content.strip()

    # 5) 끊김 방지 이어쓰기(pass-2)
    if not draft.rstrip().endswith(("다.", "요.", ".", "!", "?")) or _missing_sections(draft):
        cont = client.chat.completions.create(
            model=gen_model,
            messages=[
                {"role":"system","content": SYSTEM_ROLE or "한국어 심리 상담 전문가"},
                {"role":"user","content": f"아래 보고서를 중복 없이 이어서 완결성 있게 마무리해 주세요. 모든 섹션을 포함해야 합니다.\n\n{draft}"},
            ],
            temperature=0.3,
            max_tokens=700,
        ).choices[0].message.content.strip()
        draft = (draft + "\n\n" + cont).strip()

    # 6) 섹션 누락 보완(pass-3)
    draft = _complete_missing_sections(client, gen_model, draft)

    # 7) 레이아웃/문체 정리
    formatted = _refine_layout_markdown(client, gen_model, draft)
    polished = _refine_style_coherence(client, gen_model, formatted)

    # 8) 평문화
    plain = strip_markdown(polished)

    return {
        "report_text": plain,
        "emotion_trend": emotion_scores,
        "top_keywords": top_keywords,
    }
