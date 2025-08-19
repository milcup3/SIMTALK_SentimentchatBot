import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Any
from .config import REPORT_PROMPT_TEMPLATE, SYSTEM_ROLE, assert_env

assert_env()
from openai import OpenAI

def generate_final_report(
    conversation_history: List[Dict[str, Any]],
    user_info: Dict[str, str],
    analysis_rag_db,  # Should provide .similarity_search(text, k)
    openai_api_key: str,
    gen_model: str
) -> Dict[str, Any]:
    client = OpenAI(api_key=openai_api_key)
    if len(conversation_history) > 3:
        summary_conv_list = [conversation_history[0]] + conversation_history[-2:]
    else:
        summary_conv_list = conversation_history
    summary_text = "\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in summary_conv_list])
    # Retrieve similar case
    retrieved_docs = analysis_rag_db.similarity_search(summary_text, k=1)
    retrieved_case = retrieved_docs[0].page_content if retrieved_docs else "(유사 사례 없음)"
    # Collect keywords (simple noun extraction)
    all_keywords = []
    for conv in conversation_history:
        all_keywords.extend(conv.get('keywords', []))
    top_keywords = [kw[0] for kw in Counter(all_keywords).most_common(5)]
    # Collect emotion scores
    emotion_scores = [conv.get('emotion_score', 0.0) for conv in conversation_history]
    # Compose prompt
    prompt = REPORT_PROMPT_TEMPLATE.format(
        name=user_info.get('name', ''),
        age=user_info.get('age', ''),
        gender=user_info.get('gender', ''),
        summary_text=summary_text,
        retrieved_case=retrieved_case
    )
    messages = [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(model=gen_model, messages=messages, temperature=0.7, max_tokens=1100)
    report_text = resp.choices[0].message.content.strip()
    if len(report_text) > 900 and not report_text.rstrip().endswith(("다.", "요.", ".", "!", "?")):
        cont = client.chat.completions.create(
            model=gen_model,
            messages=[
                {"role": "system", "content": SYSTEM_ROLE},
                {"role": "user", "content": f"{prompt}\n\n[이어서 완결성 있게 마무리해 주세요. 중복 없이 결론과 실천 항목을 보강]"}
            ],
            temperature=0.5, max_tokens=400
        )
        report_text += "\n\n" + (cont.choices[0].message.content or "").strip()
    return {
        "report_text": report_text,
        "emotion_trend": emotion_scores,
        "top_keywords": top_keywords,
    }
