# app/utils_lang.py
from openai import OpenAI
from .config import OPENAI_API_KEY, GEN_MODEL, assert_env

assert_env()

_client = OpenAI(api_key=OPENAI_API_KEY)

def ko2en(text: str) -> str:
    """한국어를 자연스러운 영어 질의로 변환. 한국어가 아니면 원문을 그대로 반환."""
    sys = "You translate Korean user queries into concise English search queries. If the input isn't Korean, return it unchanged."
    msg = [
        {"role": "system", "content": sys},
        {"role": "user", "content": text},
    ]
    out = _client.chat.completions.create(model=GEN_MODEL, messages=msg, temperature=0)
    return out.choices[0].message.content.strip()
