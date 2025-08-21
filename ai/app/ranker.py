from typing import List, Tuple
from openai import OpenAI
from .config import OPENAI_API_KEY, GEN_MODEL, assert_env

assert_env()

from konlpy.tag import Okt
from transformers import pipeline

_client = OpenAI(api_key=OPENAI_API_KEY)

def gpt_rerank(question: str, candidates: List[Tuple[str, dict, float]], top_k: int = 4):
    """Ask the model to pick the most relevant snippets. Returns same tuple list."""
    if not candidates: return []
    snippets = [c[0] for c in candidates]
    prompt = (
        "질문과 후보 문맥들이 주어집니다. 가장 관련도 높은 문맥 상위 {k}개를 번호로만, 콤마로 구분해 출력하세요.\n"
        "형식: 1,3,4\n\n"
        f"질문: {question}\n"
        "후보:\n" + "\n".join(f"{i+1}) {s[:1000]}" for i, s in enumerate(snippets))
    )
    msg = [{"role": "user", "content": prompt}]
    resp = _client.chat.completions.create(model=GEN_MODEL, messages=msg, temperature=0)
    text = resp.choices[0].message.content.strip()
    try:
        picks = [int(x)-1 for x in text.split(",") if x.strip().isdigit()]
    except Exception:
        picks = list(range(min(top_k, len(candidates))))
    picks = [p for p in picks if 0 <= p < len(candidates)]
    picks = picks[:top_k]
    return [candidates[i] for i in picks]


# === Emotion/Keyword Analyzer for dialogue/report integration ===
class EmotionKeywordAnalyzer:
    def __init__(self, emotion_model_name: str = "Jinuuuu/KoELECTRA_fine_tunning_emotion"):
        self.classifier = pipeline(
            "text-classification",
            model=emotion_model_name,
            tokenizer=emotion_model_name
        )
        self.tagger = Okt()

    def analyze(self, text: str):
        emotions = self.classifier(text)[0]
        keywords = list(set(noun for noun in self.tagger.nouns(text) if len(noun) > 1))
        return {
            "emotion_label": emotions['label'],
            "emotion_score": emotions['score'],
            "keywords": keywords
        }