# main.py
import argparse
import os
import sys
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from openai import OpenAI

from app.retriever import SimpleIndex
from app.chatbot import answer_naive, answer_multiquery, answer_hyde
from app.config import load_env, assert_env  # assert_env가 없다면 load_env만 사용하세요
from app.question_manager import QuestionManager
from app.ranker import EmotionKeywordAnalyzer
from app.report_generator import generate_final_report
from app.prompting import (
    NEXT_QUESTION_SYSTEM,
    build_next_question_user_prompt,
    REFINE_QUESTION_SYSTEM,
    build_refine_question_user_prompt,
)


#===========경고 문구,디버깅 문구 제거===========
import warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
# ============ 상수/헬퍼 ============
PROMPT_PREFIX = "아래 대화 내용을 참고하여 자연스럽게 이어지는 다음 질문을 한 문장으로 만들어주세요.\n"


def resolve_sct_path(user_path: str, root: Path) -> Path:
    """SCT 질문 템플릿 경로 안전화"""
    p = Path(user_path)
    cands = [p if p.is_absolute() else (root / p)]
    # 흔한 위치 후보
    cands += [
        root / "sct_questions.jsonl",
        root.parent / "sct_questions.jsonl",
    ]
    for c in cands:
        if c.exists():
            #print(f"[INFO] SCT 템플릿 사용: {c}")
            return c
    raise FileNotFoundError(
        "SCT 질문 파일을 찾을 수 없습니다. 시도한 경로:\n  " + "\n  ".join(str(x) for x in cands)
    )


def resolve_analysis_csv(user_path: str, root: Path) -> Path:
    """리포트용 분석 CSV 경로 자동 탐색"""
    p = Path(user_path)
    cands = [p if p.is_absolute() else (root / p)]
    cands += [
        root / "Data" / "train.csv",
        root / "Data" / "train_data.csv",
        root.parent / "Data" / "train.csv",
        root.parent / "Data" / "train_data.csv",
    ]
    for c in cands:
        if c.exists():
            #print(f"[INFO] 분석 CSV 사용: {c}")
            return c
    raise FileNotFoundError(
        "분석용 CSV 파일을 찾을 수 없습니다. 시도한 경로:\n  " + "\n  ".join(str(x) for x in cands)
    )


def build_next_question(client: OpenAI, gen_model: str, history: List[dict], fallback_pool: List[str]) -> str:
    """직전 2턴 문맥을 사용해서 다음 질문 1문장 생성"""
    ctx = "".join(
        f"상담가: {conv['question']}\n사용자: {conv['answer']}\n"
        for conv in history[-2:]
    )
    prompt = f"{PROMPT_PREFIX}{ctx}\n상담가:"
    resp = client.chat.completions.create(
        model=gen_model,
        messages=[
            {"role": "system", "content": "한국어 심리 상담 전문가입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=100,
    )
    q = (resp.choices[0].message.content or "").strip()
    return q if q else random.choice(fallback_pool)


# ============ 리포트용 경량 RAG (파일 내 포함) ============
class _MiniDoc:
    def __init__(self, text: str):
        self.page_content = text

class MiniRAG:
    """Response 텍스트 리스트만으로 간단 유사도 검색하는 경량 RAG"""
    def __init__(self, texts: List[str], client: OpenAI, embed_model: str):
        self.texts = texts
        self.client = client
        self.embed_model = embed_model
        self.vecs = self._embed(texts)  # (N, D) L2 정규화

    def _embed(self, texts: List[str]) -> np.ndarray:
        BATCH = 100  # 한번에 100개씩
        arrs = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i+BATCH]
            embs = self.client.embeddings.create(model=self.embed_model, input=chunk).data
            arrs.append(np.array([e.embedding for e in embs], dtype="float32"))
        arr = np.vstack(arrs)
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
        return arr

    def similarity_search(self, query: str, k: int = 1) -> List[_MiniDoc]:
        q = self._embed([query])[0]
        sims = self.vecs @ q
        top = sims.argsort()[-k:][::-1]
        return [_MiniDoc(self.texts[i]) for i in top]


# ============ 메인 ============
def main():
    # ---- 인자
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "multiquery", "hyde", "sct"], default="naive")
    ap.add_argument("--mq", type=int, default=4, help="multiquery 개수")
    ap.add_argument("--sct_questions", type=str, default="../sct_questions.jsonl", help="SCT 질문 템플릿 경로")
    ap.add_argument("--analysis_csv", type=str, default="../Data/train.csv", help="분석용 RAG csv 경로")
    args = ap.parse_args()

    # ---- 루트/경로
    ROOT = Path(__file__).resolve().parent
    sct_path = resolve_sct_path(args.sct_questions, ROOT)
    analysis_csv_path = resolve_analysis_csv(args.analysis_csv, ROOT)

    # ---- 환경 & 클라이언트
    env = load_env()
    # assert_env()가 있다면 호출 (없다면 load_env 결과만 사용)
    try:
        assert_env()
    except Exception:
        pass

    openai_api_key = env["OPENAI_API_KEY"]
    embed_model = env["EMBED_MODEL"]
    gen_model = env["GEN_MODEL"]
    client = OpenAI(api_key=openai_api_key)

    # ---- 기본 인덱스 (대화 RAG)
    index = SimpleIndex("index")  # index/vectors.npz, meta.json 필요

    # ===================== SCT 모드 =====================
    if args.mode == "sct":
        print("🧠 SCT 5턴 대화 및 리포트 모드")

        # 1) 질문 템플릿 로드
        qm = QuestionManager(str(sct_path))

        # 2) 사용자 정보
        print("=" * 50)
        print("🤖 상담을 시작하기 전에 몇 가지 정보를 여쭤볼게요.")
        print("=" * 50)
        user_info = {
            "name": input("👤 이름: "),
            "age": input("👤 나이: "),
            "gender": input("👤 성별: "),
        }

        # 3) 카테고리 선택
        cats = qm.get_categories()
        print("\n" + "=" * 50)
        print("💬 어떤 주제에 대해 이야기하고 싶으신가요?")
        print("=" * 50) 
        for i, cat in enumerate(cats):
            print(f"  [{i+1}] {cat}")
        while True:
            try:
                choice = int(input("\n👤 원하는 주제의 번호 선택: "))
                if 1 <= choice <= len(cats):
                    selected_category = cats[choice - 1]
                    break
                print("⚠️ 잘못된 번호입니다. 다시 선택해주세요.")
            except ValueError:
                print("⚠️ 숫자로 입력해주세요.")
        print(f"\n✅ '{selected_category}' 카테고리를 선택하셨습니다.")

        question_pool = qm.get_questions_by_category(selected_category)
        conversation_history = []
        analyzer = EmotionKeywordAnalyzer()
        current_question = random.choice(question_pool)
        seed_q = current_question

        ref = client.chat.completions.create(
            model=gen_model,
            messages=[
                {"role": "system", "content": REFINE_QUESTION_SYSTEM},
                {"role": "user", "content": build_refine_question_user_prompt(seed_q or current_question)},
            ],
            temperature=0.2, max_tokens=60,
        )
        current_question = (ref.choices[0].message.content or current_question).strip()


        # 4) 대화(최대 5턴)
        MAX_TURNS = 5

        # ... (루프]]]]]]

        for turn in range(MAX_TURNS):
            print(f"\nAI 🤖 (질문 {turn+1}/{MAX_TURNS}): {current_question}")
            user_response = input("나 👤: ")

            if user_response.lower() in {"종료", "exit", "quit"}:
                print("🤖 대화를 종료합니다. 분석 리포트를 생성합니다.")
                break

            # 감정/키워드 분석 기록 (그대로)
            analysis = analyzer.analyze(user_response)
            conversation_history.append({
                "turn": turn + 1,
                "question": current_question,
                "answer": user_response,
                "emotion_label": analysis["emotion_label"],
                "emotion_score": analysis["emotion_score"],
                "keywords": analysis["keywords"],
            })

            # ── 다음 질문 생성 (직전 2턴 문맥 사용) ─────────────────────────────
            recent = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history[-1:]
            turns_text = "".join(
                f"상담가: {c['question']}\n사용자: {c['answer']}\n" for c in recent
            )

            # 1) 1차 질문 생성
            gen = client.chat.completions.create(
                model=gen_model,
                messages=[
                    {"role": "system", "content": NEXT_QUESTION_SYSTEM},
                    {"role": "user", "content": build_next_question_user_prompt(turns_text)},
                ],
                temperature=0.3,             # 톤 안정
                max_tokens=80,
                frequency_penalty=0.2,       # 중복 줄이기
                presence_penalty=0.0,
            )
            next_q = (gen.choices[0].message.content or "").strip()

            # 2) (선택) 후편집으로 더 부드럽게 교정
            if next_q:
                ref = client.chat.completions.create(
                    model=gen_model,
                    messages=[
                        {"role": "system", "content": REFINE_QUESTION_SYSTEM},
                        {"role": "user", "content": build_refine_question_user_prompt(next_q)},
                    ],
                    temperature=0.2,
                    max_tokens=60,
                )
                next_q = (ref.choices[0].message.content or "").strip()

            # 비상시 fallback
            if not next_q or len(next_q) < 3 or "?" not in next_q:
                next_q = random.choice(question_pool)

            current_question = next_q
    # ─────────────────────────────────────────────────────────────────────

        # 5) 리포트용 RAG 준비(MiniRAG)
        from mini_text_400 import mini_texts

        df = pd.read_csv(analysis_csv_path)
        if "Response" not in df.columns:
            raise RuntimeError(f"[오류] CSV에 'Response' 컬럼이 없습니다: {analysis_csv_path}")

        responses = df["Response"].dropna().astype(str).tolist()
        responses_small = mini_texts(responses)  # 400자 단위로 압축

        analysis_rag_db = MiniRAG(responses_small, client, embed_model)
        
        # 6) 리포트 생성/출력
        print("\n📊 최종 심리 분석 리포트를 생성 중입니다...")
        final_report = generate_final_report(
            conversation_history=conversation_history,
            user_info=user_info,
            analysis_rag_db=analysis_rag_db,  # similarity_search(text,k) 지원
            openai_api_key=openai_api_key,
            gen_model=gen_model,
        )

        print("\n[기본 정보]")
        print(f"  - 이름: {user_info['name']}")
        print(f"  - 나이: {user_info['age']}")
        print(f"  - 성별: {user_info['gender']}")
        print(f"  - 선택 카테고리: {selected_category}\n")
        print("[주요 키워드]")
        print(f"  이번 대화에서 자주 언급한 키워드: {', '.join(final_report['top_keywords'])}\n")
        print("[감정 트렌드]")
        #print(f"  감정 점수: {final_report['emotion_trend']}")
        print("[심리 해석 및 조언]")
        print(final_report["report_text"])
        return

    # ===================== 기본 RAG 챗봇 =====================
    print("💬 OpenAI-only RAG Chatbot")
    print("타이핑을 시작하세요. 종료: exit")
    while True:
        q = input("\n🙋 질문: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        if args.mode == "naive":
            ans = answer_naive(index, q)
        elif args.mode == "multiquery":
            ans = answer_multiquery(index, q, mq=args.mq)
        else:
            ans = answer_hyde(index, q)
        print("\n" + ans)


if __name__ == "__main__":
    main()
