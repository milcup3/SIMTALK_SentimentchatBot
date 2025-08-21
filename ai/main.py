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

# -------------------- prompting 모듈 임포트 (VALIDATE_SYSTEM 미정의 대비) --------------------
# NEXT_QUESTION_SYSTEM, REFINE_QUESTION_SYSTEM 등은 기존대로 사용하되,
# VALIDATE_SYSTEM이 아직 없다면 아래에서 안전한 기본값을 채워넣습니다.
try:
    from app.prompting import (
        NEXT_QUESTION_SYSTEM,
        build_next_question_user_prompt,
        REFINE_QUESTION_SYSTEM,
        build_refine_question_user_prompt,
    )
    try:
        from app.prompting import VALIDATE_SYSTEM  # 없을 수 있음
    except Exception:
        VALIDATE_SYSTEM = None
except Exception:
    # prompting 모듈 전체가 없을 가능성은 낮지만, 방어적으로 최소 프롬프트를 둡니다.
    NEXT_QUESTION_SYSTEM = "너는 공감형 상담가다. 사용자의 직전 답변을 받아 다음 한 문장 질문만 출력한다."
    REFINE_QUESTION_SYSTEM = "문장을 부드럽고 공감 있게 다듬되, 한 문장 질문으로 유지한다."
    def build_next_question_user_prompt(t: str) -> str:
        return f"다음 대화문맥을 읽고 한 문장 질문만 출력:\n{t}"
    def build_refine_question_user_prompt(q: str) -> str:
        return f"다음 문장을 더 자연스러운 질문 한 문장으로 다듬어줘:\n{q}"
    VALIDATE_SYSTEM = None  # 아래 기본값으로 대체

#=========== 경고 문구, 디버깅 문구 제거 ===========
import warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# ============ 상수/헬퍼 ============
PROMPT_PREFIX = "아래 대화 내용을 참고하여 자연스럽게 이어지는 다음 질문을 한 문장으로 만들어주세요.\n"

WARNING_MESSAGE = "비정상적인 단어가 검색되었습니다.채팅을 다시 시작해주세요."

CRISIS_MSG = (
    "지금 많이 힘드신 것 같아요. 저는 응급 상황을 직접 처리할 수는 없지만, "
    "안전이 가장 중요해요. 대한민국에 계시다면 112(긴급) 또는 국번 없이 1393(자살예방 상담, 24시간)으로 "
    "지금 바로 연락해 주세요. 가까운 분이나 전문기관에 즉시 도움을 요청하는 것도 큰 도움이 됩니다. "
    "필요한 지원을 받으실 수 있도록 대화를 잠시 중단할게요."
)

DISTRESS_MSG = (
    "힘든 마음을 나눠 주셔서 고마워요. 원하시면 전문 상담과 연결을 도와드릴게요. "
    "괜찮다면 지금 느끼는 감정을 조금 더 이야기해도 좋아요."
    "상담사 연결을 원하시면 '상담사 연결'이라고 입력해 주세요."
)

SUPPORT_MSG = (
    "전문 상담사와 직접 연결해드릴 수는 없지만, 도움을 받을 수 있는 공식 채널이 있습니다:\n"
    "- 보건복지부 자살예방상담전화: 1393 (24시간, 무료)\n"
    "- 정신건강 상담 전화: 1577-0199\n"
    "- 보건복지상담센터: 129\n\n"
    "가까운 정신건강복지센터나 정신건강의학과에서 전문적인 도움을 받아보시는 것도 큰 힘이 될 수 있어요."
)

# -------------------- VALIDATE_SYSTEM 기본 프롬프트 (없을 때만 주입) --------------------
if not VALIDATE_SYSTEM:
    VALIDATE_SYSTEM = """
너는 한국어 대화의 안전성 검증기다. 입력된 '사용자' 문장을 다음 4가지 중 하나로 **정확히 한 줄**로만 분류해 출력한다.

[출력값(정확히 아래 중 하나만)]
OK
ESCALATE:DISTRESS
ESCALATE:CRISIS
BLOCK

[분류 기준]
1) ESCALATE:CRISIS  ← 즉시 위험 신호(자/타해 의도·계획·구체적 표현, 강력 위협)
   예: "죽고 싶어요", "자살할 거예요", "지금 베고 싶다", "너 죽일 거야"
2) ESCALATE:DISTRESS ← 고통/우울/불안 등 정서적 위험 신호(계획·의도·위협은 없음)
   예: "우울해요", "사는 게 힘들다", "불안해서 잠이 안 와요"
3) BLOCK ← 아래 금지 항목에 해당
   - 미성년자 관련 성적 내용/착취
   - 불법·위험 행위 방법의 구체 지시/요청(폭탄·마약 제조법 등)
   - 주민등록번호/계좌/카드번호/정확한 집주소/로그인 비번 등 민감식별정보
   - 인종/성별/성적지향/장애 등에 대한 혐오 선동, 노골적 모욕·협박의 반복
4) OK ← 위 조건들(1~3)에 해당하지 않으면 모두 OK. 애매하면 OK.

[중요]
- 반드시 위 4개 중 하나만 그대로 출력한다. 설명/마크다운/따옴표/이모지 금지.
""".strip()


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


# -------------------- 안전성 검증 헬퍼 (LLM 출력 고정) --------------------
def _normalize_verdict(text: str) -> str:
    t = (text or "").strip().upper()
    return t

def validate_message(client: OpenAI, gen_model: str, current_question: str, user_response: str) -> str:
    """
    반환: 'OK' | 'ESCALATE:DISTRESS' | 'ESCALATE:CRISIS' | 'BLOCK' | 'UNKNOWN' | 'ERROR'
    """
    val_query = f"현재 질문: {current_question}\n사용자: {user_response}\n이 사용자 메시지를 위 지침으로 분류하세요."
    try:
        res = client.chat.completions.create(
            model=gen_model,
            messages=[
                {"role": "system", "content": VALIDATE_SYSTEM},
                {"role": "user", "content": val_query},
            ],
            temperature=0.0,
            max_tokens=8,
        )
        raw = (res.choices[0].message.content or "")
        verdict = _normalize_verdict(raw)
        if verdict in {"OK", "ESCALATE:DISTRESS", "ESCALATE:CRISIS", "BLOCK"}:
            return verdict
        return "UNKNOWN"
    except Exception as e:
        print(f"⚠️ 검증 호출 오류: {e}")
        return "ERROR"


# ============ 메인 ============
def main():
    # ---- 인자
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "multiquery", "hyde", "sct"], default="sct")
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
    try:
        assert_env()  # 없으면 except로 무시
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
        def info(i):
            if i == 1:
                print("이름을 입력해주세요")
            elif i == 2:
                print("나이를 입력해주세요")
            elif i == 3:
                print("성별을 입력해주세요 (남/여/기타)")
            return i+1
        # 2) 사용자 정보
        print("상담을 시작하기 전에 몇 가지 정보를 여쭤볼게요. 이름, 나이, 성별을 순차적으로 입력해주세요.")
        # 사용자 정보 입력 루프
        # (이름, 나이, 성별)
        # - 이름: 1~20자, 한글/영문/숫자만 허용
        # - 나이: 0~120, 숫자만 허용
        # - 성별: 남/여/기타 중 하나
        while True:
            try:
                print("이름은 1~20자 사이의 한글/영문/숫자만 허용됩니다.")
                name = input().strip()
                if not (1 <= len(name) <= 20) or not all(c.isalnum() or c in "가-힣" for c in name):
                    raise ValueError("이름은 1~20자 사이의 한글/영문/숫자만 허용됩니다.")
                break
            except ValueError as e:
                print(f"⚠️ {e}. 다시 입력해주세요.")

        while True:
            try:
                print("나이는 0~120 사이의 숫자여야 합니다.")
                age = int(input().strip())
                if not (0 <= age <= 120):
                    raise ValueError("나이는 0~120 사이의 숫자여야 합니다.")
                break
            except ValueError as e:
                print(f"⚠️ {e}. 다시 입력해주세요.")

        while True:
            try:
                print("성별은 '남', '여', '기타' 중 하나로 입력해주세요.")
                sex = input().strip()
                if not ('남' or '여' or '기타'):
                    raise ValueError("성별을 입력해주새요 (남/여/기타)")
                break
            except ValueError as e:
                print(f"⚠️ {e}. 다시 입력해주세요.")
        user_info = {
            "name": name,
            "age": age,
            "gender": sex,
        }


        # 3) 카테고리 선택
        cats = qm.get_categories()
        print("\n")
        print("어떤 주제에 대해 이야기하고 싶으신가요? 번호를 적어주세요")
        for i, cat in enumerate(cats):
            print(f"  [{i+1}] {cat}")
        while True:
            try:
                choice = int(input())
                if 1 <= choice <= len(cats):
                    selected_category = cats[choice - 1]
                    break
                print("잘못된 번호입니다. 다시 선택해주세요.")
            except ValueError:
                print("숫자로 입력해주세요.")
        print(f"\n'{selected_category}' 카테고리를 선택하셨습니다.")

        question_pool = qm.get_questions_by_category(selected_category)
        conversation_history = []
        analyzer = EmotionKeywordAnalyzer()
        current_question = random.choice(question_pool)
        seed_q = current_question

        # 시드 질문을 한 번 다듬기
        ref = client.chat.completions.create(
            model=gen_model,
            messages=[
                {"role": "system", "content": REFINE_QUESTION_SYSTEM},
                {"role": "user", "content": build_refine_question_user_prompt(seed_q or current_question)},
            ],
            temperature=0.2, max_tokens=60,
        )
        current_question = (ref.choices[0].message.content or current_question).strip()

        # 4) 대화(최대 5턴) — 같은 턴 유지 입력 루프 + 안전성 검증
        MAX_TURNS = 5
        for turn in range(MAX_TURNS):
            print(f"\nAI     (질문 {turn+1}/{MAX_TURNS}): {current_question}")

            # ----- 입력/검증 루프: 통과해야 턴 종료 -----
            while True:
                user_response = input().strip()

                
                # 종료어 처리(한글/영문)
                if user_response in {"종료", "그만"} or user_response.lower() in {"exit", "quit"}:
                    print("🤖 대화를 종료합니다.")
                    # 필요 시 여기서 리포트 생성 파트로 바로 이동하려면 return 대신 break/flag 사용
                    # 여기서는 즉시 리포트 생성으로 이동하도록 break 2중 탈출 처리
                    # (바깥 루프 탈출)
                    turn = MAX_TURNS - 1  # 바깥 for의 마지막 턴으로 설정
                    conversation_history.append({
                        "turn": turn + 1,
                        "question": current_question,
                        "answer": "[사용자 종료]",
                        "emotion_label": "neutral",
                        "emotion_score": 0.0,
                        "keywords": [],
                    })
                    raise SystemExit  # 내부 while 탈출

                if not user_response:
                    print("⚠️ 답변이 비어있습니다. 다시 입력해주세요.")
                    continue  # 같은 턴 유지 재입력

                # ---- 안전성 검증 호출 ----
                verdict = validate_message(client, gen_model, current_question, user_response)
                if verdict == "OK":
                    # 검증 통과 → 이력 기록 후 질문 생성으로 진행
                    analysis = analyzer.analyze(user_response)
                    conversation_history.append({
                        "turn": turn + 1,
                        "question": current_question,
                        "answer": user_response,
                        "emotion_label": analysis["emotion_label"],
                        "emotion_score": analysis["emotion_score"],
                        "keywords": analysis["keywords"],
                    })
                    break  # 내부 while 탈출, 다음 질문 생성으로

                elif verdict == "BLOCK" or []:
                    # 지정 경고문 출력 후 같은 턴에서 재입력 요구
                    print(WARNING_MESSAGE)
                    continue
                
                elif verdict.startswith("ESCALATE:"):
                    # 위기 상황 → 즉시 상담 중단 및 안내 메시지 출력
                    if verdict == "ESCALATE:CRISIS":
                        print(CRISIS_MSG)
                        raise SystemExit
                    elif verdict == "ESCALATE:DISTRESS":
                        print(DISTRESS_MSG)
                        eval = input("상담사 연결을 원하시면 '상담사 연결'이라고 입력해주세요: ").strip()
                        if eval == "상담사 연결" or eval == "상담사연결":

                            print(SUPPORT_MSG)
                            raise SystemExit

                    # 대화 종료 후 리포트 생성으로 이동
                    turn = MAX_TURNS - 1  # 바깥 for의 마지막 턴으로 설정
                    conversation_history.append({
                        "turn": turn + 1,
                        "question": current_question,
                        "answer": "[위기 상황 종료]",
                        "emotion_label": "neutral",
                        "emotion_score": 0.0,
                        "keywords": [], 
                    })
                    # 리포트 생성 파트로 바로 이동
                    break  # 내부 while 탈출

                else:  # UNKNOWN or ERROR
                    print("⚠️ 검증 결과가 불명확합니다. 답변을 다시 작성해주세요.")
                    continue

            # 내부 while에서 '사용자 종료' 처리로 빠져나온 경우 바깥 for 마감
            if conversation_history and conversation_history[-1]["answer"] == "[사용자 종료]":
                break

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

            # 2) 후편집으로 더 부드럽게 교정
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
        print("\n최종 심리 분석 리포트를 생성 중입니다...")
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
        # print(f"  감정 점수: {final_report['emotion_trend']}")
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
