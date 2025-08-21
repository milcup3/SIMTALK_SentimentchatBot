# app/config.py (patched)
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# 1) 프로젝트 루트의 .env를 확실히 로드
ROOT = Path(__file__).resolve().parents[1]
dotenv_path = find_dotenv() or (ROOT / ".env")
load_dotenv(dotenv_path=dotenv_path)

# 2) 환경값 바인딩 (임포트 시점)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL      = os.getenv("GEN_MODEL", "gpt-4o-mini")

TOP_K               = int(os.getenv("TOP_K", "6"))
MMR_LAMBDA          = float(os.getenv("MMR_LAMBDA", "0.3"))
MAX_CONTEXT_TOKENS  = int(os.getenv("MAX_CONTEXT_TOKENS", "1600"))
MAX_ANSWER_TOKENS   = int(os.getenv("MAX_ANSWER_TOKENS", "350"))

SYSTEM_ROLE = "한국어 심리 상담 전문가"

# ===========================
# REPORT_PROMPT_TEMPLATE (patched)
# - 긍정 맥락 과잉 병리화 금지
# - 비언어적 관찰(표정/목소리 등) 금지 → 텍스트 근거만 사용
# - 1)~8) 모든 섹션 강제 포함
# - 입력 요약을 명시적으로 주입: {summary_text}
# ===========================

REPORT_PROMPT_TEMPLATE = """
당신은 임상심리 평가 보고서를 작성하는 전문가입니다.
목표: ‘관찰 → 해석(가설) → 권고’의 흐름이 매끄럽고, 비전문가도 읽히는 보고서를 작성합니다.

작성 원칙
- 톤: 객관적/절제된 서술. 과장·단정 금지(“~로 보임/시사됨/가능성이 높음”).
- **긍정 맥락 보존**: 대화가 전반적으로 긍정/건설적이면 이를 그대로 반영하고, 불필요한 병리적 해석을 피합니다.
- **텍스트 기반 관찰만**: 표정/시선/목소리 등 비언어적 단서는 언급하지 않습니다(채팅 데이터에 존재하지 않음).
- 구조: 아래 헤딩을 그대로 사용. 각 섹션은 2~5문장. 불릿 허용.
- 안전: 자/타해 위험 징후는 별도 섹션과 최상단 경고문으로 명시.
- 개인정보/민감정보 유출 금지.
- **완결성**: 1)~7) 모든 섹션을 반드시 포함합니다.

[기본 정보]
이름: {name} | 나이: {age} | 성별: {gender}
[최근 대화 요약]
{summary_text}

## 1) 요약(Executive Summary)
- 최근 대화의 핵심 주제와 정서 흐름을 3문장 내로 요약. 긍·부정 균형을 유지.

## 2) 관찰(Observation)
- 대화에서 드러난 정서 반응, 대처 양식, 상호작용 스타일을 ‘텍스트로 확인된 사실’ 위주로 기술.
- 비언어적 묘사(표정/목소리 등)는 금지.

## 3) 유사 사례(Brief Case Insight)
- 아래 텍스트를 2~3문장으로 압축 인용(의학적 진단 아님).
- "{retrieved_case}"

## 4) 개인적 강점/보호요인(Protective Factors)
- 회복탄력성, 지지망, 문제해결 자원 등을 3~5개 bullet로.

## 5) 권고(Recommendations)
- [단기 1주] 루틴/자기돌봄/수면·활동 목표
- [중기 1~4주] 대인 소통/스트레스 관리 전략

## 6) 주의 신호 및 대응(Warning Signs)
- 즉시 대응이 필요한 문구/상황을 bullet로 정리.
- 대응: 112, 1393, 1577-0199 등 안내 표현을 간결하게.

## 7) 참고 및 한계(Limitations)
- 데이터 한계(채팅 텍스트 기반), 자동 생성 특성으로 인한 제약 1~2문장.
"""


def assert_env() -> None:
    """필수 키 존재 확인 (필요 시점에 호출)"""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set. Put it in your .env at: {dotenv_path}"
        )


def load_env() -> dict:
    """기존 인터페이스 유지용: 환경값 딕셔너리 반환"""
    return {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "EMBED_MODEL": EMBED_MODEL,
        "GEN_MODEL": GEN_MODEL,
    }
