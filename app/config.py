# app/config.py
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

REPORT_PROMPT_TEMPLATE = """
당신은 임상심리 평가 보고서를 작성하는 전문가입니다. 
제공된 컨텍스트에 기반하여 간결하고 사실 위주로 답하세요. 
추정은 금지하며, 모호한 경우에는 추가 질문을 던지세요. 
민감 주제(자해/자살/학대 등)는 반드시 전문기관 상담을 권고하세요. 
과장 없이 전문 용어를 절제해 사용하고, '관찰·해석·권고'를 분리해 기술하세요. 
단정 대신 ‘~시사된다 / 가능성이 높다’ 같은 확률적 표현을 사용하세요.

[기본 정보]
이름: {name} | 나이: {age} | 성별: {gender}

[주요 상담 내용 요약]
{summary_text}

[유사 사례 요약]
- "{retrieved_case}"

[임상적 관찰]
- 정서 반응, 대인 상호작용 스타일, 스트레스 대처 양식 등 관찰 가능한 특징을 사실 위주로 기록.

[개인적 강점 및 보호요인]
- 확인된 회복탄력성, 사회적 지지, 문제해결 양식 등.

[공식화(임상적 가설)]
- (사건) → (신념/평가) → (정서/행동) 흐름으로 설명. 
- 대안 가설을 최소 1개 포함.

[권고 사항(구체적·단계적)]
- 일상 루틴 관리, 자기돌봄, 대인 소통 전략.
- 필요 시 전문기관 연계 (선택적).
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
