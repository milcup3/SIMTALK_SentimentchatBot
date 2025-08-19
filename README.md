# OpenAI-only RAG Chatbot (Korean)

이 프로젝트는 OpenAI API만 사용하여 작동하는 경량 RAG 챗봇입니다.  
외부 프레임워크(예: LangChain, HF Transformers)를 사용하지 않으며, 임베딩/생성/재랭킹 모두 OpenAI API로만 수행합니다.

## 구성
```
openai_only_chatbot/
├─ app/
│  ├─ config.py             # 환경변수/설정
│  ├─ embeddings.py         # OpenAI 임베딩 (배치 지원)
│  ├─ indexer.py            # CSV → 청크 → 임베딩 → 로컬 인덱스(.npz, .json)
│  ├─ retriever.py          # 코사인 기반 검색 (NumPy)
│  ├─ ranker.py             # (선택) GPT 재랭커 / 정서-일치 가중치
│  ├─ prompting.py          # 시스템/지시 프롬프트
│  └─ chatbot.py            # naive/multiquery/HyDE 파이프라인
├─ index/                   # 인덱스 파일 저장 폴더
├─ main.py                  # CLI 챗봇
├─ webui.py                 # Streamlit UI (선택)
├─ requirements.txt
└─ .env.example
```

## 빠른 시작
1) 의존성 설치
```bash
pip install -r requirements.txt
```

2) `.env` 생성
```
OPENAI_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
GEN_MODEL=gpt-4o-mini
```
> 주의: 코드 내에 API 키를 하드코딩하지 마세요. (이번 리팩토링의 핵심 개선점)

3) 인덱스 생성 (CSV 예시)
```bash
python -m app.indexer --csv ./Data/train_data.csv --text-cols Context Response   --chunk-size 900 --chunk-overlap 120
```

4) CLI 실행
```bash
python main.py --mode naive
# 또는
python main.py --mode multiquery --mq 4
python main.py --mode hyde
```

5) (선택) 웹 UI
```bash
streamlit run webui.py
```

## 핵심 설계 원칙
- OpenAI API only: 임베딩/생성/재랭킹 전부 OpenAI API로 통일
- 토큰 상한 엄수: 질의/문맥/시스템합 1,600~2,000토큰 이내로 유지
- 간결한 프롬프트: 한국어 지시 최소화, 구조화 출력(JSON) 옵션 제공
- 로컬 인덱스: `.npz`(벡터) + `.json`(메타/텍스트)로 간단/투명
- 안전 모드: 민감 주제 감지 시 완화된 응답/상담기관 안내 (프롬프트로 제어)

## 기존 파일 대비 개선 포인트
- `config.py`에 키 하드코딩 → .env로 이전 (보안 향상)  
- LangChain/Transformers/Mistral 의존 → OpenAI 단일화  
- 감성 랭커(HF) → GPT 재랭커/정서 스코어 (선택, 비용/지연 관리)  
- 벡터DB(FAISS + LangChain) → NumPy 코사인 (의존성 최소화)

---
SCT 기반
python main.py --mode sct --sct_questions ..\sct_questions.jsonl --analysis_csv ..\Data\train.csv

단순 챗봇 모드(편집 및 디버깅 모드)
python main.py --mode naive

왜 써야 했는지(감정 트레이싱 및 RAG를 이용한 챗봇->질문 만들 때 신뢰도를 위해서)