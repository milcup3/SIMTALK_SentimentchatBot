# SimTok SentimentCare ChatBot

감정 분석 및 멘탈 헬스 케어를 위한 AI 챗봇 프로젝트입니다.

## 🚀 빠른 시작 (Quick Start) !! 경로 설정 유의 !!

0. 저장소 클론
	```bash
	git clone https://github.com/milcup3/SimTok_SentimmentCareChatBot.git
	cd SimTok_SentimmentCareChatBot/project
	```

1) 의존성 설치 (경로 설정 유의;ai Folder)
```bash
pip install -r requirements.txt
```
2) `.env` 생성
```
OPENAI_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
GEN_MODEL=gpt-4o-mini
```
3) 인덱스 생성 (CSV 예시)
```bash
python -m app.indexer --csv ./Data/train_data.csv --text-cols Context Response   --chunk-size 900 --chunk-overlap 120
```
4) 서버폴더로 이동
```
cd server
```
5) 서버 실행
```
node server.js
```
6) 서버 실행후 아무것도 뜨지 않을 경우

```
cd ai
python main.py
```

## 📂 폴더 구조

```
project/
├── ai/
│   ├── main.py
│   ├── mini_text_400.py
│   ├── app/
│   │   ├── chatbot.py
│   │   ├── config.py
│   │   └── ...
│   ├── Data/
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── ...
│   └── ...
├── front/
│   ├── css/
│   ├── html/
│   └── js/
├── server/
│   ├── server.js
│   └── ...
└── README.md
```

## 주요 기능

- 감정 분석 기반 대화 지원
- 멘탈 헬스 관련 데이터셋 활용
- 사용자 맞춤형 상담 및 피드백 제공
- 대화 데이터 관리 및 분석

## 상세 설명

### 1. AI 챗봇 (ai/)
- `main.py`, `mini_text_400.py`: 챗봇의 핵심 로직 및 텍스트 처리
- `app/`: 챗봇, 임베딩, 인덱싱, 프롬프트, 랭킹, 리포트 등 세부 기능 모듈
- `Data/`: 멘탈 헬스 대화 데이터, 학습/테스트 데이터, 데이터 분할 스크립트 포함

### 2. 프론트엔드 (front/)
- `html/`, `css/`, `js/`: 사용자 인터페이스 및 채팅 화면 구현

### 3. 서버 (server/)
- `server.js`: Node.js 기반 API 서버
- `.env`: 환경 변수 파일

## 데이터

- `ai/Data/` 폴더에 다양한 CSV 데이터셋 포함
  - `train_data.csv`, `test_data.csv`: 챗봇 학습/테스트용
  - `amod-mental-health-counseling-conversations-data.csv`: 멘탈 헬스 대화 데이터
  - `test_split_data/`: 분할된 테스트 데이터

## 기여 방법

1. 이슈 등록 또는 Pull Request 요청
2. 코드 컨벤션 및 문서화 준수
3. 데이터 및 기능 추가 시 상세 설명 필수

## 라이선스

MIT License
실행후 터미널에서 작동하는지 확인


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

<<<<<<< HEAD
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

=======
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

<<<<<<< HEAD
---

단순 챗봇 모드(편집 및 디버깅 모드)
python main.py --mode naive
=======

왜 써야 했는지(감정 트레이싱 및 RAG를 이용한 챗봇->질문 만들 때 신뢰도를 위해서)
