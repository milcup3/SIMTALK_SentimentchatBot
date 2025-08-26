# 🧠 SimTok: SentimentCare ChatBot

AI 기반 감정 분석 및 멘탈 헬스케어 챗봇 프로젝트입니다.  
사용자의 대화를 분석해 감정 변화, 핵심 키워드, 맞춤형 조언을 제공하며, 위험 신호가 감지되면 전문 기관 안내를 권고합니다.  

---

## 🚀 프로젝트 개요
- **목표:** 정신적 장벽을 낮추고, 1차 상담/검증 역할을 수행할 수 있는 AI 챗봇 개발
- **핵심 기능:**
  1. SCT 기반 구조적 대화 (심리검사 응용 질문)
  2. RAG 기반 신뢰도 높은 답변 (상담 사례 검색)
  3. 감정 분석 (KoELECTRA) 및 키워드 추출
  4. 자동 리포트 생성 (대화 요약, 감정 추이, 맞춤형 조언)
  5. 위험 신호 탐지 → 전문 기관 안내

---

## 🛠️ 기술 스택
- **Frontend:** HTML, CSS, JS (Chat UI)
- **Backend:** FastAPI (Python), Node.js (서버 라우팅)
- **AI/NLP:**
  - OpenAI API (GPT 기반 RAG)
  - KoELECTRA 감정 분석 모델
  - FAISS/NumPy 기반 벡터 검색
- **Infra:** 환경 변수 관리(.env), 모듈화된 폴더 구조

---

## 📂 프로젝트 구조
```bash
project/
├── ai/ # 핵심 AI 로직
│ ├── main.py
│ ├── app/
│ │ ├── chatbot.py # RAG + 감정 분석
│ │ ├── retriever.py # 검색 모듈
│ │ ├── ranker.py # 감정 기반 재랭킹
│ │ └── ...
│ ├── Data/ # 상담 데이터셋
├── front/ # 사용자 인터페이스
│ ├── html/
│ ├── css/
│ └── js/
├── server/ # Node.js API 서버
│ └── server.js
└── README.md
```
---

## ⚡ 실행 방법
### 1) 환경 설정
```bash
git clone https://github.com/<username>/SimTok_SentimentCareChatBot.git
cd SimTok_SentimentCareChatBot/project
pip install -r requirements.txt
```
.env 파일 생성
OPENAI_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
GEN_MODEL=gpt-4o-mini

2) 인덱스 생성 (예시)
python -m app.indexer --csv ./Data/train_data.csv --text-cols Context Response \
--chunk-size 900 --chunk-overlap 120

3) 서버 실행
cd server
node server.js

브라우저에서 http://localhost:3000
 접속.


## 📊 성과 및 결과물

대화 리포트 자동 생성 (요약·감정 추이·조언 포함)

RAG 검색 정확도 향상 (LLM 단독 대비 일관성 개선)

파일럿 테스트 (n=)

상담 경험 만족도: % 긍정

“실제 상담 대비 초기 접근성 우수”라는 피드백 확보

## 🎥 데모

link https...

## 🔮 향후 개선 방향

사용자 집단 확장(대학생 → 직장인, 시니어)

다국어 상담 지원 (영어/일본어)

리포트 시각화 고도화 (키워드 클라우드, 감정 변화 그래프)

## 👤 팀

김동현: 아이디어 기획, 백엔드/AI 모듈 구현, 데이터 파이프라인, 보고서·PPT 작성

김현서: 프론트엔드 협업(UI), 발표 지원

## License
MIT License
