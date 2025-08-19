# tools_smoke_test_index.py
import json, os, re, sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# 0) 환경 변수 준비
load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")
if not KEY:
    print("[FAIL] OPENAI_API_KEY가 로드되지 않았습니다. src\\.env 확인 필요")
    sys.exit(1)

# 1) 파일 존재/크기 확인
idx_dir = Path("index")
vec_path = idx_dir / "vectors.npz"
meta_path = idx_dir / "meta.json"

if not vec_path.exists() or not meta_path.exists():
    print("[FAIL] index 폴더에 vectors.npz / meta.json 이 없습니다.")
    sys.exit(1)

print(f"[OK] Found: {vec_path} ({vec_path.stat().st_size} bytes)")
print(f"[OK] Found: {meta_path} ({meta_path.stat().st_size} bytes)")

# 2) 로딩 및 일관성 검사
try:
    data = np.load(vec_path)
    vectors = data["vectors"].astype("float32")
except Exception as e:
    print(f"[FAIL] vectors.npz 로딩 실패: {e}")
    sys.exit(1)

try:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    texts = payload["texts"]
    metas = payload["metas"]
except Exception as e:
    print(f"[FAIL] meta.json 로딩 실패: {e}")
    sys.exit(1)

n_vec, dim = vectors.shape
n_txt = len(texts)
n_meta = len(metas)
print(f"[INFO] vectors shape = {vectors.shape}")
print(f"[INFO] texts = {n_txt}, metas = {n_meta}")

if n_vec == 0 or n_txt == 0:
    print("[FAIL] 청크 수가 0입니다. 인덱싱 실패.")
    sys.exit(1)

if n_vec != n_txt or n_vec != n_meta:
    print("[FAIL] vectors/texts/metas 개수가 일치하지 않습니다.")
    sys.exit(1)

print("[OK] 개수 일치")

# 3) 샘플 미리보기
def preview(i):
    t = texts[i]
    m = metas[i]
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 120:
        t = t[:120] + "..."
    return f"#{i} | {m.get('source','?')}#{m.get('row','?')} | {t}"

print("[SAMPLE] 일부 청크 미리보기:")
for i in [0, n_vec//2, n_vec-1]:
    try:
        print("  " + preview(i))
    except Exception:
        pass

# 4) 실제 검색 테스트 (OpenAI 임베딩 필요)
try:
    from app.retriever import SimpleIndex
except Exception as e:
    print(f"[FAIL] app.retriever 임포트 실패: {e}")
    sys.exit(1)

try:
    si = SimpleIndex("index")
except Exception as e:
    print(f"[FAIL] SimpleIndex 로딩 실패: {e}")
    sys.exit(1)

# 질의 예시: 데이터셋에 맞게 바꾸세요.
test_queries = [
    "불안, 우울에 대한 상담 조언",
    "가족과의 갈등을 해결하는 방법",
    "스스로를 지키는 감정 조절 팁",
]

try:
    for q in test_queries:
        hits = si.search(q, top_k=5)
        print(f"\n[QUERY] {q}")
        if not hits:
            print("  [WARN] 결과 없음")
            continue
        best_sim = hits[0][2]
        # 간단 프리뷰
        for j, (txt, meta, sim) in enumerate(hits, 1):
            pv = re.sub(r"\s+", " ", txt).strip()
            if len(pv) > 80:
                pv = pv[:80] + "..."
            print(f"  [{j}] sim={sim:.3f} | {meta.get('source','?')}#{meta.get('row','?')} | {pv}")
        # 관련도 임계치
        if best_sim < 0.35:
            print("  [WARN] 최상위 sim이 낮습니다(0.35 미만). 텍스트 컬럼/언어/클린업 확인 요망.")
        else:
            print("  [OK] 관련도 양호")
except Exception as e:
    print(f"[FAIL] 검색 테스트 실패: {e}")
    sys.exit(1)

print("\n[OK] 인덱스 스모크 테스트 통과(치명 오류 없음)")
