# face_cleaner/src/dataset_maker.py
import shutil, json, numpy as np
from pathlib import Path

# ── 수정이 필요한 기본 경로만 여기서 관리 ──────────────────────────────
DEFAULT_EMB = Path('/Users/yunsunghwang/Documents/학교/2025-1 인공지능/face_cleaner/embeddings')   # embeddings.npy 등이 있는 곳
DEFAULT_OUT = Path('/Users/yunsunghwang/Documents/학교/2025-1 인공지능/face_cleaner/dataset')      # ImageFolder 형식 결과 위치


# ─────────────────────────────────────────────────────────────────────

def main(emb_dir: Path = DEFAULT_EMB, out_dir: Path = DEFAULT_OUT):
    """
    embeddings/labels → 클래스별 폴더 복사
    Spyder·Jupyter 환경에 맞춰 최소 입력만 받도록 설계
    """
    if not emb_dir.exists():
        raise FileNotFoundError(f'🔍 {emb_dir} 경로가 없습니다.')
    
    labels = np.load(emb_dir / 'labels.npy')
    files  = json.load(open(emb_dir / 'cluster_report.json'))['files']

    for lab, src in zip(labels, files):
        if lab == -1:            # DBSCAN 노이즈 클러스터
            continue
        dst_dir = out_dir / f'class_{lab}'
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst_dir)

    print(f'✅ Done!  → {out_dir} (clusters: {len(set(labels))-("-1" in set(labels))})')

# ── Spyder/Jupyter에서 파일을 직접 실행(F5/F6)할 때 자동 동작 ──────
if __name__ == '__main__':
    print(f"[Spyder] EMB='{DEFAULT_EMB}'  →  OUT='{DEFAULT_OUT}'")
    main()