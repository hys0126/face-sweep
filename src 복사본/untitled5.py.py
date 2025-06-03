# face_cleaner/src/eval_threshold.py
import cv2, numpy as np, tqdm, json
from pathlib import Path
from insightface.app import FaceAnalysis

VAL_DIR = Path('val')              # 위 폴더 구조
DEVICE  = 'cpu'

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

scores, labels = [], []
for lab_name in ('pos', 'neg'):
    for fp in (VAL_DIR/lab_name).glob('*'):
        if fp.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
        f = app.get(cv2.imread(str(fp)), max_num=1)
        if not f: continue
        scores.append(f[0].embedding)            # raw 벡터
        labels.append(1 if lab_name=='pos' else 0)
scores = np.vstack(scores)                       # (N,512)
labels = np.array(labels)                        # (N,)