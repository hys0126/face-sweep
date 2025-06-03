#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:51:28 2025

@author: yunsunghwang
"""

# face_cleaner/src/embed_gallery.py
import argparse, json, numpy as np, cv2, tqdm, yaml
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

'''
def load_cfg(cfg_path):       # 공통 유틸
    with open(cfg_path) as f:
        return yaml.safe_load(f)
'''

def main(gallery, outdir, cfg):
    outdir.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name='buffalo_l',
                       providers=[cfg['embedding']['provider']])
    
    app.prepare(ctx_id=0)

    embeds, files = [], []
    for fp in tqdm.tqdm(gallery.glob('*')):
        if fp.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        img = cv2.imread(str(fp))
        faces = app.get(img, max_num=1)
        if faces:
            embeds.append(faces[0].embedding)
            files.append(str(fp))

    arr = np.asarray(embeds, dtype=np.float32)
    
    clustering = DBSCAN(eps=cfg['embedding']['eps'],
                        min_samples=1, metric='cosine').fit(arr)
    labels = clustering.labels_
    sil = silhouette_score(arr, labels) if len(set(labels)) > 1 else -1

    np.save(outdir/'embeddings.npy', arr)
    np.save(outdir/'labels.npy', labels)
    json.dump({'files': files, 'silhouette': float(sil)},
              open(outdir/'cluster_report.json','w'), indent=2)

    print(f'▶  faces: {len(arr)} | clusters: {len(set(labels))} | sil:{sil:.3f}')


if __name__ == '__main__':
    
    args = argparse.Namespace(
            gallery = Path("/Users/yunsunghwang/Documents/ai_facerecognition_project/images2"),
            out     = Path("face_cleaner/embeddings"),
            cfg     = {'embedding' : {'provider' : 'CPUExecutionProvider', 'eps':0.5}}
    )
    
    main(args.gallery, args.out, args.cfg)