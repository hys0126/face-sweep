# face_cleaner/src/train_head.py
# ――― Spyder / Jupyter 친화 버전 ――――――――――――――――――――――――――――――――
import sys
from pathlib import Path

# 현재 파일의 경로를 기준으로 face_cleaner 루트를 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backbones.iresnet import iresnet50  # 이제 정상 임포트 가능
import yaml, torch, tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import FaceFolder, trans              # FaceFolder 는 ImageFolder 래퍼



# ── ❶ 기본 경로 & 하이퍼파라미터 ------------------------------------
DEFAULT_DATA = Path('/Users/yunsunghwang/Documents/학교/2025-1 인공지능/face_cleaner/dataset')
DEFAULT_OUT  = Path('/Users/yunsunghwang/Documents/학교/2025-1 인공지능/face_cleaner/checkpoints')
CFG = {                                            # 필요하면 바로 고치세요
    'train': {
        'batch' : 32,
        'lr'    : 3e-4,
        'epochs': 8
    },
    'device': 'cpu'                               # 'cuda:0' 가능
}
# ---------------------------------------------------------------------

def main(data_dir: Path, out_dir: Path, cfg: dict = CFG):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 데이터
    ds = FaceFolder(data_dir, transform=trans)    # dataset.py 에 정의
    dl = DataLoader(ds,
                    batch_size=cfg['train']['batch'],
                    shuffle=True,
                    num_workers=2)

    # 2) 백본(동결) + 새 헤드
    backbone = iresnet50(num_features=512).to(cfg['device'])
    for p in backbone.parameters():
        p.requires_grad = False                   # feature extractor freeze

    n_cls = len({l for _, l in ds.samples})
    head  = torch.nn.Linear(512, n_cls).to(cfg['device'])

    # 3) 옵티마이저 / 손실
    opt = torch.optim.AdamW(head.parameters(), lr=cfg['train']['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    # 4) 학습 루프
    backbone.eval(); head.train()
    for ep in range(1, cfg['train']['epochs'] + 1):
        total, hit = 0, 0
        for img, lbl in tqdm.tqdm(dl, desc=f'Epoch {ep}'):
            img, lbl = img.to(cfg['device']), lbl.to(cfg['device'])
            with torch.no_grad():
                feat = backbone(img)
            logit = head(feat)
            loss  = criterion(logit, lbl)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += lbl.size(0)
            hit   += (logit.argmax(1) == lbl).sum().item()

        acc = hit / total
        print(f'  ↳ loss: {loss.item():.4f} | acc: {acc:.3%}')

    # 5) 저장
    torch.save(head.state_dict(), out_dir / f'head_ep{ep}.pth')
    print(f'✅ 학습 완료 • 저장: {out_dir}/head_ep{ep}.pth')

# ―― 스파이더에서 직접 실행될 때 자동 호출 ――――――――――――――――――――――――
if __name__ == '__main__':
    print(f'[Spyder] DATA={DEFAULT_DATA}\n        OUT ={DEFAULT_OUT}')
    main(DEFAULT_DATA, DEFAULT_OUT)