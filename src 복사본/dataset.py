# face_cleaner/src/dataset.py
import torch               # ➜ Tensor 변환용
from torch.utils.data import Dataset
import cv2, numpy as np, albumentations as A, os

trans = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.95,1.05), translate_percent=0.05, rotate=(-5,5), p=0.5),
    A.RandomBrightnessContrast(p=0.5)
])

class FaceFolder(Dataset):
    def __init__(self, root, transform=trans):
        self.samples = []                                   # [(path, label), …]
        for cls in sorted(os.listdir(root)):
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir):       # 잡동사니 파일 무시
                continue
            lab = int(cls.split('_')[-1])     # class_0 → 0
            for f in os.listdir(cdir):
                self.samples.append((os.path.join(cdir,f), lab))
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, lab = self.samples[idx]
        img = cv2.imread(path)[:,:,::-1]             # BGR→RGB
        img = self.transform(image=img)['image']
        img = cv2.resize(img,(112,112))
        img = (img/255.).astype('float32')           # HWC, 0-1 정규화
        img = torch.from_numpy(img).permute(2,0,1)   # CHW Tensor
        return img, lab