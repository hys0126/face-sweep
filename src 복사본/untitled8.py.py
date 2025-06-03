#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from insightface.app import FaceAnalysis
import albumentations as A
import tqdm

# 🔶 1. 설정
gallery_dir = './images2'
dataset_dir = './dataset'
augmented_dir = './dataset_aug'
output_model_path = './output/finetuned_model.pt'
pretrained_model_path = '/path/to/pretrained/model.pt'

# 🔶 2. 얼굴 추출 및 임베딩
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)
embeddings, image_paths = [], []

for filename in tqdm(os.listdir(gallery_dir)):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(gallery_dir, filename)
        img = cv2.imread(img_path)
        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].embedding)
            image_paths.append(img_path)

# 🔶 3. 클러스터링 (DBSCAN)
embedding_array = np.array(embeddings)
clustering = DBSCAN(eps=0.5, min_samples=1, metric='cosine').fit(embedding_array)
labels = clustering.labels_

# 🔶 4. 증강 및 데이터셋 준비 (ImageFolder 형식)
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
])
Path(augmented_dir).mkdir(parents=True, exist_ok=True)

for label, img_p in zip(labels, image_paths):
    if label == -1: continue
    img = cv2.imread(img_p)[:,:,::-1]
    for i in range(5):
        aug_img = augment(image=img)['image']
        save_path = Path(augmented_dir) / f'class_{label}' / f'{Path(img_p).stem}_aug_{i}.jpg'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), aug_img[:,:,::-1])

for label, img_p in zip(labels, image_paths):
    if label == -1: continue
    label_dir = os.path.join(dataset_dir, f'class_{label}')
    os.makedirs(label_dir, exist_ok=True)
    shutil.copy(img_p, label_dir)

# 🔶 5. 파인튜닝 학습
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

from backbones.iresnet import iresnet50
model = iresnet50(num_features=512)
model.load_state_dict(torch.load(pretrained_model_path))
model.train()

num_classes = len(dataset.classes)
classifier = nn.Linear(512, num_classes)

class FullModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(FullModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

full_model = FullModel(model, classifier)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
full_model = full_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(full_model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for imgs, labels_batch in dataloader:
        imgs, labels_batch = imgs.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = full_model(imgs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

torch.save(full_model.state_dict(), output_model_path)
print(f'🎉 파인튜닝된 모델 저장 완료: {output_model_path}')

# 🔶 6. 갤러리에서 특정 인물 탐색
def get_target_embedding(img_path):
    img = cv2.imread(img_path)
    faces = app.get(img)
    if not faces:
        print(f"❌ 얼굴 탐지 실패: {img_path}")
        return None
    return faces[0].embedding

def search_gallery(target_emb, gallery_dir, threshold=0.5):
    for filename in os.listdir(gallery_dir):
        if filename.lower().endswith(('.jpg','.png','.jpeg')):
            img_path = os.path.join(gallery_dir, filename)
            img = cv2.imread(img_path)
            faces = app.get(img)
            if not faces:
                continue
            for face in faces:
                sim = cosine_similarity([target_emb], [face.embedding])[0][0]
                if sim > threshold:
                    print(f"⭕ 유사 얼굴 발견: {filename} | 유사도: {sim:.4f}")
                    # os.remove(img_path)  # 원하면 삭제
                else:
                    print(f"✅ 유사도 낮음: {filename} | 유사도: {sim:.4f}")

target_img_path = input("🔎 찾을 인물 사진 경로 입력: ")
target_emb = get_target_embedding(target_img_path)
if target_emb is not None:
    search_gallery(target_emb, gallery_dir, threshold=0.5)