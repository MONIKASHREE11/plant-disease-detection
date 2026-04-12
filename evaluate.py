"""
evaluate.py — Evaluate the trained ResNet-18 model on the test set.

Dataset folder structure expected:
    data/
        Test/  ...

Run:
    python evaluate.py
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model import get_model

# ── Settings ──────────────────────────────────────────────────
DATA_DIR   = 'data'
BATCH_SIZE = 32
MODEL_PATH = 'weed_crop_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load Test Data ────────────────────────────────────────────
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'Test'), transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── Load Model ────────────────────────────────────────────────
model = get_model(num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ── Predict ───────────────────────────────────────────────────
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ── Report ────────────────────────────────────────────────────
print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# ── Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes,
            cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix — ResNet-18')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
