"""
evaluate_efficientnet.py — Evaluate EfficientNet-B0 on the test set.

Run:
    python evaluate_efficientnet.py
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────
DATA_DIR   = 'data'
BATCH_SIZE = 32
MODEL_PATH = 'best_efficientnet_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transform & Dataset ───────────────────────────────────────
weights       = EfficientNet_B0_Weights.DEFAULT
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, 'Test'), transform=weights.transforms())
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(test_dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ── Predict ───────────────────────────────────────────────────
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
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
plt.ylabel('True')
plt.title('Confusion Matrix — EfficientNet-B0')
plt.tight_layout()
plt.savefig('efficientnet_confusion_matrix.png')
plt.show()
