"""
train_efficientnet.py — Train EfficientNet-B0 on tomato leaf disease dataset.

Dataset folder structure expected:
    data/
        Train/  ...
        Val/    ...

Run:
    python train_efficientnet.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────
DATA_DIR   = 'data'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR         = 0.001
SAVE_PATH  = 'best_efficientnet_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Transforms ────────────────────────────────────────────────
weights = EfficientNet_B0_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    weights.transforms()
])
val_transform = weights.transforms()

# ── Datasets & Loaders ────────────────────────────────────────
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'Train'), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, 'Val'),   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# ── Model ─────────────────────────────────────────────────────
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.classes))
model = model.to(device)

# ── Loss, Optimizer, Scheduler ────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Training Loop ─────────────────────────────────────────────
train_losses, val_losses         = [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # — Train —
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * inputs.size(0)
        _, preds       = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels).item()
        train_total   += labels.size(0)

    scheduler.step()
    train_loss /= train_total
    train_acc   = train_correct / train_total

    # — Validate —
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss    += loss.item() * inputs.size(0)
            _, preds     = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()
            val_total   += labels.size(0)

    val_loss /= val_total
    val_acc   = val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print("  ✅ Saved best model.")

# ── Plot ──────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies,   label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('efficientnet_training_curves.png')
plt.show()
