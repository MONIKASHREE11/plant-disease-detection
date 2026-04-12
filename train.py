"""
train.py — Train ResNet-18 on tomato leaf disease dataset.

Dataset folder structure expected:
    data/
        Train/
            Bacterial_Spot/
            Early_Blight/
            Healthy/
            Late_Blight/
            Septoria_Leaf_Spot/
            Yellow_Leaf_Curl_Virus/
        Val/
            ...

Run:
    python train.py
"""

import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import get_model
from focal_loss import FocalLoss

# ── Settings ──────────────────────────────────────────────────
DATA_DIR   = 'data'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR         = 0.001
IMG_SIZE   = 224
SAVE_PATH  = 'weed_crop_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ── Transforms ────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Datasets & Loaders ────────────────────────────────────────
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'Train'), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, 'Val'),   transform=val_transform)

print(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model, Loss, Optimizer ────────────────────────────────────
model     = get_model(num_classes=len(train_dataset.classes)).to(device)
criterion = FocalLoss(alpha=1.0, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Training Loop ─────────────────────────────────────────────
train_losses, val_losses         = [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # — Train —
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    train_loss = running_loss / total
    train_acc  = 100.0 * correct / total

    # — Validate —
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total   += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_acc  = 100.0 * val_correct / val_total

    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Saved best model (val acc: {val_acc:.2f}%)")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")

# ── Plot ──────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies,   label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
