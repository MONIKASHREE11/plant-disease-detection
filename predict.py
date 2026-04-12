"""
predict.py — Run inference on a single image from the command line.

Run:
    python predict.py path/to/leaf_image.jpg
"""

import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import get_model

# ── Settings ──────────────────────────────────────────────────
MODEL_PATH   = 'weed_crop_model.pth'
CLASS_NAMES  = [
    'Bacterial Spot', 'Early Blight', 'Healthy',
    'Late Blight', 'Septoria Leaf Spot', 'Yellow Leaf Curl Virus'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load Model ────────────────────────────────────────────────
model = get_model(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ── Transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Predict ───────────────────────────────────────────────────
def predict_image(image_path):
    image     = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output   = model(img_tensor)
        probs    = F.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    print(f"Prediction : {CLASS_NAMES[pred_idx]}")
    print(f"Confidence : {probs[pred_idx]*100:.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(image)
    ax1.set_title(f"Predicted: {CLASS_NAMES[pred_idx]}")
    ax1.axis('off')

    colors = ['green' if i == pred_idx else 'steelblue' for i in range(len(CLASS_NAMES))]
    ax2.barh(CLASS_NAMES, probs, color=colors)
    ax2.set_xlim([0, 1])
    ax2.set_title("Confidence Scores")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict_image(sys.argv[1])
