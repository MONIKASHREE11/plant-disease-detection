# 🌿 Intelligent Plant Disease Detection & Expert Assistance System

> Final Year UG Project — Ramaiah University of Applied Sciences  
> **K Shruthi | Pranay V R | Monika Shree**

---

## What This Project Does

Upload a photo of a tomato leaf → the system identifies the disease → and gives you eco-friendly treatment advice in a chat interface.

**Three components working together:**
1. **ResNet-18 CNN** — classifies the disease from the leaf image (99.5% accuracy)
2. **TinyLLaMA chatbot** — gives sustainable, organic treatment advice via Ollama
3. **Streamlit web app** — ties it all together in a browser-based UI

---

## Diseases Detected

| Class | Description |
|-------|-------------|
| Bacterial Spot | Dark water-soaked spots on leaves |
| Early Blight | Concentric ring-shaped lesions |
| Healthy | No disease detected |
| Late Blight | Irregular dark brown blotches |
| Septoria Leaf Spot | Small circular spots with dark borders |
| Yellow Leaf Curl Virus | Yellowing and upward curling of leaves |

---

## Project Structure

```
plant-disease-detection/
│
├── app.py                    # Streamlit web app (main entry point)
├── model.py                  # ResNet-18 model definition
├── focal_loss.py             # Custom Focal Loss function
├── custom_dataset.py         # Dataset class with per-class augmentations
│
├── train.py                  # Train ResNet-18
├── train_efficientnet.py     # Train EfficientNet-B0 (alternate model)
│
├── evaluate.py               # Evaluate ResNet-18 + confusion matrix
├── evaluate_efficientnet.py  # Evaluate EfficientNet-B0
│
├── predict.py                # Single image prediction (command line)
│
├── data/
│   └── crop_disease_qa_dataset.json   # Q&A dataset for chatbot fine-tuning
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama
Download Ollama from [ollama.com](https://ollama.com), then pull the model:
```bash
ollama pull tinyllama
```

### 4. Download the trained model weights
Download `weed_crop_model.pth` from [Google Drive link here] and place it in the root folder.

### 5. Run the app
```bash
streamlit run app.py
```

---

## Training From Scratch

If you want to retrain the model on your own data, organize your dataset like this:
```
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
    Test/
        ...
```

Then run:
```bash
python train.py
```

---

## Results

| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| ResNet-18 (224×224) | 99.51% | 99.52% | 100% |
| EfficientNet-B0 | 99.68% | — | 100% |

---

## Tech Stack

- **PyTorch** — model training & inference
- **Streamlit** — web UI
- **Ollama (TinyLLaMA)** — LLM chatbot
- **Albumentations** — image augmentation
- **scikit-learn** — evaluation metrics

---

## Dataset

Based on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), filtered for tomato leaf classes.
