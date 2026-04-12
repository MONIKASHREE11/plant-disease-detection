#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import time

# Model name (ensure it's available in Ollama)
MODEL_NAME = "llama3"

# Keywords related to crop diseases and eco-friendly solutions
KEYWORDS = [
    "blight", "fungicide", "yellowing", "rust", "powdery mildew",
    "root rot", "aphids", "bacterial wilt", "leaf spot", "mosaic virus",
    "nematodes", "stem borer", "downy mildew", "dieback", "fruit rot",
    "crop rotation", "organic treatment", "integrated pest management", "soil health",
    "Bacterial Spot", "Early Blight", "Late Blight", "Septoria Leaf Spot", "Yellow Leaf Curl Virus"
]

# Number of Q&A to generate
NUM_QA = len(KEYWORDS)

# Output file name
OUTPUT_FILE = "crop_disease_qa_dataset.json"

# Generate prompt for each keyword
def generate_prompt(i):
    keyword = KEYWORDS[i - 1]
    return (
        f"Generate a unique question and answer for farmers about identifying and naturally treating '{keyword}'. "
        f"Use simple and clear language, and only suggest eco-friendly, organic, or sustainable methods. "
        f"Avoid any mention of chemical pesticides or synthetic products."
    )

# Call Ollama locally with the generated prompt
def call_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()["response"]

# Generate the Q&A dataset
def generate_qa_dataset():
    qa_list = []
    for i in range(1, NUM_QA + 1):
        prompt = generate_prompt(i)
        print(f"🌱 Generating Q&A {i}/{NUM_QA} with keyword: {KEYWORDS[i-1]}")
        try:
            result = call_ollama(prompt)
            qa_list.append({"id": i, "keyword": KEYWORDS[i - 1], "qa": result.strip()})
            time.sleep(1)  # Optional: delay to avoid overloading the local API
        except Exception as e:
            print(f"❌ Error on Q&A {i} ({KEYWORDS[i-1]}): {e}")
    return qa_list

# Save the result to a JSON file
def save_to_file(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(data)} Q&A pairs to '{OUTPUT_FILE}'")

# Main execution
if __name__ == "__main__":
    print("🌾 Starting Q&A generation for crop disease classification (Natural Solutions)...")
    dataset = generate_qa_dataset()
    save_to_file(dataset)

