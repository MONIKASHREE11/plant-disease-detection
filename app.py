"""
app.py — Streamlit web app for tomato disease detection + expert chatbot.

Run:
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import ollama
from deep_translator import GoogleTranslator
from gtts import gTTS
import io

from model import get_model

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

# ── Model Setup ───────────────────────────────────────────────
MODEL_PATH  = 'weed_crop_model.pth'
CLASS_NAMES = [
    'Bacterial Spot', 'Early Blight', 'Healthy',
    'Late Blight', 'Septoria Leaf Spot', 'Yellow Leaf Curl Virus'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    m = get_model(num_classes=len(CLASS_NAMES))
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.to(device)
    m.eval()
    return m

model = load_model()

# ── Transform ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Chatbot Helper ────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a plant pathology expert. Provide accurate, eco-friendly, and sustainable "
    "preventive and cure measures when asked for plant disease advice."
)

def get_chatbot_response(disease_name):
    response = ollama.chat(
        model="tinyllama",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"What are the treatment and prevention methods for {disease_name} in tomato plants?"}
        ]
    )
    return response['message']['content']

# ── Multilingual Translation + TTS ────────────────────────────
LANGUAGES = {
    "English":   ("en", "en"),
    "Hindi":     ("hi", "hi"),
    "Tamil":     ("ta", "ta"),
    "Kannada":   ("kn", "kn"),
    "Malayalam": ("ml", "ml"),
}

def translate_text(text, target_lang_code):
    if target_lang_code == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang_code).translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        return None

# ── Main UI ───────────────────────────────────────────────────
st.title("🌿 Tomato Disease Detection & Expert Chatbot")
st.write("Upload an image of a tomato leaf to detect the disease and get expert eco-friendly advice.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output    = model(img_tensor)
        probs     = F.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx  = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]

    st.markdown(f"### 🔍 Prediction: **{pred_class}** ({probs[pred_idx]*100:.2f}% confidence)")

    fig, ax = plt.subplots()
    colors = ['green' if i == pred_idx else 'skyblue' for i in range(len(CLASS_NAMES))]
    ax.barh(CLASS_NAMES, probs, color=colors)
    ax.set_xlim([0, 1])
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

    st.markdown("### 💡 Expert Advice")
    with st.spinner("Consulting plant pathology expert..."):
        advice = get_chatbot_response(pred_class)
        st.write(advice)

    # ── Multilingual Tabs ─────────────────────────────────────
    st.markdown("### 🌐 Multilingual Advice")
    st.write("Read and listen to the expert advice in your preferred language.")

    tabs = st.tabs(list(LANGUAGES.keys()))

    for tab, (lang_name, (translate_code, tts_code)) in zip(tabs, LANGUAGES.items()):
        with tab:
            with st.spinner(f"Translating to {lang_name}..."):
                translated = translate_text(advice, translate_code)
            st.write(translated)
            if st.button(f"🔊 Read Aloud in {lang_name}", key=f"tts_{lang_name}"):
                audio = text_to_speech(translated, tts_code)
                if audio:
                    st.audio(audio, format="audio/mp3")
                else:
                    st.error("Audio generation failed. Please try again.")

# ── Sidebar Chatbot ───────────────────────────────────────────
with st.sidebar.expander("🧠 Chat with Plant Expert", expanded=False):
    st.markdown("#### 🌿 Plant Health Chatbot")
    st.markdown("_Ask me about disease cures, natural manures, or crop care tips._")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! 🌱 I'm your plant health assistant. How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me anything about your plants...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ollama.chat(
                        model="tinyllama",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            *st.session_state.messages
                        ]
                    )
                    reply = response["message"]["content"]
                except Exception as e:
                    reply = "Sorry, I couldn't generate a response. Make sure Ollama is running."
                    print(f"Ollama error: {e}")

            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
