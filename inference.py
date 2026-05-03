import os
import pickle
import numpy as np
import librosa
from groq import Groq
from dotenv import load_dotenv
from gtts import gTTS
import base64
import tempfile
from utils import extract_features

# .env dosyasindan anahtarlari yukle
load_dotenv()

# Groq istemcisini baslat
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model ve Scaler'i yukle
def load_models():
    if not os.path.exists("data/svm_model.pkl"):
        return None, None
    svm = pickle.load(open("data/svm_model.pkl", "rb"))
    scaler = pickle.load(open("data/scaler.pkl", "rb"))
    return svm, scaler

# Speech to Text (Groq - Whisper)
def speech_to_text(audio_file):
    if not os.path.exists(audio_file):
        return ""
    try:
        with open(audio_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3",
                language="tr"
            )
        return transcription.text
    except Exception as e:
        print(f"STT Hatası: {e}")
        return ""

# Metni LLM ile Isleme (Groq - Llama 3)
def process_with_llm(text):
    if not text or len(text.strip()) < 2:
        return "Sizi tam olarak anlayamadım Operatör, lütfen tekrar eder misiniz?"
        
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Sen AEGIS OS adlı fütüristik bir biyometrik güvenlik sisteminin yapay zekasısın. Kısa, öz ve otoriter ama yardımcı bir tonla konuş. Kullanıcıya 'Operatör' diye hitap et."},
                {"role": "user", "content": text}
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Hata: {str(e)}"

def text_to_speech(text):
    """Metni sese çevirir ve bayt formatında döner"""
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang='tr')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)
        
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        
        os.remove(temp_path)
        return audio_bytes
    except Exception as e:
        print(f"TTS Hatası: {e}")
        return None
