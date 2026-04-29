import os
import pickle
import numpy as np
import librosa
from groq import Groq
from dotenv import load_dotenv

# .env dosyasindan anahtarlari yukle
load_dotenv()

# Groq istemcisini baslat
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model ve Scaler'i yukle
def load_models():
    svm = pickle.load(open("data/svm_model.pkl", "rb"))
    scaler = pickle.load(open("data/scaler.pkl", "rb"))
    return svm, scaler

# MFCC cikarimi (main.py ile ayni parametreler)
def extract_mfcc(file_path, n_mfcc=40, sr=16000):
    try:
        # Sesi yukle ve on islemler
        audio, _ = librosa.load(file_path, sr=sr, duration=5.0)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        audio = librosa.effects.preemphasis(audio)
        
        mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Mean
        return np.vstack([mfcc, delta, delta2]).mean(axis=1)
    except Exception as e:
        print(f"Hata: {e}")
        return None

# Biyometrik Dogrulama
def verify_speaker(file_path, svm, scaler):
    feat = extract_mfcc(file_path)
    if feat is None:
        return None, 0
    
    feat_scaled = scaler.transform([feat])
    prediction = svm.predict(feat_scaled)[0]
    probability = svm.predict_proba(feat_scaled).max()
    
    return prediction, probability

# Whisper ile Sesi Metne Donusturme (Groq)
def speech_to_text(file_path):
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, file.read()),
            model="whisper-large-v3",
            response_format="text",
            language="tr"
        )
    return transcription

# Metni LLM ile Isleme (Groq - Llama 3)
def process_with_llm(text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Sen biyometrik ses dogrulama sistemiyle entegre calisan bir asistansin. Kullanicinin sesinden donusturulen metni analiz et ve kisa, oz cevaplar ver."
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def main(audio_file):
    print(f"\n--- Analiz Basliyor: {audio_file} ---")
    
    svm, scaler = load_models()
    
    # 1. Biyometrik Kontrol
    speaker_id, prob = verify_speaker(audio_file, svm, scaler)
    
    # Esik degeri (EER %0.11 oldugu icin yuksek tutabiliriz)
    THRESHOLD = 0.70 
    
    if prob >= THRESHOLD:
        print(f"✅ Kimlik Dogrulandi: Konusmaci ID {speaker_id} (Guven: %{prob*100:.2f})")
        
        # 2. Whisper ile Metne Donustur
        print("🎙️ Ses metne donusturuluyor...")
        text = speech_to_text(audio_file)
        print(f"📝 Metin: {text}")
        
        # 3. LLM ile Cevap Uret
        print("🤖 Yapay zeka isliyor...")
        response = process_with_llm(text)
        print(f"💬 Yanit: {response}")
        
    else:
        print(f"❌ Kimlik Dogrulanamadi! (Guven: %{prob*100:.2f})")
        print("Uyari: Yetkisiz erisim denemesi veya dusuk ses kalitesi.")

if __name__ == "__main__":
    # Test etmek icin bir ses dosyasi yolu girmelisin
    # Ornek: main("test_audio.wav")
    print("Lutfen bir ses dosyasi ile calistirin.")
