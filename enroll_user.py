import os
import sounddevice as sd
import soundfile as sf
import time

def record_audio(filename, duration, fs=16000):
    print(f"\n[{duration} saniye boyunca konusun...]")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Kaydin bitmesini bekle
    sf.write(filename, myrecording, fs)
    print(f"✅ Kaydedildi: {filename}")

def main():
    print("========================================")
    print(" AEGIS OS - KULLANICI SES KAYIT SISTEMI")
    print("========================================")
    print("Bu arac, sesinizi sisteme tanitmak icin 5 adet kayit alacaktir.")
    print("Her kayit 10 saniye surecektir.")
    print("Lutfen mikrofonunuzun calistigindan emin olun.")
    
    input("\nBaslamak icin ENTER'a basin...")
    
    save_dir = "data/user_voice"
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = 5
    duration = 10 # saniye
    
    for i in range(1, num_samples + 1):
        print(f"\n--- Kayit {i}/{num_samples} ---")
        print("Lutfen net bir sekilde bir metin okuyun (orn: bir kitap sayfasi veya rastgele kelimeler).")
        input("Hazir oldugunuzda ENTER'a basin...")
        
        filename = os.path.join(save_dir, f"user_sample_{i}.wav")
        record_audio(filename, duration)
        time.sleep(1)
        
    print("\n✅ Tum kayitlar basariyla alindi!")
    print(f"Dosyalar '{save_dir}' klasorune kaydedildi.")
    print("\nSimdi modeli egitmek icin lutfen 'main.py' dosyasini calistirin.")

if __name__ == "__main__":
    main()
