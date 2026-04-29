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
    
    # Isim sorma
    user_name = input("Lutfen isminizi girin (orn: Berkay, Elcin): ").strip()
    if not user_name:
        user_name = "USER_OPERATOR"
    
    print(f"\nMerhaba {user_name}! Sesini sisteme tanitmak icin 5 adet kayit alacagiz.")
    print("Her kayit 5 saniye surecektir.")
    
    input("\nBaslamak icin ENTER'a basin...")
    
    # Isme ozel klasor olustur
    save_dir = os.path.join("data", "user_voice", user_name)
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = 5
    duration = 5 # saniye (Hizli kayit icin)
    
    for i in range(1, num_samples + 1):
        print(f"\n--- Kayit {i}/{num_samples} ---")
        print("Lutfen net bir sekilde konusun.")
        input("Hazir oldugunuzda ENTER'a basin...")
        
        filename = os.path.join(save_dir, f"sample_{i}.wav")
        record_audio(filename, duration)
        time.sleep(0.5)
        
    print(f"\n✅ Tum kayitlar basariyla alindi, {user_name}!")
    print(f"Dosyalar '{save_dir}' klasorune kaydedildi.")
    print("\nSimdi modeli bu yeni sesle egitmek icin 'python main.py' calistirin.")

if __name__ == "__main__":
    main()
