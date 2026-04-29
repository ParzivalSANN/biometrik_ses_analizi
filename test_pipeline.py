import librosa
import soundfile as sf
import os
from inference import main

def create_test_audio():
    # Librosa'nin ornek ses dosyasini yukle
    filename = librosa.ex('trumpet')
    y, sr = librosa.load(filename, duration=3)
    test_path = "test_audio.wav"
    sf.write(test_path, y, sr)
    return test_path

if __name__ == "__main__":
    if not os.path.exists("data/svm_model.pkl"):
        print("Hata: Model dosyasi bulunamadi!")
    else:
        test_file = create_test_audio()
        try:
            main(test_file)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
