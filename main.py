#MFCC
import os
import librosa
import numpy as np

DATA_DIR = r"C:\Users\Elcin Erdemir\Desktop\speaker_recognition_project\data\dev-clean\LibriSpeech\dev-clean"

def extract_mfcc(file_path, n_mfcc=40, sr=16000):
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=5.0)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        if len(audio) < sr * 0.5:
            return None
        mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, delta, delta2]).mean(axis=1)
    except Exception as e:
        print(f"Hata: {file_path} -> {e}")
        return None

def load_dataset(data_dir, max_speakers=20, max_files=50):
    X, y = [], []
    speakers = sorted(os.listdir(data_dir))[:max_speakers]

    for speaker_id, speaker in enumerate(speakers):
        speaker_path = os.path.join(data_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue

        files = []
        for root, _, filenames in os.walk(speaker_path):
            for f in filenames:
                if f.endswith('.flac') or f.endswith('.wav'):
                    files.append(os.path.join(root, f))

        for fpath in files[:max_files]:
            feat = extract_mfcc(fpath)
            if feat is not None:
                X.append(feat)
                y.append(speaker_id)

        print(f"[{speaker_id+1}/{len(speakers)}] {speaker} islendi.")

    return np.array(X), np.array(y), speakers

print("Veri yukleniyor...")
X, y, speakers = load_dataset(DATA_DIR)
print(f"\nToplam ornek: {X.shape[0]}")
print(f"Konusmaci sayisi: {len(speakers)}")
print(f"Ozellik vektoru boyutu: {X.shape[1]}")

np.save("data/X.npy", X)
np.save("data/y.npy", y)
print("\nX.npy ve y.npy kaydedildi!")





#SVM Modeli
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Veriyi yukle
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Train/Test ayir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize et
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# SVM egit
print("\nSVM egitiliyor...")
svm = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
svm.fit(X_train, y_train)

# Dogruluk
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test dogrulugu: %{acc*100:.2f}")

# Modeli kaydet
pickle.dump(svm, open("data/svm_model.pkl", "wb"))
pickle.dump(scaler, open("data/scaler.pkl", "wb"))
print("Model kaydedildi: svm_model.pkl")