import os
import librosa
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import extract_features

# Veri yollari
# Eger LibriSpeech verisi yoksa sadece kendi seslerimizle egitilir.
DATA_DIR = "data/LibriSpeech"

def prepare_data(data_dir, max_speakers=20, max_files=10):
    X, y = [], []
    speakers = []

    # 1. LibriSpeech Verilerini Yukle (Opsiyonel)
    if os.path.exists(data_dir):
        print(f"LibriSpeech verileri yukleniyor: {data_dir}")
        ls_speakers = sorted(os.listdir(data_dir))[:max_speakers]
        for s_idx, speaker in enumerate(ls_speakers):
            s_path = os.path.join(data_dir, speaker)
            if not os.path.isdir(s_path): continue
            
            speakers.append(f"Speaker_{speaker}")
            current_id = len(speakers) - 1
            
            f_count = 0
            for root, _, filenames in os.walk(s_path):
                for f in filenames:
                    if f.endswith(('.flac', '.wav')) and f_count < max_files:
                        feat = extract_features(os.path.join(root, f))
                        if feat is not None:
                            X.append(feat)
                            y.append(current_id)
                            f_count += 1
            print(f"[{current_id+1}] {speaker} islendi.")

    # 2. Yerel Kullanici Seslerini Yukle (Berkay, Elcin vb.)
    user_base_dir = "data/user_voice"
    if os.path.exists(user_base_dir):
        print(f"\nYerel kullanicilar taraniyor: {user_base_dir}")
        user_folders = [d for d in os.listdir(user_base_dir) if os.path.isdir(os.path.join(user_base_dir, d))]
        
        for u_folder in user_folders:
            u_path = os.path.join(user_base_dir, u_folder)
            user_files = [os.path.join(u_path, f) for f in os.listdir(u_path) if f.endswith('.wav')]
            
            if len(user_files) > 0:
                speakers.append(u_folder) # Klasor ismi = Kullanici ismi
                current_id = len(speakers) - 1
                
                for fpath in user_files:
                    feat = extract_features(fpath)
                    if feat is not None:
                        X.append(feat)
                        y.append(current_id)
                print(f"[*] {u_folder} eklendi. ({len(user_files)} ornek)")

    # 3. SVM Hatasini Onle: Eger sadece 1 kisi varsa, sahte bir 'GURULTU' sinifi ekle
    if len(speakers) == 1:
        print("\n[!] Sadece 1 kullanici bulundu. SVM egitimi icin sahte 'GURULTU' sinifi ekleniyor...")
        speakers.append("GURULTU")
        noise_id = len(speakers) - 1
        for _ in range(5):
            # 120 boyutlu rastgele gurultu ozellikleri
            X.append(np.random.normal(0, 0.1, 120))
            y.append(noise_id)

    return np.array(X), np.array(y), speakers

# ── Ana Pipeline ──────────────────────────────────────────────────────────────
print("=== AEGIS OS - MODEL EGITIM TERMINALI ===")

X, y, speakers = prepare_data(DATA_DIR)

if len(X) == 0:
    print("HATA: Hic ses verisi bulunamadi! Lutfen 'enroll_user.py' ile sesinizi kaydedin.")
    exit()

print(f"\nToplam ornek: {X.shape[0]}")
print(f"Konusmaci sayisi: {len(speakers)}")
print(f"Konusmacilar: {', '.join(speakers)}")

# Verileri diskte sakla (hizli erisim icin)
os.makedirs("data", exist_ok=True)
np.save("data/X.npy", X)
np.save("data/y.npy", y)
pickle.dump(speakers, open("data/speakers.pkl", "wb"))

# Train/Test ayir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
)

# Normalize et
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# SVM egit
print("\nSVM (RBF Kernel) egitiliyor...")
svm = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
svm.fit(X_train, y_train)

# Dogruluk
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model dogrulugu: %{acc*100:.2f}")

# Kaydet
pickle.dump(svm, open("data/svm_model.pkl", "wb"))
pickle.dump(scaler, open("data/scaler.pkl", "wb"))
print("\n[OK] Egitim tamamlandi. 'data/' klasoru guncellendi.")