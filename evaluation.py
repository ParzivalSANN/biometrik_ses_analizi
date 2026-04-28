import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Veri ve modeli yukle
X = np.load("data/X.npy")
y = np.load("data/y.npy")
svm    = pickle.load(open("data/svm_model.pkl", "rb"))
scaler = pickle.load(open("data/scaler.pkl",    "rb"))

# Train/Test ayir (main.py ile ayni seed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test = scaler.transform(X_test)

# Olasilik skorlari al
probs = svm.predict_proba(X_test)

# FAR, FRR, EER hesapla
thresholds = np.linspace(0, 1, 200)
FARs, FRRs = [], []

for thresh in thresholds:
    FA = 0  # yanlis kabul
    FR = 0  # yanlis red
    total_pos = 0
    total_neg = 0

    for i in range(len(y_test)):
        true_label   = y_test[i]
        max_prob     = probs[i].max()
        pred_label   = probs[i].argmax()

        if max_prob >= thresh:
            # Sistem kabul etti
            if pred_label != true_label:
                FA += 1   # yanlis kisi kabul edildi
            total_neg += (len(np.unique(y_test)) - 1)
        else:
            # Sistem reddetti
            if pred_label == true_label:
                FR += 1   # dogru kisi reddedildi
            total_pos += 1

    FAR = FA / max(total_neg, 1)
    FRR = FR / max(total_pos, 1)
    FARs.append(FAR)
    FRRs.append(FRR)

FARs = np.array(FARs)
FRRs = np.array(FRRs)

# EER bul (FAR ve FRR en yakin oldugu nokta)
eer_idx = np.argmin(np.abs(FARs - FRRs))
EER     = (FARs[eer_idx] + FRRs[eer_idx]) / 2
print(f"EER: %{EER*100:.2f}")
print(f"FAR @ EER: %{FARs[eer_idx]*100:.2f}")
print(f"FRR @ EER: %{FRRs[eer_idx]*100:.2f}")

# DET egrisi ciz
plt.figure(figsize=(8, 6))
plt.plot(FARs * 100, FRRs * 100, color='steelblue', linewidth=2, label='SVM')
plt.scatter(FARs[eer_idx]*100, FRRs[eer_idx]*100,
            color='red', zorder=5, s=100, label=f'EER = %{EER*100:.2f}')
plt.xlabel("FAR (False Acceptance Rate) %")
plt.ylabel("FRR (False Rejection Rate) %")
plt.title("DET Curve - Speaker Recognition")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/det_curve.png", dpi=150)
plt.show()
print("DET egrisi kaydedildi: data/det_curve.png")