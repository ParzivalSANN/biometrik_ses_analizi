import numpy as np
import librosa
import os
import traceback

# static_ffmpeg opsiyoneldir
try:
    from static_ffmpeg import add_paths
    add_paths()
except ImportError:
    pass


# ─── Ortak çekirdek: numpy array'den özellik çıkar ───────────────────────────
def _extract_from_array(audio_np: np.ndarray, orig_sr: int, n_mfcc: int = 40, target_sr: int = 16000) -> np.ndarray:
    """Float32 numpy array'den MFCC özellik vektörü üretir."""

    # Stereo → Mono
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    # Float32'ye çevir
    audio_np = audio_np.astype(np.float32)

    # Kırpılmış ses çok küçükse normalize et (integer PCM'den gelmiş olabilir)
    if np.abs(audio_np).max() > 10.0:
        audio_np = audio_np / np.abs(audio_np).max()

    # Sample rate normalizasyonu
    if orig_sr != target_sr:
        audio_np = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)

    # Çok kısa ses kontrolü
    if len(audio_np) < target_sr * 0.3:
        raise ValueError("Ses çok kısa (< 0.3 sn). En az 2-3 saniye konuşun.")

    # Ön işleme
    audio_np, _ = librosa.effects.trim(audio_np, top_db=20)
    audio_np = librosa.effects.preemphasis(audio_np)

    # MFCC + Delta + Delta-Delta → 120 boyutlu vektör
    mfcc   = librosa.feature.mfcc(y=audio_np, sr=target_sr, n_mfcc=n_mfcc)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.vstack([mfcc, delta, delta2]).mean(axis=1)


# ─── MIC RECORDER AUDIO DICT (streamlit_mic_recorder) ────────────────────────
def extract_features_from_mic(audio_dict: dict, n_mfcc: int = 40, target_sr: int = 16000) -> np.ndarray:
    """
    streamlit_mic_recorder'dan gelen audio dict'ini doğrudan numpy'a çevirir.
    Dosyaya yazmaz, format sorununu tamamen atlar.

    audio_dict içerikleri:
        bytes        : ham PCM baytları
        sample_rate  : Hz (ör. 44100, 48000)
        sample_width : bayt başına byte sayısı (1=8bit, 2=16bit, 4=32bit)
        channels     : kanal sayısı (1=mono, 2=stereo)
    """
    raw      = audio_dict.get("bytes", b"")
    orig_sr  = audio_dict.get("sample_rate", 44100)
    sw       = audio_dict.get("sample_width", 2)   # byte cinsinden
    channels = audio_dict.get("channels", 1)

    if not raw:
        raise ValueError("Ses verisi boş.")

    # Bytes → numpy float32
    if sw == 1:          # 8-bit unsigned
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    elif sw == 2:        # 16-bit signed  ← browser'dan genellikle bu gelir
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:        # 32-bit float VEYA int
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.max() > 2.0 or arr.min() < -2.0:   # float değilse int dene
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Desteklenmeyen sample_width: {sw} byte")

    # Stereo → Mono
    if channels > 1 and len(arr) % channels == 0:
        arr = arr.reshape(-1, channels).mean(axis=1)

    return _extract_from_array(arr, orig_sr, n_mfcc, target_sr)


# ─── DOSYADAN (eğitim için) ───────────────────────────────────────────────────
def extract_features(audio_path: str, n_mfcc: int = 40, target_sr: int = 16000) -> np.ndarray:
    """
    .wav / .flac dosyasından özellik çıkarır.
    Eğitim pipeline'ı (main.py) bu fonksiyonu kullanır.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_path}")

    try:
        # 1) soundfile ile dene
        try:
            import soundfile as sf
            audio_np, orig_sr = sf.read(audio_path, dtype="float32", always_2d=False)
        except Exception:
            # 2) librosa ile dene (audioread fallback dahil)
            audio_np, orig_sr = librosa.load(audio_path, sr=None, mono=True)

        return _extract_from_array(audio_np, orig_sr, n_mfcc, target_sr)

    except Exception as e:
        detail = str(e) if str(e) else traceback.format_exc(limit=3)
        raise RuntimeError(f"Ozellik Cikarim Hatasi: {detail}")
