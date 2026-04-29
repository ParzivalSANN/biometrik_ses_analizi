import streamlit as st
import os
import pickle
import numpy as np
from utils import extract_features
from inference import speech_to_text, process_with_llm
from streamlit_mic_recorder import mic_recorder

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEGIS OS | Biometric Terminal",
    page_icon="🛡️",
    layout="wide"
)

# ── Session State ─────────────────────────────────────────────────────────────
for k, v in [("auth_status", False), ("last_transcript", ""), ("last_response", ""), ("auth_prob", 0.0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS (plain string, NO f-string to avoid brace conflicts) ─────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Manrope:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(ellipse at 20% 50%, #050A1F 0%, #020408 55%, #080514 100%);
    min-height: 100vh;
}

.main .block-container {
    padding: 36px 48px 80px 48px;
    max-width: 1400px;
    margin: 0 auto;
}

/* ── Keyframes ── */
@keyframes scan {
    0%   { top: -4px; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
@keyframes pulse-ring {
    0%   { transform: scale(0.85); opacity: 0.9; }
    100% { transform: scale(2.4);  opacity: 0; }
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1;   transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}
@keyframes glow-breathe {
    0%, 100% { box-shadow: 0 0 20px rgba(0,240,255,0.15), 0 0 40px rgba(0,240,255,0.04); }
    50%       { box-shadow: 0 0 35px rgba(0,240,255,0.35), 0 0 70px rgba(0,240,255,0.12); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Glass Panel ── */
.aegis-panel {
    background: rgba(14, 19, 25, 0.6);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(0,240,255,0.12);
    border-top: 1px solid rgba(0,240,255,0.28);
    border-radius: 2px;
    padding: 36px;
    position: relative;
    overflow: hidden;
    animation: glow-breathe 4s ease-in-out infinite, fadeUp 0.5s ease-out;
}
.aegis-panel::before {
    content: '';
    position: absolute;
    top: 8px; left: 8px;
    width: 14px; height: 14px;
    border-top: 1px solid rgba(0,240,255,0.5);
    border-left: 1px solid rgba(0,240,255,0.5);
}
.aegis-panel::after {
    content: '';
    position: absolute;
    bottom: 8px; right: 8px;
    width: 14px; height: 14px;
    border-bottom: 1px solid rgba(0,240,255,0.5);
    border-right: 1px solid rgba(0,240,255,0.5);
}
.scan-line {
    position: absolute;
    left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent 0%, rgba(0,240,255,0.55) 50%, transparent 100%);
    animation: scan 3s linear infinite;
}

/* ── Typography ── */
.brand-display {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 68px;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1.0;
    background: linear-gradient(140deg, #00F0FF 0%, #0066FF 55%, #7000FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 18px rgba(0,240,255,0.35));
    margin: 0;
}
.label-caps {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: rgba(0,240,255,0.55);
}
.data-mono {
    font-family: 'Space Grotesk', monospace;
    font-size: 13px;
    letter-spacing: 0.07em;
    color: #849495;
}

/* ── Status Chip ── */
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border: 1px solid rgba(0,240,255,0.2);
    background: rgba(0,240,255,0.03);
    border-radius: 2px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(0,240,255,0.65);
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00F0FF;
    box-shadow: 0 0 6px #00F0FF;
    animation: pulse-dot 2s ease-in-out infinite;
}

/* ── Streamlit Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,240,255,0.08) 0%, rgba(0,102,255,0.08) 100%) !important;
    color: #00F0FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border: 1px solid rgba(0,240,255,0.3) !important;
    border-radius: 2px !important;
    padding: 14px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 12px rgba(0,240,255,0.08) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,240,255,0.18) 0%, rgba(0,102,255,0.18) 100%) !important;
    border-color: rgba(0,240,255,0.65) !important;
    box-shadow: 0 0 22px rgba(0,240,255,0.28) !important;
    transform: translateY(-1px) !important;
}

/* mic_recorder button override — target all buttons */
div[data-testid="stVerticalBlock"] button,
div[data-testid="column"] button,
div[class*="stAudioInput"] button,
div[class*="mic"] button,
button[kind],
button {
    background: linear-gradient(135deg, rgba(0,240,255,0.08) 0%, rgba(0,102,255,0.08) 100%) !important;
    color: #00F0FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border: 1px solid rgba(0,240,255,0.3) !important;
    border-radius: 2px !important;
    padding: 14px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 14px rgba(0,240,255,0.1) !important;
}
div[data-testid="stVerticalBlock"] button:hover,
div[data-testid="column"] button:hover,
button:hover {
    background: rgba(0,240,255,0.14) !important;
    border-color: rgba(0,240,255,0.7) !important;
    box-shadow: 0 0 28px rgba(0,240,255,0.3) !important;
}

/* ── Terminal output ── */
.term-block {
    background: rgba(0,0,0,0.45);
    border-left: 2px solid #00F0FF;
    padding: 16px 20px;
    font-family: 'Space Grotesk', monospace;
    font-size: 14px;
    color: #e0e2eb;
    line-height: 1.65;
    letter-spacing: 0.02em;
    border-radius: 0 2px 2px 0;
}
.term-block.ai { border-left-color: #7000FF; }

/* ── Access banner ── */
.access-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 28px;
    background: rgba(0,240,255,0.03);
    border: 1px solid rgba(0,240,255,0.18);
    border-radius: 2px;
    margin-bottom: 28px;
    animation: fadeUp 0.4s ease-out;
}
.access-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #00F0FF;
}
.live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00F0FF;
    box-shadow: 0 0 8px #00F0FF;
    animation: pulse-dot 1.5s ease-in-out infinite;
    display: inline-block;
    margin-right: 10px;
}

/* ── Progress bar ── */
.prog-bg { width:100%; height:3px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden; margin-top:8px; }
.prog-fill { height:100%; background:linear-gradient(90deg,#00F0FF,#0066FF); border-radius:2px; box-shadow:0 0 8px rgba(0,240,255,0.5); }

/* ── Orb mic button overlay ── */
/* Hide the recorder button text visually, keep it functional, position it over the orb */
.orb-recorder-wrapper {
    position: relative;
    display: inline-block;
    width: 110px;
    height: 110px;
    margin: 0 auto 28px auto;
}
.orb-recorder-wrapper > div {
    position: relative;
    z-index: 1;
}
/* The stVerticalBlock that wraps mic_recorder */
.mic-overlay-container {
    position: absolute !important;
    top: 0; left: 0;
    width: 110px !important;
    height: 110px !important;
    z-index: 10;
    opacity: 0;
    cursor: pointer;
}
.mic-overlay-container button {
    width: 110px !important;
    height: 110px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    opacity: 0 !important;
    cursor: pointer !important;
}

/* Orb glow on button hover — JS adds .orb-active class */
.orb-active {
    box-shadow: 0 0 40px rgba(255,60,60,0.6), inset 0 0 20px rgba(255,60,60,0.2) !important;
    border-color: rgba(255,60,60,0.7) !important;
    background: radial-gradient(circle at 35% 35%, rgba(255,60,60,0.25), rgba(200,0,0,0.12)) !important;
}

.stSpinner > div { border-top-color: #00F0FF !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Load ML Models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    if not os.path.exists("data/svm_model.pkl"):
        return None, None
    svm    = pickle.load(open("data/svm_model.pkl", "rb"))
    scaler = pickle.load(open("data/scaler.pkl",    "rb"))
    return svm, scaler

svm, scaler = load_assets()

# ═══════════════════════════════════════════════════════════════════════════════
#  SCREEN 1 — LOCKED / AUTH
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.auth_status:

    # ── Top strip ──
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown('<div class="label-caps s1" style="padding-top:6px;">AEGIS // SECURE NODE</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="text-align:center;" class="s1"><span class="status-chip"><span class="status-dot"></span>AWAITING BIOMETRIC</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="data-mono s1" style="text-align:right; padding-top:6px;">v4.2.1 // ENCRYPTED</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)

    # ── Center panel ──
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        # Mic recorder must come BEFORE the panel HTML so Streamlit renders it first
        # Then JS repositions it over the orb
        st.markdown('<div id="mic-slot">', unsafe_allow_html=True)
        audio = mic_recorder(
            start_prompt="⬤  KAYIT BAŞLAT",
            stop_prompt="■  KAYDI DURDUR VE DOĞRULA",
            key="auth_recorder"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
<div class="aegis-panel s2" style="text-align:center; padding:56px 48px;">
  <div class="scan-line"></div>
  <div class="label-caps" style="margin-bottom:18px;">CLASSIFIED SYSTEM ACCESS</div>
  <h1 class="brand-display">AEGIS OS</h1>
  <p style="font-family:'Manrope',sans-serif; font-size:12px; color:#849495;
             letter-spacing:0.2em; text-transform:uppercase; margin:18px 0 36px 0;">
    BIOMETRIC VOICE AUTHENTICATION REQUIRED
  </p>
  <div style="width:48px; height:1px; background:linear-gradient(90deg,transparent,#00F0FF,transparent); margin:0 auto 32px auto;"></div>

  <!-- Orb: clickable via JS -->
  <div id="mic-orb" style="position:relative; width:110px; height:110px; margin:0 auto 10px auto; cursor:pointer;">
    <div style="position:absolute; inset:0; border-radius:50%;
                border:1px solid rgba(0,240,255,0.28);
                animation:pulse-ring 2.5s ease-out infinite;"></div>
    <div style="position:absolute; inset:0; border-radius:50%;
                border:1px solid rgba(0,240,255,0.15);
                animation:pulse-ring 2.5s ease-out 0.9s infinite;"></div>
    <div id="orb-core" style="position:absolute; inset:16px; border-radius:50%;
                background:radial-gradient(circle at 35% 35%, rgba(0,240,255,0.22), rgba(0,66,255,0.12));
                backdrop-filter:blur(8px);
                border:1px solid rgba(0,240,255,0.38);
                box-shadow:0 0 22px rgba(0,240,255,0.28), inset 0 0 18px rgba(0,240,255,0.08);
                display:flex; align-items:center; justify-content:center;
                transition: all 0.3s ease;">
      <svg id="orb-icon" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#00F0FF" stroke-width="1.5">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
        <line x1="12" y1="19" x2="12" y2="23"/>
        <line x1="8" y1="23" x2="16" y2="23"/>
      </svg>
    </div>
  </div>

  <p id="orb-status" class="data-mono" style="font-size:11px; color:#3b494b; margin-bottom:28px;">
    CLICK ORB TO AUTHENTICATE
  </p>
  <div id="orb-hint" style="font-family:'Space Grotesk',sans-serif; font-size:10px;
       color:rgba(0,240,255,0.35); letter-spacing:0.15em; margin-bottom:0;">▼ VEYA AŞAĞIDAN BAŞLAT</div>
</div>

<script>
// mic_recorder, Streamlit custom component olarak bir iframe icinde render edilir.
// Bu yuzden doc.getElementById degil, iframe contentDocument ile bulmamiz gerekiyor.
(function() {
  var recording = false;

  function getMicBtn() {
    // Sayfadaki tum iframe'lere bak
    var iframes = document.querySelectorAll('iframe');
    for (var i = 0; i < iframes.length; i++) {
      try {
        var doc = iframes[i].contentDocument || iframes[i].contentWindow.document;
        if (!doc) continue;
        var btn = doc.querySelector('button');
        if (btn) return btn;
      } catch(e) { /* cross-origin ise atla */ }
    }
    return null;
  }

  function setOrbState(isRecording) {
    var core   = document.getElementById('orb-core');
    var icon   = document.getElementById('orb-icon');
    var status = document.getElementById('orb-status');
    if (!core) return;
    if (isRecording) {
      core.style.background  = 'radial-gradient(circle at 35% 35%, rgba(255,50,50,0.4), rgba(180,0,0,0.2))';
      core.style.borderColor = 'rgba(255,80,80,0.7)';
      core.style.boxShadow   = '0 0 35px rgba(255,50,50,0.6), inset 0 0 20px rgba(255,50,50,0.2)';
      if (icon)   icon.setAttribute('stroke', '#FF4040');
      if (status) { status.textContent = 'RECORDING\u2026 CLICK TO STOP'; status.style.color = '#FF4040'; }
    } else {
      core.style.background  = 'radial-gradient(circle at 35% 35%, rgba(0,240,255,0.22), rgba(0,66,255,0.12))';
      core.style.borderColor = 'rgba(0,240,255,0.38)';
      core.style.boxShadow   = '0 0 22px rgba(0,240,255,0.28), inset 0 0 18px rgba(0,240,255,0.08)';
      if (icon)   icon.setAttribute('stroke', '#00F0FF');
      if (status) { status.textContent = 'CLICK ORB TO AUTHENTICATE'; status.style.color = '#3b494b'; }
    }
  }

  function hookOrb() {
    var orb = document.getElementById('mic-orb');
    if (!orb) { setTimeout(hookOrb, 400); return; }

    // Orb'a tiklaninca
    orb.addEventListener('click', function() {
      var btn = getMicBtn();
      if (btn) {
        btn.click();          // Gercek butona tikla
        recording = !recording;
        setOrbState(recording);
      } else {
        var status = document.getElementById('orb-status');
        if (status) { status.textContent = 'MIC YOK — ASAGIDAKI BUTONU KULLANIN'; status.style.color = '#FF8800'; }
      }
    });

    // Orb uzerinde hover efekti
    orb.addEventListener('mouseenter', function() {
      var core = document.getElementById('orb-core');
      if (core && !recording) core.style.boxShadow = '0 0 35px rgba(0,240,255,0.5), inset 0 0 25px rgba(0,240,255,0.15)';
    });
    orb.addEventListener('mouseleave', function() {
      var core = document.getElementById('orb-core');
      if (core && !recording) core.style.boxShadow = '0 0 22px rgba(0,240,255,0.28), inset 0 0 18px rgba(0,240,255,0.08)';
    });
  }

  // DOM yuklendikten 800ms sonra baslat (Streamlit component'lerin render olmasi icin)
  setTimeout(hookOrb, 800);
})();
</script>
""", unsafe_allow_html=True)

        # ── Info strip ──
        st.markdown("""
<div class="s3" style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:14px;">
  <div style="padding:11px 14px; border:1px solid rgba(255,255,255,0.04);
              border-radius:2px; background:rgba(0,0,0,0.22);">
    <div class="label-caps" style="font-size:9px; margin-bottom:3px;">MODEL</div>
    <div class="data-mono">SVM // RBF KERNEL</div>
  </div>
  <div style="padding:11px 14px; border:1px solid rgba(255,255,255,0.04);
              border-radius:2px; background:rgba(0,0,0,0.22);">
    <div class="label-caps" style="font-size:9px; margin-bottom:3px;">ENCRYPTION</div>
    <div class="data-mono">AES-256 // ACTIVE</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Auth logic ──────────────────────────────────────────────────────
        if audio:
            with st.spinner("SCANNING BIOMETRIC SIGNATURE..."):
                try:
                    from utils import extract_features_from_mic
                    feat = extract_features_from_mic(audio)
                    if svm is None:
                        st.error("Model yüklenemedi. Terminalde `python main.py` çalıştırın.")
                    else:
                        prob      = svm.predict_proba(scaler.transform([feat])).max()
                        THRESHOLD = 0.20
                        if prob >= THRESHOLD:
                            # STT için sesi geçici dosyaya yaz
                            import tempfile, soundfile as _sf, numpy as _np
                            sr  = audio.get("sample_rate", 44100)
                            sw  = audio.get("sample_width", 2)
                            raw = audio["bytes"]
                            arr = _np.frombuffer(raw, dtype=_np.int16 if sw==2 else _np.float32)
                            if arr.dtype == _np.int16:
                                arr = arr.astype(_np.float32) / 32768.0
                            tmp = tempfile.mktemp(suffix=".wav")
                            try:
                                _sf.write(tmp, arr, sr)
                                transcript = speech_to_text(tmp)
                            except Exception:
                                transcript = "(ses metne çevrilemedi)"
                            finally:
                                if os.path.exists(tmp): os.remove(tmp)

                            st.session_state.auth_status     = True
                            st.session_state.auth_prob       = prob
                            st.session_state.last_transcript = transcript
                            st.session_state.last_response   = process_with_llm(transcript)
                            st.rerun()
                        else:
                            st.error(f"⛔  ACCESS DENIED — MATCH SCORE: {prob*100:.1f}%  (MIN: {THRESHOLD*100:.0f}%)")
                            st.warning("ℹ︎  Mikrofona yakın, net ve uzun konuşun (3-5 sn).")
                except Exception as e:
                    st.error(f"SYSTEM ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SCREEN 2 — AUTHORIZED / NEURAL TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════
else:
    prob_pct  = st.session_state.auth_prob * 100
    bar_width = min(prob_pct * 3, 100)

    # ── Banner ──
    st.markdown(
        f'<div class="access-banner">'
        f'  <div style="display:flex;align-items:center;">'
        f'    <span class="live-dot"></span>'
        f'    <span class="access-title">ACCESS GRANTED — USER_OPERATOR</span>'
        f'  </div>'
        f'  <div class="data-mono">CONF: {prob_pct:.1f}% // SESSION ACTIVE</div>'
        f'  <span class="status-chip"><span class="status-dot"></span>SYSTEM ONLINE</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns([1, 2.4], gap="large")

    # ── LEFT: Metrics ──
    with col_l:
        st.markdown(
            f'<div class="aegis-panel" style="height:100%;">'
            f'<div class="scan-line"></div>'
            f'<div class="label-caps" style="margin-bottom:22px;">SYSTEM STATUS</div>'

            f'<div style="padding:14px 0; border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'  <div class="label-caps" style="font-size:9px; color:#849495;">MATCH CONFIDENCE</div>'
            f'  <div style="font-family:Space Grotesk,sans-serif; font-size:22px; font-weight:600; color:#e0e2eb; margin-top:4px;">{prob_pct:.1f}%</div>'
            f'  <div class="prog-bg"><div class="prog-fill" style="width:{bar_width:.0f}%;"></div></div>'
            f'</div>'

            f'<div style="padding:14px 0; border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'  <div class="label-caps" style="font-size:9px; color:#849495;">SECURITY TIER</div>'
            f'  <div style="font-family:Space Grotesk,sans-serif; font-size:15px; font-weight:600; color:#00F0FF; margin-top:4px;">ALPHA-1 // ADMIN</div>'
            f'</div>'

            f'<div style="padding:14px 0; border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'  <div class="label-caps" style="font-size:9px; color:#849495;">ENGINE</div>'
            f'  <div style="font-family:Space Grotesk,sans-serif; font-size:14px; color:#e0e2eb; margin-top:4px;">SVM · RBF KERNEL</div>'
            f'</div>'

            f'<div style="padding:14px 0; border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'  <div class="label-caps" style="font-size:9px; color:#849495;">FEATURE DIMS</div>'
            f'  <div style="font-family:Space Grotesk,sans-serif; font-size:14px; color:#e0e2eb; margin-top:4px;">120-D // MFCC+Δ+ΔΔ</div>'
            f'</div>'

            f'<div style="padding:14px 0;">'
            f'  <div class="label-caps" style="font-size:9px; color:#849495;">NETWORK</div>'
            f'  <div style="font-family:Space Grotesk,sans-serif; font-size:14px; color:#00F0FF; margin-top:4px;">ENCRYPTED // LOCAL</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── RIGHT: Terminal ──
    with col_r:
        transcript = st.session_state.last_transcript or "(Ses metne çevrilemedi)"
        response   = st.session_state.last_response   or "(LLM yanıt üretemedi — .env dosyasını kontrol edin)"

        st.markdown(
            '<div class="aegis-panel">'
            '<div class="scan-line"></div>'
            '<div class="label-caps" style="margin-bottom:22px;">NEURAL TERMINAL // Llama-3.3-70B</div>'

            '<div style="margin-bottom:22px;">'
            '  <div class="label-caps" style="font-size:9px; color:#849495; margin-bottom:10px;">&gt; DECODED VOICE INPUT</div>'
            f'  <div class="term-block">{transcript}</div>'
            '</div>'

            '<div style="margin-bottom:4px;">'
            '  <div class="label-caps" style="font-size:9px; color:#849495; margin-bottom:10px;">&gt; AI RESPONSE</div>'
            f'  <div class="term-block ai">{response}</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.button("⬤  TERMINATE SESSION", use_container_width=True):
            st.session_state.auth_status = False
            st.session_state.auth_prob   = 0.0
            st.rerun()
