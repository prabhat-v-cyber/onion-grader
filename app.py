import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import time
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Onion Quality Grader",
    page_icon="🧅",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ================= CUSTOM CSS (premium glass/dark theme) =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 10%, #2a1a3d 0%, #1a1030 35%, #0d0818 100%);
    color: #f5f0ff;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

/* ---- Hero ---- */
.hero {
    text-align: center;
    padding: 2rem 1rem 1.2rem 1rem;
}
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #d8c9ff;
    margin-bottom: 0.9rem;
}
.hero h1 {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    margin: 0;
    background: linear-gradient(90deg, #ff9a6c, #ff6ec7 45%, #7d6cff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% auto;
    animation: shine 5s linear infinite;
}
@keyframes shine {
    to { background-position: 200% center; }
}
.hero p {
    color: #b8adcf;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* ---- Glass card ---- */
.glass-card {
    background: rgba(255,255,255,0.055);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 1.6rem;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 1.4rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.04);
    padding: 6px;
    border-radius: 14px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #cfc3ea;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ff6ec7, #7d6cff);
    color: white !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.03);
    border: 1.5px dashed rgba(255,255,255,0.25);
    border-radius: 16px;
    padding: 1rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploader"] label p {
    color: #d8cdee !important;
}
/* Make the Browse/Upload button clearly visible on dark background */
[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #ff6ec7, #7d6cff) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] button:hover {
    filter: brightness(1.1);
}
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button div {
    color: #ffffff !important;
}

/* ---- Result card ---- */
.result-card {
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.1rem;
    position: relative;
    overflow: hidden;
    animation: pop 0.5s cubic-bezier(.2,.9,.3,1.3);
    border: 1px solid rgba(255,255,255,0.12);
}
@keyframes pop {
    0% {opacity: 0; transform: scale(0.94) translateY(10px);}
    100% {opacity: 1; transform: scale(1) translateY(0);}
}
.result-good { background: linear-gradient(135deg, rgba(46,158,91,0.25), rgba(46,158,91,0.06)); }
.result-warn { background: linear-gradient(135deg, rgba(224,168,0,0.25), rgba(224,168,0,0.06)); }
.result-bad  { background: linear-gradient(135deg, rgba(214,69,80,0.28), rgba(214,69,80,0.07)); }
.result-unknown { background: linear-gradient(135deg, rgba(150,150,160,0.22), rgba(150,150,160,0.05)); }

.result-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: #ffffff;
}
.result-sub {
    color: #d8d0ea;
    font-size: 0.98rem;
    line-height: 1.4;
}
.conf-pill {
    display: inline-block;
    margin-top: 0.7rem;
    background: rgba(255,255,255,0.14);
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}

/* Probability rows */
.prob-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.88rem;
    color: #cabce6;
    margin-top: 0.7rem;
    margin-bottom: 2px;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #ff6ec7, #7d6cff) !important;
}

.footer-note {
    text-align: center;
    color: #7a6f92;
    font-size: 0.8rem;
    margin-top: 3rem;
    letter-spacing: 0.03em;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1230 0%, #140c24 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}
.side-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
}
.side-card h4 {
    font-family: 'Poppins', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.55rem 0;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.side-card p {
    color: #c3b8dc;
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 0;
}
.side-steps {
    list-style: none;
    margin: 0;
    padding: 0;
}
.side-steps li {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    font-size: 0.86rem;
    color: #c3b8dc;
    line-height: 1.5;
    margin-bottom: 0.6rem;
}
.side-steps li:last-child { margin-bottom: 0; }
.side-steps .num {
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: linear-gradient(90deg, #ff6ec7, #7d6cff);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 0.05rem;
}
.side-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: #9a8fb8;
    letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("""
    <div class="side-card">
        <h4>🧅 About this app</h4>
        <p>Powered by a trained AI model, this tool instantly classifies onions into
        <b style="color:#fff;">Grade A</b>, <b style="color:#fff;">Defective</b>, or
        <b style="color:#fff;">Sprouted</b> — helping ensure quality before they reach the market.</p>
    </div>

    <div class="side-card">
        <h4>📋 How it works</h4>
        <ul class="side-steps">
            <li><span class="num">1</span> Take a clear photo of the onion — a plain background works best</li>
            <li><span class="num">2</span> Upload it, or capture directly using your camera</li>
            <li><span class="num">3</span> Get an instant grade with a confidence score</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-tag">🇮🇳 Built for Smart India Hackathon</div>', unsafe_allow_html=True)

# ================= HERO HEADER =================
st.markdown("""
<div class="hero">
    <div class="badge">AI Powered · Real-time</div>
    <h1>🧅 Onion Quality Grader</h1>
    <p>Apne onion ki photo upload karo — AI turant batayega quality kaisi hai.</p>
</div>
""", unsafe_allow_html=True)

# ================= UPLOAD SECTION =================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📁 Upload Photo", "📷 Camera se kheencho"])

image = None
with tab1:
    uploaded_file = st.file_uploader(
        "Onion ki photo choose karo", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

with tab2:
    camera_file = st.camera_input("Photo kheencho", label_visibility="collapsed")
    if camera_file is not None:
        image = Image.open(camera_file)

st.markdown('</div>', unsafe_allow_html=True)

# ================= LOAD MODEL =================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras as legacy_keras

@st.cache_resource(show_spinner=False)
def load_model():
    model = legacy_keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [
            line.strip().split(" ", 1)[1] if " " in line.strip() else line.strip()
            for line in f.readlines()
        ]
    return model, class_names

with st.spinner("Model load ho raha hai... ⏳"):
    try:
        model, class_names = load_model()
    except Exception as e:
        st.error(f"Model file error: {e}")
        st.stop()

def preprocess_image(image: Image.Image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image.convert("RGB"))
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
    data = np.expand_dims(normalized_image_array, axis=0)
    return data

# Agar top prediction ki confidence isse kam ho, to "onion nahi lag raha" maana jayega
CONFIDENCE_THRESHOLD = 65.0

# ================= PREDICTION =================
if image is not None:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        progress_text = st.empty()
        bar = st.progress(0)
        steps = ["Image process ho rahi hai...", "Model dekh raha hai...", "Result taiyaar ho raha hai..."]
        for i, txt in enumerate(steps):
            progress_text.write(f"🔍 {txt}")
            bar.progress(int((i + 1) / len(steps) * 100))
            time.sleep(0.3)
        progress_text.empty()
        bar.empty()

        data = preprocess_image(image)
        prediction = model.predict(data, verbose=0)
        index = int(np.argmax(prediction))
        raw_class = class_names[index]
        confidence = float(prediction[0][index]) * 100

        class_key = raw_class.replace(" ", "_")
        is_recognized = confidence >= CONFIDENCE_THRESHOLD

        messages = {
            "Grade_A": ("result-good", "✅ Grade A — Badiya Onion!",
                        "Ye onion fresh aur bikri ke liye bilkul tayyar hai."),
            "Defective": ("result-bad", "❌ Defective — Kharaab Onion",
                          "Ye onion sadha hua ya kharaab lag raha hai, bikri ke layak nahi."),
            "Defective_Onion": ("result-bad", "❌ Defective — Kharaab Onion",
                                "Ye onion sadha hua ya kharaab lag raha hai, bikri ke layak nahi."),
            "Sprouted": ("result-warn", "⚠️ Sprouted — Ankurit Onion",
                         "Isme sprouting shuru ho chuki hai, jaldi use kar lo ya alag rakho."),
        }

        if is_recognized:
            css_class, title, sub = messages.get(
                class_key, ("result-warn", raw_class, "")
            )
            st.markdown(f"""
            <div class="result-card {css_class}">
                <div class="result-title">{title}</div>
                <div class="result-sub">{sub}</div>
                <div class="conf-pill">Confidence · {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-unknown">
                <div class="result-title">🚫 Not Recognized</div>
                <div class="result-sub">Ye image onion jaisi nahi lag rahi, ya photo unclear hai. Kripya ek onion ki clear, well-lit photo upload karo.</div>
                <div class="conf-pill">Best match: {raw_class} · {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 📊 Sabhi classes ki probability")
    for i, name in enumerate(class_names):
        prob = float(prediction[0][i]) * 100
        st.markdown(f'<div class="prob-row"><span>{name}</span><span>{prob:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(min(int(prob), 100))

    if not is_recognized:
        st.caption(f"ℹ️ Model ko koi bhi class {CONFIDENCE_THRESHOLD:.0f}% se zyada confidence ke saath match nahi mili, isliye result 'Not Recognized' diya gaya.")

    st.button("🔄 Doosri photo test karo", on_click=lambda: st.rerun(), use_container_width=True)

else:
    st.info("👆 Upar se ek onion ki photo upload karo ya camera se kheencho, result turant dikhega.")

# ================= FOOTER =================
st.markdown(
    '<div class="footer-note">Made with ❤️ using Streamlit + Teachable Machine · SIH Project</div>',
    unsafe_allow_html=True,
)
