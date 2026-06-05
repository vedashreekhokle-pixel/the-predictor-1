import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8e6f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem; max-width: 720px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1d27 0%, #12141e 100%);
    border: 1px solid #2a2d3e;
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    color: #a78bfa;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(124,58,237,0.3);
    margin-bottom: 0.8rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #f0eeff;
    margin: 0 0 0.5rem;
    line-height: 1.2;
}
.hero p {
    font-size: 0.95rem;
    color: #9490b0;
    margin: 0;
    font-weight: 300;
}

/* ── Section card ── */
.section-card {
    background: #13151f;
    border: 1px solid #1f2235;
    border-radius: 16px;
    padding: 1.8rem 1.8rem 1rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6c69a0;
    margin-bottom: 1.2rem;
}

/* ── Streamlit widget overrides ── */
label { color: #c4c0e0 !important; font-size: 0.88rem !important; font-weight: 500 !important; }

div[data-baseweb="input"] > div,
div[data-baseweb="slider"] {
    background: #1a1d2a !important;
    border-color: #2a2d42 !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 0 !important;
    margin-top: 0.5rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.4rem;
    border: 1px solid;
    animation: fadeUp 0.4s ease both;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-excellent { background:#0f1f12; border-color:#22c55e; }
.result-good      { background:#0f1a25; border-color:#3b82f6; }
.result-average   { background:#1a1a0f; border-color:#eab308; }
.result-low       { background:#1f0f0f; border-color:#ef4444; }

.result-grade {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.result-label {
    font-size: 1.05rem;
    font-weight: 500;
    margin-bottom: 0.2rem;
}
.result-sub {
    font-size: 0.82rem;
    opacity: 0.6;
    font-weight: 300;
}

/* ── Divider ── */
hr { border-color: #1f2235 !important; margin: 1.4rem 0 !important; }

/* ── Metric row ── */
.metric-row {
    display: flex; gap: 12px; margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: #1a1d2a;
    border: 1px solid #2a2d42;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-lbl {
    font-size: 0.72rem;
    color: #6c69a0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">ML · Education Analytics</div>
  <h1>Student Performance<br>Predictor</h1>
  <p>Predict final exam grades using study habits &amp; past performance — powered by Random Forest.</p>
</div>
""", unsafe_allow_html=True)

# ── Load & train ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    data = pd.read_csv("student_data.csv")
    X = data[["age", "studytime", "G1", "G2"]]
    y = data["G3"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model, len(data)

model, dataset_size = load_and_train()

# ── Stats row ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="metric-val">{dataset_size}</div><div class="metric-lbl">Students</div></div>
  <div class="metric-box"><div class="metric-val">RF</div><div class="metric-lbl">Algorithm</div></div>
  <div class="metric-box"><div class="metric-val">4</div><div class="metric-lbl">Features</div></div>
  <div class="metric-box"><div class="metric-val">0–20</div><div class="metric-lbl">Grade Scale</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Student Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=10, max_value=25, value=16)
    G1  = st.number_input("G1 — First Period Grade (0–20)", min_value=0, max_value=20, value=10)
with col2:
    studytime = st.select_slider(
        "Weekly Study Time",
        options=[1, 2, 3, 4],
        value=2,
        format_func=lambda x: {1:"<2 hrs", 2:"2–5 hrs", 3:"5–10 hrs", 4:">10 hrs"}[x]
    )
    G2 = st.number_input("G2 — Second Period Grade (0–20)", min_value=0, max_value=20, value=10)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("✦ Predict Final Grade"):
    new_student = pd.DataFrame({
        "age": [age], "studytime": [studytime],
        "G1": [G1], "G2": [G2]
    })
    predicted_G3 = model.predict(new_student)[0]

    if predicted_G3 >= 16:
        css_cls = "result-excellent"
        emoji   = "🌟"
        label   = "Excellent Performance"
        sub     = "Outstanding result — keep up the great work!"
        color   = "#22c55e"
    elif predicted_G3 >= 12:
        css_cls = "result-good"
        emoji   = "👍"
        label   = "Good Performance"
        sub     = "Solid work — a little more effort reaches excellence."
        color   = "#3b82f6"
    elif predicted_G3 >= 8:
        css_cls = "result-average"
        emoji   = "📊"
        label   = "Average Performance"
        sub     = "There's room to grow — try increasing study time."
        color   = "#eab308"
    else:
        css_cls = "result-low"
        emoji   = "📚"
        label   = "Needs Improvement"
        sub     = "Don't give up — consistent study makes a big difference."
        color   = "#ef4444"

    st.markdown(f"""
    <div class="result-card {css_cls}">
      <div class="result-grade" style="color:{color};">{predicted_G3:.1f} <span style="font-size:1.5rem">{emoji}</span></div>
      <div class="result-label" style="color:{color};">{label}</div>
      <div class="result-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)
