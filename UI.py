import streamlit as st
import numpy as np
import pandas as pd
import os
import sys

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Score Predictor 🏏",
    page_icon="🏏",
    layout="centered"
)

# ─────────────────────────────────────────────
# Custom Dark Theme CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* ── Global Background & Font ── */
.stApp {
    background: linear-gradient(160deg, #0a0e1a 0%, #101829 40%, #0d1522 100%) !important;
    font-family: 'Poppins', sans-serif !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}
html, body, p, span, label, div {
    font-family: 'Poppins', sans-serif !important;
}

/* ── Title ── */
h1 {
    text-align: center !important;
    font-weight: 800 !important;
    font-size: 2.6rem !important;
    background: linear-gradient(135deg, #fbbf24 0%, #f97316 50%, #ef4444 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    padding-bottom: 10px !important;
    letter-spacing: -0.5px !important;
}

/* ── Section Headers ── */
h2 {
    color: #fbbf24 !important;
    font-weight: 700 !important;
    font-size: 1.35rem !important;
    border-bottom: 2px solid rgba(251, 191, 36, 0.15) !important;
    padding-bottom: 8px !important;
    margin-top: 35px !important;
}

/* ── All text white ── */
p, span, label, .stMarkdown, [data-testid="stText"] {
    color: #e2e8f0 !important;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(251, 191, 36, 0.2) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    transition: border-color 0.3s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(251, 191, 36, 0.5) !important;
}
[data-testid="stSelectbox"] label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* ── Number inputs ── */
[data-testid="stNumberInput"] > div > div > input {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(251, 191, 36, 0.2) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    transition: border-color 0.3s ease !important;
}
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #fbbf24 !important;
    box-shadow: 0 0 12px rgba(251, 191, 36, 0.15) !important;
}
[data-testid="stNumberInput"] label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 14px 40px !important;
    width: 100% !important;
    box-shadow: 0 6px 25px rgba(245, 158, 11, 0.3) !important;
    transition: all 0.35s ease !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 35px rgba(245, 158, 11, 0.45) !important;
    background: linear-gradient(135deg, #fbbf24 0%, #f97316 100%) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Success message (prediction result) ── */
[data-testid="stAlert"] {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(239, 68, 68, 0.08) 100%) !important;
    border: 1px solid rgba(251, 191, 36, 0.25) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    backdrop-filter: blur(8px) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(251, 191, 36, 0.12) !important;
    border-radius: 12px !important;
    padding: 18px !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(251, 191, 36, 0.35) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(251, 191, 36, 0.1) !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    color: #fbbf24 !important;
    font-weight: 700 !important;
}

/* ── Horizontal rule ── */
hr {
    border-color: rgba(251, 191, 36, 0.1) !important;
    margin: 25px 0 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #fbbf24 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: rgba(251, 191, 36, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(251, 191, 36, 0.5); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Caching the model loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model instantly if cached, otherwise train it."""
    from main import get_or_train_model
    return get_or_train_model()


# Title
st.title("🏏 IPL Score Predictor")

with st.spinner("🔄 Loading model..."):
    model, scaler, label_encoders = load_model()

# ─────────────────────────────────────────────
# Match Setup
# ─────────────────────────────────────────────
st.header("🏟️ Match Setup")

col1, col2 = st.columns(2)
with col1:
    bat_team = st.selectbox("🏏 Batting Team", label_encoders['bat_team'].classes_)
with col2:
    default_bowl_idx = min(1, len(label_encoders['bowl_team'].classes_) - 1)
    bowl_team = st.selectbox("⚾ Bowling Team", label_encoders['bowl_team'].classes_, index=default_bowl_idx)

venue = st.selectbox("📍 Venue", label_encoders['venue'].classes_)

# ─────────────────────────────────────────────
# Player Details
# ─────────────────────────────────────────────
st.header("👤 Player Details")

col3, col4 = st.columns(2)
with col3:
    batsman = st.selectbox("🏏 Batsman (on strike)", label_encoders['batsman'].classes_)
with col4:
    default_bowler_idx = min(1, len(label_encoders['bowler'].classes_) - 1)
    bowler = st.selectbox("🎯 Bowler", label_encoders['bowler'].classes_, index=default_bowler_idx)

# ─────────────────────────────────────────────
# Match Status
# ─────────────────────────────────────────────
st.header("📊 Current Match Status")

col5, col6, col7 = st.columns(3)
with col5:
    runs = st.number_input("Runs Scored", min_value=0, max_value=400, step=1, value=50)
with col6:
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, step=1, value=2)
with col7:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1, value=6.0)

striker = st.number_input("Striker's Current Score", min_value=0, max_value=300, step=1, value=20)

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
st.markdown("---")

if st.button("⚡ Predict Final Score"):

    # Validate inputs
    if bat_team == bowl_team:
        st.error("❌ Batting team and bowling team cannot be the same!")
    elif overs == 0.0 and runs == 0 and wickets == 0:
        st.warning("⚠️ Please enter some match data to get a meaningful prediction.")
    else:
        with st.spinner("Crunching numbers..."):
            input_df = pd.DataFrame([[
                label_encoders['bat_team'].transform([bat_team])[0],
                label_encoders['bowl_team'].transform([bowl_team])[0],
                label_encoders['venue'].transform([venue])[0],
                runs,
                wickets,
                overs,
                striker,
                label_encoders['batsman'].transform([batsman])[0],
                label_encoders['bowler'].transform([bowler])[0]
            ]], columns=[
                "bat_team", "bowl_team", "venue",
                "runs", "wickets", "overs",
                "striker", "batsman", "bowler"
            ])

            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            predicted_score = max(int(prediction), runs)

        st.success(f"🏆 Predicted Final Score: {predicted_score}")

        if overs > 0:
            current_rr = runs / overs
            projected_rr = predicted_score / 20.0
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Current Run Rate", f"{current_rr:.2f}")
            with col_b:
                st.metric("Projected Run Rate", f"{projected_rr:.2f}")
            with col_c:
                remaining = predicted_score - runs
                st.metric("Runs Remaining", f"{remaining}")
