import streamlit as st
import numpy as np
import os
import sys

# Ensure correct working directory for CSV loading
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Score Predictor 🏏",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# Custom CSS for a premium look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f7971e, #ffd200, #f7971e);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        letter-spacing: -0.5px;
    }

    @keyframes shimmer {
        to { background-position: 200% center; }
    }

    .sub-title {
        text-align: center;
        color: #b0a8d0;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Cards */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    .card-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffd200;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(247,151,30,0.15), rgba(255,210,0,0.10));
        border: 1px solid rgba(255,210,0,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-label {
        color: #b0a8d0;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }

    .result-score {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    .result-sub {
        color: #8b83a8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Loading */
    .loading-msg {
        text-align: center;
        color: #ffd200;
        font-size: 1.2rem;
        padding: 2rem;
    }

    /* Selectbox and inputs */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    .stSelectbox label,
    .stNumberInput label {
        color: #d0c8f0 !important;
        font-weight: 500 !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #f7971e, #ffd200) !important;
        color: #1a1a2e !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        cursor: pointer;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(247,151,30,0.35) !important;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.5rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #5a5478;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load model with caching
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Train and cache the model so it only trains once."""
    from main import train_model
    return train_model()


# Show loading UI while model trains
st.markdown('<div class="main-title">🏏 IPL Score Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered match score prediction using Deep Learning</div>', unsafe_allow_html=True)

with st.spinner("🔄 Training the neural network model... This may take a minute on first load."):
    model, scaler, label_encoders = load_model()


# ─────────────────────────────────────────────
# Team Selection Card
# ─────────────────────────────────────────────
st.markdown('<div class="glass-card"><div class="card-header">🏟️ Match Setup</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    bat_team = st.selectbox("🏏 Batting Team", label_encoders['bat_team'].classes_)
with col2:
    bowl_team = st.selectbox("⚾ Bowling Team", label_encoders['bowl_team'].classes_)

venue = st.selectbox("📍 Venue", label_encoders['venue'].classes_)

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Player Selection Card
# ─────────────────────────────────────────────
st.markdown('<div class="glass-card"><div class="card-header">👤 Player Details</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    batsman = st.selectbox("🏏 Batsman (on strike)", label_encoders['batsman'].classes_)
with col4:
    bowler = st.selectbox("🎯 Bowler", label_encoders['bowler'].classes_)

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Current Match Status Card
# ─────────────────────────────────────────────
st.markdown('<div class="glass-card"><div class="card-header">📊 Current Match Status</div>', unsafe_allow_html=True)

col5, col6, col7 = st.columns(3)
with col5:
    runs = st.number_input("Runs Scored", min_value=0, max_value=400, step=1, value=50)
with col6:
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, step=1, value=2)
with col7:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1, value=6.0)

striker = st.number_input("Striker's Current Score", min_value=0, max_value=300, step=1, value=20)

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
st.markdown("")  # spacer

if st.button("⚡ Predict Final Score"):

    # Validate inputs
    if bat_team == bowl_team:
        st.error("❌ Batting team and bowling team cannot be the same!")
    elif overs == 0.0 and runs == 0 and wickets == 0:
        st.warning("⚠️ Please enter some match data to get a meaningful prediction.")
    else:
        with st.spinner("Crunching numbers..."):
            # Encode inputs
            encoded_input = [[
                label_encoders['bat_team'].transform([bat_team])[0],
                label_encoders['bowl_team'].transform([bowl_team])[0],
                label_encoders['venue'].transform([venue])[0],
                runs,
                wickets,
                overs,
                striker,
                label_encoders['batsman'].transform([batsman])[0],
                label_encoders['bowler'].transform([bowler])[0]
            ]]

            # Scale and predict
            scaled_input = scaler.transform(encoded_input)
            prediction = model.predict(scaled_input, verbose=0)[0][0]
            predicted_score = max(int(prediction), runs)  # Score can't be less than current runs

        # Display result
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">PREDICTED FINAL SCORE</div>
            <div class="result-score">{predicted_score}</div>
            <div class="result-sub">
                {bat_team} vs {bowl_team} • {overs} overs bowled • {wickets} wickets down
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Extra insight
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


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="footer">Built with TensorFlow & Streamlit • IPL Score Prediction using Neural Networks</div>', unsafe_allow_html=True)
