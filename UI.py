import streamlit as st
import numpy as np
import pandas as pd
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
    layout="centered"
)

# ─────────────────────────────────────────────
# Caching the model loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Train and cache the model so it only trains once."""
    from main import train_model
    return train_model()


# Title & Subtitle
st.title("🏏 IPL Score Predictor")
st.write("AI-powered match score prediction using Random Forest Regressor")

with st.spinner("🔄 Training the machine learning model..."):
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
            # Create a DataFrame with correct feature names to avoid MinMaxScaler warnings
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

            # Scale and predict
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            predicted_score = max(int(prediction), runs)  # Score can't be less than current runs

        # Display result
        st.success(f"🏆 Predicted Final Score: {predicted_score}")

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

# Footer
st.markdown("---")
st.markdown("<center>Built with Scikit-Learn & Streamlit • IPL Score Prediction using Random Forest Regressor</center>", unsafe_allow_html=True)
