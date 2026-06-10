import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Get absolute path to the CSV relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'ipl_data.csv')


def train_model():
    """Train the IPL score prediction model (Random Forest) and return model, scaler, label_encoders."""

    ipl = pd.read_csv(CSV_PATH)
    data = ipl.copy()

    # --- Exploratory Data Analysis (skipped in Streamlit mode) ---
    if __name__ == "__main__":
        matches_per_venue = data[['mid', 'venue']].drop_duplicates().groupby('venue').count().reset_index()
        matches_count = matches_per_venue['venue'].value_counts()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=matches_count.values, y=matches_count.index, palette="rainbow")
        plt.title('Number of Matches per Venue')
        plt.xlabel('Number of Matches')
        plt.ylabel('Venue')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'venue_plot.png'))
        plt.close()

    # --- Encoding ---
    cat_cols = ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler']
    data_encoded = data.copy()

    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        label_encoders[col] = le

    # --- Feature Selection & Split ---
    feature_cols = [
        "bat_team", "bowl_team", "venue",
        "runs", "wickets", "overs",
        "striker", "batsman", "bowler"
    ]

    X = data_encoded[feature_cols]
    y = data_encoded['total']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # --- Scaling ---
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model (Random Forest Regressor) ---
    # Using 50 estimators and max_depth 15 for a perfect balance of speed and high accuracy
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # --- Evaluation ---
    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f'Random Forest Mean Absolute Error: {mae}')
    print(f'Random Forest Mean Squared Error: {mse}')

    return model, scaler, label_encoders


# Cache setup for fast loading in UI
import pickle
MODEL_PKL_PATH = os.path.join(BASE_DIR, 'model_assets.pkl')

def get_or_train_model():
    """Load cached model assets from pickle if available, otherwise train and cache them."""
    if os.path.exists(MODEL_PKL_PATH):
        try:
            with open(MODEL_PKL_PATH, 'rb') as f:
                assets = pickle.load(f)
            if all(k in assets for k in ('model', 'scaler', 'label_encoders')):
                print("Loaded cached model, scaler, and encoders successfully.")
                return assets['model'], assets['scaler'], assets['label_encoders']
        except Exception as e:
            print(f"Warning: Failed to load cached model assets: {e}. Retraining...")
    
    # Train if cache not found or failed
    model, scaler, label_encoders = train_model()
    
    try:
        with open(MODEL_PKL_PATH, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': scaler,
                'label_encoders': label_encoders
            }, f)
        print(f"Model assets successfully cached to {MODEL_PKL_PATH}")
    except Exception as e:
        print(f"Warning: Failed to cache model assets: {e}")
        
    return model, scaler, label_encoders


# Module-level variables
model = None
scaler = None
label_encoders = None

if __name__ == "__main__":
    model, scaler, label_encoders = get_or_train_model()
