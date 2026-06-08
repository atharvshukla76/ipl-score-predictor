import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents plt.show() from blocking
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Get absolute path to the CSV relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'ipl_data.csv')


def train_model():
    """Train the IPL score prediction model and return model, scaler, label_encoders."""

    ipl = pd.read_csv(CSV_PATH)
    data = ipl.copy()

    # --- Exploratory Data Analysis (skipped in Streamlit mode) ---
    # These plots are only generated when running main.py directly
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

        runs_by_batsman = data.groupby('batsman')['runs'].max().sort_values(ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=runs_by_batsman.values, y=runs_by_batsman.index, palette="pastel")
        plt.title('Top 10 Batsmen by Maximum Runs in an Innings')
        plt.xlabel('Maximum Runs in an Innings')
        plt.ylabel('Batsman')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'batsmen_plot.png'))
        plt.close()

        wickets_by_bowler = data.groupby('bowler')['wickets'].max().sort_values(ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=wickets_by_bowler.values, y=wickets_by_bowler.index, palette="muted")
        plt.title('Top 10 Bowlers by Maximum Wickets in an Innings')
        plt.xlabel('Maximum Wickets in an Innings')
        plt.ylabel('Bowler')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'bowlers_plot.png'))
        plt.close()

    # --- Encoding ---
    cat_cols = ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler']
    data_encoded = data.copy()

    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        label_encoders[col] = le

    # --- Correlation (skipped in Streamlit mode) ---
    if __name__ == "__main__":
        data_corr = data_encoded.drop(columns=["date", "mid"], axis=1).corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(data_corr, annot=True)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'correlation_plot.png'))
        plt.close()

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

    # --- Model ---
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])

    huber_loss = tf.keras.losses.Huber(delta=1.0)

    model.compile(optimizer='adam', loss=huber_loss)

    model.fit(
        X_train_scaled,
        y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )

    # --- Evaluation ---
    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

    return model, scaler, label_encoders


# Module-level variables
model = None
scaler = None
label_encoders = None

if __name__ == "__main__":
    model, scaler, label_encoders = train_model()
