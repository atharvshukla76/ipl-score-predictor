<div align="center">

# 🏏 IPL Score Predictor

### *Predict the unpredictable. Every ball counts.*

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

> 🏆 A machine learning powered web app that predicts the **final innings score** of an IPL match in real-time — using current match situation, teams, players, and venue data.

<br>

</div>

---

## 🎬 See It In Action

<div align="center">

https://github.com/user-attachments/assets/ipl-score-predictor-demo

> *Select teams, set the match situation, and get an instant AI-powered score prediction!*

| Match Setup | Prediction Result |
|:-----------:|:-----------------:|
| Choose batting & bowling teams, venue, batsman & bowler | Get predicted final score with live run-rate metrics |

</div>

### 🖥️ Run It Yourself

```bash
git clone https://github.com/atharvshukla76/ipl-score-predictor.git
cd ipl-score-predictor
pip install -r requirements.txt
streamlit run UI.py
```

Then open **http://localhost:8501** and start predicting! 🏏

---

## ⚡ Key Features

<table>
<tr>
<td width="50%">

### 🤖 Smart Prediction Engine
- **Random Forest Regressor** trained on real IPL ball-by-ball data
- **MAE of ~7.8 runs** — highly accurate match projections
- Instant predictions with cached model loading

</td>
<td width="50%">

### 📊 Live Match Insights
- **Current Run Rate** calculated from your inputs
- **Projected Run Rate** based on AI prediction
- **Runs Remaining** to reach the predicted target

</td>
</tr>
<tr>
<td width="50%">

### 🎨 Premium Dark UI
- Sleek dark theme with IPL-inspired gold & amber accents
- Smooth hover effects and gradient buttons
- Clean, modern layout built with custom CSS

</td>
<td width="50%">

### ⚙️ Optimized Performance
- One-time model training with **pickle caching**
- All CPU cores utilized via parallel processing
- Sub-second predictions after initial load

</td>
</tr>
</table>

---

## 🏗️ How It Works

```
📂 IPL Ball-by-Ball Data (76,000+ deliveries)
        │
        ▼
┌─────────────────────────┐
│   Data Preprocessing    │
│  • Label Encoding       │
│  • MinMax Scaling       │
│  • Feature Selection    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Random Forest Model    │
│  • 50 Decision Trees    │
│  • Max Depth: 15        │
│  • Parallel Training    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Streamlit Web App     │
│  • Team & Player Input  │
│  • Real-time Prediction │
│  • Run Rate Analytics   │
└─────────────────────────┘
```

---

## 🏏 Input Features

| Feature | Description | Example |
|---------|-------------|---------|
| 🏏 **Batting Team** | Team currently batting | Mumbai Indians |
| ⚾ **Bowling Team** | Team currently bowling | Chennai Super Kings |
| 📍 **Venue** | Stadium where the match is played | Wankhede Stadium |
| 🏏 **Batsman** | Current striker on crease | V Kohli |
| 🎯 **Bowler** | Current bowler | JJ Bumrah |
| 🔢 **Runs** | Runs scored so far | 85 |
| 🚫 **Wickets** | Wickets fallen | 3 |
| ⏱️ **Overs** | Overs completed | 10.2 |
| 💥 **Striker Score** | Current batsman's individual score | 42 |

---

## 📦 Tech Stack

<div align="center">

| Layer | Technology |
|:-----:|:----------:|
| **Frontend** | Streamlit + Custom CSS |
| **ML Model** | Scikit-Learn (Random Forest) |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Caching** | Pickle + Streamlit Cache |

</div>

---

## 💻 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/atharvshukla76/ipl-score-predictor.git
cd ipl-score-predictor

# 2. Create virtual environment
python -m venv tf_env
tf_env\Scripts\activate        # Windows
# source tf_env/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pre-train and cache the model (optional, auto-trains on first run)
python main.py

# 5. Launch the app
streamlit run UI.py
```

---

## 📁 Project Structure

```
ipl-score-predictor/
├── UI.py               # Streamlit frontend with dark theme
├── main.py             # Model training, caching & evaluation
├── ipl_data.csv        # Ball-by-ball IPL dataset (76K+ rows)
├── requirements.txt    # Python dependencies
├── .gitignore          # Excluded files
└── README.md           # You are here 👋
```

---

## 🏆 Model Performance

| Metric | Score |
|--------|-------|
| **Mean Absolute Error** | ~7.8 runs |
| **Mean Squared Error** | ~138 |
| **Training Time** | ~8 seconds (cached after first run) |
| **Prediction Time** | < 0.01 seconds |

---

<div align="center">

### Made with ❤️ for Cricket & AI

*If you found this useful, drop a ⭐ on the repo!*

</div>
