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

### 🔴 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://atharvshukla76-ipl-score-predictor.streamlit.app)

👉 **[Click here to try the live app](https://atharvshukla76-ipl-score-predictor.streamlit.app)**

</div>

---

## 🎬 How It Works

<div align="center">

```
 ┌────────────────────────────────────────────────────┐
 │              🏏 IPL Score Predictor                │
 │                                                    │
 │   🏟️ Match Setup                                   │
 │   ┌──────────────────┐  ┌──────────────────┐      │
 │   │ Mumbai Indians   │  │ Chennai Super K. │      │
 │   └──────────────────┘  └──────────────────┘      │
 │   📍 Wankhede Stadium, Mumbai                     │
 │                                                    │
 │   📊 Current Status                                │
 │   Runs: 85  │  Wickets: 3  │  Overs: 10.2        │
 │                                                    │
 │   ┌────────────────────────────────────────┐      │
 │   │        ⚡ Predict Final Score           │      │
 │   └────────────────────────────────────────┘      │
 │                                                    │
 │   🏆 Predicted Final Score: 178                    │
 │                                                    │
 │   Current RR    Projected RR    Runs Remaining    │
 │     8.25           8.90              93           │
 └────────────────────────────────────────────────────┘
```

> *Select teams, set the match situation, and get an instant AI-powered score prediction!*

</div>

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

## 🏗️ Architecture

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
