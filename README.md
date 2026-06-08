# 🏏 IPL Score Predictor

An AI-powered web application built using **TensorFlow/Keras** and **Streamlit** to predict the final score of an Indian Premier League (IPL) match based on real-time match dynamics.

<table>
  <tr>
    <td valign="middle"><strong>Live Application:</strong></td>
    <td valign="middle">
      <a href="https://share.streamlit.io/deploy?repository=atharvshukla76/ipl-score-predictor&branch=master&main_file=UI.py" target="_blank">
        <img src="https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white" height="28" />
      </a>
    </td>
  </tr>
</table>

---

## 🚀 Key Features

* **Deep Learning Model:** Powered by a Multi-Layer Perceptron (MLP) built with TensorFlow/Keras.
* **Smart Data Preprocessing:** Utilizes Label Encoding for categorical data (teams, venues, players) and Min-Max scaling for numerical inputs.
* **Premium User Interface:** A modern glassmorphism design with responsive widgets, progress spinners, and dynamic stats layout.
* **Live Run-Rate Insights:** Provides real-time metrics such as Current Run Rate, Projected Run Rate, and Runs Remaining to win/reach target.
* **Automatic Caching:** Optimized model loading via Streamlit's resource caching so training only occurs once.

---

## 🛠️ Model Architecture

The neural network is trained using the **Huber Loss** function (which is robust to outliers) and uses the **Adam** optimizer:
* **Input Layer:** Normalized match state variables
* **Hidden Layer 1:** 512 nodes with ReLU activation
* **Hidden Layer 2:** 256 nodes with ReLU activation
* **Output Layer:** Linear activation predicting the final target/first innings score

---

## 📦 Tech Stack

* **Front-end:** Streamlit
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Tools:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

---

## 💻 Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/atharvshukla76/ipl-score-predictor.git
   cd ipl-score-predictor
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv tf_env
   tf_env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App:**
   ```bash
   streamlit run UI.py
   ```

---

## ☁️ Deployment to Streamlit Cloud

To host this repository live on Streamlit Community Cloud:
1. Log in to [Streamlit Share](https://share.streamlit.io/).
2. Click **New App**.
3. Select your repository: `atharvshukla76/ipl-score-predictor`.
4. Set **Main file path** to `UI.py`.
5. Click **Deploy**!
