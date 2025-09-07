# 📊 YouTube Trending Video Analysis & Prediction

This project analyzes the **YouTube Trending Videos (US) dataset** from Kaggle and predicts video views using Machine Learning models.  
It also provides interactive **visualizations** and **live predictions** via Streamlit.

---

## 🚀 Features

- Load and analyze **US YouTube Trending dataset**
- Exploratory Data Analysis (EDA):
  - Distribution of views
  - Likes vs Views scatterplot
  - Correlation heatmap
- Feature engineering:
  - `title_length`
  - `likes_ratio`
  - `engagement_rate`
- Machine Learning Models:
  - Linear Regression
  - Random Forest
  - XGBoost
- Model evaluation metrics:
  - R² Score
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
- Interactive Streamlit app for predictions:
  - Enter likes, dislikes, comments, title length → get **predicted views**

---

## 📂 Dataset

The dataset is taken from Kaggle:  
👉 [YouTube Trending Dataset (USvideos.csv)](https://www.kaggle.com/datasnaek/youtube-new)

Place the file inside the `data/` folder:

## Tech Stack

Python

Streamlit (App UI)

Pandas, NumPy (Data processing)

Matplotlib, Seaborn (Visualizations)

Scikit-learn (ML Models & Metrics)

XGBoost (Boosted Trees Model)

## folder structure
```
YouTube-Trending-ML/
│── app.py # Main Streamlit app
│── requirements.txt # Dependencies list
│── README.md # Project description
│
├── data/ # Dataset folder (not pushed if file is too big)
│ └── USvideos.csv # Kaggle dataset (US trending videos)
│
├── src/ # Optional - helper scripts
│ ├── preprocessing.py # Data cleaning & feature engineering
│ ├── model.py # Model training functions
│ └── evaluate.py # Evaluation metrics & plots
```
## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/YouTube-Trending-ML.git
   cd YouTube-Trending-ML
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```
