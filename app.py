import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------
# Streamlit App Title
# ------------------------------
st.title("📊 YouTube Trending Video Analysis & Prediction")
st.write("Analyze YouTube trending videos dataset and predict video views using ML models.")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload your YouTube dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Dataset")
    st.dataframe(df.head())

    # ------------------------------
    # Basic EDA
    # ------------------------------
    st.subheader("📈 Exploratory Data Analysis")

    if "views" in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df["views"], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Views")
        st.pyplot(fig)

    if "likes" in df.columns and "views" in df.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(x="likes", y="views", data=df, ax=ax)
        ax.set_title("Likes vs Views")
        st.pyplot(fig)

    # ------------------------------
    # Feature Engineering
    # ------------------------------
    st.subheader("🛠 Feature Engineering")

    df["title_length"] = df["title"].astype(str).apply(len) if "title" in df.columns else 0
    df["likes_ratio"] = df["likes"] / (df["likes"] + df["dislikes"] + 1)
    df["engagement_rate"] = (df["likes"] + df["comment_count"]) / (df["views"] + 1)

    st.write("✅ Added Features: `title_length`, `likes_ratio`, `engagement_rate`")

    # Select features for ML
    features = ["likes", "dislikes", "comment_count", "title_length", "likes_ratio", "engagement_rate"]
    df = df.dropna(subset=["views"])  # Ensure no missing target values

    X = df[features]
    y = np.log1p(df["views"])  # Log-transform target for stability

    # ------------------------------
    # Train-Test Split
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ------------------------------
    # Model Selection
    # ------------------------------
    st.sidebar.subheader("⚙️ Model Training Options")
    model_choice = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Random Forest", "XGBoost"])

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)

    # Train Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_exp = np.expm1(y_pred)   # Reverse log transform
    y_test_exp = np.expm1(y_test)

    # ------------------------------
    # Model Evaluation
    # ------------------------------
    st.subheader("📊 Model Performance")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))

    st.write(f"**Model:** {model_choice}")
    st.write(f"✅ R² Score: {r2:.3f}")
    st.write(f"✅ MAE: {mae:,.2f}")
    st.write(f"✅ RMSE: {rmse:,.2f}")

    # Plot Predictions vs Actual
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test_exp, y=y_pred_exp, ax=ax)
    ax.set_xlabel("Actual Views")
    ax.set_ylabel("Predicted Views")
    ax.set_title("Actual vs Predicted Views")
    st.pyplot(fig)

    # ------------------------------
    # Live Prediction
    # ------------------------------
    st.subheader("🔮 Predict Views for New Video")

    likes = st.number_input("Likes", 0, 10_000_000, 5000)
    dislikes = st.number_input("Dislikes", 0, 1_000_000, 100)
    comments = st.number_input("Comment Count", 0, 500_000, 1000)
    title_length = st.number_input("Title Length", 1, 150, 20)

    likes_ratio = likes / (likes + dislikes + 1)
    engagement_rate = (likes + comments) / (likes + 1)

    input_data = pd.DataFrame([[likes, dislikes, comments, title_length, likes_ratio, engagement_rate]],
                              columns=features)

    if st.button("Predict Views"):
        pred = model.predict(input_data)
        pred_views = int(np.expm1(pred[0]))
        st.success(f"📌 Predicted Views: {pred_views:,}")

else:
    st.info("👆 Please upload a CSV file to begin.")
