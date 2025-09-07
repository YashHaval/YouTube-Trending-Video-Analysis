import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(df: pd.DataFrame):
    """
    Train a simple regression model to predict views.
    Input: DataFrame with features + 'views' column.
    Output: trained model, evaluation metrics
    """
    
    # Features (X) and Target (y)
    X = df.drop(columns=['views'])
    y = df['views']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "Mean Squared Error": mse,
        "R² Score": r2
    }

    return model, metrics
