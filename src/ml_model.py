import os
import joblib
from sklearn.linear_model import LinearRegression

def train_model(X, y, output_dir):
    model = LinearRegression()
    model.fit(X, y)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))
    return model
