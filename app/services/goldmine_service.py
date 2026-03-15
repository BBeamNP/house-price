import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "ml_models" / "goal_minemap_model.pkl")
scaler = joblib.load(BASE_DIR / "ml_models" / "goal_minemap_scaler.pkl")


def predict_cluster(price_sqft, connectivity):

    X = np.array([[price_sqft, connectivity]])

    X_scaled = scaler.transform(X)

    distances, indices = model.kneighbors(X_scaled)

    cluster = int(indices[0][0])

    return cluster