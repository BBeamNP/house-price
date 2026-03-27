import joblib
import numpy as np
import pandas as pd

model = scaler = features = weights = None

def load_model():
    global model, scaler, features, weights
    try:
        model   = joblib.load("app/ml_models/matchmaker_model.pkl")
        scaler  = joblib.load("app/ml_models/matchmaker_scaler.pkl")
        features = joblib.load("app/ml_models/matchmaker_features.pkl")
        weights  = joblib.load("app/ml_models/matchmaker_weights.pkl")
    except:
        model = None

load_model()

def predict_roi(price, sqft):
    if model is None:
        return ["model_not_loaded"]
    # Build input with defaults for beds/bath/distance (not provided by ROI endpoint)
    user_input = pd.DataFrame([[price, 2, 1, sqft, 3.0]], columns=features)
    user_scaled   = scaler.transform(user_input)
    user_weighted = user_scaled * weights
    distances, _ = model.kneighbors(user_weighted)
    match_score = float((1 / (1 + distances[0][0])) * 100)
    return [round(match_score, 2)]
