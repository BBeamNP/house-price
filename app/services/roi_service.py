import joblib
import numpy as np

model = None

def load_model():
    global model
    try:
        model = joblib.load("app/ml_models/matchmaker_model.pkl")
    except:
        model = None

load_model()

def predict_roi(price, sqft):
    if model is None:
        return ["model_not_loaded"]
    features = np.array([[price, sqft]])
    prediction = model.predict(features)
    return prediction.tolist()
