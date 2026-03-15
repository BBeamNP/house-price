import joblib
import numpy as np

model = None

def load_model():
    global model
    try:
        model = joblib.load("app/ml_models/goal_minemap_model.pkl")
    except:
        model = None

load_model()

def predict_cluster(price_sqft, connectivity):
    if model is None:
        return ["model_not_loaded"]
    features = np.array([[price_sqft, connectivity]])
    cluster = model.predict(features)
    return cluster.tolist()
