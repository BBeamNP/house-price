import joblib
import numpy as np
import pandas as pd
from geopy.distance import geodesic

ECONOMIC_HUBS = [
    (40.7580, -73.9855), (40.7081, -74.0093),
    (40.6925, -73.9868),
    (40.7447, -73.9485), (40.7654, -73.8282), (40.7024, -73.7966),
    (40.8162, -73.9165),
    (40.6437, -74.0759)
]

model = None
scaler = None
features = None
weights = None
data = None
train_indices = None

def load_model():
    global model, scaler, features, weights, data, train_indices
    try:
        model   = joblib.load("app/ml_models/matchmaker_model.pkl")
        scaler  = joblib.load("app/ml_models/matchmaker_scaler.pkl")
        features = joblib.load("app/ml_models/matchmaker_features.pkl")
        weights = joblib.load("app/ml_models/matchmaker_weights.pkl")

        data = pd.read_csv("data/NY_House_Cleaned.csv")

        if 'DISTANCE_TO_CENTER' not in data.columns:
            data['DISTANCE_TO_CENTER'] = data.apply(
                lambda row: min(
                    geodesic((row['LATITUDE'], row['LONGITUDE']), hub).miles
                    for hub in ECONOMIC_HUBS
                ),
                axis=1
            )

        # Build the same training index the model was fitted on
        train_indices = data[features].dropna().index

    except Exception as e:
        model = None

load_model()

def predict_roi(price, sqft, beds, bath, distance):
    if model is None:
         return {"error": "model_not_loaded"}

    user_input = pd.DataFrame(
        [[price, beds, bath, sqft, distance]],
        columns=features
    )

    user_scaled = scaler.transform(user_input)
    user_weighted = user_scaled * weights
    distances, _ = model.kneighbors(user_weighted)
    distances, indices = model.kneighbors(user_weighted)

    scores = (1 / (1 + distances[0])) * 100
    final_score = float(np.mean(scores[:5]))

    # Map positional indices back to correct rows in data
    top_matches = data.loc[train_indices[indices[0]]].copy()
    top_matches['MATCH_SCORE'] = scores[:len(top_matches)]

    result = top_matches[
        ['PRICE', 
            'BEDS', 
            'BATH', 
            'PROPERTYSQFT', 
            'DISTANCE_TO_CENTER', 
            'SUBLOCALITY_CLEAN', 
            'MATCH_SCORE',
            'LATITUDE',  # เพิ่มเพื่อให้แผนที่ปักหมุดได้
            'LONGITUDE'  # เพิ่มเพื่อให้แผนที่ปักหมุดได้]
    ]].head(5)

    return {
        "match_score": round(final_score, 2),
        "top_matches": result.to_dict(orient="records")
    }
