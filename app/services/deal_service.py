import joblib
import numpy as np

model = joblib.load("app/ml_models/deal_hunter_model.pkl")

def predict_deal(beds, bath, sqft, lat, lon, property_type, sublocality):

    features = {
        "BEDS": beds,
        "BATH": bath,
        "PROPERTYSQFT": sqft,
        "LAT_SCALED": lat,
        "LON_SCALED": lon,

        "TYPE_Co-op for sale": 0,
        "TYPE_Condo for sale": 0,
        "TYPE_House for sale": 0,
        "TYPE_Multi-family home for sale": 0,
        "TYPE_Townhouse for sale": 0,

        "SUBLOCALITY_CLEAN_Bronx": 0,
        "SUBLOCALITY_CLEAN_Brooklyn": 0,
        "SUBLOCALITY_CLEAN_Manhattan": 0,
        "SUBLOCALITY_CLEAN_Queens": 0,
        "SUBLOCALITY_CLEAN_Staten Island": 0
    }

    features[f"TYPE_{property_type}"] = 1
    features[f"SUBLOCALITY_CLEAN_{sublocality}"] = 1

    X = np.array([list(features.values())])

    prediction = model.predict(X)

    price = np.exp(prediction[0])

    return round(price,2)