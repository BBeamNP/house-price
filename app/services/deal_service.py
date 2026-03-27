import joblib
import numpy as np
import pandas as pd
from geopy.distance import geodesic

model       = joblib.load("app/ml_models/deal_hunter_model.pkl")
scaler      = joblib.load("app/ml_models/deal_hunter_scaler.pkl")
FEATURES    = joblib.load("app/ml_models/deal_hunter_features.pkl")
TYPE_ENC    = joblib.load("app/ml_models/deal_hunter_type_enc.pkl")
BOROUGH_ENC = joblib.load("app/ml_models/deal_hunter_borough_enc.pkl")

ECONOMIC_HUBS = {
    'Midtown':    (40.7580,-73.9855),
    'FiDi':       (40.7081,-74.0093),
    'Brooklyn':   (40.6925,-73.9868),
    'LIC':        (40.7447,-73.9485),
    'Flushing':   (40.7654,-73.8282),
    'Jamaica':    (40.7024,-73.7966),
    'SouthBronx': (40.8162,-73.9165),
    'StGeorge':   (40.6437,-74.0759),
}

def predict_deal(beds, bath, sqft, lat, lon, property_type, sublocality):
    lat_scaled, lon_scaled = scaler.transform([[lat, lon]])[0]

    hub_dists = {k: geodesic((lat, lon), v).miles for k, v in ECONOMIC_HUBS.items()}
    dist_min  = min(hub_dists.values())

    global_mean = np.mean(list(TYPE_ENC.values()))

    row = {
        'BEDS':             beds,
        'BATH':             bath,
        'PROPERTYSQFT':     sqft,
        'LOG_SQFT':         np.log1p(sqft),
        'SQFT_PER_BED':     sqft / max(beds, 1),
        'BEDS_BATH_RATIO':  beds / max(bath, 1),
        'LAT_SCALED':       lat_scaled,
        'LON_SCALED':       lon_scaled,
        'DISTANCE_TO_CENTER': dist_min,
        'DIST_Midtown':     hub_dists['Midtown'],
        'DIST_FiDi':        hub_dists['FiDi'],
        'DIST_Brooklyn':    hub_dists['Brooklyn'],
        'DIST_LIC':         hub_dists['LIC'],
        'DIST_Flushing':    hub_dists['Flushing'],
        'DIST_Jamaica':     hub_dists['Jamaica'],
        'DIST_SouthBronx':  hub_dists['SouthBronx'],
        'DIST_StGeorge':    hub_dists['StGeorge'],
        'TYPE_ENCODED':     TYPE_ENC.get(property_type, global_mean),
        'BOROUGH_ENCODED':  BOROUGH_ENC.get(sublocality, global_mean),
    }

    X = pd.DataFrame([row])[FEATURES]
    prediction = model.predict(X)
    return round(float(np.expm1(prediction[0])), 2)
