import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load("app/ml_models/deal_hunter_model.pkl")

# ดู feature ที่โมเดลใช้
print("Features:", model.feature_names_in_)

# สร้าง input ตาม feature ของโมเดล
features = {
    'BEDS': 3,
    'BATH': 2,
    'PROPERTYSQFT': 1400,
    'LAT_SCALED': 40.7,
    'LON_SCALED': -74.0,

    'TYPE_Co-op for sale': 0,
    'TYPE_Condo for sale': 0,
    'TYPE_House for sale': 1,
    'TYPE_Multi-family home for sale': 0,
    'TYPE_Townhouse for sale': 0,

    'SUBLOCALITY_CLEAN_Bronx': 0,
    'SUBLOCALITY_CLEAN_Brooklyn': 0,
    'SUBLOCALITY_CLEAN_Manhattan': 1,
    'SUBLOCALITY_CLEAN_Queens': 0,
    'SUBLOCALITY_CLEAN_Staten Island': 0
}

# แปลงเป็น array
X = np.array([list(features.values())])

# predict
prediction = model.predict(X)

price = np.exp(prediction[0])

print("Predicted Price:", round(price, 2))