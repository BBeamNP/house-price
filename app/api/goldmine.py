from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import json

router = APIRouter()

model = joblib.load("app/ml_models/goldmine_model.pkl")
df = pd.read_csv("data/NY_House_ValueMap.csv")


@router.get("/goldmine-map-data")
def goldmine_map_data():
    result = df[[
        "LATITUDE", "LONGITUDE", "PRICE", "PRICE_PER_SQFT",
        "VALUE_INDEX", "CLUSTER", "CLUSTER_NAME", "STATE", "SUBLOCALITY_CLEAN"
    ]].rename(columns={"LATITUDE": "latitude", "LONGITUDE": "longitude"})
    # Use pandas to_json to safely handle numpy types and NaN values
    return JSONResponse(content=json.loads(result.to_json(orient="records")))


@router.get("/goldmine-best-area")
def best_area():
    gold = df[df["CLUSTER_NAME"] == "Gold Mine"]
    result = gold[["LATITUDE", "LONGITUDE", "PRICE"]].rename(
        columns={"LATITUDE": "lat", "LONGITUDE": "lng"}
    )
    return JSONResponse(content=json.loads(result.to_json(orient="records")))