from fastapi import APIRouter
import pandas as pd
import joblib


router = APIRouter()

# โหลด model
model = joblib.load("app/ml_models/goldmine_model.pkl")

# โหลด dataset ที่มี feature แล้ว
df = pd.read_csv("data/NY_House_ValueMap.csv")


@router.get("/goldmine-map-data")
def goldmine_map_data():

    return df[
        [
            "LATITUDE",
            "LONGITUDE",
            "PRICE",
            "PRICE_PER_SQFT",
            "VALUE_INDEX",
            "CLUSTER",
            "CLUSTER_NAME",
            "STATE"
        ]
    ].rename(
        columns={
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude"
        }
    ).to_dict(orient="records")
@router.get("/goldmine-best-area")
def best_area():

    gold = df[df["CLUSTER_NAME"]=="Gold Mine"]

    return gold[["LATITUDE","LONGITUDE","PRICE"]].rename(
        columns={"LATITUDE":"lat","LONGITUDE":"lng"}
    ).to_dict(orient="records")