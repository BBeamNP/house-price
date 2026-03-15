from fastapi import APIRouter
from app.services.deal_service import predict_deal

router = APIRouter()

@router.post("/deal")
def deal_prediction(
    beds: int,
    bath: int,
    sqft: float,
    lat: float,
    lon: float,
    property_type: str,
    sublocality: str
):
        
    result = predict_deal(beds, bath, sqft, lat, lon, property_type, sublocality)

    return {
        "deal_prediction": result
    }