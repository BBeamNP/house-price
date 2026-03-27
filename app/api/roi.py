from fastapi import APIRouter
from app.services.roi_service import predict_roi

router = APIRouter()

@router.post("/api/matchmaker")
def matchmaker(
    price: float,
    sqft: float,
    beds: int,
    bath: float,
    distance: float
):
    result = predict_roi(price, sqft, beds, bath, distance)
    return result