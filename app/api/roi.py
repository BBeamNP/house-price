from fastapi import APIRouter
from app.services.roi_service import predict_roi

router = APIRouter()

@router.post("/roi")
def roi_prediction(price:float, sqft:float):
    result = predict_roi(price, sqft)
    return {"roi_prediction": result}
