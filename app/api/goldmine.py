from fastapi import APIRouter
from app.services.goldmine_service import predict_cluster

router = APIRouter()

@router.post("/goldmine")
def goldmine_prediction(price_sqft:float, connectivity:float):
    result = predict_cluster(price_sqft, connectivity)
    return {"cluster": result}
