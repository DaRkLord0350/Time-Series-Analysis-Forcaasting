from fastapi import APIRouter 
from pydantic import BaseModel 
from app.models.model_wrapper import ModelWrapper 
 
router = APIRouter() 
 
class PredictRequest(BaseModel): 
    steps: int = 14 
 
@router.post("/") 
def predict(req: PredictRequest): 
    mw = ModelWrapper() 
    forecast = mw.predict(steps=req.steps) 
    return {"forecast": forecast} 
