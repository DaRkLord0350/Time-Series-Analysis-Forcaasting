from fastapi import APIRouter, BackgroundTasks 
from pydantic import BaseModel 
from app.models.model_wrapper import ModelWrapper 
 
router = APIRouter() 
 
class TrainRequest(BaseModel): 
    retrain: bool = True 
 
@router.post("/") 
def train(req: TrainRequest, background_tasks: BackgroundTasks): 
    background_tasks.add_task(ModelWrapper().train_and_log) 
    return {"message":"training started"} 
