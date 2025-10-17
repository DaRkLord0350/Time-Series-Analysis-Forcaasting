from fastapi import FastAPI 
from app.api import predict, train 
 
app = FastAPI(title="Forecasting API") 
app.include_router(predict.router, prefix="/predict") 
app.include_router(train.router, prefix="/train") 
 
@app.get("/health") 
def health(): 
    return {"status": "ok"} 
