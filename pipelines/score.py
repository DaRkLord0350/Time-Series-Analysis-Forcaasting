from app.models.model_wrapper import ModelWrapper 
def score(steps=14): 
    mw = ModelWrapper() 
    return mw.predict(steps) 
