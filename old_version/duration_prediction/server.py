from fastapi import FastAPI, HTTPException
from .models import TaskInput, PredictionOutput
from .engine import DurationPredictor
import logging

# Initialize App and Predictor
app = FastAPI(title="Construction Duration Predictor API")
predictor = DurationPredictor()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "duration-predictor"}

@app.post("/predict", response_model=PredictionOutput)
def predict_duration(task: TaskInput):
    """
    Predict duration for a single construction task.
    """
    try:
        logger.info(f"Received prediction request for task: {task.id}")
        result = predictor.predict(task)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
