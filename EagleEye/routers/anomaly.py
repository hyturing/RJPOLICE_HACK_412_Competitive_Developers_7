from fastapi import APIRouter, BackgroundTasks, Request, HTTPException
from utils.anomaly_detection import IsolationForestTrainer

router = APIRouter()


@router.post("/train_model")
def train_model(background_tasks: BackgroundTasks):

    trainer = IsolationForestTrainer()
    try:
        background_tasks.add_task(trainer.train_and_save_model)
        return {"message": "Model training initiated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
