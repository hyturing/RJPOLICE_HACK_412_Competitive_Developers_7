from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from utils.anomaly_detection import IsolationForestTrainer, IsolationForestInference

router = APIRouter()

# class Anomaly:
#     CustomerID: int
#     AccountBalance: float
#     LastLogin: str
#     Age: int
#     TransactionID: int
#     Amount: float

@router.post("/anomaly_train_model")
def anomaly_train_model(background_tasks: BackgroundTasks):

    trainer = IsolationForestTrainer()
    try:
        background_tasks.add_task(trainer.train_and_save_model)
        return {"message": "Model training initiated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/anomaly_inference")
# async def anomaly_inference(
#     anomaly: Anomaly = Body(...)
# ):
#     inference = IsolationForestInference()