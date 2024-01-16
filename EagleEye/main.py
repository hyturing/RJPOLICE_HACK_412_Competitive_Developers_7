import time
import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import data_pipeline, anomaly, monitoring

origins = ["http://localhost", "http://localhost:3000", "*"]

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Start-Time"] = str(f"{start_time:0.4f} sec")
    response.headers["X-Process-Time"] = str(f"{process_time:0.4f} sec")
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    traceback.print_exc()
    return JSONResponse({"error": "Internal Server Error: " + str(exc)}, status_code=500)


@app.get("/")
def index():
    return {"Root of EagleEye, Thanks for using the service."}


app.include_router(data_pipeline.router, prefix="/data_pipeline", tags=["data_pipeline"])

app.include_router(anomaly.router, prefix="/anomaly", tags=["anomaly"])

app.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
