from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.dataset_routes import router as dataset_router
from backend.analysis_routes import router as analysis_router


app = FastAPI(
    title="DataAgent API",
    version="0.1.0",
)

app.include_router(dataset_router)
app.include_router(analysis_router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "dataagent",
    }


