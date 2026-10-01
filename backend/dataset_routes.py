from pathlib import Path

from fastapi import APIRouter, Query

from backend.data_loader import load_parquet
from backend.data_quality import summarize_quality


router = APIRouter(prefix="/dataset", tags=["数据集"])

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "nextgig_jobs_2026-06.parquet"


@router.get("/overview")
def dataset_overview():
    df = load_parquet(DATA_PATH)

    return {
        "dataset": DATA_PATH.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
    }


@router.get("/quality")
def dataset_quality():
    df = load_parquet(DATA_PATH)
    summary = summarize_quality(df).head(10)
    fields = summary.rename_axis("field").reset_index()

    return {
        "dataset": DATA_PATH.name,
        "fields": fields.to_dict(orient="records"),
    }


@router.get("/preview")
def dataset_preview(
    limit: int = Query(default=10, ge=1, le=50),
):
    df = load_parquet(DATA_PATH)

    preview = df.head(limit).astype(object)
    preview = preview.where(preview.notna(), None)

    return {
        "dataset": DATA_PATH.name,
        "limit": limit,
        "rows": preview.to_dict(orient="records"),
    }
