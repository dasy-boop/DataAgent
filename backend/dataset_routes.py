from pathlib import Path

from fastapi import APIRouter, Query

from backend.data_loader import load_parquet
from backend.data_quality import summarize_quality
from backend.country_cleaning import normalize_country
from tools.pandas_tools import count_skills
from tools.analysis_tools import count_values


router = APIRouter(prefix="/dataset", tags=["数据集"])

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "nextgig_jobs_2026-06.parquet"


@router.get("/overview")
def dataset_overview():
    df = load_parquet(DATA_PATH)
    countries = normalize_country(df["country"])

    def present(values):
        return values.notna() & values.astype("string").str.strip().ne("").fillna(False)

    quality = summarize_quality(df)
    quality_fields = {
        "title": "岗位名称",
        "country": "国家/地区",
        "skills_required": "技能要求",
    }
    quality_rates = [
        {"label": label, "rate": round(100 - float(quality.loc[field, "缺失率(%)"]), 2)}
        for field, label in quality_fields.items()
    ]
    # 薪资上下限任意一个有数值即可；不把缺失值解释成 0。
    salary_present = df[["salary_min", "salary_max"]].notna().any(axis=1)
    quality_rates.append({"label": "薪资信息", "rate": round(float(salary_present.mean() * 100), 2)})

    return {
        "dataset": DATA_PATH.name,
        "display_name": "2026年6月全球招聘岗位数据",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "countries": int(countries.nunique(dropna=True)),
        "companies": int(df.loc[present(df["company_name"]), "company_name"].nunique()),
        "quality_rates": quality_rates,
        "preview": df[["title", "company_name", "country", "work_model", "experience_level"]]
        .head(8).where(lambda rows: rows.notna(), None).to_dict(orient="records"),
    }


@router.get("/dashboard")
def dataset_dashboard():
    """供图表页使用的真实数据汇总；保留原有分析接口和统计口径。"""
    df = load_parquet(DATA_PATH)
    country_df = df.assign(country_display=normalize_country(df["country"]))
    countries = count_values(country_df, "country_display", 8)
    companies = count_values(df, "company_name", 8)
    title_column = "normalized_title" if "normalized_title" in df.columns else "title"
    titles = count_values(df, title_column, 8)
    skills_result = count_skills(df, 8)

    def points(rows, column):
        return [
            {"label": str(row[column]), "count": int(row["count"])}
            for row in rows.to_dict(orient="records")
        ]

    return {
        "rows": int(len(df)),
        "charts": [
            {"kind": "country", "title": "招聘岗位最多的国家/地区", "note": "已合并已知国家别名；统计非空国家信息。", "items": points(countries, "country_display")},
            {"kind": "company", "title": "招聘岗位最多的公司", "note": "按公司名称统计非空岗位记录。", "items": points(companies, "company_name")},
            {"kind": "title", "title": "常见岗位", "note": "按标准岗位名称统计；排除仅表示全职/兼职的取值。", "items": points(titles, title_column)},
            {"kind": "skill", "title": "常见技能", "note": f"{skills_result['valid_skill_records']:,} 条岗位的技能可解析；同一岗位的同一技能只计一次。", "items": [
                {"label": item["skill"], "count": int(item["岗位记录数"])}
                for item in skills_result["skills"]
            ]},
        ],
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
