from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.data_loader import load_parquet
from backend.skill_analysis import summarize_skills
from backend.skill_cleaning import normalize_skills, parse_skills


router = APIRouter(prefix="/analysis", tags=["数据分析"])

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "nextgig_jobs_2026-06.parquet"


@router.get("/skills")
def analyze_skills(
    keyword: str = Query(min_length=1, max_length=100),
    top_n: int = Query(default=10, ge=1, le=50),
):
    keyword = keyword.strip()
    if not keyword:
        raise HTTPException(status_code=422, detail="关键词不能全为空格")

    df = load_parquet(DATA_PATH)

    # 按岗位标题筛选，关键词按普通文字匹配
    mask = df["title"].astype("string").str.contains(
        keyword, case=False, regex=False, na=False
    )
    jobs = df.loc[mask].copy()

    # 复用已经测试过的技能处理函数
    parsed = jobs["skills_required"].apply(parse_skills)
    jobs["skills_list"] = parsed.apply(lambda result: result[0])
    jobs["skills_parse_status"] = parsed.apply(lambda result: result[1])
    jobs["skills_normalized"] = jobs["skills_list"].apply(normalize_skills)

    ranking, valid_count = summarize_skills(jobs, top_n=top_n)

    return {
        "dataset": DATA_PATH.name,
        "keyword": keyword,
        "matched_records": int(len(jobs)),
        "valid_skill_records": int(valid_count),
        "top_n": top_n,
        "counting_rule": "每条岗位记录的同一技能最多计数一次",
        "percentage_denominator": "匹配岗位中技能解析成功的记录数",
        "skills": ranking.reset_index().to_dict(orient="records"),
    }