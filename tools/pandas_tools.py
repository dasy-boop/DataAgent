import pandas as pd


class ToolError(ValueError):
    """工具参数错误。"""


class NoNumericDataError(ToolError):
    """字段存在，但本次筛选范围没有可用于统计的数值。"""


def filter_rows(
    df: pd.DataFrame,
    column: str,
    keyword: str,
) -> pd.DataFrame:
    """按指定字段筛选包含关键词的岗位记录。"""
    if column not in df.columns:
        raise ToolError(f"字段不存在: {column}")

    if not keyword or not keyword.strip():
        raise ToolError("关键词不能为空")

    mask = df[column].astype("string").str.contains(
        keyword.strip(),
        case=False,
        regex=False,
        na=False,
    )
    return df.loc[mask].copy()

def count_values(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """统计指定字段的非空值频次。"""
    if column not in df.columns:
        raise ToolError(f"字段不存在: {column}")

    if top_n < 1 or top_n > 100:
        raise ToolError("top_n 必须在 1 到 100 之间")

    values = df[column].dropna().astype("string").str.strip()
    values = values[values != ""]

    return (
        values.value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="count")
    )
def summarize_numeric(
    df: pd.DataFrame,
    column: str,
) -> dict:
    """统计数值字段的基本分布。"""
    if column not in df.columns:
        raise ToolError(f"字段不存在: {column}")

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        raise NoNumericDataError(f"字段没有可用数值: {column}")

    return {
        "column": column,
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
from backend.skill_cleaning import normalize_skills, parse_skills
from backend.skill_analysis import summarize_skills


def count_skills(df: pd.DataFrame, top_n: int = 10) -> dict:
    """统计当前岗位范围内的技能出现次数。"""
    if "skills_required" not in df.columns:
        raise ToolError("字段不存在: skills_required")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 50:
        raise ToolError("top_n 必须是 1 到 50 之间的整数")

    jobs = df[["skills_required"]].copy()
    parsed = jobs["skills_required"].apply(parse_skills)
    jobs["skills_parse_status"] = parsed.apply(lambda item: item[1])
    jobs["skills_normalized"] = parsed.apply(
        lambda item: normalize_skills(item[0])
    )

    ranking, valid_count = summarize_skills(jobs, top_n=top_n)
    return {
        "matched_records": len(jobs),
        "valid_skill_records": int(valid_count),
        "skills": ranking.reset_index().to_dict(orient="records"),
    }
def compare_skills(
    df: pd.DataFrame,
    column: str,
    keyword_a: str,
    keyword_b: str,
    top_n: int = 10,
) -> dict:
    """分别统计两个岗位范围的技能。"""
    if not isinstance(keyword_a, str) or not keyword_a.strip():
        raise ToolError("keyword_a 不能为空")
    if not isinstance(keyword_b, str) or not keyword_b.strip():
        raise ToolError("keyword_b 不能为空")

    groups = []
    for keyword in (keyword_a.strip(), keyword_b.strip()):
        matched = filter_rows(df, column, keyword)
        groups.append({
            "keyword": keyword,
            **count_skills(matched, top_n),
        })

    return {"column": column, "groups": groups}
