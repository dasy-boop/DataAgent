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
    column: str | None = None,
    keyword_a: str | None = None,
    keyword_b: str | None = None,
    top_n: int = 10,
    groups: list[dict] | None = None,
    skill: str | None = None,
) -> dict:
    """Compare independent scopes; retain the original two-keyword API."""
    from .analysis_tools import filter_rows as filter_conditions

    if skill is not None and (not isinstance(skill, str) or not skill.strip()):
        raise ToolError("skill 必须是非空技能名称")
    target = normalize_skills([skill])[0] if skill is not None else None
    if groups is not None:
        if any(value is not None for value in (column, keyword_a, keyword_b)):
            raise ToolError("groups 不能与 column、keyword_a、keyword_b 混用")
        if not isinstance(groups, list) or len(groups) != 2:
            raise ToolError("groups 必须包含两个独立比较组")
        for group in groups:
            if (not isinstance(group, dict) or set(group) != {"label", "conditions"}
                    or not isinstance(group["label"], str) or not group["label"].strip()):
                raise ToolError("每组必须包含非空 label 和 conditions")
        if groups[0]["label"].strip() == groups[1]["label"].strip():
            raise ToolError("比较组 label 不能相同")
        scopes = [(group["label"].strip(), filter_conditions(df, conditions=group["conditions"]))
                  for group in groups]
    else:
        for name, value in (("keyword_a", keyword_a), ("keyword_b", keyword_b)):
            if not isinstance(value, str) or not value.strip():
                raise ToolError(f"{name} 不能为空")
        scopes = [(keyword.strip(), filter_rows(df, column, keyword))
                  for keyword in (keyword_a, keyword_b)]

    results = []
    for index, (label, matched) in enumerate(scopes):
        result = {"keyword": label, **count_skills(matched, top_n)}
        if groups is not None:
            result["conditions"] = groups[index]["conditions"]
        if target is not None:
            # Use the very same parser, normalization, deduplication and percentage
            # calculation as rankings; lookup is independent of the top_n cutoff.
            jobs = matched[["skills_required"]].copy()
            parsed = jobs["skills_required"].apply(parse_skills)
            jobs["skills_parse_status"] = parsed.apply(lambda item: item[1])
            jobs["skills_normalized"] = parsed.apply(lambda item: normalize_skills(item[0]))
            ranking, valid = summarize_skills(jobs, top_n=max(1, int(jobs["skills_normalized"].map(len).sum())))
            count = int(ranking.loc[target, "岗位记录数"]) if target in ranking.index else 0
            percentage = (float(ranking.loc[target, "占可解析记录比例(%)"])
                          if target in ranking.index else (0.0 if valid else None))
            result["target_skill"] = {"skill": target, "numerator": count,
                                      "denominator": int(valid), "percentage": percentage}
            result["skills"] = [{"skill": target, "岗位记录数": count,
                                 "占可解析记录比例(%)": percentage}]
        results.append(result)

    output = {"column": column, "groups": results}
    if groups is not None or target is not None:
        enough = all(group["valid_skill_records"] > 0 for group in results)
        output["status"] = "ok" if enough else "insufficient_data"
        if target is not None:
            winner = None
            tied = None
            if enough:
                left, right = [group["target_skill"] for group in results]
                # Compare exact ratios, not rounded display percentages.
                delta = left["numerator"] * right["denominator"] - right["numerator"] * left["denominator"]
                tied = delta == 0
                winner = results[0 if delta > 0 else 1]["keyword"] if delta else None
            output["comparison"] = {"skill": target, "winner": winner, "tied": tied}
    return output
