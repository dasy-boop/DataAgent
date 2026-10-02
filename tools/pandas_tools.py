import pandas as pd


class ToolError(ValueError):
    """工具参数错误。"""


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
        raise ToolError(f"字段没有可用数值: {column}")

    return {
        "column": column,
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }

