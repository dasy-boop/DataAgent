import pandas as pd
import pytest

from tools.pandas_tools import ToolError, count_values, filter_rows
from tools.pandas_tools import ToolError, filter_rows
from tools.pandas_tools import (
    ToolError,
    count_values,
    filter_rows,
    summarize_numeric,
)


def test_filter_rows_matches_keyword():
    df = pd.DataFrame(
        {
            "title": ["Data Analyst", "Software Engineer", None],
        }
    )

    result = filter_rows(df, "title", "data analyst")

    assert len(result) == 1
    assert result.iloc[0]["title"] == "Data Analyst"


def test_filter_rows_rejects_unknown_column():
    df = pd.DataFrame({"title": ["Data Analyst"]})

    with pytest.raises(ToolError):
        filter_rows(df, "missing", "test")


def test_filter_rows_rejects_empty_keyword():
    df = pd.DataFrame({"title": ["Data Analyst"]})

    with pytest.raises(ToolError):
        filter_rows(df, "title", "  ")
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
def test_count_values_returns_sorted_counts():
    df = pd.DataFrame(
        {
            "country": ["USA", "USA", "Canada", None, "Canada"],
        }
    )

    result = count_values(df, "country", top_n=2)

    assert result["country"].tolist() == ["USA", "Canada"]
    assert result["count"].tolist() == [2, 2]
def test_summarize_numeric():
    df = pd.DataFrame({"salary": [50000, 70000, None]})

    result = summarize_numeric(df, "salary")

    assert result["count"] == 2
    assert result["mean"] == 60000
    assert result["median"] == 60000
    assert result["min"] == 50000
    assert result["max"] == 70000