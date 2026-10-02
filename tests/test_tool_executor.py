import pandas as pd
import pytest
from tools.pandas_tools import ToolError
from tools.executor import execute_tool


def test_execute_count_values():
    df = pd.DataFrame(
        {
            "country": ["USA", "USA", "Canada"],
        }
    )

    result = execute_tool(
        "count_values",
        df,
        {"column": "country", "top_n": 2},
    )

    assert result.iloc[0]["country"] == "USA"
    assert result.iloc[0]["count"] == 2


def test_execute_unknown_tool():
    df = pd.DataFrame({"country": ["USA"]})

    with (pytest.raises(KeyError)):
        execute_tool("unknown_tool", df, {})
def test_execute_rejects_unknown_argument():
    df = pd.DataFrame({"country": ["USA", "Canada"]})

    with pytest.raises(ToolError, match="不支持参数"):
        execute_tool(
            "count_values",
            df,
            {"field": "country"},
        )


def test_execute_rejects_unknown_column():
    df = pd.DataFrame({"country": ["USA", "Canada"]})

    with pytest.raises(ToolError, match="字段不存在"):
        execute_tool(
            "count_values",
            df,
            {"column": "missing_column"},
        )