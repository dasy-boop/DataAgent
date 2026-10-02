from typing import Any

import pandas as pd

from .pandas_tools import ToolError
from .registry import get_tool


def _validate_arguments(
    name: str,
    df: pd.DataFrame,
    arguments: dict[str, Any],
) -> None:
    allowed_arguments = {
        "filter_rows": {"column", "keyword"},
        "count_values": {"column", "top_n"},
        "summarize_numeric": {"column"},
    }

    allowed = allowed_arguments.get(name)
    if allowed is None:
        return

    extra = sorted(set(arguments) - allowed)
    if extra:
        raise ToolError(
            f"工具 {name} 不支持参数: {', '.join(extra)}"
        )

    column = arguments.get("column")
    if not isinstance(column, str) or not column.strip():
        raise ToolError(f"工具 {name} 必须提供有效的 column 参数")

    if column not in df.columns:
        raise ToolError(f"字段不存在: {column}")

    if name == "filter_rows":
        keyword = arguments.get("keyword")
        if not isinstance(keyword, str) or not keyword.strip():
            raise ToolError("filter_rows 的 keyword 不能为空")

    if name == "count_values" and "top_n" in arguments:
        top_n = arguments["top_n"]
        if (
            isinstance(top_n, bool)
            or not isinstance(top_n, int)
            or not 1 <= top_n <= 100
        ):
            raise ToolError("count_values 的 top_n 必须是 1 到 100 之间的整数")


def execute_tool(
    name: str,
    df: pd.DataFrame,
    arguments: dict[str, Any],
) -> Any:
    if not isinstance(arguments, dict):
        raise ToolError("工具参数必须是 JSON 对象")

    tool = get_tool(name)
    _validate_arguments(name, df, arguments)

    return tool.func(df, **arguments)