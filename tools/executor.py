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
        "filter_rows": {"column", "keyword", "conditions"},
        "count_values": {"column", "top_n"},
        "count_skills": {"top_n"},
        "compare_skills": {"column", "keyword_a", "keyword_b", "top_n", "groups", "skill"},
        "summarize_numeric": {"column"},
        "calculate_proportion": {"column", "value"},
        "compare_groups": {"group_column", "value_a", "value_b", "metric_column", "top_n"},
        "summarize_salary": {"currency", "period"},
        "sample_records": {"limit"},
        "sample_text": {"column", "limit"},
        "summarize_dates": {"column", "top_n"},
    }

    allowed = allowed_arguments.get(name)
    if allowed is None:
        return

    extra = sorted(set(arguments) - allowed)
    if extra:
        raise ToolError(
            f"工具 {name} 不支持参数: {', '.join(extra)}"
        )

    if name == "filter_rows" and "conditions" in arguments:
        if "column" in arguments or "keyword" in arguments:
            raise ToolError("多条件筛选不能混用旧式参数")
        return

    if name in {"calculate_proportion", "compare_groups", "summarize_salary",
                "sample_records", "sample_text", "summarize_dates"}:
        required = {
            "calculate_proportion": {"column", "value"},
            "compare_groups": {"group_column", "value_a", "value_b", "metric_column"},
            "summarize_salary": {"currency", "period"},
            "sample_records": set(),
            "sample_text": {"column"},
            "summarize_dates": set(),
        }[name]
        missing = required - set(arguments)
        if missing:
            raise ToolError(f"工具 {name} 缺少参数: {', '.join(sorted(missing))}")
        return

    if name == "compare_skills" and "groups" in arguments:
        return  # The tool validates nested groups and mutually exclusive legacy arguments.

    # The skill tool has a fixed source field and validates top_n itself.
    if name == "count_skills":
        return

    column = arguments.get("column")
    if not isinstance(column, str) or not column.strip():
        raise ToolError(f"工具 {name} 必须提供有效的 column 参数")

    if column not in df.columns:
        raise ToolError(f"字段不存在: {column}")

    if name == "compare_skills":
        for key in ("keyword_a", "keyword_b"):
            if not isinstance(arguments.get(key), str) or not arguments[key].strip():
                raise ToolError(f"compare_skills 的 {key} 不能为空")

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
