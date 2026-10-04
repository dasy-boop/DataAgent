from typing import Any

import pandas as pd

from agent.schemas import AgentPlan
from tools.executor import execute_tool
from tools.pandas_tools import NoNumericDataError


def run_plan(
    df: pd.DataFrame,
    plan: AgentPlan,
) -> list[Any]:
    """按 Agent 计划顺序执行工具。"""
    current_df = df
    results: list[Any] = []

    for tool_call in plan.tool_calls:
        try:
            result = execute_tool(
                tool_call.name,
                current_df,
                tool_call.arguments,
            )
        except NoNumericDataError:
            # Missing measurements are evidence limitations, not execution errors.
            result = {
                "column": tool_call.arguments["column"], "count": 0,
                "mean": None, "median": None, "min": None, "max": None,
                "status": "no_numeric_data",
                "message": "该字段在本次筛选范围内没有可用数值，不能据此推断为零或没有要求。请结合岗位描述和任职要求；仍无依据时说明无法判断。",
            }
        results.append(result)

        # Aggregates are outputs, not a replacement for the filtered records.
        if tool_call.name == "filter_rows" and isinstance(result, pd.DataFrame):
            current_df = result

    return results
