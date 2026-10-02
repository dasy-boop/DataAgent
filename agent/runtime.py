from typing import Any

import pandas as pd

from agent.schemas import AgentPlan
from tools.executor import execute_tool


def run_plan(
    df: pd.DataFrame,
    plan: AgentPlan,
) -> list[Any]:
    """按 Agent 计划顺序执行工具。"""
    current_df = df
    results: list[Any] = []

    for tool_call in plan.tool_calls:
        result = execute_tool(
            tool_call.name,
            current_df,
            tool_call.arguments,
        )
        results.append(result)

        if isinstance(result, pd.DataFrame):
            current_df = result

    return results