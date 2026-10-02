from dataclasses import dataclass
from typing import Any, Callable

from .pandas_tools import count_values, filter_rows, summarize_numeric


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]


TOOL_REGISTRY = {
    "filter_rows": ToolSpec(
        name="filter_rows",
        description="按字段和关键词筛选岗位记录",
        func=filter_rows,
    ),
    "count_values": ToolSpec(
        name="count_values",
        description="统计字段值的出现次数",
        func=count_values,
    ),
    "summarize_numeric": ToolSpec(
        name="summarize_numeric",
        description="统计数值字段的平均值、中位数、最小值和最大值",
        func=summarize_numeric,
    ),
}


def get_tool(name: str) -> ToolSpec:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"未知工具: {name}")

    return TOOL_REGISTRY[name]