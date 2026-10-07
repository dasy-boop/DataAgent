from dataclasses import dataclass
from typing import Any, Callable

from .pandas_tools import compare_skills, count_skills
from .analysis_tools import (
    calculate_proportion, compare_groups, count_values, filter_rows,
    sample_records, sample_text, summarize_dates, summarize_numeric,
    summarize_salary,
)


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
    "count_skills": ToolSpec(
        name="count_skills",
        description="统计当前岗位范围内各项技能的出现次数",
        func=count_skills,
    ),
    "compare_skills": ToolSpec(
        name="compare_skills",
        description="分别筛选两组岗位并比较各组技能出现次数",
        func=compare_skills,
    ),
    "summarize_numeric": ToolSpec(
        name="summarize_numeric",
        description="统计数值字段的平均值、中位数、最小值和最大值",
        func=summarize_numeric,
    ),
    "calculate_proportion": ToolSpec(
        name="calculate_proportion", description="按可判定记录计算分类比例及分子分母",
        func=calculate_proportion,
    ),
    "compare_groups": ToolSpec(
        name="compare_groups", description="在相同当前范围内比较两组分类分布",
        func=compare_groups,
    ),
    "summarize_salary": ToolSpec(
        name="summarize_salary", description="按单一币种及薪资周期统计薪资上下限",
        func=summarize_salary,
    ),
    "sample_records": ToolSpec(
        name="sample_records", description="返回有限条真实岗位的必要字段",
        func=sample_records,
    ),
    "sample_text": ToolSpec(
        name="sample_text", description="从长文本字段抽取有限样本供审慎总结",
        func=sample_text,
    ),
    "summarize_dates": ToolSpec(
        name="summarize_dates", description="按月份汇总已有招聘日期",
        func=summarize_dates,
    ),
}


def get_tool(name: str) -> ToolSpec:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"未知工具: {name}")

    return TOOL_REGISTRY[name]
