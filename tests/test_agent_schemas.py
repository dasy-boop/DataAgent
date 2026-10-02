import pytest
from pydantic import ValidationError

from agent.schemas import AgentPlan


def test_agent_plan_accepts_tool_calls():
    plan = AgentPlan(
        question="统计 Data Analyst 岗位技能",
        reasoning="先筛选岗位，再统计技能频次",
        tool_calls=[
            {
                "name": "filter_rows",
                "arguments": {
                    "column": "title",
                    "keyword": "Data Analyst",
                },
            }
        ],
    )

    assert plan.tool_calls[0].name == "filter_rows"


def test_agent_plan_rejects_empty_question():
    with pytest.raises(ValidationError):
        AgentPlan(
            question="",
            reasoning="测试",
            tool_calls=[],
        )
@pytest.mark.parametrize(
    "status, reason_code, message",
    [
        (
            "cannot_answer",
            "missing_data",
            "当前数据没有面试评价，无法分析面试难度。",
        ),
        (
            "cannot_answer",
            "unsupported_operation",
            "当前工具不支持预测未来薪资。",
        ),
        (
            "needs_clarification",
            "ambiguous_request",
            "请说明需要分析哪个岗位，以及关注什么指标。",
        ),
    ],
)
def test_plan_accepts_non_executable_status(
    status, reason_code, message
):
    plan = AgentPlan(
        question="测试无法直接回答的问题",
        reasoning="检查数据和工具能否支持分析",
        status=status,
        reason_code=reason_code,
        message=message,
        tool_calls=[],
    )

    assert plan.status == status
    assert plan.message == message
    assert plan.tool_calls == []


def test_cannot_answer_rejects_tool_calls():
    with pytest.raises(ValidationError):
        AgentPlan(
            question="预测明年的薪资",
            reasoning="当前工具不支持预测",
            status="cannot_answer",
            reason_code="unsupported_operation",
            message="当前只能统计已有数据，不能预测未来薪资。",
            tool_calls=[
                {
                    "name": "count_values",
                    "arguments": {
                        "column": "country_clean",
                        "top_n": 5,
                    },
                }
            ],
        )