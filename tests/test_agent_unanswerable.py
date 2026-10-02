import pytest
from fastapi.testclient import TestClient

import agent.routes as agent_routes
from backend.main import app


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
def test_ask_does_not_execute_unavailable_plan(
    monkeypatch, status, reason_code, message
):
    question = "测试无法直接完成的问题"

    def fake_create_plan(request):
        return {
            "question": request.question,
            "reasoning": "检查数据和工具是否支持请求",
            "status": status,
            "reason_code": reason_code,
            "message": message,
            "tool_calls": [],
        }

    def forbidden_call(*args, **kwargs):
        pytest.fail("无法执行的计划不应加载执行数据或调用工具")

    monkeypatch.setattr(
        agent_routes,
        "create_agent_plan",
        fake_create_plan,
    )
    monkeypatch.setattr(
        agent_routes,
        "load_agent_data",
        forbidden_call,
    )
    monkeypatch.setattr(
        agent_routes,
        "run_plan",
        forbidden_call,
    )

    with TestClient(app) as client:
        response = client.post(
            "/agent/ask",
            json={"question": question},
        )

    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == status
    assert body["reason_code"] == reason_code
    assert body["message"] == message
    assert body["steps"] == []
    assert body["plan"]["tool_calls"] == []