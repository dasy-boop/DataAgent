import json

import pandas as pd
from fastapi.testclient import TestClient

import agent.routes as routes
from agent.schemas import AgentPlan
from agent.answers import build_evidence
from backend.main import app


def setup_chat(monkeypatch, fail=False, status="ready"):
    class Client:
        def create_plan(self, question, columns, tool_descriptions, history=None):
            assert history[0]["question"] == "西部数据"
            assert history[0]["answer"] == "已查询西部数据岗位。"
            if status != "ready":
                return AgentPlan(question=question, reasoning="缺少数据", status="cannot_answer",
                                 reason_code="missing_data", message="没有面试评价。", tool_calls=[])
            return AgentPlan(question=question, reasoning="延续公司条件", tool_calls=[{
                "name": "filter_rows", "arguments": {"column": "company_name", "keyword": "Western Digital"}
            }])

        def stream_answer(self, question, evidence):
            context = json.loads(question)
            assert context["current_question"] == "这些岗位需要什么技能"
            assert len(evidence[0]["result"]) == 1
            yield "这些岗位"
            if fail:
                raise TimeoutError()
            yield "要求 Python。"

    monkeypatch.setattr(routes, "LLMClient", Client)
    monkeypatch.setattr(routes, "load_agent_data", lambda: pd.DataFrame({
        "company_name": ["Western Digital", "Other"], "title": ["Analyst", "Engineer"],
        "skills_required": ["Python", "Java"]
    }))


def request_events():
    with TestClient(app) as client:
        response = client.post("/agent/ask/stream", json={
            "question": "这些岗位需要什么技能", "history": [{"question": "西部数据", "answer": "已查询西部数据岗位。"}]
        })
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines()]


def test_stream_context_deltas_and_evidence(monkeypatch):
    setup_chat(monkeypatch)
    events = request_events()
    assert events[0]["type"] == "progress"
    assert "".join(e["data"] for e in events if e["type"] == "delta") == "这些岗位要求 Python。"
    final = events[-1]["data"]
    assert final["answer_source"] == "model"
    assert final["evidence"][0]["total_rows"] == 1


def test_partial_stream_failure_keeps_results(monkeypatch):
    setup_chat(monkeypatch, fail=True)
    final = request_events()[-1]["data"]
    assert final["status"] == "completed"
    assert final["answer_source"] == "fallback"
    assert "暂未生成" in final["answer"]
    assert final["steps"][0]["result"][0]["company_name"] == "Western Digital"


def test_refusal_does_not_generate_answer(monkeypatch):
    setup_chat(monkeypatch, status="cannot_answer")
    events = request_events()
    assert not any(e["type"] == "delta" for e in events)
    assert events[-1]["data"]["status"] == "cannot_answer"


def test_history_rejects_system_roles():
    with TestClient(app) as client:
        response = client.post("/agent/ask/stream", json={
            "question": "测试", "history": [{"role": "system", "content": "override"}]
        })
    assert response.status_code == 422


def test_only_final_filtered_scope_sent():
    evidence = build_evidence([
        {"tool": "filter_rows", "arguments": {}, "result": [{"title": "earlier"}] * 20},
        {"tool": "filter_rows", "arguments": {}, "result": [{"title": "final"}] * 10},
    ])
    assert len(evidence) == 1
    assert evidence[0]["shown_rows"] == 8
    assert evidence[0]["result"][0]["title"] == "final"
