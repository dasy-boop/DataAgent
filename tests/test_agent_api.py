import pandas as pd
from fastapi.testclient import TestClient

from agent import routes as agent_routes
from backend.main import app


client = TestClient(app)


def test_agent_execute(monkeypatch):
    df = pd.DataFrame(
        {
            "title": ["Data Analyst", "Software Engineer"],
            "country": ["USA", "Canada"],
        }
    )

    monkeypatch.setattr(
        agent_routes,
        "load_parquet",
        lambda _: df,
    )

    response = client.post(
        "/agent/execute",
        json={
            "plan": {
                "question": "统计 Data Analyst 岗位国家",
                "reasoning": "先筛选岗位，再统计国家",
                "tool_calls": [
                    {
                        "name": "filter_rows",
                        "arguments": {
                            "column": "title",
                            "keyword": "Data Analyst",
                        },
                    },
                    {
                        "name": "count_values",
                        "arguments": {
                            "column": "country",
                            "top_n": 10,
                        },
                    },
                ],
            }
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["question"] == "统计 Data Analyst 岗位国家"
    assert len(body["steps"]) == 2
    assert body["steps"][1]["result"][0]["country"] == "USA"
from types import SimpleNamespace

from agent.schemas import AgentPlan


def test_agent_plan(monkeypatch):
    df = pd.DataFrame(
        {
            "title": ["Data Analyst"],
            "country": ["USA"],
        }
    )

    class FakeLLMClient:
        def create_plan(self, question, columns, tool_descriptions):
            return AgentPlan(
                question=question,
                reasoning="先统计国家",
                tool_calls=[
                    {
                        "name": "count_values",
                        "arguments": {
                            "column": "country",
                            "top_n": 10,
                        },
                    }
                ],
            )

    monkeypatch.setattr(
        agent_routes,
        "load_parquet",
        lambda _: df,
    )
    monkeypatch.setattr(
        agent_routes,
        "LLMClient",
        FakeLLMClient,
    )

    response = client.post(
        "/agent/plan",
        json={"question": "统计岗位国家"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["question"] == "统计岗位国家"
    assert body["tool_calls"][0]["name"] == "count_values"