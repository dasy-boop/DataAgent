import pandas as pd
from fastapi.testclient import TestClient

import agent.routes as agent_routes
from agent.schemas import AgentPlan
from backend.main import app


def test_ask_generates_plan_and_executes(monkeypatch):
    question = "统计 Data Analyst 岗位的国家分布"

    df = pd.DataFrame({
        "title": [
            "Data Analyst",
            "Senior Data Analyst",
            "Data Analyst",
            "Software Engineer",
        ],
        "country_clean": [
            "United States",
            "United States",
            "Canada",
            "Canada",
        ],
    })

    class FakeLLMClient:
        def create_plan(self, question, columns, tool_descriptions):
            assert "country_clean" in columns

            return AgentPlan.model_validate({
                "question": question,
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
                            "column": "country_clean",
                            "top_n": 5,
                        },
                    },
                ],
            })

    monkeypatch.setattr(
        agent_routes,
        "load_agent_data",
        lambda: df.copy(),
    )
    monkeypatch.setattr(
        agent_routes,
        "LLMClient",
        FakeLLMClient,
    )

    with TestClient(app) as client:
        response = client.post(
            "/agent/ask",
            json={"question": question},
        )

    assert response.status_code == 200, response.text

    body = response.json()
    assert body["question"] == question
    assert len(body["plan"]["tool_calls"]) == 2
    assert len(body["steps"]) == 2
    assert body["steps"][-1]["result"] == [
        {"country_clean": "United States", "count": 2},
        {"country_clean": "Canada", "count": 1},
    ]