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


def test_skill_comparison_exposes_tool_and_counts_filtered_jobs(monkeypatch):
    question = "Data Analyst 岗位中 SQL 和 Python 哪个更常见？"
    df = pd.DataFrame({
        "title": ["Data Analyst", "Data Analyst", "Software Engineer"],
        "skills_required": ['["SQL", "Python"]', '["SQL"]', '["Python"]'],
    })

    class FakeLLMClient:
        def create_plan(self, question, columns, tool_descriptions):
            skill_tool = next(tool for tool in tool_descriptions
                              if tool["name"] == "count_skills")
            assert skill_tool["parameters"]["required"] == []
            assert skill_tool["parameters"]["properties"]["top_n"]["maximum"] == 50
            return AgentPlan.model_validate({
                "question": question,
                "reasoning": "先筛选岗位，再统计单项技能",
                "tool_calls": [
                    {"name": "filter_rows", "arguments": {
                        "column": "title", "keyword": "Data Analyst"}},
                    {"name": "count_skills", "arguments": {"top_n": 2}},
                ],
            })

        def create_answer(self, question, evidence):
            result = evidence[-1]["result"]
            assert result["matched_records"] == 2
            assert result["valid_skill_records"] == 2
            assert [(row["skill"], row["岗位记录数"]) for row in result["skills"]] == [
                ("sql", 2), ("python", 1),
            ]
            return "在这 2 条岗位中，SQL 出现于 2 条，Python 出现于 1 条。"

    monkeypatch.setattr(agent_routes, "load_agent_data", lambda: df.copy())
    monkeypatch.setattr(agent_routes, "LLMClient", FakeLLMClient)

    with TestClient(app) as client:
        response = client.post("/agent/ask", json={"question": question})

    assert response.status_code == 200, response.text
    body = response.json()
    assert [step["tool"] for step in body["steps"]] == ["filter_rows", "count_skills"]
    assert body["answer_source"] == "model"


def test_two_job_groups_use_independent_skill_counts(monkeypatch):
    question = "比较 Data Analyst 和 Software Engineer 的技能要求。"
    df = pd.DataFrame({
        "title": ["Data Analyst", "Data Analyst", "Software Engineer", "Software Engineer"],
        "skills_required": ['["SQL", "Python"]', '["SQL"]',
                            '["Python", "Java"]', '["Java"]'],
    })

    class FakeLLMClient:
        def create_plan(self, question, columns, tool_descriptions):
            tool = next(tool for tool in tool_descriptions if tool["name"] == "compare_skills")
            assert tool["parameters"]["required"] == ["column", "keyword_a", "keyword_b"]
            return AgentPlan.model_validate({
                "question": question,
                "reasoning": "对两个岗位组分别筛选和统计技能",
                "tool_calls": [{"name": "compare_skills", "arguments": {
                    "column": "title", "keyword_a": "Data Analyst",
                    "keyword_b": "Software Engineer", "top_n": 2,
                }}],
            })

        def create_answer(self, question, evidence):
            analyst, engineer = evidence[-1]["result"]["groups"]
            assert analyst["matched_records"] == engineer["matched_records"] == 2
            assert analyst["skills"][0]["skill"] == "sql"
            assert analyst["skills"][0]["岗位记录数"] == 2
            assert engineer["skills"][0]["skill"] == "java"
            assert engineer["skills"][0]["岗位记录数"] == 2
            return "两组各有 2 条岗位；分析岗 SQL 最常见，软件工程岗 Java 最常见。"

    monkeypatch.setattr(agent_routes, "load_agent_data", lambda: df.copy())
    monkeypatch.setattr(agent_routes, "LLMClient", FakeLLMClient)

    with TestClient(app) as client:
        response = client.post("/agent/ask", json={"question": question})

    assert response.status_code == 200, response.text
    body = response.json()
    assert [step["tool"] for step in body["steps"]] == ["compare_skills"]
    assert body["answer_source"] == "model"
