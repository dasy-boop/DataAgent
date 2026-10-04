import pandas as pd

import agent.routes as routes
from agent.answers import build_evidence
from agent.runtime import run_plan
from agent.schemas import AgentPlan


def test_aggregate_does_not_destroy_filtered_records():
    df = pd.DataFrame({"company_name": ["Western Digital", "Other"],
                       "title": ["Data Analyst", "Engineer"]})
    plan = AgentPlan(question="岗位", reasoning="筛选后统计，再筛选岗位", tool_calls=[
        {"name": "filter_rows", "arguments": {"column": "company_name", "keyword": "Western Digital"}},
        {"name": "count_values", "arguments": {"column": "title"}},
        {"name": "filter_rows", "arguments": {"column": "title", "keyword": "Data Analyst"}},
    ])
    results = run_plan(df, plan)
    assert results[-1].to_dict("records") == [{"company_name": "Western Digital", "title": "Data Analyst"}]


def test_evidence_marks_sampling_and_limits_description():
    evidence = build_evidence([{"tool": "filter_rows", "arguments": {}, "result": [
        {"title": "Analyst", "job_description": "a" * 4000, "irrelevant": "excluded"}
        for _ in range(20)
    ]}])[0]
    assert evidence["total_rows"] == 20
    assert evidence["sampled"] is True
    assert evidence["shown_rows"] == 8
    assert len(evidence["result"][0]["job_description"]) == 3000
    assert "irrelevant" not in evidence["result"][0]


def test_ask_returns_answer_and_preserves_data_on_model_failure(monkeypatch):
    plan = AgentPlan(question="岗位职责", reasoning="筛选公司", tool_calls=[
        {"name": "filter_rows", "arguments": {"column": "company_name", "keyword": "Western Digital"}}
    ])
    monkeypatch.setattr(routes, "create_agent_plan", lambda request: plan.model_dump())
    monkeypatch.setattr(routes, "load_agent_data", lambda: pd.DataFrame({
        "company_name": ["Western Digital"], "title": ["Analyst"], "job_description": ["Build reports"]
    }))

    class Client:
        def create_answer(self, question, evidence):
            assert evidence[0]["result"][0]["job_description"] == "Build reports"
            return "该公司的 Analyst 岗位负责制作报告。"

    monkeypatch.setattr(routes, "LLMClient", Client)
    result = routes.ask_agent(routes.PlanRequest(question="岗位职责"))
    assert result["answer_source"] == "model"
    assert "制作报告" in result["answer"]

    def fail(*args):
        raise TimeoutError()

    monkeypatch.setattr(Client, "create_answer", fail)
    fallback = routes.ask_agent(routes.PlanRequest(question="岗位职责"))
    assert fallback["answer_source"] == "fallback"
    assert fallback["steps"] == result["steps"]
    assert fallback["status"] == "completed"
