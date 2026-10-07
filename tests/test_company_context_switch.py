import pandas as pd
import pytest
from fastapi.testclient import TestClient

import agent.routes as routes
from agent.context_scope import enforce_scope
from agent.schemas import AgentPlan
from backend.main import app


def plan(company=None, *, combined=False, extra=()):
    calls = []
    if company:
        if combined:
            conditions = [{"column": "company_name", "operator": "contains", "value": company}]
            conditions.extend(extra)
            calls.append({"name": "filter_rows", "arguments": {"conditions": conditions}})
        else:
            calls.append({"name": "filter_rows", "arguments": {"column": "company_name", "keyword": company}})
    calls.append({"name": "count_values", "arguments": {"column": "title", "top_n": 10}})
    return AgentPlan(question="test", reasoning="test", tool_calls=calls)


def filters(value):
    result = []
    for call in value.tool_calls:
        if call.name != "filter_rows":
            continue
        if "conditions" in call.arguments:
            result.extend(call.arguments["conditions"])
        else:
            result.append({"column": call.arguments["column"], "value": call.arguments["keyword"]})
    return result


@pytest.mark.parametrize("company,question,combined", [
    ("Western Digital", "西部数据有哪些岗位？", True),
    ("Bosch", "Bosch 有哪些岗位？", False),
    ("Apple", "Apple 有哪些岗位？", True),
])
def test_new_company_discards_unmentioned_historical_dimensions(company, question, combined):
    history = [{"question": "美国的数据分析师有哪些远程岗位？"}]
    current = plan(company, combined=combined, extra=(
        {"column": "country_clean", "operator": "equals", "value": "United States"},
        {"column": "title", "operator": "contains", "value": "Data Analyst"},
        {"column": "work_model", "operator": "equals", "value": "Remote"},
    ) if combined else ())
    adjusted = enforce_scope(current, history, question)
    expected = ({"column": "company_name", "value": company, "operator": "contains"}
                if combined else {"column": "company_name", "value": company})
    assert filters(adjusted) == [expected]


@pytest.mark.parametrize("question,expected", [
    ("美国 Bosch 有哪些岗位？", {"country_clean": "United States"}),
    ("Bosch 的数据分析师有哪些？", {"title": "Data Analyst"}),
    ("Bosch 的远程岗位有哪些？", {"work_model": "Remote"}),
])
def test_explicit_current_dimensions_survive_company_switch(question, expected):
    adjusted = enforce_scope(plan("Bosch"), [{"question": "英国的软件工程师有哪些现场岗位？"}], question)
    values = {item["column"]: item["value"] for item in filters(adjusted)}
    assert values == {**expected, "company_name": "Bosch"}


def test_referential_company_followup_keeps_country_and_replaces_role():
    history = [{"question": "美国的 Bosch 有哪些软件工程师？"}]
    adjusted = enforce_scope(plan("Bosch"), history, "其中的数据分析师有哪些？")
    values = {item["column"]: item["value"] for item in filters(adjusted)}
    assert values == {"country_clean": "United States", "title": "Data Analyst", "company_name": "Bosch"}


def test_country_and_work_mode_followups_keep_other_scope():
    history = [{"question": "美国的软件工程师有哪些岗位？"}]
    uk = enforce_scope(plan(), history, "那英国呢？")
    assert {item["column"]: item["value"] for item in filters(uk)} == {
        "country_clean": "United Kingdom", "title": "Software Engineer"}
    remote = enforce_scope(plan(), history, "那远程岗位呢？")
    assert {item["column"]: item["value"] for item in filters(remote)} == {
        "country_clean": "United States", "title": "Software Engineer", "work_model": "Remote"}


def test_api_new_company_uses_full_dataset_without_old_country_or_role(monkeypatch):
    frame = pd.DataFrame({"country_clean": ["United States", "United Kingdom", "Malaysia"],
                          "title": ["Data Analyst", "Software Engineer", "Staff Engineer"],
                          "company_name": ["Bosch", "Bosch", "Western Digital"]})

    class FakeLLM:
        def create_plan(self, question, columns, tool_descriptions, history=None):
            assert history and history[0]["question"] == "美国的数据分析师有哪些？"
            return plan("Bosch", combined=True, extra=(
                {"column": "country_clean", "operator": "equals", "value": "United States"},
                {"column": "title", "operator": "contains", "value": "Data Analyst"},
            ))

        def create_answer(self, *args):
            return "Bosch 有两个岗位。"

    monkeypatch.setattr(routes, "LLMClient", FakeLLM)
    monkeypatch.setattr(routes, "load_agent_data", lambda: frame.copy())
    with TestClient(app) as client:
        body = client.post("/agent/ask", json={"question": "Bosch 有哪些岗位？",
                     "history": [{"question": "美国的数据分析师有哪些？", "answer": "找到岗位。"}]}).json()
    assert body["status"] == "completed"
    assert len(body["steps"][0]["result"]) == 2
    assert {item["country_clean"] for item in body["steps"][0]["result"]} == {"United States", "United Kingdom"}
    assert [item["column"] for item in filters(AgentPlan.model_validate(body["plan"]))] == ["company_name"]
