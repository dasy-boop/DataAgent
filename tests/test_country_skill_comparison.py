import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import agent.routes as routes
from agent.answers import build_evidence, protected_answer
from agent.schemas import AgentPlan
from backend.main import app
from tools.executor import execute_tool
from tools.pandas_tools import ToolError


def arguments(role="Data Analyst", countries=("美国", "英国"), skill="Python"):
    args = {"groups": [{"label": f"{country} {role}", "conditions": [
        {"column": "country_clean", "operator": "equals", "value": country},
        {"column": "title", "operator": "contains", "value": role},
    ]} for country in countries], "top_n": 1}
    if skill is not None:
        args["skill"] = skill
    return args


def jobs():
    return pd.DataFrame({
        "country_clean": ["US", "USA", "United States", "United States", "UK", "United Kingdom", "U.K.", "Canada", "US"],
        "title": ["Data Analyst"] * 8 + ["Software Engineer"],
        "skills_required": ['["Python", "python", "SQL"]', '["SQL"]', '["SQL"]', None,
                            '["Python", "SQL"]', '["Python"]', 'broken', '["Python"]', '["Python"]'],
    })


def test_independent_countries_share_role_and_use_own_skill_denominators():
    frame = jobs()
    before = frame.copy(deep=True)
    result = execute_tool("compare_skills", frame, arguments())
    us, uk = result["groups"]
    assert (us["matched_records"], us["valid_skill_records"]) == (4, 3)
    assert (uk["matched_records"], uk["valid_skill_records"]) == (3, 2)
    assert us["target_skill"] == {"skill": "python", "numerator": 1, "denominator": 3, "percentage": 33.33}
    assert uk["target_skill"] == {"skill": "python", "numerator": 2, "denominator": 2, "percentage": 100.0}
    # Python is not US top-1, but must still be counted; duplicate mentions count once.
    assert us["skills"][0]["岗位记录数"] == 1
    assert result["comparison"]["winner"] == "英国 Data Analyst"
    assert result["comparison"]["tied"] is False
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("role,countries,skill", [
    ("Data Scientist", ("美国", "印度"), "SQL"),
    ("Software Engineer", ("英国", "加拿大"), None),
])
def test_other_countries_roles_and_whole_rankings(role, countries, skill):
    country_values = {"美国": "US", "印度": "India", "英国": "UK", "加拿大": "Canada"}
    frame = pd.DataFrame({"country_clean": [country_values[c] for c in countries],
                          "title": [role, role], "skills_required": ['["SQL"]', '["Python"]']})
    result = execute_tool("compare_skills", frame, arguments(role, countries, skill))
    assert [g["valid_skill_records"] for g in result["groups"]] == [1, 1]
    if skill:
        assert [g["target_skill"]["percentage"] for g in result["groups"]] == [100.0, 0.0]
    else:
        assert [g["skills"][0]["skill"] for g in result["groups"]] == ["sql", "python"]


@pytest.mark.parametrize("missing_group", ["no_jobs", "invalid_skills"])
def test_unavailable_group_has_no_percentage_or_winner(missing_group):
    frame = jobs()
    mask = frame["country_clean"].isin(["UK", "United Kingdom", "U.K."])
    if missing_group == "no_jobs":
        frame = frame.loc[~mask]
    else:
        frame.loc[mask, "skills_required"] = None
    args = arguments()
    result = execute_tool("compare_skills", frame, args)
    assert result["status"] == "insufficient_data"
    assert result["groups"][1]["target_skill"]["percentage"] is None
    assert result["comparison"] == {"skill": "python", "winner": None, "tied": None}
    steps = [{"tool": "compare_skills", "arguments": args, "result": result}]
    answer = protected_answer(steps, build_evidence(steps))
    assert "样本不足" in answer and "无法比较" in answer
    assert "比例更高" not in answer


def test_absent_skill_is_zero_with_usable_denominator_and_ties():
    result = execute_tool("compare_skills", jobs(), arguments(skill="Rust"))
    assert result["status"] == "ok"
    assert [g["target_skill"]["percentage"] for g in result["groups"]] == [0.0, 0.0]
    assert result["comparison"]["tied"] is True


@pytest.mark.parametrize("override", [
    {"groups": []}, {"groups": [{"label": "A", "conditions": []}] * 2},
    {"column": "title", "keyword_a": "A", "keyword_b": "B"}, {"skill": " "},
    {"groups": [{"label": "A", "conditions": [{"column": "not_a_field", "operator": "equals", "value": "x"}]},
                {"label": "B", "conditions": []}]},
])
def test_malformed_group_arguments_are_rejected(override):
    with pytest.raises(ToolError):
        execute_tool("compare_skills", jobs(), {**arguments(), **override})


@pytest.mark.parametrize("path", ["/agent/ask", "/agent/ask/stream"])
def test_country_comparison_api_exposes_groups_and_generates_tool_based_conclusion(monkeypatch, path):
    class FakeLLM:
        def create_plan(self, question, columns, tool_descriptions, **kwargs):
            spec = next(t for t in tool_descriptions if t["name"] == "compare_skills")
            assert {"required": ["groups"]} in spec["parameters"]["oneOf"]
            assert "skill" in spec["parameters"]["properties"]
            return AgentPlan(question=question, reasoning="分别按国家和岗位筛选", tool_calls=[
                {"name": "compare_skills", "arguments": arguments()}])

        def create_answer(self, *args):
            raise AssertionError("The comparison conclusion should come from computed ratios")

        def stream_answer(self, *args):
            raise AssertionError("The comparison conclusion should come from computed ratios")

    monkeypatch.setattr(routes, "load_agent_data", jobs)
    monkeypatch.setattr(routes, "LLMClient", FakeLLM)
    with TestClient(app) as client:
        response = client.post(path, json={"question": "美国的数据分析师和英国的数据分析师哪个更常要求 Python？"})
    assert response.status_code == 200
    body = response.json() if path.endswith("ask") else next(
        json.loads(line)["data"] for line in response.text.splitlines() if json.loads(line)["type"] == "result")
    assert body["answer_source"] == "data_guard"
    assert "英国 Data Analyst要求 python 的比例更高" in body["answer"]
    assert "33.33%" in body["answer"] and "100.0%" in body["answer"]
    assert [g["matched_records"] for g in body["steps"][-1]["result"]["groups"]] == [4, 3]
