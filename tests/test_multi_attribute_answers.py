import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import agent.routes as routes
from agent.answers import build_evidence, protected_answer
from agent.schemas import AgentPlan
from backend.main import app


def dataset():
    degrees = ["Bachelor", "Bachelor’s", "B.Tech", "Master", "MSc", "PhD", "Doctorate", "Bachelor or Master", None]
    return pd.DataFrame({
        "country_clean": ["United States"] * 9 + ["United Kingdom", "United States"],
        "title": ["Data Scientist"] * 10 + ["Data Analyst"],
        "education_level": degrees + ["PhD", "PhD"],
        "years_experience_numeric": [None] * 9 + [20, 30],
        "experience_level": ["Mid"] * 5 + ["Senior"] * 6,
        "skills_required": ['["Python", "SQL", "python"]'] * 8 + [None, '["Rust"]', '["Java"]'],
        "work_model": ["Hybrid"] * 5 + ["Remote"] * 6,
    })


def plan_for(question, columns):
    calls = [{"name": "filter_rows", "arguments": {"conditions": [
        {"column": "country_clean", "operator": "equals", "value": "United States"},
        {"column": "title", "operator": "contains", "value": "Data Scientist"},
    ]}}]
    for column in columns:
        if column == "skills_required":
            calls.append({"name": "count_skills", "arguments": {"top_n": 10}})
        else:
            calls.append({"name": "summarize_numeric" if column == "years_experience_numeric" else "count_values",
                          "arguments": {"column": column}})
    return AgentPlan(question=question, reasoning="共同筛选后分别分析每个属性", tool_calls=calls)


def ask(monkeypatch, frame, question, columns, path="/agent/ask"):
    class FakeLLM:
        def create_plan(self, question, columns, tool_descriptions):
            return plan_for(question, requested_columns)

        def create_answer(self, *args):
            raise AssertionError("Structured multi-attribute results must remain complete without model text")

        def stream_answer(self, *args):
            raise AssertionError("Structured multi-attribute results must remain complete without model text")

    requested_columns = columns
    monkeypatch.setattr(routes, "LLMClient", FakeLLM)
    monkeypatch.setattr(routes, "load_agent_data", lambda: frame.copy())
    with TestClient(app) as client:
        response = client.post(path, json={"question": question})
    assert response.status_code == 200, response.text
    if path.endswith("stream"):
        return next(json.loads(line)["data"] for line in response.text.splitlines()
                    if json.loads(line)["type"] == "result")
    return response.json()


@pytest.mark.parametrize("path", ["/agent/ask", "/agent/ask/stream"])
@pytest.mark.parametrize("years_first", [False, True])
def test_country_role_education_and_experience_preserve_both_sections(monkeypatch, path, years_first):
    columns = ["education_level", "years_experience_numeric", "experience_level"]
    if years_first:
        columns = columns[1:] + columns[:1]
    body = ask(monkeypatch, dataset(), "美国的数据科学家岗位通常要求什么学历和工作经验？", columns, path)
    results = {s["arguments"].get("column"): s["result"] for s in body["steps"][1:]}
    education = {row["education_level"]: row["count"] for row in results["education_level"]}
    assert education == {"本科": 4, "硕士": 2, "博士": 2}
    assert sum(education.values()) == 8  # alternatives count once; missing stays missing
    assert results["years_experience_numeric"]["count"] == 0
    assert results["years_experience_numeric"]["status"] == "no_numeric_data"
    assert {r["experience_level"]: r["count"] for r in results["experience_level"]} == {"Mid": 5, "Senior": 4}
    assert len(body["steps"][0]["result"]) == 9
    assert "**学历要求**" in body["answer"] and "本科 4 个" in body["answer"]
    assert "**工作经验**" in body["answer"] and "不足以可靠判断通常要求几年经验" in body["answer"]
    assert "岗位级别不等同于具体工作年限" in body["answer"]
    assert "20" not in body["answer"] and "30" not in body["answer"]


@pytest.mark.parametrize("columns,headings", [
    (["education_level", "skills_required"], ["学历要求", "技能要求"]),
    (["work_model", "years_experience_numeric", "experience_level"], ["工作方式", "工作经验"]),
    (["education_level", "skills_required", "work_model"], ["学历要求", "技能要求", "工作方式"]),
])
def test_multiple_attribute_combinations_share_scope(monkeypatch, columns, headings):
    body = ask(monkeypatch, dataset(), "请分别分析" + "、".join(headings), columns)
    for heading in headings:
        assert f"**{heading}**" in body["answer"]
    assert len(body["steps"][0]["result"]) == 9
    for step in body["steps"][1:]:
        if step["tool"] == "count_skills":
            result = step["result"]
            assert (result["matched_records"], result["valid_skill_records"]) == (9, 8)
            assert [(r["skill"], r["岗位记录数"]) for r in result["skills"]] == [("python", 8), ("sql", 8)]
        if step["arguments"].get("column") == "work_model":
            assert {r["work_model"]: r["count"] for r in step["result"]} == {"Hybrid": 5, "Remote": 4}


def test_missing_education_does_not_suppress_skills(monkeypatch):
    frame = dataset()
    frame["education_level"] = None
    body = ask(monkeypatch, frame, "学历和技能要求", ["skills_required", "education_level"])
    assert "**学历要求**" in body["answer"] and "暂时无法分析这一项" in body["answer"]
    assert "**技能要求**" in body["answer"] and "python（8 个，100.0%）" in body["answer"]


def test_some_numeric_experience_still_uses_existing_sparsity_guard(monkeypatch):
    frame = dataset()
    frame.loc[0, "years_experience_numeric"] = 2
    body = ask(monkeypatch, frame, "学历和经验要求", ["education_level", "years_experience_numeric", "experience_level"])
    assert body["steps"][2]["result"]["median"] == 2.0
    assert "**学历要求**" in body["answer"]
    assert "不足以可靠判断通常要求几年经验" in body["answer"]
    assert "中位数为 2" not in body["answer"]


def test_sufficient_numeric_experience_keeps_original_measurement(monkeypatch):
    frame = pd.concat([dataset().iloc[[0]]] * 35, ignore_index=True)
    frame["years_experience_numeric"] = 4
    body = ask(monkeypatch, frame, "学历和经验要求", ["education_level", "years_experience_numeric"])
    assert body["steps"][2]["result"]["median"] == 4.0
    assert "本科 35 个" in body["answer"] and "中位数为 4.0 年" in body["answer"]


def test_single_attribute_guard_remains_unchanged():
    steps = [{"tool": "summarize_numeric", "arguments": {"column": "years_experience_numeric"},
              "result": {"count": 0, "median": None}}]
    answer = protected_answer(steps, build_evidence(steps))
    assert "不足以可靠判断通常要求几年经验" in answer
    assert "学历" not in answer
