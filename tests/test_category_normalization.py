"""Regression checks for conservative user-facing category analysis."""
from pathlib import Path

import pandas as pd
import pytest

from agent.answers import build_evidence, protected_answer
from agent.routes import ExecuteRequest, execute_agent
from agent.schemas import AgentPlan
from tools.analysis_tools import count_values
from tools.category_normalization import (
    classify_visa, low_coverage, normalize_education, normalize_industries,
)


@pytest.mark.parametrize(("raw", "expected"), [
    ("Bachelor's", "本科"), ("Master's", "硕士"), ("PhD", "博士"),
    ("B.Tech", "本科"),
    ("Bachelor's degree required; Master's preferred", "本科"),
    ("Bachelor's or Master's", "本科"),
    ('["Bachelors", "Masters"]', "本科"),
    ("Master's or PhD in quantitative field", "硕士"),
    ("Unknown qualification", "无法明确判断"),
])
def test_education_minimum_rule(raw, expected):
    assert normalize_education(raw) == expected


def test_education_ranking_uses_categories_not_raw_lists():
    jobs = pd.DataFrame({"education_level": ["Bachelor's", '["Bachelors", "Masters"]',
                                             "Master's", "PhD"]})
    result = count_values(jobs, "education_level")
    assert result.to_dict("records") == [
        {"education_level": "本科", "count": 2},
        {"education_level": "硕士", "count": 1},
        {"education_level": "博士", "count": 1},
    ]


def test_industry_list_is_parsed_and_counted_per_job():
    assert normalize_industries('["Finance", "FinTech"]') == ["金融服务", "金融科技"]
    assert normalize_industries("['Finance', 'Finance']") == ["金融服务"]
    jobs = pd.DataFrame({"industry": ['["Finance", "FinTech"]', "Financial Services", None]})
    result = count_values(jobs, "industry")
    assert dict(zip(result.industry, result["count"])) == {"金融服务": 2, "金融科技": 1}


def test_sparse_industry_answer_is_guarded():
    rows = [{"industry": "Financial Services"}] * 11 + [{"industry": None}] * 158
    steps = [
        {"tool": "filter_rows", "arguments": {}, "result": rows},
        {"tool": "count_values", "arguments": {"column": "industry"},
         "result": [{"industry": "金融服务", "count": 11}]},
    ]
    evidence = build_evidence(steps)
    assert low_coverage(169, 11)
    assert evidence[-1]["low_coverage"] is True
    answer = protected_answer(steps, evidence)
    assert "11 个提供了行业信息" in answer
    assert "不足以可靠判断整体行业分布" in answer


def test_unfiltered_industry_coverage_is_reported(monkeypatch):
    jobs = pd.DataFrame({"industry": ["Financial Services"] + [None] * 99})
    monkeypatch.setattr("agent.routes.load_agent_data", lambda: jobs)
    plan = AgentPlan(question="行业", reasoning="统计", tool_calls=[
        {"name": "count_values", "arguments": {"column": "industry"}}
    ])
    execution = execute_agent(ExecuteRequest(plan=plan))
    assert execution["steps"][0]["coverage"] == {
        "matched_records": 100, "available_records": 1, "low_coverage": True,
    }
    assert "不足以可靠判断整体行业分布" in protected_answer(
        execution["steps"], build_evidence(execution["steps"]))


@pytest.mark.parametrize(("raw", "expected"), [
    (None, "未提供相关信息"), ("US", "信息不明确"), ("WW", "信息不明确"),
    ("Not available", "信息不明确"), ("No sponsorship required", "信息不明确"),
    ("Canadian work authorization required", "要求已有当地工作许可或身份"),
    ("US citizen required", "要求已有当地工作许可或身份"),
    ("Yes", "明确支持签证"), ("No", "明确不提供签证支持"),
    ("H1B sponsorship available", "明确支持签证"),
    ("TN sponsorship available", "明确支持签证"),
    ("No visa sponsorship available", "明确不提供签证支持"),
])
def test_visa_support_is_not_confused_with_work_permission(raw, expected):
    assert classify_visa(raw) == expected


def test_visa_missing_is_not_no_sponsorship():
    jobs = pd.DataFrame({"visa_sponsorship_available": ["Yes", "No", "US", None]})
    result = count_values(jobs, "visa_sponsorship_available")
    counts = dict(zip(result.visa_sponsorship_available, result["count"]))
    assert counts["未提供相关信息"] == 1
    assert counts["明确不提供签证支持"] == 1
    assert counts["信息不明确"] == 1


def test_aggregate_uses_all_jobs_while_text_evidence_is_bounded():
    rows = [{"work_model": "On-site", "title": "Data Scientist"}] * 20
    steps = [
        {"tool": "filter_rows", "arguments": {}, "result": rows},
        {"tool": "count_values", "arguments": {"column": "work_model"},
         "result": count_values(pd.DataFrame(rows), "work_model").to_dict("records")},
    ]
    evidence = build_evidence(steps)
    assert evidence[0]["shown_rows"] == 8
    assert evidence[0]["total_rows"] == 20
    assert evidence[1]["available_records"] == 20
    assert evidence[1]["result"][0]["count"] == 20


def test_sparse_numeric_experience_uses_level_distribution_without_claiming_typical_years():
    steps = [
        {"tool": "count_values", "arguments": {"column": "experience_level"},
         "coverage": {"matched_records": 169, "available_records": 143, "low_coverage": False},
         "result": [{"experience_level": "Mid", "count": 84},
                    {"experience_level": "Senior", "count": 54}]},
        {"tool": "summarize_numeric", "arguments": {"column": "years_experience_numeric"},
         "result": {"count": 2, "median": 2.0}},
    ]
    answer = protected_answer(steps, build_evidence(steps))
    assert "不足以可靠判断通常要求几年经验" in answer
    assert "143 个岗位写明了级别" in answer
    assert "中级 84 个" in answer
    assert "通常要求2年" not in answer


def test_frontend_has_chinese_work_and_experience_labels():
    html = (Path(__file__).resolve().parents[1] / "frontend/agent.html").read_text(encoding="utf-8")
    for label in ("现场办公", "混合办公", "远程办公", "中级", "高级", "入门级", "初级", "实习"):
        assert label in html
    assert "图表使用本次筛选后所有具有相关信息的岗位" in html
