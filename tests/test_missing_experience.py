import pandas as pd
import pytest

from agent.runtime import run_plan
from agent.schemas import AgentPlan
from agent.answers import build_evidence
from tools.pandas_tools import ToolError


def test_missing_experience_preserves_text_and_later_steps():
    df = pd.DataFrame({"company_name": ["Western Digital"], "title": ["Analyst"],
                       "years_experience_numeric": [None],
                       "minimum_qualifications": ["Minimum 3 years of experience"]})
    calls = [
        {"name": "filter_rows", "arguments": {"column": "company_name", "keyword": "Western Digital"}},
        {"name": "summarize_numeric", "arguments": {"column": "years_experience_numeric"}},
        {"name": "count_values", "arguments": {"column": "title"}},
    ]
    result = run_plan(df, AgentPlan(question="应届生", reasoning="检查经验", tool_calls=calls))
    assert result[1]["status"] == "no_numeric_data"
    assert result[1]["count"] == 0
    assert result[1]["mean"] is None
    assert result[2].iloc[0]["count"] == 1
    evidence = build_evidence([{"tool": "filter_rows", "arguments": {}, "result": result[0].to_dict("records")}])
    assert evidence[0]["result"][0]["minimum_qualifications"] == "Minimum 3 years of experience"
    assert evidence[0]["result"][0]["years_experience_numeric"] is None


def test_nonexistent_field_is_still_an_error():
    plan = AgentPlan(question="test", reasoning="test", tool_calls=[
        {"name": "summarize_numeric", "arguments": {"column": "typo"}}
    ])
    with pytest.raises(ToolError, match="字段不存在"):
        run_plan(pd.DataFrame({"title": ["Analyst"]}), plan)


def test_evidence_budget_keeps_partial_records_instead_of_dropping_all():
    rows = [{"title": "Analyst", "minimum_qualifications": "x"*3000,
             "job_description": "y"*3000, "responsibilities": "z"*3000} for _ in range(8)]
    evidence = build_evidence([{"tool": "filter_rows", "arguments": {}, "result": rows}])
    assert 0 < evidence[0]["shown_rows"] < 8
    assert evidence[0]["sampled"] is True
