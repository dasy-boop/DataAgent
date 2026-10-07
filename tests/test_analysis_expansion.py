"""Regression coverage against the committed June 2026 job snapshot."""
from pathlib import Path

import pandas as pd
import pytest

from agent.context_scope import enforce_scope
from agent.llm_client import SYSTEM_PROMPT
from agent.schemas import AgentPlan, ToolCall
from backend.field_metadata import FIELD_SPECS
from tools.analysis_tools import period_name
from tools.executor import execute_tool
from tools.pandas_tools import ToolError


DATA_PATH = Path(__file__).resolve().parents[1] / "data/raw/nextgig_jobs_2026-06.parquet"


@pytest.fixture(scope="module")
def jobs():
    return pd.read_parquet(DATA_PATH)


def call(name, jobs, **arguments):
    return execute_tool(name, jobs, arguments)


def test_metadata_covers_every_raw_field(jobs):
    assert len(jobs) == 112816
    assert len(jobs.columns) == 47
    assert set(jobs.columns) <= FIELD_SPECS.keys()
    assert "work_model" in jobs and "work_mode" not in jobs


def test_education_frequency_uses_real_field(jobs):
    result = call("count_values", jobs, column="education_level", top_n=5)
    assert not result.empty
    assert result["count"].sum() <= jobs["education_level"].notna().sum()
    assert "本科" in result["education_level"].tolist()
    assert "Bachelor's degree" not in result["education_level"].tolist()


def test_years_experience_uses_only_numeric_records(jobs):
    result = call("summarize_numeric", jobs, column="years_experience_numeric")
    assert result["count"] == pd.to_numeric(jobs["years_experience_numeric"], errors="coerce").count()
    assert result["count"] < 300
    assert result["median"] is not None


def test_work_model_frequency(jobs):
    result = call("count_values", jobs, column="work_model", top_n=10)
    assert "Remote" in result["work_model"].tolist()
    assert result["count"].sum() <= len(jobs)


def test_industry_ranking_marks_sparse_values(jobs):
    result = call("count_values", jobs, column="industry", top_n=10)
    assert not result.empty
    assert jobs["industry"].notna().sum() < len(jobs) // 10


def test_proportion_has_known_denominator(jobs):
    result = call("calculate_proportion", jobs, column="work_model", value="Remote")
    assert result["numerator"] > 0
    assert result["denominator"] > result["numerator"]
    assert result["denominator"] + result["unknown_records"] == len(jobs)


def test_visa_proportion_excludes_ambiguous_values(jobs):
    result = call("calculate_proportion", jobs, column="visa_sponsorship_available", value="Yes")
    assert result["denominator"] > 0
    assert result["denominator"] < jobs["visa_sponsorship_available"].notna().sum()
    assert result["unknown_records"] > 0


def test_combined_uk_remote_analyst_filter(jobs):
    from backend.country_cleaning import normalize_country
    jobs = jobs.assign(country_clean=normalize_country(jobs["country"]))
    selected = call("filter_rows", jobs, conditions=[
        {"column": "country_clean", "operator": "equals", "value": "英国"},
        {"column": "work_model", "operator": "equals", "value": "Remote"},
        {"column": "title", "operator": "contains", "value": "数据分析师"},
    ])
    assert all(selected["country_clean"] == "United Kingdom")
    assert all(selected["title"].str.contains("Data Analyst", case=False, regex=False))
    assert all(selected["work_model"].astype(str).str.casefold() == "remote")


@pytest.mark.parametrize("raw", ["year", "YEAR", "yearly", "Annual", "per annum", "Yr", "年薪"])
def test_real_year_period_aliases(raw):
    assert period_name(raw) == "year"


def test_salary_currency_isolation(jobs):
    usd = call("summarize_salary", jobs, currency="USD", period="year")
    gbp = call("summarize_salary", jobs, currency="GBP", period="year")
    assert usd["matched_records"] > 0
    assert gbp["matched_records"] > 0
    assert usd["currency"] == "USD" and gbp["currency"] == "GBP"
    assert usd["salary_min"]["median"] != gbp["salary_min"]["median"]


def test_annual_salary_no_longer_returns_zero(jobs):
    result = call("summarize_salary", jobs, currency="USD", period="年薪")
    assert result["status"] == "ok"
    assert result["matched_records"] > 17000
    assert result["valid_records"] > 17000
    assert result["salary_max"]["median"] is not None


def test_salary_without_period_cannot_use_generic_numeric(jobs):
    with pytest.raises(ToolError, match="薪资"):
        call("summarize_numeric", jobs, column="salary_max")
    with pytest.raises(ToolError):
        call("summarize_salary", jobs, currency="USD", period="")


def test_group_comparison_calculates_each_group(jobs):
    result = call("compare_groups", jobs, group_column="title", value_a="Data Analyst",
                  value_b="Data Scientist", metric_column="education_level", top_n=5)
    a, b = result["groups"]
    assert a["matched_records"] > 0 and b["matched_records"] > 0
    assert a["ranking"] and b["ranking"]
    assert all(row["count"] <= group["valid_metric_records"]
               for group in (a, b) for row in group["ranking"])


def test_title_ranking_omits_only_non_job_labels(jobs):
    result = call("count_values", jobs, column="title", top_n=20)
    assert "Full-time" not in result["title"].tolist()
    assert not result.empty


def test_bounded_samples_and_dates(jobs):
    sample = call("sample_text", jobs, column="minimum_qualifications", limit=4)
    assert 0 < sample["shown_records"] <= 4
    assert all(len(row["text"]) <= 1200 for row in sample["records"])
    dates = call("summarize_dates", jobs, column="date_posted", top_n=5)
    assert dates["valid_records"] > 0
    assert dates["valid_records"] + dates["unknown_records"] == len(jobs)


def _plan():
    return AgentPlan(question="追问", status="ready", reasoning="分析", tool_calls=[
        ToolCall(name="filter_rows", arguments={"column": "title", "keyword": "Data Scientist"}),
        ToolCall(name="count_skills", arguments={"top_n": 5}),
    ])


def _filter_values(plan):
    values = []
    for step in plan.tool_calls:
        if step.name == "filter_rows":
            values.extend((c["column"], c["value"]) for c in step.arguments.get("conditions", []))
            if "column" in step.arguments:
                values.append((step.arguments["column"], step.arguments["keyword"]))
    return values


def test_followup_inherits_country_when_role_changes():
    plan = enforce_scope(_plan(), [{"question": "英国的数据分析师技能有哪些？"}], "那数据科学家呢？")
    assert ("country_clean", "United Kingdom") in _filter_values(plan)
    assert ("title", "Data Scientist") in _filter_values(plan)


def test_followup_replaces_country_but_keeps_role():
    plan = enforce_scope(_plan(), [
        {"question": "英国的数据分析师技能有哪些？"},
        {"question": "那数据科学家呢？"}], "那美国呢？")
    assert ("country_clean", "United States") in _filter_values(plan)
    assert ("country_clean", "United Kingdom") not in _filter_values(plan)
    assert ("title", "Data Scientist") in _filter_values(plan)


def test_followup_explicitly_clears_country():
    plan = enforce_scope(_plan(), [{"question": "英国的数据分析师技能有哪些？"}], "全球的数据科学家呢？")
    assert not any(column == "country_clean" for column, _ in _filter_values(plan))
    assert ("title", "Data Scientist") in _filter_values(plan)


def test_two_country_comparison_does_not_keep_old_country():
    plan = enforce_scope(_plan(), [{"question": "英国的数据科学家有哪些岗位？"}],
                         "英国和美国的数据科学家工作方式有什么区别？")
    assert not any(column == "country_clean" for column, _ in _filter_values(plan))
    assert ("title", "Data Scientist") in _filter_values(plan)


def test_empty_result_is_explicit(jobs):
    empty = jobs.iloc[:0]
    result = call("calculate_proportion", empty, column="work_model", value="Remote")
    assert result["status"] == "no_data" and result["percentage"] is None
    salary = call("summarize_salary", empty, currency="USD", period="year")
    assert salary["status"] == "no_data" and salary["salary_max"]["median"] is None


def test_future_salary_prediction_remains_prohibited():
    assert "不能预测明年薪资" in SYSTEM_PROMPT
