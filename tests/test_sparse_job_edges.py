import pandas as pd

from agent.answers import fallback_answer
from agent.runtime import run_plan
from agent.schemas import AgentPlan, ToolCall
from tools.analysis_tools import calculate_proportion, count_values, sample_text, summarize_salary
from tools.pandas_tools import count_skills


def sparse_jobs():
    return pd.DataFrame({
        "title": ["Analyst", "Analyst"],
        "company_name": ["Acme", "Acme"],
        "country_clean": ["United States", "United States"],
        "skills_required": ["not json", None],
        "salary_min": [None, None],
        "salary_max": [None, None],
        "salary_currency": [None, None],
        "salary_rate_unit": [None, None],
        "education_level": [None, None],
        "work_model": [None, None],
        "experience_level": [None, None],
        "years_experience_numeric": [None, None],
        "minimum_qualifications": [None, ""],
    })


def test_missing_company_and_empty_scope_keep_results_empty():
    plan = AgentPlan(question="不存在的公司有多少岗位？", status="ready", reasoning="筛选",
                     tool_calls=[
                         ToolCall(name="filter_rows", arguments={"column": "company_name", "keyword": "Absent Corp"}),
                         ToolCall(name="count_values", arguments={"column": "title"}),
                     ])
    filtered, ranking = run_plan(sparse_jobs(), plan)
    assert filtered.empty and ranking.empty
    assert "没有找到" in fallback_answer([{"result": ranking.to_dict("records")}])


def test_sparse_two_job_scope_keeps_unknowns_out_of_denominators():
    jobs = sparse_jobs()
    assert count_values(jobs, "education_level").empty
    assert count_values(jobs, "experience_level").empty
    assert count_values(jobs, "work_model").empty
    work = calculate_proportion(jobs, "work_model", "Remote")
    assert work["matched_records"] == 2
    assert work["denominator"] == 0 and work["unknown_records"] == 2
    assert work["percentage"] is None and work["status"] == "no_data"
    skills = count_skills(jobs)
    assert skills["matched_records"] == 2 and skills["valid_skill_records"] == 0
    assert skills["skills"] == []
    salary = summarize_salary(jobs, "USD", "year")
    assert salary["valid_records"] == 0 and salary["salary_min"]["median"] is None
    assert salary["status"] == "no_data"
    qualifications = sample_text(jobs, "minimum_qualifications")
    assert qualifications["available_records"] == 0 and qualifications["records"] == []


def test_missing_experience_years_returns_unknown_instead_of_zero():
    plan = AgentPlan(question="这两个岗位要几年经验？", status="ready", reasoning="核对年限",
                     tool_calls=[ToolCall(name="summarize_numeric",
                                          arguments={"column": "years_experience_numeric"})])
    result = run_plan(sparse_jobs(), plan)[0]
    assert result["status"] == "no_numeric_data"
    assert result["count"] == 0 and result["median"] is None
    assert "不能据此推断为零" in result["message"]
