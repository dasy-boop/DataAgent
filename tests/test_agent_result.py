import pandas as pd

from agent.runtime import run_plan
from agent.schemas import AgentPlan


def test_filter_then_count_uses_filtered_data():
    df = pd.DataFrame({
        "title": [
            "Data Analyst",
            "Senior Data Analyst",
            "Data Analyst",
            "Software Engineer",
            "Software Engineer",
            "Software Engineer",
        ],
        "country_clean": [
            "United States",
            "United States",
            "Canada",
            "Canada",
            "Canada",
            "Canada",
        ],
    })

    plan = AgentPlan.model_validate({
        "question": "统计 Data Analyst 岗位的国家分布",
        "reasoning": "先筛选岗位，再统计筛选结果中的国家",
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

    results = run_plan(df, plan)

    assert len(results) == 2
    assert len(results[0]) == 3
    assert results[-1].to_dict(orient="records") == [
        {"country_clean": "United States", "count": 2},
        {"country_clean": "Canada", "count": 1},
    ]
def test_no_matches_returns_empty_result():
    df = pd.DataFrame({
        "title": ["Software Engineer", "Product Manager"],
        "country_clean": ["United States", "Canada"],
    })

    plan = AgentPlan.model_validate({
        "question": "统计 Data Analyst 岗位的国家分布",
        "reasoning": "先筛选，再统计；无匹配记录时返回空结果",
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

    results = run_plan(df, plan)

    assert len(results) == 2
    assert results[0].empty
    assert results[-1].empty