import pandas as pd

from agent.runtime import run_plan
from agent.schemas import AgentPlan


def test_run_plan_executes_tools_in_order():
    df = pd.DataFrame(
        {
            "title": ["Data Analyst", "Software Engineer"],
            "country": ["USA", "Canada"],
        }
    )

    plan = AgentPlan(
        question="统计 Data Analyst 岗位国家",
        reasoning="先筛选岗位，再统计国家",
        tool_calls=[
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
                    "column": "country",
                    "top_n": 10,
                },
            },
        ],
    )

    results = run_plan(df, plan)

    assert len(results) == 2
    assert results[1].iloc[0]["country"] == "USA"
    assert results[1].iloc[0]["count"] == 1