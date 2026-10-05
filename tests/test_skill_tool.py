import pandas as pd
import pytest

from tools.executor import execute_tool
from tools.pandas_tools import ToolError, compare_skills, count_skills


def test_count_skills_uses_current_rows_and_deduplicates():
    jobs = pd.DataFrame({
        "skills_required": [
            '["SQL", "SQL", "Python"]',
            '["sql"]',
            None,
        ]
    })

    result = count_skills(jobs, top_n=2)

    assert result["matched_records"] == 3
    assert result["valid_skill_records"] == 2
    assert [(row["skill"], row["岗位记录数"]) for row in result["skills"]] == [
        ("sql", 2),
        ("python", 1),
    ]


def test_agent_executor_can_call_skill_tool():
    jobs = pd.DataFrame({
        "skills_required": ['["SQL", "Python"]', '["SQL"]']
    })

    result = execute_tool("count_skills", jobs, {"top_n": 1})

    assert result["matched_records"] == 2
    assert result["skills"][0]["skill"] == "sql"
    assert result["skills"][0]["岗位记录数"] == 2


def test_agent_executor_rejects_unexpected_skill_arguments():
    jobs = pd.DataFrame({"skills_required": ['["SQL"]']})

    with pytest.raises(ToolError, match="不支持参数"):
        execute_tool("count_skills", jobs, {"column": "title"})


def test_compare_skills_keeps_job_groups_separate():
    jobs = pd.DataFrame({
        "title": [
            "Data Analyst", "Senior Data Analyst",
            "Software Engineer", "Software Engineer",
        ],
        "skills_required": [
            '["SQL", "Python"]', '["SQL"]',
            '["Python", "Java"]', '["Java"]',
        ],
    })

    result = compare_skills(
        jobs, "title", "Data Analyst", "Software Engineer", top_n=2
    )

    analyst, engineer = result["groups"]
    assert analyst["matched_records"] == 2
    assert engineer["matched_records"] == 2
    assert analyst["skills"][0]["skill"] == "sql"
    assert analyst["skills"][0]["岗位记录数"] == 2
    assert engineer["skills"][0]["skill"] == "java"
    assert engineer["skills"][0]["岗位记录数"] == 2
