import pandas as pd
import pytest

from tools.executor import execute_tool
from tools.pandas_tools import ToolError, count_skills


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
