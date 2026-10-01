import pandas as pd
from fastapi.testclient import TestClient

from backend import main, analysis_routes


def test_analysis_skills(monkeypatch):
    # 前三条标题匹配，第四条不匹配
    sample = pd.DataFrame({
        "title": [
            "Data Analyst",
            "Senior Data Analyst",
            "Data Analyst Intern",
            "Software Engineer",
        ],
        "skills_required": [
            '["SQL", "sql", "Python"]',
            '["SQL"]',
            None,
            '["Java"]',
        ],
    })

    def fake_load_parquet(path):
        return sample.copy()

    monkeypatch.setattr(
        analysis_routes, "load_parquet", fake_load_parquet
    )

    with TestClient(main.app) as client:
        response = client.get(
            "/analysis/skills",
            params={"keyword": "data analyst", "top_n": 10},
        )

    assert response.status_code == 200

    body = response.json()
    assert body["matched_records"] == 3
    assert body["valid_skill_records"] == 2

    assert body["skills"] == [
        {
            "skill": "sql",
            "岗位记录数": 2,
            "占可解析记录比例(%)": 100.0,
        },
        {
            "skill": "python",
            "岗位记录数": 1,
            "占可解析记录比例(%)": 50.0,
        },
    ]
def test_analysis_skills_no_match(monkeypatch):
    sample = pd.DataFrame({
        "title": ["Software Engineer"],
        "skills_required": ['["Java"]'],
    })

    def fake_load_parquet(path):
        return sample.copy()

    monkeypatch.setattr(
        analysis_routes, "load_parquet", fake_load_parquet
    )

    with TestClient(main.app) as client:
        response = client.get(
            "/analysis/skills",
            params={"keyword": "Data Analyst"},
        )

    assert response.status_code == 200

    body = response.json()
    assert body["matched_records"] == 0
    assert body["valid_skill_records"] == 0
    assert body["skills"] == []
import pytest


@pytest.mark.parametrize(
    "params",
    [
        {"keyword": "Data Analyst", "top_n": 0},
        {"keyword": "Data Analyst", "top_n": 51},
        {"keyword": "   "},
    ],
)
def test_analysis_skills_invalid_params(params):
    with TestClient(main.app) as client:
        response = client.get(
            "/analysis/skills",
            params=params,
        )

    assert response.status_code == 422