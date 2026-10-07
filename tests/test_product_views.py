import pandas as pd
from fastapi.testclient import TestClient

from backend import dataset_routes, main
from tools.pandas_tools import filter_rows


def sample_jobs():
    return pd.DataFrame({
        "title": ["Data Analyst", "Data Analyst", "Software Engineer"],
        "company_name": ["Acme", "Acme", "Other"],
        "country": ["UK", "United Kingdom", "Canada"],
        "skills_required": ['["SQL", "Python"]', '["SQL"]', '["Java"]'],
        "salary_min": [None, 50000.0, None],
        "salary_max": [None, None, None],
        "work_model": ["Remote", "Hybrid", "On-site"],
        "experience_level": ["Entry", "Mid", None],
    })


def test_overview_uses_dataframe_counts_quality_and_short_preview(monkeypatch):
    jobs = sample_jobs()
    monkeypatch.setattr(dataset_routes, "load_parquet", lambda _: jobs.copy())

    with TestClient(main.app) as client:
        response = client.get("/dataset/overview")

    assert response.status_code == 200
    data = response.json()
    assert data["rows"] == 3
    assert data["columns"] == len(jobs.columns)
    assert data["countries"] == 2  # UK and United Kingdom are one country
    assert data["companies"] == 2
    assert data["display_name"] == "2026年6月全球招聘岗位数据"
    assert data["quality_rates"][-1] == {"label": "薪资信息", "rate": 33.33}
    assert len(data["preview"]) == 3
    assert set(data["preview"][0]) == {
        "title", "company_name", "country", "work_model", "experience_level"
    }
    assert "dataset" in data  # keep the existing API field for callers


def test_dashboard_charts_use_existing_statistics(monkeypatch):
    monkeypatch.setattr(dataset_routes, "load_parquet", lambda _: sample_jobs())

    with TestClient(main.app) as client:
        response = client.get("/dataset/dashboard")

    assert response.status_code == 200
    charts = {chart["kind"]: chart for chart in response.json()["charts"]}
    assert set(charts) == {"country", "company", "title", "skill"}
    assert charts["country"]["items"][0] == {"label": "United Kingdom", "count": 2}
    assert charts["company"]["items"][0] == {"label": "Acme", "count": 2}
    assert charts["skill"]["items"][0] == {"label": "sql", "count": 2}


def test_dashboard_prefers_normalized_title_and_excludes_employment_labels(monkeypatch):
    jobs = sample_jobs()
    jobs["title"] = ["Full-time", "Data Analyst I", "Data Analyst II"]
    jobs["normalized_title"] = ["Full-time", "Data Analyst", "Data Analyst"]
    monkeypatch.setattr(dataset_routes, "load_parquet", lambda _: jobs.copy())

    with TestClient(main.app) as client:
        response = client.get("/dataset/dashboard")

    assert response.status_code == 200
    titles = next(chart for chart in response.json()["charts"] if chart["kind"] == "title")
    assert titles["items"] == [{"label": "Data Analyst", "count": 2}]
    assert "标准岗位名称" in titles["note"]


def test_chart_entry_is_served_and_display_mapping_does_not_change_filtering():
    with TestClient(main.app) as client:
        response = client.get("/agent.html")
    assert response.status_code == 200
    html = response.text
    assert 'href="#charts" id="chartsNav"' in html
    assert 'fetch("/dataset/dashboard")' in html
    assert "function jobTitle(value)" in html
    assert '"data analyst":"数据分析师"' in html
    assert "国家别名归一统计" not in html

    # Chinese labels are a presentation choice; filtering still uses original values.
    selected = filter_rows(sample_jobs(), "title", "Data Analyst")
    assert selected["title"].tolist() == ["Data Analyst", "Data Analyst"]
