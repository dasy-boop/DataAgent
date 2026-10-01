import pandas as pd
import pytest

from fastapi.testclient import TestClient
from backend import main, dataset_routes


def test_dataset_preview(monkeypatch):
    # 三条小型样例，其中第一条薪资缺失
    sample = pd.DataFrame({
        "title": ["Data Analyst", "Developer", "Tester"],
        "salary_min": [None, 90000.0, 70000.0],
    })

    # 测试期间，用样例数据代替读取真实文件
    def fake_load_parquet(path):
        return sample.copy()

    monkeypatch.setattr(dataset_routes, "load_parquet", fake_load_parquet)

    with TestClient(main.app) as client:
        response = client.get("/dataset/preview", params={"limit": 2})

    assert response.status_code == 200

    body = response.json()
    assert body["limit"] == 2
    assert body["rows"] == [
        {"title": "Data Analyst", "salary_min": None},
        {"title": "Developer", "salary_min": 90000.0},
    ]
@pytest.mark.parametrize("limit", [0, 51, "abc"])
def test_dataset_preview_invalid_limit(monkeypatch, limit):
    # 非法参数应在读取数据之前被拒绝
    def unexpected_load(path):
        pytest.fail("非法 limit 不应触发数据读取")

    monkeypatch.setattr(dataset_routes, "load_parquet", unexpected_load)

    with TestClient(main.app) as client:
        response = client.get(
            "/dataset/preview",
            params={"limit": limit},
        )

    assert response.status_code == 422