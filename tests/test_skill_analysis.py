import pandas as pd
import pytest

from backend.data_loader import load_parquet
from backend.skill_analysis import summarize_skills
from backend.skill_cleaning import parse_skills
from backend.data_quality import summarize_quality


def test_skill_ranking():
    # 构造 5 条记录：只有前 3 条技能解析成功
    jobs = pd.DataFrame({
        "skills_parse_status": [
            "ok", "ok", "ok", "missing", "invalid_json",
        ],
        "skills_normalized": [
            ["sql", "python"],
            ["sql"],
            ["python"],
            [],
            [],
        ],
    })

    ranking, valid_count = summarize_skills(jobs)

    # 分母应为 3，而不是全部 5 条
    assert valid_count == 3

    # 两项技能均出现两次，并列时按名称升序排列
    assert ranking.index.tolist() == ["python", "sql"]
    assert ranking["岗位记录数"].tolist() == [2, 2]

    # 2 / 3 × 100，保留两位小数
    assert ranking.loc["sql", "占可解析记录比例(%)"] == pytest.approx(66.67)
def test_skill_ranking_with_no_valid_jobs():
    jobs = pd.DataFrame({
        "skills_parse_status": ["missing", "invalid_json"],
        "skills_normalized": [[], []],
    })

    ranking, valid_count = summarize_skills(jobs)

    assert valid_count == 0
    assert ranking.empty
def test_skill_ranking_with_duplicate_skill_in_one_job():
    jobs = pd.DataFrame({
        "skills_parse_status": ["ok", "ok"],
        "skills_normalized": [
            ["sql", "sql", "python"],
            ["sql"],
        ],
    })

    ranking, valid_count = summarize_skills(jobs)

    assert valid_count == 2
    assert ranking.loc["sql", "岗位记录数"] == 2
    assert ranking.loc["python", "岗位记录数"] == 1
@pytest.mark.parametrize(
    "raw_value, expected_skills, expected_status",
    [
        ('["SQL", " Python "]', ["SQL", "Python"], "ok"),
        (None, [], "missing"),
        ("", [], "missing"),
        ("SQL,Python", [], "invalid_json"),
        ("0", [], "invalid_structure"),
    ],
)
def test_parse_skills_cases(
    raw_value,
    expected_skills,
    expected_status,
):
    skills, status = parse_skills(raw_value)

    assert skills == expected_skills
    assert status == expected_status
def test_load_parquet(tmp_path):
    source = pd.DataFrame({
        "title": ["Data Analyst", "Python Developer"],
        "salary_min": [80000, 90000],
    })

    data_path = tmp_path / "sample.parquet"
    source.to_parquet(data_path, index=False)

    loaded = load_parquet(data_path)

    pd.testing.assert_frame_equal(loaded, source)


def test_load_parquet_missing_file(tmp_path):
    missing_path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match="数据文件不存在"):
        load_parquet(missing_path)
def test_summarize_quality():
    data = pd.DataFrame({
        "name": ["A", "B", "C", "D"],
        "score": [90, None, 80, None],
    })

    summary = summarize_quality(data)

    assert summary.loc["score", "非空数量"] == 2
    assert summary.loc["score", "缺失数量"] == 2
    assert summary.loc["score", "缺失率(%)"] == 50.0
    assert summary.index[0] == "score"