from pathlib import Path
from backend.skill_cleaning import normalize_skills, parse_skills
from backend.skill_analysis import summarize_skills
from backend.data_loader import load_parquet
from backend.data_quality import summarize_quality

import json
import pandas as pd



# 根据脚本位置确定项目目录
project_dir = Path(__file__).resolve().parent

# 拼接原始数据文件的路径
data_path = project_dir / "data" / "raw" / "nextgig_jobs_2026-06.parquet"

# 将文件读取为 Pandas 表格
df = load_parquet(data_path)

print("数据行数：", df.shape[0])
print("字段数量：", df.shape[1])

print("\n字段列表：")
for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")
    # 汇总每个字段的数据类型和缺失情况
field_summary = summarize_quality(df)
# 终端只显示缺失率最高的 10 个字段
print("\n缺失率最高的 10 个字段：")
print(field_summary.head(10).to_string())

print("\n字段列表：")
for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")
core_columns = [
    "title", "company_name", "skills_required",
    "experience_level", "education_level", "work_model",
    "salary_min", "salary_max", "salary_currency",
    "salary_rate_unit", "country", "date_posted",
]

print("\n核心字段缺失情况：")
print(
    field_summary.loc[
        core_columns, ["非空数量", "缺失率(%)"]
    ].to_string()
)
# 查看三个分类字段中最常出现的值
for column in ["salary_rate_unit", "salary_currency", "work_model"]:
    print(f"\n{column}：出现次数最多的 10 个值")
    counts = df[column].value_counts(dropna=False)
    print(counts.head(10).to_string())

# 检查同一条岗位记录是否同时具备四个薪资字段
salary_columns = [
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_rate_unit",
]

complete_salary = df[salary_columns].notna().all(axis=1)

print("\n四个薪资字段均非空的岗位数：", int(complete_salary.sum()))
print("占全部岗位比例：", f"{complete_salary.mean():.2%}")
# 将原始写法映射为统一名称
work_model_mapping = {
    "On-site": "On-site",
    "Onsite": "On-site",
    "ON_SITE": "On-site",
    "On‑site": "On-site",
    "On-Site": "On-site",
    "Hybrid": "Hybrid",
    "Remote": "Remote",
}

# 新建清洗后的字段，保留原始 work_model
df["work_model_clean"] = (
    df["work_model"]
    .astype("string")
    .str.strip()
    .map(work_model_mapping)
    .fillna("Unknown")
)

print("\n统一后的办公模式：")
print(df["work_model_clean"].value_counts().to_string())
# 先统一大小写、去掉前后空格，保留原始字段
salary_unit_normalized = (
    df["salary_rate_unit"]
    .astype("string")
    .str.strip()
    .str.lower()
)

print("\n初步规范化后的全部薪资周期：")
print(
    salary_unit_normalized
    .value_counts(dropna=False)
    .to_string()
)
# 每个标准周期对应的已确认写法
salary_unit_groups = {
    "hour": ["hour", "hourly", "hr"],
    "year": [
        "year", "yearly", "annual", "annually",
        "per annum", "yr", "annum", "annuelle", "p.a.",
    ],
    "month": ["month", "monthly"],
    "week": ["week", "weekly"],
    "day": ["day", "daily"],
}

# 转换为“原始写法 → 标准周期”的映射
salary_unit_mapping = {}
for standard_unit, aliases in salary_unit_groups.items():
    for alias in aliases:
        salary_unit_mapping[alias] = standard_unit

# 有内容但暂未纳入标准分类的值，标记为 Other
df["salary_rate_unit_clean"] = (
    salary_unit_normalized.map(salary_unit_mapping).fillna("Other")
)

# 将缺失值和无意义占位符单独标记为 Unknown
unknown_mask = (
    salary_unit_normalized.isna()
    | salary_unit_normalized.isin(["", "na", "..."])
)
df.loc[unknown_mask, "salary_rate_unit_clean"] = "Unknown"

print("\n标准化后的薪资周期：")
print(df["salary_rate_unit_clean"].value_counts().to_string())
salary_min = df["salary_min"]
salary_max = df["salary_max"]

# 两个金额都存在时，才能检查上下界顺序
both_present = salary_min.notna() & salary_max.notna()

salary_checks = pd.Series({
    "最低薪资缺失": salary_min.isna().sum(),
    "最高薪资缺失": salary_max.isna().sum(),
    "两个金额都缺失": (
        salary_min.isna() & salary_max.isna()
    ).sum(),
    "任一金额为负数": (
        (salary_min < 0) | (salary_max < 0)
    ).sum(),
    "任一金额为零": (
        (salary_min == 0) | (salary_max == 0)
    ).sum(),
    "最低薪资大于最高薪资": (
        both_present & (salary_min > salary_max)
    ).sum(),
})

print("\n薪资金额检查：")
print(salary_checks.to_string())
# 找出两个金额均存在、但最低薪资大于最高薪资的记录
reversed_salary = both_present & (salary_min > salary_max)

review_columns = [
    "title",
    "company_name",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_rate_unit",
]

# 每个岗位单独打印，避免一行太宽
print("\n薪资上下界颠倒的岗位：")
for row_index, row in df.loc[reversed_salary, review_columns].iterrows():
    print(f"\n原始行索引：{row_index}")
    print(row.to_string())
# 每类问题单独标记，同一岗位可以有多个问题
df["salary_missing_bound"] = salary_min.isna() | salary_max.isna()
df["salary_has_negative"] = (salary_min < 0) | (salary_max < 0)
df["salary_has_zero"] = (salary_min == 0) | (salary_max == 0)
df["salary_bounds_reversed"] = reversed_salary

issue_columns = [
    "salary_missing_bound",
    "salary_has_negative",
    "salary_has_zero",
    "salary_bounds_reversed",
]

# 任意一项为 True，就表示未通过本轮金额基础检查
has_amount_issue = df[issue_columns].any(axis=1)
df["salary_amount_basic_ok"] = ~has_amount_issue

print("\n薪资金额基础检查汇总：")
print("存在至少一项问题：", int(has_amount_issue.sum()))
print("通过基础检查：", int(df["salary_amount_basic_ok"].sum()))
print("总岗位数：", len(df))
# 统一币种的大小写与前后空格，保留原始字段
df["salary_currency_clean"] = (
    df["salary_currency"]
    .astype("string")
    .str.strip()
    .str.upper()
    .replace("", pd.NA)
)

# 仅选取通过金额基础检查的岗位
salary_candidates = df.loc[df["salary_amount_basic_ok"]]

# 按币种和周期分组，统计每组岗位数
salary_groups = (
    salary_candidates
    .groupby(
        ["salary_currency_clean", "salary_rate_unit_clean"],
        dropna=False,
    )
    .size()
    .sort_values(ascending=False)
)

print("\n通过金额基础检查的岗位：币种与周期分组（前 10 组）")
print(salary_groups.head(10).to_string())
print("\n全部分组数量合计：", int(salary_groups.sum()))
# 筛选通过金额基础检查的 USD 年薪岗位
usd_year_mask = (
    df["salary_amount_basic_ok"]
    & df["salary_currency_clean"].eq("USD")
    & df["salary_rate_unit_clean"].eq("year")
).fillna(False)

usd_year_jobs = df.loc[
    usd_year_mask, ["salary_min", "salary_max"]
].copy()

# 计算发布薪资区间的中点
usd_year_jobs["salary_midpoint"] = (
    usd_year_jobs["salary_min"] + usd_year_jobs["salary_max"]
) / 2

print("\nUSD 年薪候选岗位数：", len(usd_year_jobs))
print("\n金额分布（单位：USD/年）：")
print(
    usd_year_jobs.describe(
        percentiles=[0.01, 0.5, 0.99]
    ).round(2).to_string()
)
# 按薪资中点选出两端的样例
extreme_groups = {
    "中点最低的 3 条": usd_year_jobs.nsmallest(3, "salary_midpoint"),
    "中点最高的 3 条": usd_year_jobs.nlargest(3, "salary_midpoint"),
}

review_columns = [
    "title",
    "company_name",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_rate_unit",
    "job_description",
]

for group_name, sample in extreme_groups.items():
    print(f"\n=== {group_name} ===")

    for row_index in sample.index:
        print(f"\n原始行索引：{row_index}")

        for column in review_columns:
            value = df.at[row_index, column]

            # 描述较长，只打印前 400 个字符
            if column == "job_description":
                value = str(value)[:400]

            print(f"{column}: {value}")

        print("salary_midpoint:", sample.at[row_index, "salary_midpoint"])

# 根据当前 USD 年薪候选样本计算复核边界
midpoints = usd_year_jobs["salary_midpoint"]
lower_bound = midpoints.quantile(0.01)
upper_bound = midpoints.quantile(0.99)

# 标记两端的记录
usd_year_jobs["salary_review_reason"] = "not_flagged"

usd_year_jobs.loc[
    midpoints < lower_bound, "salary_review_reason"
] = "below_p01"

usd_year_jobs.loc[
    midpoints > upper_bound, "salary_review_reason"
] = "above_p99"

# 提取需要复核的记录
review_rows = usd_year_jobs.loc[
    usd_year_jobs["salary_review_reason"] != "not_flagged"
    ]

# 按原始索引补回岗位信息
review_report = df.loc[
    review_rows.index,
    [
        "title", "company_name", "country",
        "salary_min", "salary_max",
        "salary_currency", "salary_rate_unit",
        "job_description",
    ],
].copy()

review_report["salary_midpoint"] = review_rows["salary_midpoint"]
review_report["salary_review_reason"] = review_rows["salary_review_reason"]

# 保存复核报告，不覆盖原始数据
report_dir = project_dir / "data" / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

report_path = report_dir / "usd_year_salary_review.csv"
review_report.to_csv(
    report_path,
    encoding="utf-8-sig",
    index_label="source_row_index",
)

print("\n复核下界：", round(lower_bound, 2))
print("复核上界：", round(upper_bound, 2))
print("待复核记录数：", len(review_report))
print("报告位置：", report_path)

# 取前 3 条技能字段非空的岗位
skill_samples = df.loc[
    df["skills_required"].notna(),
    ["title", "skills_required"],
].head(3)

print("\n技能解析结果：")

for row_index, row in skill_samples.iterrows():
    raw_value = row["skills_required"]
    skills = json.loads(raw_value)

    print(f"\n原始行索引：{row_index}")
    print("岗位名称：", row["title"])
    print("转换前类型：", type(raw_value).__name__)
    print("转换后类型：", type(skills).__name__)
    print("技能数量：", len(skills))
    print("第一个技能：", skills[0] if skills else "无技能")
# 每条记录会得到一个“列表、状态”的二元组
parsed_skills = df["skills_required"].apply(parse_skills)

df["skills_list"] = parsed_skills.apply(lambda result: result[0])
df["skills_parse_status"] = parsed_skills.apply(lambda result: result[1])

status_counts = df["skills_parse_status"].value_counts()

print("\n整列技能解析状态：")
print(status_counts.to_string())
print("状态数量合计：", int(status_counts.sum()))
print("\n技能格式异常样例：")

for status in ["invalid_json", "invalid_structure"]:
    samples = df.loc[
        df["skills_parse_status"].eq(status),
        ["title", "skills_required"],
    ].head(2)

    print(f"\n状态：{status}")

    for row_index, row in samples.iterrows():
        raw_value = row["skills_required"]

        print("原始行索引：", row_index)
        print("岗位名称：", row["title"])
        print("内容长度：", len(raw_value))
        print("内容开头：", repr(raw_value[:300]))
        print("内容结尾：", repr(raw_value[-100:]))
        print()



df["skills_normalized"] = df["skills_list"].apply(normalize_skills)

print("\n技能标准化样例：")
for row_index in skill_samples.index:
    print(f"\n原始行索引：{row_index}")
    print("标准化前：", df.at[row_index, "skills_list"])
    print("标准化后：", df.at[row_index, "skills_normalized"])
# 只使用技能解析成功的岗位
valid_skill_jobs = df.loc[
    df["skills_parse_status"].eq("ok")
]

# 将每个岗位的技能列表展开为一行一个技能
skill_counts = (
    valid_skill_jobs["skills_normalized"]
    .explode()
    .dropna()
    .value_counts()
)

# 计算每项技能在可解析岗位中的出现比例
top_skills = skill_counts.head(10).rename("岗位数").to_frame()
top_skills["占可解析岗位比例(%)"] = (
    top_skills["岗位数"] / len(valid_skill_jobs) * 100
).round(2)

print("\n技能统计覆盖岗位数：", len(valid_skill_jobs))
print("\n全部可解析岗位的技能 Top 10：")
print(top_skills.to_string())
# 按标题中的连续文本匹配，不区分大小写
data_analyst_mask = (
    df["title"]
    .astype("string")
    .str.contains(
        "data analyst",
        case=False,
        regex=False,
        na=False,
    )
)

data_analyst_jobs = df.loc[data_analyst_mask]

valid_count = data_analyst_jobs["skills_parse_status"].eq("ok").sum()

print("\n标题包含 Data Analyst 的岗位记录数：", len(data_analyst_jobs))
print("其中技能可解析的记录数：", int(valid_count))

print("\n前 10 个不同岗位标题：")
for title in data_analyst_jobs["title"].drop_duplicates().head(10):
    print("-", title)
# 保留有效岗位表，供后面的报告元数据使用
analyst_skill_jobs = data_analyst_jobs.loc[
    data_analyst_jobs["skills_parse_status"].eq("ok")
]

# 使用统一函数计算排名
analyst_top_skills, valid_count = summarize_skills(data_analyst_jobs)

print("\n本次技能统计覆盖记录数：", valid_count)

if valid_count == 0:
    print("没有技能可解析的记录，暂时无法统计。")
else:
    print("\n标题包含 Data Analyst 的岗位：技能 Top 10")
    print(analyst_top_skills.to_string())

# 保存结果的目录
analysis_dir = project_dir / "data" / "reports"
analysis_dir.mkdir(parents=True, exist_ok=True)

if not analyst_skill_jobs.empty:
    # 保存技能排名
    ranking_path = analysis_dir / "data_analyst_skills_top10.csv"
    analyst_top_skills.to_csv(
        ranking_path,
        encoding="utf-8-sig",
        index_label="skill",
    )

    # 保存本次分析的统计口径
    analysis_info = {
        "source_file": data_path.name,
        "filter": "title contains 'data analyst', case-insensitive",
        "matched_records": int(len(data_analyst_jobs)),
        "valid_skill_records": int(len(analyst_skill_jobs)),
        "counting_rule": "Each normalized skill counted once per record",
        "denominator": "Matched records with skills_parse_status == 'ok'",
        "job_records_deduplicated": False,
    }

    info_path = analysis_dir / "data_analyst_skills_metadata.json"
    info_path.write_text(
        json.dumps(analysis_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n技能排名已保存：", ranking_path)
    print("统计口径已保存：", info_path)
print("\n技能标准化函数来源：", normalize_skills.__module__)
print("技能解析函数来源：", parse_skills.__module__)
