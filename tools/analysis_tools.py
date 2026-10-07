"""Additional deterministic analysis tools; existing Pandas algorithms stay intact."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from backend.country_cleaning import normalize_country
from backend.field_metadata import COUNTRY_INPUT, FIELD_SPECS, canonical_value
from .category_normalization import classify_visa, normalize_education, normalize_industries
from .pandas_tools import ToolError, count_values as raw_count_values
from .pandas_tools import filter_rows as legacy_filter_rows
from .pandas_tools import summarize_numeric as raw_summarize_numeric


NON_JOB_TITLES = {"full-time", "full time", "part-time", "part time"}
TEXT_FIELDS = {"minimum_qualifications", "preferred_qualifications",
               "responsibilities", "job_description"}
BOOLEAN_FIELDS = {"visa_sponsorship_available", "relocation_assistance",
                  "travel_required"}
PERIOD_ALIASES = {
    "year": "year", "yearly": "year", "annual": "year",
    "annually": "year", "annum": "year", "per annum": "year",
    "yr": "year", "p.a.": "year", "年": "year", "年薪": "year",
    "hour": "hour", "hourly": "hour", "hr": "hour",
    "小时": "hour", "时薪": "hour",
    "month": "month", "monthly": "month", "月": "month", "月薪": "month",
    "week": "week", "weekly": "week", "周": "week", "周薪": "week",
    "day": "day", "daily": "day", "天": "day", "日薪": "day",
}
ROLE_ALIASES = {"数据分析师": "Data Analyst", "数据科学家": "Data Scientist",
                "数据工程师": "Data Engineer", "软件工程师": "Software Engineer"}


def period_name(value: object) -> str | None:
    return PERIOD_ALIASES.get(str(value).strip().casefold())


def _boolean_name(value: object, column: str) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"true", "yes", "是", "1"}:
        return True
    if text in {"false", "no", "否", "0"}:
        return False
    if column == "visa_sponsorship_available":
        if text in {"no sponsorship", "no new h1b sponsorship available",
                    "no new h1b sponsorship"}:
            return False
    if column == "relocation_assistance" and text in {
        "relocation available", "relocation assistance available",
        "relocation assistance",
    }:
        return True
    return None


def _clean_categories(values: pd.Series, column: str) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    if column == "country_clean":
        return normalize_country(cleaned)
    if column in {"work_model", "employment_type", "experience_level",
                  "job_level_normalized"}:
        return cleaned.map(lambda item: canonical_value(column, item)
                           if pd.notna(item) else pd.NA).astype("string")
    if column == "education_level":
        return cleaned.map(normalize_education).astype("string")
    return cleaned


def _one_condition(df: pd.DataFrame, condition: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(condition, dict) or set(condition) != {"column", "operator", "value"}:
        raise ToolError("每个筛选条件必须包含 column、operator、value")
    column, operator, value = (condition[key] for key in ("column", "operator", "value"))
    if column not in df.columns or column not in FIELD_SPECS:
        raise ToolError(f"不可筛选的字段: {column}")
    if not FIELD_SPECS[column].filterable:
        raise ToolError(f"该字段不适合直接筛选: {column}")
    if operator in {"gte", "lte"}:
        if not FIELD_SPECS[column].numeric:
            raise ToolError("该字段不能直接比较数值；薪资请使用专门的薪资分析")
        try:
            target = float(value)
        except (TypeError, ValueError) as exc:
            raise ToolError("数值筛选条件无效") from exc
        numeric = pd.to_numeric(df[column], errors="coerce")
        mask = numeric.ge(target) if operator == "gte" else numeric.le(target)
    elif operator == "boolean":
        if column not in BOOLEAN_FIELDS:
            raise ToolError("该字段不支持布尔筛选")
        target = _boolean_name(value, column)
        if target is None:
            raise ToolError("布尔筛选值必须为是或否")
        mask = df[column].map(lambda item: _boolean_name(item, column) is target)
    elif operator == "in":
        if not isinstance(value, list) or not value:
            raise ToolError("in 筛选必须提供非空列表")

        series = _clean_categories(df[column], column)
        targets = []

        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ToolError("in 筛选值必须是非空文本")

            target = str(
                _clean_categories(
                    pd.Series([item.strip()]), column
                ).iloc[0]
            )
            targets.append(target.casefold())

        mask = series.str.casefold().isin(targets)
    elif operator in {"equals", "contains"}:
        if not isinstance(value, str) or not value.strip():
            raise ToolError("文本筛选值不能为空")
        target = value.strip()
        for chinese, english in ROLE_ALIASES.items():
            if column in {"title", "normalized_title"} and chinese in target:
                target = target.replace(chinese, english)
        if column == "country_clean":
            target = COUNTRY_INPUT.get(target, target)
            target = str(normalize_country(pd.Series([target])).iloc[0])
        if operator == "equals":
            series = _clean_categories(df[column], column)
            target = str(_clean_categories(pd.Series([target]), column).iloc[0])
            mask = series.str.casefold().eq(target.casefold())
        else:
            mask = df[column].astype("string").str.contains(
                target, case=False, regex=False, na=False)
    else:
        raise ToolError(f"不支持的筛选方式: {operator}")
    return df.loc[mask.fillna(False)].copy()


def filter_rows(df: pd.DataFrame, column: str | None = None,
                keyword: str | None = None,
                conditions: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Keep the old single-filter API; add an explicit AND condition list."""
    if conditions is None:
        return legacy_filter_rows(df, column, keyword)
    if column is not None or keyword is not None:
        raise ToolError("多条件筛选不能同时提供旧式 column 或 keyword")
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= 12:
        raise ToolError("conditions 必须包含 1 到 12 个条件")
    result = df
    for condition in conditions:
        result = _one_condition(result, condition)
    return result


def count_values(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """Preserve ordinary counts; omit only exact employment labels in title rankings."""
    if column not in df.columns:
        raise ToolError(f"字段不存在: {column}")
    if column in TEXT_FIELDS or column in {"skills_required", "job_description"}:
        raise ToolError("长文本或技能列表不能直接做分类排名；请使用受控样本或技能统计")
    if not isinstance(top_n, int) or not 1 <= top_n <= 100:
        raise ToolError("top_n 必须是 1 到 100 之间的整数")
    if column == "industry":
        values = df[column].map(normalize_industries).explode().dropna()
        return values.value_counts().head(top_n).rename_axis(column).reset_index(name="count")
    if column == "visa_sponsorship_available":
        values = df[column].map(classify_visa)
        return values.value_counts().head(top_n).rename_axis(column).reset_index(name="count")
    selected = df
    if column in {"title", "normalized_title"}:
        names = df[column].astype("string").str.strip().str.casefold()
        selected = df.loc[~names.isin(NON_JOB_TITLES)].copy()
    if column in {"work_model", "employment_type", "experience_level",
                  "education_level",
                  "job_level_normalized"}:
        selected = selected.copy()
        selected[column] = _clean_categories(selected[column], column)
    return raw_count_values(selected, column, top_n)


def summarize_numeric(df: pd.DataFrame, column: str) -> dict:
    if column in {"salary_min", "salary_max", "on_target_earnings"}:
        raise ToolError("薪资字段必须使用按币种和周期隔离的薪资分析")
    if not FIELD_SPECS.get(column, FIELD_SPECS["job_description"]).numeric:
        raise ToolError("该字段不适合做普通数值统计")
    return raw_summarize_numeric(df, column)


def calculate_proportion(df: pd.DataFrame, column: str, value: str) -> dict:
    if column not in df.columns or column not in FIELD_SPECS:
        raise ToolError(f"字段不存在: {column}")
    if not FIELD_SPECS[column].groupable:
        raise ToolError("该字段不适合直接计算分类比例")
    if column in BOOLEAN_FIELDS:
        target = _boolean_name(value, column)
        if target is None:
            raise ToolError("请指定是或否")
        known = df[column].map(lambda item: _boolean_name(item, column))
        valid = known.notna()
        selected = known.eq(target).fillna(False)
        label = "是" if target else "否"
    else:
        target = str(_clean_categories(pd.Series([value]), column).iloc[0])
        known = _clean_categories(df[column], column)
        valid = known.notna() & known.ne("").fillna(False)
        if column == "work_model":
            valid &= known.isin({"Remote", "Hybrid", "On-site"})
        selected = known.str.casefold().eq(target.casefold()).fillna(False)
        label = target
    denominator = int(valid.sum())
    numerator = int((valid & selected).sum())
    return {"column": column, "value": label, "matched_records": int(len(df)),
            "numerator": numerator, "denominator": denominator,
            "unknown_records": int(len(df) - denominator),
            "percentage": round(100 * numerator / denominator, 2) if denominator else None,
            "status": "ok" if denominator else "no_data"}


def compare_groups(df: pd.DataFrame, group_column: str, value_a: str,
                   value_b: str, metric_column: str, top_n: int = 10) -> dict:
    if group_column not in df.columns or metric_column not in df.columns:
        raise ToolError("比较字段不存在")
    if not 1 <= top_n <= 20:
        raise ToolError("top_n 必须在 1 到 20 之间")
    if not FIELD_SPECS.get(metric_column, FIELD_SPECS["job_description"]).groupable:
        raise ToolError("该字段不适合分类比较")
    operator = "contains" if group_column in {"title", "normalized_title", "company_name"} else "equals"
    groups = []
    for value in (value_a, value_b):
        scoped = _one_condition(df, {"column": group_column, "operator": operator, "value": value})
        ranking = count_values(scoped, metric_column, top_n)
        available = _clean_categories(scoped[metric_column], metric_column)
        valid = int((available.notna() & available.ne("").fillna(False)).sum())
        groups.append({"value": value, "matched_records": int(len(scoped)),
                       "valid_metric_records": int(valid),
                       "ranking": [{"label": str(row[metric_column]),
                                    "count": int(row["count"]),
                                    "percentage": round(100 * row["count"] / valid, 2) if valid else None}
                                   for row in ranking.to_dict(orient="records")]})
    return {"group_column": group_column, "metric_column": metric_column,
            "groups": groups}


def summarize_salary(df: pd.DataFrame, currency: str, period: str) -> dict:
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Za-z]{3}", currency.strip()):
        raise ToolError("必须指定三字母薪资币种，例如 USD")
    canonical_period = period_name(period)
    if canonical_period is None:
        raise ToolError("必须指定可识别的薪资周期，例如年薪、月薪或时薪")
    required = {"salary_min", "salary_max", "salary_currency", "salary_rate_unit"}
    if not required.issubset(df.columns):
        raise ToolError("缺少完整的薪资字段")
    currency = currency.strip().upper()
    currency_match = df["salary_currency"].astype("string").str.strip().str.upper().eq(currency)
    period_match = df["salary_rate_unit"].map(period_name).eq(canonical_period)
    scoped = df.loc[(currency_match & period_match).fillna(False)]
    minimum = pd.to_numeric(scoped["salary_min"], errors="coerce").where(lambda s: s.gt(0))
    maximum = pd.to_numeric(scoped["salary_max"], errors="coerce").where(lambda s: s.gt(0))

    def stats(series):
        values = series.dropna()
        return {"count": int(len(values)),
                "mean": round(float(values.mean()), 2) if len(values) else None,
                "median": round(float(values.median()), 2) if len(values) else None,
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None}

    valid = minimum.notna() | maximum.notna()
    return {"currency": currency, "period": canonical_period,
            "matched_records": int(len(scoped)), "valid_records": int(valid.sum()),
            "salary_min": stats(minimum), "salary_max": stats(maximum),
            "status": "ok" if valid.any() else "no_data",
            "counting_rule": "仅比较同一币种和原始薪资周期；未进行汇率换算或年化"}


def sample_records(df: pd.DataFrame, limit: int = 8) -> dict:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ToolError("limit 必须在 1 到 20 之间")
    fields = [name for name in ("title", "company_name", "country_clean", "country",
                                "city", "work_model", "experience_level", "education_level")
              if name in df.columns]
    sample = df[fields].head(limit).astype(object)
    sample = sample.where(sample.notna(), None)
    return {"matched_records": int(len(df)), "shown_records": int(len(sample)),
            "sampled": len(sample) < len(df), "records": sample.to_dict(orient="records")}


def sample_text(df: pd.DataFrame, column: str, limit: int = 8) -> dict:
    if column not in TEXT_FIELDS or column not in df.columns:
        raise ToolError("该字段不适合文本样本分析")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
        raise ToolError("limit 必须在 1 到 8 之间")
    available = df.loc[df[column].notna() & df[column].astype("string").str.strip().ne("").fillna(False)]
    if len(available) > limit:
        positions = [round(i * (len(available) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
        selected = available.iloc[positions]
    else:
        selected = available
    records = []
    for _, row in selected.iterrows():
        title = row.get("title")
        company = row.get("company_name")
        records.append({"title": str(title) if pd.notna(title) else "",
                        "company_name": str(company) if pd.notna(company) else "",
                        "text": str(row[column])[:1200]})
    return {"column": column, "matched_records": int(len(df)),
            "available_records": int(len(available)), "shown_records": len(records),
            "sampled": len(records) < len(available), "records": records}


def summarize_dates(df: pd.DataFrame, column: str = "date_posted", top_n: int = 12) -> dict:
    if column not in {"date_posted", "closing_date"} or column not in df.columns:
        raise ToolError("仅支持发布日期或截止日期")
    if not 1 <= top_n <= 24:
        raise ToolError("top_n 必须在 1 到 24 之间")
    parsed = pd.to_datetime(df[column], format="mixed", errors="coerce", dayfirst=False)
    valid = parsed.dropna()
    months = valid.dt.to_period("M").astype(str).value_counts().head(top_n)
    return {"column": column, "matched_records": int(len(df)),
            "valid_records": int(len(valid)), "unknown_records": int(len(df) - len(valid)),
            "months": [{"month": str(month), "count": int(count)}
                       for month, count in months.items()],
            "status": "ok" if len(valid) else "no_data"}
