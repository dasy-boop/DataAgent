import json

import pandas as pd


def normalize_skills(skills):
    """统一技能大小写和空白，并在单条岗位内去重。"""
    normalized = []

    for skill in skills:
        name = " ".join(skill.casefold().split())

        if name and name not in normalized:
            normalized.append(name)

    return normalized


def parse_skills(value):
    # 原字段缺失或只有空白
    if pd.isna(value):
        return [], "missing"

    if not isinstance(value, str):
        return [], "invalid_type"

    if not value.strip():
        return [], "missing"

    # 尝试解析 JSON 字符串
    try:
        skills = json.loads(value)
    except json.JSONDecodeError:
        return [], "invalid_json"

    # 必须是列表，而且每个元素都必须是字符串
    if not isinstance(skills, list):
        return [], "invalid_structure"

    if not all(isinstance(skill, str) for skill in skills):
        return [], "invalid_structure"

    # 去掉每项前后的空格，过滤空白技能名称
    cleaned_skills = [
        skill.strip()
        for skill in skills
        if skill.strip()
    ]

    if not cleaned_skills:
        return [], "empty"

    return cleaned_skills, "ok"
