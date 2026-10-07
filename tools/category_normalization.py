"""Conservative labels for the June 2026 job snapshot.

These functions classify only text that explicitly identifies a category. They do
not infer qualifications, industries or immigration policy from a job title.
"""
from __future__ import annotations

import json
import ast
import re


EDUCATION_ORDER = ("高中及以下", "专科", "本科", "硕士", "博士")
_DEGREES = (
    ("高中及以下", re.compile(r"\bhigh[ -]?school\b|\bged\b|\bsecondary school\b", re.I)),
    ("专科", re.compile(r"\bassociate(?:['’]?s)?(?: degree)?\b", re.I)),
    ("本科", re.compile(r"\bbachelor(?:['’]?s)?\b|\bundergraduate\b|\bb\.?tech\b|\bbsc\b|\bb\.s\.?\b|\bbsn\b", re.I)),
    ("硕士", re.compile(r"\bmaster(?:['’]?s)?\b|\bmsc\b|\bm\.s\.?\b|\bmba\b", re.I)),
    ("博士", re.compile(r"\bph\.?d\.?\b|\bdoctorate\b|\bdoctoral\b", re.I)),
)


def _parts(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
        except (ValueError, TypeError):
            try:
                decoded = ast.literal_eval(text)
                if isinstance(decoded, list):
                    return [str(item).strip() for item in decoded if str(item).strip()]
            except (ValueError, SyntaxError):
                pass
    return [text]


def normalize_education(value: object) -> str | None:
    """Lowest explicitly acceptable degree; an explicit 'required' clause wins.

    An 'A or B' requirement and a list of alternatives use the lower named degree.
    This is a minimum among stated options, not the most frequently mentioned
    degree. Ambiguous equivalency with experience is left unclassified.
    """
    parts = _parts(value)
    if not parts:
        return None
    text = "; ".join(parts).replace("’", "'").replace("�", "'")
    if re.search(r"\bno degree (?:required|needed)\b|\bdegree not required\b", text, re.I):
        return "学历不限"
    if re.search(r"\bor equivalent (?:work )?experience\b", text, re.I):
        return "无法明确判断"
    clauses = re.split(r"[;。]", text)
    required = [part for part in clauses if re.search(r"\brequired\b|\bminimum\b|\bmust have\b", part, re.I)]
    chosen = " ".join(required) if required else text
    found = [label for label, pattern in _DEGREES if pattern.search(chosen)]
    if not found and required:
        found = [label for label, pattern in _DEGREES if pattern.search(text)]
    return found[0] if found else "无法明确判断"


_INDUSTRIES = {
    "financial services": "金融服务", "finance": "金融服务", "fintech": "金融科技",
    "banking": "银行", "insurance": "保险", "automotive": "汽车",
    "consulting": "咨询", "construction": "建筑业", "homebuilding": "建筑业",
    "technology": "科技", "technology, digital and data": "科技",
    "information technology": "信息技术", "marketing": "市场营销",
    "healthcare": "医疗健康", "health care": "医疗健康", "retail": "零售",
    "manufacturing": "制造业", "hospitality": "酒店与餐饮", "education": "教育",
    "engineering": "工程", "transportation": "交通运输", "logistics": "物流",
    "security": "安保", "real estate": "房地产", "food service": "餐饮",
    "pharmaceutical": "制药", "government": "政府", "legal": "法律",
    "energy": "能源", "human resources": "人力资源", "sales": "销售",
    "defense": "国防",
}


def normalize_industries(value: object) -> list[str]:
    """One posting can name several industries; count each once per posting."""
    labels = []
    for part in _parts(value):
        key = " ".join(part.casefold().split())
        label = _INDUSTRIES.get(key, "其他/未标准化")
        if label not in labels:
            labels.append(label)
    return labels


def classify_visa(value: object) -> str:
    """Unknown geography/flags never become sponsorship evidence."""
    parts = _parts(value)
    if not parts:
        return "未提供相关信息"
    text = " ".join(parts).strip().casefold()
    if text in {"yes", "true", "1", "是"} or re.fullmatch(
        r"(?:visa |h1b |tn )?sponsorship (?:is )?(?:available|provided)", text
    ):
        return "明确支持签证"
    if text in {"no", "false", "0", "否", "no sponsorship", "no visa sponsorship",
                "no new h1b sponsorship", "no new h1b sponsorship available"} or re.fullmatch(
                    r"no (?:visa )?sponsorship (?:available|provided)", text
                ):
        return "明确不提供签证支持"
    if re.search(r"\b(?:work authorization|right to work|citizen(?:ship)?|resident)\b.*\b(?:required|needed)\b|\beligible to work\b|\b(?:uk|us|canadian) resident\b", text):
        return "要求已有当地工作许可或身份"
    return "信息不明确"


def low_coverage(matched: int, available: int) -> bool:
    """A distribution is tentative if <30 records or <30% of matched jobs."""
    return matched > 0 and (available < 30 or available / matched < 0.30)
