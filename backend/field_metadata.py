"""Field capabilities and conservative value aliases for the June 2026 snapshot.

Names and categorical values were checked against the committed Parquet file.
Unlisted values are retained as raw data rather than guessed into a category.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldSpec:
    label: str
    kind: str
    filterable: bool = True
    groupable: bool = False
    numeric: bool = False
    rankable: bool = False
    comparable: bool = False
    aliases: tuple[str, ...] = ()
    value_aliases: dict[str, str] = field(default_factory=dict)


def category(label, *aliases, values=None):
    return FieldSpec(label, "category", groupable=True, rankable=True,
                     comparable=True, aliases=aliases, value_aliases=values or {})


def text(label, *aliases):
    return FieldSpec(label, "text", groupable=False, rankable=False,
                     comparable=False, aliases=aliases)


def number(label, *aliases, comparable=True):
    return FieldSpec(label, "number", filterable=comparable, numeric=comparable,
                     comparable=comparable, aliases=aliases)


# Maps only observed spellings whose meaning is unambiguous. No currency or
# period conversion is performed here.
WORK_MODE = {
    "remote": "Remote", "on-site": "On-site", "onsite": "On-site",
    "on_site": "On-site", "on‑site": "On-site", "hybrid": "Hybrid",
    "远程": "Remote", "现场": "On-site", "线下": "On-site", "混合": "Hybrid",
}
EMPLOYMENT = {
    "full-time": "Full-time", "full time": "Full-time", "全职": "Full-time",
    "part-time": "Part-time", "part time": "Part-time", "兼职": "Part-time",
    "contract": "Contract", "合同制": "Contract",
    "internship": "Internship", "intern": "Internship", "实习": "Internship",
}
EXPERIENCE = {
    "entry": "Entry", "entry level": "Entry", "entry-level": "Entry", "初级": "Entry", "应届": "Entry",
    "junior": "Junior", "mid": "Mid", "中级": "Mid",
    "mid-level": "Mid", "mid level": "Mid",
    "senior": "Senior", "高级": "Senior", "intern": "Intern", "实习": "Intern",
}
BOOLEAN = {"true": "True", "yes": "True", "是": "True",
           "false": "False", "no": "False", "否": "False"}
COUNTRY_INPUT = {"英国": "United Kingdom", "美国": "United States",
                 "加拿大": "Canada", "德国": "Germany", "法国": "France",
                 "印度": "India", "日本": "Japan", "澳大利亚": "Australia"}


FIELD_SPECS = {
    "title": category("岗位名称", "职位", "岗位"),
    "normalized_title": category("标准岗位名称", "标准职位"),
    "company_name": category("公司", "企业"),
    "ats_name": category("招聘系统"),
    "industry": category("行业"),
    "function": category("职能"),
    "occupational_category": category("职业类别"),
    "employment_type": category("雇佣类型", "全职兼职", values=EMPLOYMENT),
    "work_model": category("工作方式", "远程", "现场", values=WORK_MODE),
    "schedule": category("工作安排", "排班"),
    "shift": category("班次"),
    "experience_level": category("经验级别", "职级", values=EXPERIENCE),
    "job_level_normalized": category("标准岗位级别", values=EXPERIENCE),
    "years_experience_numeric": number("经验年限", "工作年限"),
    "education_level": category("学历要求", "学历"),
    "skills_required": text("技能要求", "技能"),
    "minimum_qualifications": text("最低任职要求"),
    "preferred_qualifications": text("优先任职条件"),
    "responsibilities": text("岗位职责"),
    "certifications": text("证书要求"),
    "languages_required": text("语言要求"),
    "salary_min": number("薪资下限", comparable=False),
    "salary_max": number("薪资上限", comparable=False),
    "salary_currency": category("薪资币种", "货币"),
    "salary_rate_unit": category("薪资周期", "年薪", "时薪"),
    "salary_type": text("薪资类型"),
    "pay_frequency": text("发薪频率"),
    "bonus": text("奖金"),
    "equity": text("股权"),
    "on_target_earnings": text("目标总收入"),
    "benefits": text("福利"),
    "city": category("城市"),
    "country": category("国家/地区"),
    "location_resolved": category("标准地点"),
    "locations": text("工作地点"),
    "remote_eligibility_regions": category("远程适用地区"),
    "latitude": number("纬度", comparable=False),
    "longitude": number("经度", comparable=False),
    "visa_sponsorship_available": category("签证支持", values=BOOLEAN),
    "work_authorization_requirements": text("工作许可要求"),
    "security_clearance": text("安全审查要求"),
    "relocation_assistance": category("搬迁支持", values=BOOLEAN),
    "travel_required": category("出差要求", values=BOOLEAN),
    "travel_percentage": number("出差比例"),
    "date_posted": FieldSpec("发布日期", "date", groupable=True,
                             rankable=True, comparable=True, aliases=("发布时间",)),
    "closing_date": FieldSpec("截止日期", "date", groupable=True,
                              rankable=True, comparable=True),
    "job_description": text("岗位描述"),
    "country_clean": category("国家/地区（已合并已知别名）"),
}


def canonical_value(column: str, value: object) -> str:
    raw = str(value).strip()
    spec = FIELD_SPECS.get(column)
    return spec.value_aliases.get(raw.casefold(), raw) if spec else raw


def planner_fields(columns):
    return [
        {"name": name, "label": spec.label, "type": spec.kind,
         "filter": spec.filterable, "group": spec.groupable,
         "numeric": spec.numeric, "rank": spec.rankable,
         "compare": spec.comparable, "aliases": spec.aliases}
        for name in columns if (spec := FIELD_SPECS.get(name))
    ]
