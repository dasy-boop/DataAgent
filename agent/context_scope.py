"""Carry explicit country, role and work-mode constraints across follow-ups.

Only the recognized dimensions are repaired here. The LLM still plans all
other operations, and every turn executes against the full dataset.
"""

import re

from agent.schemas import AgentPlan, ToolCall


COUNTRY_TERMS = {
    "United Kingdom": r"英国|\b(?:united kingdom|uk|u\.k\.)\b",
    "United States": r"美国|\b(?:united states(?: of america)?|usa|us)\b",
    "Canada": r"加拿大|\bcanada\b",
    "Germany": r"德国|\bgermany\b",
    "France": r"法国|\bfrance\b",
    "India": r"印度|\bindia\b",
    "Japan": r"日本|\bjapan\b",
    "Australia": r"澳大利亚|\baustralia\b",
}

ROLE_TERMS = {
    "Data Analyst": r"数据分析师|\bdata analyst\b",
    "Data Scientist": r"数据科学家|\bdata scientist\b",
    "Data Engineer": r"数据工程师|\bdata engineer\b",
    "Software Engineer": r"软件工程师|\bsoftware engineer\b",
}

WORK_TERMS = {
    "Remote": r"远程|\bremote\b",
    "On-site": r"现场|线下|\bon[ -]?site\b",
    "Hybrid": r"混合办公|\bhybrid\b",
}


def _mentioned(text, terms):
    return [
        value
        for value, pattern in terms.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def _update(scope, question):
    text = str(question)

    if re.search(
        r"最常见的?\d*个?岗位|岗位排行|职位排行|招聘最多的?\d*个?岗位",
        text,
    ):
        scope.pop("country_clean", None)
        scope.pop("title", None)

    if re.search(r"重新查询|重新开始|清除所有条件", text):
        scope.clear()

    if re.search(
        r"全球|所有国家|不限地区|不限制地区|清除国家条件",
        text,
    ):
        scope.pop("country_clean", None)
        clear_country = True
    else:
        clear_country = False

    if re.search(r"不限岗位|不限制岗位", text):
        scope.pop("title", None)

    if re.search(r"不限工作方式|不限制工作方式", text):
        scope.pop("work_model", None)

    for key, terms in (
        ("country_clean", COUNTRY_TERMS),
        ("title", ROLE_TERMS),
        ("work_model", WORK_TERMS),
    ):
        matches = _mentioned(text, terms)

        if len(matches) > 1:
            scope.pop(key, None)

        elif len(matches) == 1 and not (
            key == "country_clean" and clear_country
        ):
            scope[key] = matches[0]


def resolve_scope(history, question):
    scope = {}

    for turn in history or []:
        _update(scope, turn.get("question", ""))

    _update(scope, question)

    return scope


def _has_company_filter(plan):
    """Use the planned filter field, regardless of the company's name."""
    for call in plan.tool_calls:
        if call.name != "filter_rows":
            continue
        args = call.arguments
        if args.get("column") == "company_name":
            return True
        if any(isinstance(item, dict) and item.get("column") == "company_name"
               for item in args.get("conditions", [])):
            return True
    return False


def enforce_scope(plan: AgentPlan, history, question) -> AgentPlan:
    if plan.status != "ready" or not history:
        return plan

    scope = resolve_scope(history, question)

    # A named company in this turn starts a new analysis subject. A referential
    # follow-up such as "其中..." may include an inherited company filter in the
    # plan, so it must continue to use the existing history scope.
    company_focus = _has_company_filter(plan) and not re.search(
        r"其中|这些|这家(?:公司|企业)|该(?:公司|企业)|上述|前面", question
    )
    company_cleared = set()
    if company_focus:
        for dimension, terms in (
            ("country_clean", COUNTRY_TERMS),
            ("title", ROLE_TERMS),
            ("work_model", WORK_TERMS),
        ):
            if not _mentioned(question, terms):
                scope.pop(dimension, None)
                company_cleared.add(dimension)

    reset = bool(
        re.search(
            r"重新查询|重新开始|清除所有条件",
            question,
        )
    )

    cleared = company_cleared

    if (
        reset
        or re.search(
            r"全球|所有国家|不限地区|不限制地区|清除国家条件",
            question,
        )
        or len(_mentioned(question, COUNTRY_TERMS)) > 1
    ):
        cleared.add("country_clean")

    if (
        reset
        or re.search(
            r"不限岗位|不限制岗位",
            question,
        )
        or len(_mentioned(question, ROLE_TERMS)) > 1
    ):
        cleared.add("title")

    if (
        reset
        or re.search(
            r"不限工作方式|不限制工作方式",
            question,
        )
        or len(_mentioned(question, WORK_TERMS)) > 1
    ):
        cleared.add("work_model")

    selected = set()
    kept = []

    for call in plan.tool_calls:
        if call.name != "filter_rows":
            kept.append(call)
            continue

        args = call.arguments

        if "conditions" in args:
            conditions = []

            for condition in args["conditions"]:
                column = condition.get("column")
                dimension = (
                    "country_clean"
                    if column == "country"
                    else column
                )

                if (
                    column == "country"
                    and scope.get("country_clean")
                ):
                    continue

                if (
                    dimension in cleared
                    and dimension not in scope
                ):
                    continue

                expected = scope.get(dimension)

                if (
                    expected
                    and dimension
                    in {
                        "country_clean",
                        "title",
                        "work_model",
                    }
                ):
                    if expected.casefold() in str(
                        condition.get("value", "")
                    ).casefold():
                        selected.add(dimension)
                        conditions.append(condition)

                    continue

                conditions.append(condition)

            if conditions:
                kept.append(
                    ToolCall(
                        name="filter_rows",
                        arguments={
                            "conditions": conditions
                        },
                    )
                )

        else:
            column = args.get("column")
            dimension = (
                "country_clean"
                if column == "country"
                else column
            )

            if (
                column == "country"
                and scope.get("country_clean")
            ):
                continue

            if (
                dimension in cleared
                and dimension not in scope
            ):
                continue

            expected = scope.get(dimension)

            if (
                expected
                and dimension
                in {
                    "country_clean",
                    "title",
                    "work_model",
                }
            ):
                if expected.casefold() in str(
                    args.get("keyword", "")
                ).casefold():
                    selected.add(dimension)
                    kept.append(call)

                continue

            kept.append(call)

    prefix = []

    for dimension in (
        "country_clean",
        "title",
        "work_model",
    ):
        expected = scope.get(dimension)

        if expected and dimension not in selected:
            prefix.append(
                ToolCall(
                    name="filter_rows",
                    arguments={
                        "conditions": [
                            {
                                "column": dimension,
                                "operator": (
                                    "contains"
                                    if dimension == "title"
                                    else "equals"
                                ),
                                "value": expected,
                            }
                        ]
                    },
                )
            )

    if prefix or kept:
        plan.tool_calls = prefix + kept

    return plan
