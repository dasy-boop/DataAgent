"""Prepare bounded, explicitly labelled evidence for natural-language answers."""
import json

from tools.category_normalization import low_coverage


def _has_value(value):
    return value is not None and str(value).strip().casefold() not in {"", "none", "nan", "<na>"}


def build_evidence(steps):
    evidence = []
    last_filter = max((i for i, step in enumerate(steps)
                       if step["tool"] == "filter_rows"), default=-1)
    for index, step in enumerate(steps):
        # Only the final filtered scope contributes raw records (at most eight).
        if step["tool"] == "filter_rows" and index != last_filter:
            continue
        result = step["result"]
        item = {"tool": step["tool"], "arguments": step["arguments"]}
        if isinstance(result, list):
            limit = 8 if step["tool"] == "filter_rows" else 100
            rows = []
            for row in result[:limit]:
                if step["tool"] == "filter_rows":
                    keys = ("title", "company_name", "country", "country_clean",
                            "job_description", "description", "skills_required",
                            "responsibilities", "requirements", "experience_level",
                            "years_experience_numeric", "minimum_qualifications",
                            "preferred_qualifications", "education_level", "job_level_normalized")
                    row = {key: row[key] for key in keys if key in row}
                rows.append({key: value[:3000] if isinstance(value, str) else value
                             for key, value in row.items()})
            item.update(total_rows=len(result), shown_rows=len(rows),
                        sampled=len(rows) < len(result), result=rows)
            if step["tool"] == "count_values":
                column = step["arguments"].get("column")
                if isinstance(step.get("coverage"), dict):
                    item.update(step["coverage"])
                else:
                    scope = next((prior["result"] for prior in reversed(steps[:index])
                                  if prior["tool"] == "filter_rows" and isinstance(prior["result"], list)), None)
                    if scope is not None and column:
                        available = sum(_has_value(row.get(column)) for row in scope)
                        item.update(matched_records=len(scope), available_records=available,
                                    low_coverage=low_coverage(len(scope), available))
        else:
            item["result"] = result
        evidence.append(item)
    # Keep the final result first when there are many intermediate steps.
    budget, selected = 45000, []
    for item in reversed(evidence):
        size = len(json.dumps(item, ensure_ascii=False, default=str))
        while size > budget and isinstance(item.get("result"), list) and item["result"]:
            item["result"].pop()
            item["shown_rows"] = len(item["result"])
            item["sampled"] = item["shown_rows"] < item["total_rows"]
            size = len(json.dumps(item, ensure_ascii=False, default=str))
        if size <= budget:
            selected.append(item)
            budget -= size
    return list(reversed(selected))


# Group by meaning rather than question templates. Years and levels belong to
# the same experience section; each other attribute retains its own result.
_ATTRIBUTE_LABELS = {
    "education_level": "学历要求", "years_experience_numeric": "工作经验",
    "experience_level": "工作经验", "job_level_normalized": "工作经验",
    "skills_required": "技能要求", "work_model": "工作方式",
    "industry": "行业", "visa_sponsorship_available": "签证支持",
    "employment_type": "雇佣类型",
}
_DISPLAY_NAMES = {"Mid": "中级", "Senior": "高级", "Entry": "入门级",
                  "Junior": "初级", "Intern": "实习", "On-site": "现场办公",
                  "Hybrid": "混合办公", "Remote": "远程办公"}


def _attribute(item):
    tool = item["tool"]
    if tool == "count_skills":
        return "技能要求"
    if tool in {"count_values", "summarize_numeric"}:
        return _ATTRIBUTE_LABELS.get(item["arguments"].get("column"))
    return None


def _attribute_summary(evidence):
    notes = []
    for item in evidence:
        result = item.get("result")
        column = item["arguments"].get("column")
        if item["tool"] == "count_values":
            if not result:
                notes.append("这批岗位没有足够的相关信息，暂时无法分析这一项。")
                continue
            rows = [f"{_DISPLAY_NAMES.get(row[column], row[column])} {row['count']} 个"
                    for row in result[:3]]
            if column == "education_level":
                if item.get("matched_records") is not None and item.get("available_records") is not None:
                    notes.append(f"本次找到 {item['matched_records']} 个岗位，其中 {item['available_records']} 个提供了学历信息。")
                notes.append("按岗位写明的最低可接受学历整理，结果包括：" + "、".join(rows) + "。多个可选学历按其中最低者计一次，无法确认的单独列出。")
            else:
                notes.append("已提供的信息中，" + "、".join(rows) + "。")
            if column in {"experience_level", "job_level_normalized"}:
                notes.append("岗位级别不等同于具体工作年限。")
        elif item["tool"] == "count_skills":
            if not result.get("valid_skill_records"):
                notes.append("这批岗位没有可识别的技能信息，暂时无法分析技能要求。")
                continue
            rows = [f"{row['skill']}（{row['岗位记录数']} 个，{row['占可解析记录比例(%)']}%）"
                    for row in result["skills"][:3]]
            notes.append(f"在有技能信息的 {result['valid_skill_records']} 个岗位中，出现较多的技能包括" + "、".join(rows) + "。")
        elif item["tool"] == "summarize_numeric":
            if not result.get("count"):
                notes.append("具体工作年限数据不足，暂时无法判断通常要求几年经验。")
            else:
                notes.append(f"{result['count']} 个岗位提供了可统计的具体工作年限，中位数为 {result['median']} 年；这只反映写明年限的岗位。")
    return "".join(notes)


def protected_answer(steps, evidence):
    """Preserve per-attribute guards without letting one suppress the others."""
    grouped = {}
    for item in evidence:
        label = _attribute(item)
        if label:
            grouped.setdefault(label, []).append(item)
    if len(grouped) < 2:
        return _single_attribute_guard(steps, evidence)
    # Only compose a same-scope analysis whose outputs we can all explain.
    # Comparisons, salary and text interpretation retain their existing path.
    if any(item["tool"] != "filter_rows" and _attribute(item) is None for item in evidence):
        return _single_attribute_guard(steps, evidence)
    sections = []
    for label, items in grouped.items():
        attribute_steps = [step for step in steps if _attribute(step) == label]
        text = _single_attribute_guard(attribute_steps, items) or _attribute_summary(items)
        sections.append(f"**{label}**\n{text}")
    return "\n\n".join(sections)


def _single_attribute_guard(steps, evidence):
    """Use tool-derived wording when sparse/ambiguous data invites overclaiming."""
    for item in reversed(evidence):
        if item["tool"] == "compare_skills":
            result = item.get("result", {})
            groups = result.get("groups", [])
            if result.get("status") == "insufficient_data":
                notes = []
                for group in groups:
                    label = group["keyword"]
                    if not group["matched_records"]:
                        notes.append(f"{label}没有找到符合条件的岗位")
                    elif not group["valid_skill_records"]:
                        notes.append(f"{label}找到 {group['matched_records']} 个岗位，但都没有可识别的技能信息")
                    else:
                        notes.append(f"{label}找到 {group['matched_records']} 个岗位，其中 {group['valid_skill_records']} 个写明了可识别的技能要求")
                return "；".join(notes) + "。至少一组样本不足，暂时无法比较两组的技能要求比例。"
            comparison = result.get("comparison")
            if comparison and len(groups) == 2:
                skill = comparison["skill"]
                if comparison["tied"]:
                    lead = f"两组岗位要求 {skill} 的比例相同。"
                else:
                    lead = f"在有技能信息的岗位中，{comparison['winner']}要求 {skill} 的比例更高。"
                details = []
                for group in groups:
                    metric = group["target_skill"]
                    details.append(f"{group['keyword']}找到 {group['matched_records']} 个岗位，其中 {metric['denominator']} 个写明了可识别的技能要求，{metric['numerator']} 个要求 {skill}，占 {metric['percentage']}%")
                return lead + "\n\n" + "；".join(details) + "。\n\n数据来自2026年6月招聘快照，各组比例分别根据该组有技能信息的岗位计算。"
        if item["tool"] == "count_values" and item["arguments"].get("column") == "industry":
            total, available = item.get("matched_records"), item.get("available_records")
            if total is not None and item.get("low_coverage"):
                rows = item.get("result", [])
                named = next((row for row in rows if row.get("industry") != "其他/未标准化"), None)
                example = (f"在已提供的信息中，{named['industry']}出现 {named['count']} 次，仅供参考。"
                           if named else "已提供的信息也不足以判断主要行业。")
                return (f"本次找到 {total} 个相关岗位，只有 {available} 个提供了行业信息，"
                        f"不足以可靠判断整体行业分布。{example}")
        if item["tool"] == "count_values" and item["arguments"].get("column") == "visa_sponsorship_available":
            rows = {row["visa_sponsorship_available"]: row["count"] for row in item.get("result", [])}
            total = item.get("matched_records")
            yes = rows.get("明确支持签证", 0)
            prefix = f"本次找到的 {total} 个相关岗位中，" if total is not None else "本次结果中，"
            return (f"{prefix}只有 {yes} 个明确标注提供签证支持。"
                    "其他信息可能是工作许可要求、地区标记或未说明，不能据此判断是否提供签证支持。")
        if item["tool"] == "calculate_proportion" and item["arguments"].get("column") == "visa_sponsorship_available":
            result = item.get("result", {})
            if result.get("value") == "是":
                return (f"本次找到 {result.get('matched_records', 0)} 个相关岗位，"
                        f"其中 {result.get('numerator', 0)} 个明确标注提供签证支持。"
                        "未写明、地区标记及工作许可要求不能算作支持或不支持。")
    years = next((step["result"] for step in steps if step["tool"] == "summarize_numeric"
                  and step["arguments"].get("column") == "years_experience_numeric"
                  and isinstance(step["result"], dict)), None)
    if years is not None and years.get("count", 0) < 30:
        count = years.get("count", 0)
        level = next((item for item in evidence if item["tool"] == "count_values"
                      and item["arguments"].get("column") == "experience_level"), None)
        level_note = "岗位级别分布可以作为参考，但不等同于具体工作年限。"
        if level and level.get("available_records") is not None:
            names = {"Mid": "中级", "Senior": "高级", "Entry": "入门级",
                     "Junior": "初级", "Intern": "实习"}
            top = [f"{names.get(row['experience_level'], row['experience_level'])} {row['count']} 个"
                   for row in level.get("result", [])[:2]]
            level_note = (f"另有 {level['available_records']} 个岗位写明了级别"
                          + (f"，其中{'、'.join(top)}" if top else "")
                          + "；岗位级别不等同于具体工作年限。")
        return (f"本次只有 {count} 个相关岗位提供了可直接统计的具体工作年限，"
                f"不足以可靠判断通常要求几年经验。{level_note}")
    return None


def fallback_answer(steps):
    result = steps[-1]["result"]
    if isinstance(result, dict) and result.get("status") == "no_numeric_data":
        return "这批岗位缺少可用于计算的数字，暂时无法给出平均值或中位数。可查看下方的岗位信息；没有写明的要求不能当作零。"
    if isinstance(result, list) and not result:
        return "本次没有找到符合条件的结果。可以调整岗位、公司或其他条件后再试。"
    if isinstance(result, list):
        return f"本次找到 {len(result)} 项结果。文字解读暂未生成，具体内容见下方。"
    return "分析已完成。文字解读暂未生成，具体内容见下方。"
