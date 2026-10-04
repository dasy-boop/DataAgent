"""Prepare bounded, explicitly labelled evidence for natural-language answers."""
import json


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


def fallback_answer(steps):
    result = steps[-1]["result"]
    if isinstance(result, dict) and result.get("status") == "no_numeric_data":
        return result["message"] + " 相关查询结果保留在下方。"
    if isinstance(result, list) and not result:
        return "没有找到符合条件的记录。可以调整公司名称或岗位关键词后再试。"
    if isinstance(result, list):
        return f"查询已完成，共返回 {len(result)} 项结果。暂未生成文字解读，请查看下方数据。"
    return "统计已完成。暂未生成文字解读，请查看下方数据。"
