import json

from pathlib import Path


CACHE_PATH = Path("data/title_translations.json")


def load_title_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}

    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_title_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def translate_title(title: str, llm_client) -> str:
    title = title.strip()
    if not title:
        return title

    cache = load_title_cache()

    if title in cache:
        return cache[title]

    translated = llm_client.create_answer(
        "请将英文招聘岗位名称翻译成自然、专业的简体中文职位名称。"
        "必须遵循：Staff Engineer 译为“资深工程师”，Senior Engineer 译为“高级工程师”，"
        "Lead 译为“负责人”，Manager 译为“经理”，Analyst 译为“分析师”，Specialist 译为“专员”。"
        "专业方向应放在职位前面，例如 Staff Engineer, Equipment Engineering 应译为“资深设备工程师”。"
        "只返回最终中文职位名称，不要解释、不要括号、不要英文原文："
        + title,
        [],
    ).strip()

    if not translated:
        return title

    cache[title] = translated
    save_title_cache(cache)

    return translated
def translate_titles(titles: list[str], llm_client) -> dict[str, str]:
    titles = list(dict.fromkeys(
        title.strip()
        for title in titles
        if isinstance(title, str) and title.strip()
    ))

    if not titles:
        return {}

    cache = load_title_cache()
    missing = [title for title in titles if title not in cache]

    if missing:
        try:
            response = llm_client.client.chat.completions.create(
                model=llm_client.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你负责把英文招聘岗位名称翻译成自然、专业的简体中文职位名称。"
                            "Staff Engineer 译为“资深工程师”，Senior Engineer 译为“高级工程师”，"
                            "Lead 译为“负责人”，Manager 译为“经理”，"
                            "Analyst 译为“分析师”，Specialist 译为“专员”。"
                            "专业方向应自然放入中文职位名称中，不要逐词机械翻译。"
                            "必须返回 JSON 对象：键为原始英文岗位名称，值为中文岗位名称。"
                            "不要添加解释。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(missing, ensure_ascii=False),
                    },
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            translated = json.loads(content) if content else {}

            for title in missing:
                value = translated.get(title)
                if isinstance(value, str) and value.strip():
                    cache[title] = value.strip()

            save_title_cache(cache)

        except Exception:
            # 翻译只是展示增强，失败不能影响原始分析结果。
            pass

    return {
        title: cache.get(title, title)
        for title in titles
    }