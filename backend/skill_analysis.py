def summarize_skills(jobs, top_n=10):
    """统计传入岗位中，技能可解析记录的技能排名。"""
    valid_jobs = jobs.loc[
        jobs["skills_parse_status"].eq("ok")
    ]
    valid_count = len(valid_jobs)

    # 每条岗位内部去重，确保一条岗位最多贡献一次同一技能
    deduplicated_skills = valid_jobs["skills_normalized"].apply(
        lambda skills: list(dict.fromkeys(skills))
        if isinstance(skills, list)
        else []
    )

    counts = (
        deduplicated_skills
        .explode()
        .dropna()
        .value_counts()
    )

    # 次数相同时按技能名称排序，使结果稳定
    ranking = counts.rename_axis("skill").reset_index(name="岗位记录数")
    ranking = ranking.sort_values(
        by=["岗位记录数", "skill"],
        ascending=[False, True],
    ).head(top_n)

    if valid_count > 0:
        ranking["占可解析记录比例(%)"] = (
            ranking["岗位记录数"] / valid_count * 100
        ).round(2)
    else:
        ranking["占可解析记录比例(%)"] = float("nan")

    return ranking.set_index("skill"), valid_count
