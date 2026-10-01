import pandas as pd


def summarize_quality(df: pd.DataFrame) -> pd.DataFrame:
    """生成每个字段的数据类型、非空数量和缺失率统计。"""
    summary = pd.DataFrame({
        "数据类型": df.dtypes.astype(str),
        "非空数量": df.notna().sum(),
        "缺失数量": df.isna().sum(),
        "缺失率(%)": (df.isna().mean() * 100).round(2),
    })

    return summary.sort_values(
        by="缺失率(%)",
        ascending=False,
    )