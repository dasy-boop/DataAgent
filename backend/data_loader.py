from pathlib import Path

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    """读取 Parquet 文件，并在文件不存在时给出明确错误。"""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在：{path}")

    return pd.read_parquet(path)
