import pandas as pd


COUNTRY_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
}


def normalize_country(values: pd.Series) -> pd.Series:
    """统一已知国家别名；未知名称保留，缺失值保持缺失。"""
    cleaned = values.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.eq(""), pd.NA)

    lookup = cleaned.str.casefold()
    mapped = lookup.map(COUNTRY_ALIASES)

    return mapped.fillna(cleaned).astype("string")
