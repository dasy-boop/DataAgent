import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    base_url: str
    model: str | None


def get_llm_settings() -> LLMSettings:
    return LLMSettings(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv(
            "LLM_BASE_URL",
            "https://api.openai.com/v1",
        ),
        model=os.getenv("LLM_MODEL"),
    )