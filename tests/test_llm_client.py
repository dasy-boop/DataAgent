import json
from types import SimpleNamespace

from agent.llm_client import LLMClient


class FakeCompletions:
    def create(self, **kwargs):
        content = json.dumps(
            {
                "question": "统计岗位国家",
                "reasoning": "先筛选岗位，再统计国家",
                "tool_calls": [
                    {
                        "name": "count_values",
                        "arguments": {
                            "column": "country",
                            "top_n": 10,
                        },
                    }
                ],
            }
        )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=FakeCompletions()
        )


def test_create_plan_from_model_response(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    monkeypatch.setattr(
        "agent.llm_client.OpenAI",
        lambda **kwargs: FakeClient(),
    )

    client = LLMClient()

    plan = client.create_plan(
        question="统计岗位国家",
        columns=["title", "country"],
        tool_descriptions=[
            {
                "name": "count_values",
                "description": "统计字段值频次",
            }
        ],
    )

    assert plan.question == "统计岗位国家"
    assert plan.tool_calls[0].name == "count_values"