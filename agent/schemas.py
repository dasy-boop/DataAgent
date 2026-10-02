from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ToolCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentPlan(BaseModel):
    question: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)

    status: Literal[
        "ready",
        "cannot_answer",
        "needs_clarification",
    ] = "ready"

    reason_code: Literal[
        "missing_data",
        "unsupported_operation",
        "ambiguous_request",
    ] | None = None

    message: str | None = None

    tool_calls: list[ToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "ready":
            if not self.tool_calls:
                raise ValueError("可执行的计划必须包含工具调用")

            if self.reason_code is not None or self.message is not None:
                raise ValueError("可执行的计划不应包含无法回答的原因")

        else:
            if self.tool_calls:
                raise ValueError("无法回答或需要补充信息时，不应调用工具")

            if not self.message or not self.message.strip():
                raise ValueError("必须提供具体的中文说明")

            if self.status == "cannot_answer":
                if self.reason_code not in {
                    "missing_data",
                    "unsupported_operation",
                }:
                    raise ValueError("无法回答时必须说明缺少数据或工具不支持")

            if self.status == "needs_clarification":
                if self.reason_code != "ambiguous_request":
                    raise ValueError("需要补充信息时，原因应为问题不明确")

        return self