from pathlib import Path
from typing import Any, Literal
import json
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.llm_client import LLMClient
from agent.answers import build_evidence, fallback_answer
from agent.runtime import run_plan
from agent.schemas import AgentPlan
from backend.country_cleaning import normalize_country
from backend.data_loader import load_parquet
from tools.pandas_tools import ToolError


router = APIRouter(prefix="/agent", tags=["Agent"])

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_DIR / "data" / "raw" / "nextgig_jobs_2026-06.parquet"
)


class ExecuteRequest(BaseModel):
    plan: AgentPlan


class HistoryTurn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=4000)


class PlanRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=6)


def answer_question(request):
    if not request.history:
        return request.question
    return json.dumps({"current_question": request.question,
                       "conversation_context": [m.model_dump() for m in request.history]},
                      ensure_ascii=False)


def load_agent_data() -> pd.DataFrame:
    """读取分析数据，保留原始国家字段并新增清洗字段。"""
    df = load_parquet(DATA_PATH).copy()

    if "country" in df.columns:
        df["country_clean"] = normalize_country(df["country"])

    return df


def serialize_result(result: Any) -> Any:
    """将 DataFrame 中的缺失值转换为 JSON 支持的 null。"""
    if isinstance(result, pd.DataFrame):
        clean_result = result.astype(object).where(
            pd.notna(result),
            None,
        )
        return clean_result.to_dict(orient="records")

    return result


@router.post("/execute")
def execute_agent(request: ExecuteRequest):
    """执行可完成的计划，或返回无法执行的原因。"""
    plan = request.plan

    if plan.status != "ready":
        return {
            "question": plan.question,
            "status": plan.status,
            "reason_code": plan.reason_code,
            "message": plan.message,
            "steps": [],
        }

    df = load_agent_data()

    try:
        results = run_plan(df, plan)
    except ToolError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "question": plan.question,
        "status": "completed",
        "reason_code": None,
        "message": "分析完成",
        "steps": [
            {
                "tool": call.name,
                "arguments": call.arguments,
                "result": serialize_result(result),
            }
            for call, result in zip(plan.tool_calls, results)
        ],
    }


@router.post("/plan")
def create_agent_plan(request: PlanRequest):
    """根据问题、数据字段和工具能力生成计划。"""
    df = load_agent_data()

    tool_descriptions = [
        {
            "name": "filter_rows",
            "description": (
                "按文本包含关系筛选记录，忽略大小写。"
                "不是精确相等筛选，也不支持数值大小比较。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "数据集中实际存在的字段名称",
                    },
                    "keyword": {
                        "type": "string",
                        "minLength": 1,
                        "description": "要匹配的非空文本",
                    },
                },
                "required": ["column", "keyword"],
                "additionalProperties": False,
            },
        },
        {
            "name": "count_values",
            "description": (
                "统计指定字段各个值的出现次数。"
                "自动忽略缺失值和空白值，按次数降序排列。"
                "只接受 column 和 top_n 参数。"
                "按国家统计时优先使用 country_clean；"
                "该字段统一了已知美国和英国别名，"
                "其他国家名称尚未全面规范化。"
                "如果用户要求原始国家值，则使用 country。"
                "本工具不会拆分列表中的单个技能。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "需要统计的字段名称",
                    },
                    "top_n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "返回频次最高的前几个值",
                    },
                },
                "required": ["column"],
                "additionalProperties": False,
            },
        },
        {
            "name": "count_skills",
            "description": (
                "在当前筛选的岗位记录中，解析 skills_required 中的技能列表，"
                "统计每项技能出现于多少条岗位记录。"
                "同一岗位重复提及同一技能只计一次，忽略无法解析技能的记录。"
                "返回匹配岗位数、技能可解析岗位数和技能排名。"
                "用于技能排名，以及 SQL 与 Python 等具体技能的频次比较。"
                "只接受可选的 top_n 参数，不需要 column。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "返回频次最高的前几个技能",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_numeric",
            "description": (
                "统计数值字段的有效数量、平均值、中位数、"
                "最小值和最大值。"
                "忽略缺失及无法转换为数值的内容。"
                "薪资统计前必须确保币种和薪资周期一致。"
                "本工具不能预测未来数值。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "需要计算的数值字段名称",
                    },
                },
                "required": ["column"],
                "additionalProperties": False,
            },
        },
    ]

    client = LLMClient()

    try:
        plan = client.create_plan(
            question=request.question,
            columns=list(df.columns),
            tool_descriptions=tool_descriptions,
            **({"history": [m.model_dump() for m in request.history]} if request.history else {}),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="模型返回的计划格式不正确，请重试。",
        ) from exc

    return plan.model_dump()


@router.post("/ask")
def ask_agent(request: PlanRequest):
    """生成计划，并根据计划状态决定是否执行。"""
    plan_data = create_agent_plan(request)
    plan = AgentPlan.model_validate(plan_data)

    execution = execute_agent(
        ExecuteRequest(plan=plan)
    )

    if execution["status"] == "completed":
        steps = execution["steps"]
        execution["answer"] = fallback_answer(steps)
        execution["answer_source"] = "fallback"
        if steps[-1]["result"] != []:
            try:
                execution["answer"] = LLMClient().create_answer(
                    answer_question(request), build_evidence(steps)
                )
                execution["answer_source"] = "model"
            except Exception:
                # Successful data results remain available if explanation fails.
                pass

    return {
        **execution,
        "plan": plan.model_dump(),
        "evidence": build_evidence(execution["steps"]),
    }


@router.post("/ask/stream")
def ask_agent_stream(request: PlanRequest):
    """Stream real progress and model tokens; retain structured data on failure."""
    def event(kind, data):
        return json.dumps({"type": kind, "data": data}, ensure_ascii=False) + "\n"

    def generate():
        try:
            yield event("progress", "正在结合上下文理解问题…" if request.history else "正在理解问题…")
            plan = AgentPlan.model_validate(create_agent_plan(request))
            yield event("progress", "正在查询招聘数据…")
            execution = execute_agent(ExecuteRequest(plan=plan))
            execution["plan"] = plan.model_dump()
            evidence = build_evidence(execution["steps"])
            execution["evidence"] = evidence
            if execution["status"] != "completed":
                yield event("result", execution)
                return
            yield event("progress", "查询完成，正在整理分析…")
            answer = fallback_answer(execution["steps"])
            source = "fallback"
            if execution["steps"][-1]["result"] != []:
                try:
                    chunks = []
                    for chunk in LLMClient().stream_answer(answer_question(request), evidence):
                        chunks.append(chunk)
                        yield event("delta", chunk)
                    if not chunks or not "".join(chunks).strip():
                        raise ValueError("Empty answer")
                    answer, source = "".join(chunks), "model"
                except Exception:
                    logging.getLogger(__name__).warning("Answer stream failed; returning data fallback")
            execution.update(answer=answer, answer_source=source)
            yield event("result", execution)
        except Exception as exc:
            message = exc.detail if isinstance(exc, HTTPException) else "请求未完成，请稍后重试。"
            yield event("error", str(message))

    return StreamingResponse(generate(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
