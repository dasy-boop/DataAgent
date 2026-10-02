from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.llm_client import LLMClient
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


class PlanRequest(BaseModel):
    question: str = Field(min_length=1)


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

    return {
        **execution,
        "plan": plan.model_dump(),
    }