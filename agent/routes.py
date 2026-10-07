from pathlib import Path
from typing import Any, Literal
import json
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.llm_client import LLMClient
from agent.answers import build_evidence, fallback_answer, protected_answer
from tools.category_normalization import low_coverage
from agent.runtime import run_plan
from agent.schemas import AgentPlan
from agent.context_scope import enforce_scope
from backend.country_cleaning import normalize_country
from backend.data_loader import load_parquet
from tools.pandas_tools import ToolError
from agent.title_translation import translate_titles

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

    steps = []
    current_scope = df
    for call, result in zip(plan.tool_calls, results):
        step = {"tool": call.name, "arguments": call.arguments,
                "result": serialize_result(result)}
        if call.name == "count_values" and call.arguments.get("column") in current_scope:
            column = call.arguments["column"]
            values = current_scope[column].astype("string").str.strip()
            available = int((values.notna() & values.ne("").fillna(False)).sum())
            step["coverage"] = {"matched_records": len(current_scope),
                                "available_records": available,
                                "low_coverage": low_coverage(len(current_scope), available)}
        steps.append(step)
        if call.name == "filter_rows" and isinstance(result, pd.DataFrame):
            current_scope = result
    return {
        "question": plan.question,
        "status": "completed",
        "reason_code": None,
        "message": "分析完成",
        "steps": steps,
    }


@router.post("/plan")
def create_agent_plan(request: PlanRequest):
    """根据问题、数据字段和工具能力生成计划。"""
    df = load_agent_data()

    tool_descriptions = [
        {
            "name": "filter_rows",
            "description": (
                "旧方式按单字段文本包含筛选；新方式 conditions 可用 1 到 12 个条件取交集。"
                "每项条件包含 column、operator、value；operator 可为 contains、equals、"
                "gte、lte、boolean。地区、工作方式和学历等可组合；数值比较仅用于可比较字段。"
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
                    "conditions": {"type": "array", "description":
                                   "多条件列表，每项为 {column, operator, value}；与旧式参数二选一"},
                },
                "required": [],
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
                "岗位名称排名会排除恰好等于 Full-time、Part-time 等雇佣类型标签的异常值。"
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
            "name": "compare_skills",
            "description": (
                "分别从当前岗位记录中，按同一文本字段的两个关键词独立筛选两组岗位，"
                "再解析 skills_required 并统计每组的单项技能频次。"
                "用于比较 Data Analyst 与 Software Engineer 等两类岗位的技能要求。"
                "每组返回匹配岗位数、技能可解析岗位数和技能排名。"
                "两组不是连续筛选。多条件比较使用 groups=[{label, conditions}, {label, conditions}]，每组条件独立取交集；"
                "与旧式 column/keyword_a/keyword_b 二选一。指定技能时传 skill，返回该技能的组内分子、分母和比例，不受 top_n 截断影响。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "用于分组筛选的文本字段，如 title"},
                    "keyword_a": {"type": "string", "minLength": 1, "description": "第一组筛选关键词"},
                    "keyword_b": {"type": "string", "minLength": 1, "description": "第二组筛选关键词"},
                    "groups": {"type": "array", "minItems": 2, "maxItems": 2,
                               "items": {"type": "object", "required": ["label", "conditions"],
                                         "properties": {"label": {"type": "string", "minLength": 1},
                                                        "conditions": {"type": "array", "minItems": 1, "maxItems": 12,
                                                                       "items": {"type": "object", "required": ["column", "operator", "value"],
                                                                                 "properties": {"column": {"type": "string"}, "operator": {"type": "string", "enum": ["contains", "equals", "gte", "lte", "boolean"]}, "value": {}},
                                                                                 "additionalProperties": False}}},
                                         "additionalProperties": False}},
                    "skill": {"type": "string", "minLength": 1, "description": "需要比较出现比例的单项技能，如 Python 或 SQL"},
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 50,
                              "default": 10, "description": "每组返回的技能数量"},
                },
                "required": [],
                "oneOf": [{"required": ["column", "keyword_a", "keyword_b"]}, {"required": ["groups"]}],
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
                "薪资上下限不得使用本工具，必须使用 summarize_salary。"
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
        {"name": "calculate_proportion", "description":
         "计算某类记录的分子、可判定分母和比例；未知或缺失值单独报告。"
         "适合远程、经验级别、签证支持、搬迁或出差等分类字段。",
         "parameters": {"column": "分类字段", "value": "目标值，如 Remote、Entry、是"}},
        {"name": "compare_groups", "description":
         "从当前范围独立筛选两组，计算同一分类字段的各组频次和占比；"
         "适合岗位学历、国家工作方式等比较。",
         "parameters": {"group_column": "分组字段", "value_a": "第一组值",
                        "value_b": "第二组值", "metric_column": "比较的分类字段",
                        "top_n": "可选整数，1到20"}},
        {"name": "summarize_salary", "description":
         "专用薪资统计。currency 必须是明确的三字母币种；period 必须是用户明确指定"
         "或从对话继承的 year/month/week/day/hour。按原始币种和周期隔离，"
         "分别统计薪资下限与上限的有效数、均值、中位数、最小值、最大值。"
         "不能换汇或自动年化；缺币种或周期时先追问，不能猜。",
         "parameters": {"currency": "USD/GBP/EUR等", "period": "year/hour/month/week/day"}},
        {"name": "sample_records", "description":
         "返回当前筛选范围的有限条岗位样本，只含岗位、公司、地点等必要字段。",
         "parameters": {"limit": "可选整数，1到20，默认8"}},
        {"name": "sample_text", "description":
         "从最低任职要求、优先条件、职责或岗位描述中抽取最多8条文本样本；"
         "仅根据样本总结，并明确样本量，不能概括全部市场。",
         "parameters": {"column": "文本字段", "limit": "可选整数，1到8"}},
        {"name": "summarize_dates", "description":
         "将已有发布日期或截止日期解析后按月份计数，报告有效数和缺失数。"
         "不能预测未来趋势。",
         "parameters": {"column": "date_posted或closing_date", "top_n": "可选整数，1到24"}},
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

    plan = enforce_scope(plan, [m.model_dump() for m in request.history], request.question)
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
        evidence = build_evidence(steps)
        guarded = protected_answer(steps, evidence)
        execution["answer"] = guarded or fallback_answer(steps)
        execution["answer_source"] = "data_guard" if guarded else "fallback"
        if not guarded and steps[-1]["result"] != []:
            try:
                execution["answer"] = LLMClient().create_answer(
                    answer_question(request), evidence
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
            guarded = protected_answer(execution["steps"], evidence)

            if guarded:
                answer, source = guarded, "data_guard"

            elif execution["steps"][-1]["result"] != []:
                try:
                    chunks = []

                    for chunk in LLMClient().stream_answer(
                        answer_question(request),
                        evidence,
                    ):
                        chunks.append(chunk)
                        yield event("delta", chunk)

                    if not chunks or not "".join(chunks).strip():
                        raise ValueError("Empty answer")

                    answer, source = "".join(chunks), "model"

                except Exception:
                    logging.getLogger(__name__).warning(
                        "Answer stream failed; returning data fallback"
                    )

            execution.update(
                answer=answer,
                answer_source=source,
            )

            try:
                titles = []

                for step in execution["steps"]:
                    result = step.get("result")

                    if isinstance(result, list):
                        for row in result:
                            if isinstance(row, dict):
                                for key in ("title", "normalized_title"):
                                    value = row.get(key)

                                    if isinstance(value, str) and value.strip():
                                        titles.append(value.strip())

                if titles:
                    execution["title_translations"] = translate_titles(
                        list(dict.fromkeys(titles)),
                        LLMClient(),
                    )

            except Exception:
                # 中文岗位名只是展示增强，失败不能影响分析结果。
                pass

            yield event("result", execution)

        except Exception as exc:
            message = (
                exc.detail
                if isinstance(exc, HTTPException)
                else "请求未完成，请稍后重试。"
            )
            yield event("error", str(message))

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )