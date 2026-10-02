import json
from typing import Any

from openai import OpenAI

from agent.schemas import AgentPlan
from agent.settings import get_llm_settings


SYSTEM_PROMPT = """
你是招聘数据分析规划器。
先判断问题是否能用提供的数据字段和工具完成，再生成计划。

一、可完成的问题
status 使用 ready。
reason_code 和 message 必须为 null。
tool_calls 必须包含至少一个工具调用。
只能使用提供的字段、工具及参数。
不得用不相关的统计代替用户真正要求的分析。

二、缺少必要数据
status 使用 cannot_answer。
reason_code 使用 missing_data。
message 用简体中文说明缺少什么数据。
tool_calls 必须为空列表。
例如：数据没有面试评价，不能据此判断公司的面试难度。
字段存在不代表一定有有效值；不要仅凭字段名断言数据全部缺失。

三、工具不支持
status 使用 cannot_answer。
reason_code 使用 unsupported_operation。
message 用简体中文说明当前不支持的操作。
tool_calls 必须为空列表。
例如：只有筛选和统计工具时，不能预测明年薪资。
不要用历史均值冒充预测，也不要编造新工具。

四、问题需要补充信息
status 使用 needs_clarification。
reason_code 使用 ambiguous_request。
message 用简体中文提出一个具体的补充问题。
tool_calls 必须为空列表。
例如：用户只说“帮我分析一下”，应询问关注哪个岗位或指标。
只有缺少会影响分析的重要信息时才追问。
问题已经明确时，不要无故追问。

五、数据与字段规则
岗位名称默认使用 title。
只有用户明确要求标准化岗位名称时，才使用 normalized_title。
公司名称使用 company_name。
国家统计优先使用 country_clean；没有该字段时使用 country。
用户要求原始国家值时使用 country。
技能要求字段为 skills_required，但直接统计整列取值不等于拆分单个技能后排名。

岗位数据主要使用英文。中文岗位名称应转换为对应筛选关键词：
数据分析师：Data Analyst
数据科学家：Data Scientist
数据工程师：Data Engineer
软件工程师：Software Engineer

如果用户明确指定原文关键词或字段名，应尊重其指定。
不要翻译 JSON 键、工具名和 column 参数。
reasoning 必须使用简体中文，question 保留用户原问题。

六、执行边界
filter_rows 是忽略大小写的文本包含匹配，不是精确相等或数值大小比较。
薪资统计必须明确字段、币种和周期，不能混合不同单位得出均值。
缺少必要口径时应追问；现有工具无法完成必要处理时说明不支持。
预测筛选结果可能为空，不是拒绝执行的理由。
只要字段和操作可用，就正常生成计划，由执行结果判断有无匹配记录。
不能为了给出答案而编造数据、字段、参数或计算结果。

返回一个合法 JSON 对象，不要返回 Markdown 或代码围栏。
"""


class LLMClient:
    def __init__(self):
        settings = get_llm_settings()

        if not settings.api_key:
            raise RuntimeError("未配置 LLM_API_KEY")

        if not settings.model:
            raise RuntimeError("未配置 LLM_MODEL")

        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        self.model = settings.model

    def create_plan(
        self,
        question: str,
        columns: list[str],
        tool_descriptions: list[dict[str, Any]],
    ) -> AgentPlan:
        prompt = {
            "question": question,
            "columns": columns,
            "available_tools": tool_descriptions,
            "output_format": {
                "question": "用户原问题",
                "reasoning": "中文判断依据或分析步骤",
                "status": "ready / cannot_answer / needs_clarification",
                "reason_code": (
                    "null / missing_data / "
                    "unsupported_operation / ambiguous_request"
                ),
                "message": "可执行时为 null，否则为中文解释或补充问题",
                "tool_calls": [
                    {
                        "name": "提供的工具名称",
                        "arguments": {},
                    }
                ],
            },
            "constraints": [
                "必须返回上述全部字段",
                "ready 时 reason_code 和 message 为 null，tool_calls 非空",
                "cannot_answer 时 reason_code 为 missing_data 或 unsupported_operation",
                "needs_clarification 时 reason_code 为 ambiguous_request",
                "非 ready 时 message 非空，tool_calls 为 []",
            ],
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        prompt,
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("模型未返回分析计划")

        return AgentPlan.model_validate(json.loads(content))