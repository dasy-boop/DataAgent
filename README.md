# DataAgent 智能数据分析

基于 FastAPI、Pandas 和大语言模型构建的数据分析 Agent。用户用自然语言提问，系统生成分析计划，调用受限的数据工具，并返回分析结果。项目也包含自动评测脚本和评测报告页面。

## 功能

- 查看数据集概况、质量和样例数据
- 分析岗位技能
- 根据自然语言问题生成分析计划并执行
- 结合本次查询结果生成中文回答，支持当前对话内的连续追问
- 逐步显示回答、查看数据依据，并在当前浏览器保存历史对话
- 使用工具筛选岗位、统计类别数量、汇总数值
- 合并常见国家名称别名后进行统计
- 对无法回答的问题说明原因，或请求用户补充信息
- 运行自动评测并查看 JSON 报告

## 技术栈

- Python
- FastAPI
- Pandas
- Pydantic
- pytest
- OpenAI 兼容的大语言模型接口

## 启动项目

需要 Python 3.10 或更新版本。在 PowerShell 中进入项目目录，并安装依赖：

```powershell
cd D:\DataAgent
py -3.10 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

若 `.venv` 已存在，可以跳过创建虚拟环境的命令。开发和测试依赖使用 `requirements-dev.txt` 安装。

复制 `.env.example` 为 `.env`，填入可用的模型接口地址、模型名称和密钥。调用模型生成中文分析时，会向所配置的接口发送当前问题、最多 6 轮当前对话的提问与回答，以及本次查询的统计结果和最多 8 条岗位记录样本。

启动开发服务：

```powershell
.\run_dev.cmd
```

服务启动后可以访问：

- 智能分析页面：http://127.0.0.1:8001/agent.html
- 自动评测页面：http://127.0.0.1:8001/eval.html
- API 文档：http://127.0.0.1:8001/docs

## 配置模型接口

项目从本地 `.env` 文件读取模型配置。请参考 `.env.example` 配置自己的接口地址、模型名称和密钥。

不要把包含真实密钥的 `.env` 文件提交到 GitHub。

## 数据集

项目默认读取：

```text
data/raw/nextgig_jobs_2026-06.parquet
```

请确认本地存在该数据文件。当前 Git 版本包含该数据文件；如果单独复制源码，请一并准备该文件。

## 自动测试

在项目根目录的 PowerShell 终端运行：

```powershell
& "D:\DataAgent\.venv\Scripts\python.exe" -m pytest "D:\DataAgent\tests" -q
```

最近一次全量测试结果：69 项通过。

## 自动评测

正常分析评测：

```powershell
& "D:\DataAgent\.venv\Scripts\python.exe" "D:\DataAgent\eval\run_benchmark.py"
```

无法回答和需要澄清的问题评测：

```powershell
& "D:\DataAgent\.venv\Scripts\python.exe" "D:\DataAgent\eval\run_unanswerable.py"
```

评测报告保存在：

```text
eval/reports/
```

报告可以在自动评测页面中选择并查看。最近一次记录的正常分析评测为 5/5，通过拒答与澄清评测为 3/3。这些分数只反映当前评测题目，不代表所有问题的准确率。

`eval/reports/` 中的本地生成报告不会纳入 Git 提交。历史对话只保存在当前浏览器；清除浏览器数据后无法恢复。

## API 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/dataset/overview` | 数据集概况 |
| GET | `/dataset/quality` | 数据质量检查 |
| GET | `/dataset/preview` | 预览数据 |
| GET | `/analysis/skills` | 分析技能 |
| POST | `/agent/plan` | 生成分析计划 |
| POST | `/agent/execute` | 执行已有计划 |
| POST | `/agent/ask` | 从自然语言问题生成计划并返回分析结果 |
| GET | `/health` | 健康检查 |

请求和响应示例可以在启动服务后通过 `/docs`
