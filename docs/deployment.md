# 部署 DataAgent 到 Render

前端和 FastAPI 由同一个 Web Service 提供。仓库根目录的 `render.yaml` 使用 Dockerfile 构建应用，并将健康检查设为 `/health`。镜像包含固定的 `nextgig_jobs_2026-06.parquet` 数据快照。

1. 在 Render 创建账号并连接 GitHub 仓库 `dasy-boop/DataAgent`。
2. 选择 **New → Blueprint**，选中仓库与 `main` 分支，让 Render 读取根目录的 `render.yaml`。创建前确认服务名与计划；配置默认选 `free`。
3. 在 Render 页面填写 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。使用你实际可用的 OpenAI 兼容接口地址和模型名；不要把密钥填进 GitHub、聊天记录或 `render.yaml`。这些值由 Render 保存为服务环境变量。
4. 等待构建与部署完成。打开 Render 提供的 `https://…onrender.com/health`，应返回 `status: ok`；再打开同域名的 `/agent.html`，提出一个简单的招聘数据问题，确认模型接口可用。

`render.yaml` 中的 `sync: false` 会在**首次创建 Blueprint**时要求填写这些变量；若服务已经创建，请到该服务的 Environment 页面手动设置。没有正确的模型配置时，数据接口和静态页面可能可用，但自然语言分析无法完成。公开演示会产生模型调用费用，建议在模型提供方设定预算或用量上限。

Render 免费 Web Service 空闲后会休眠，下一次访问可能需要约一分钟启动；免费实例的本地文件也不持久。项目数据已打包在镜像中，重新部署会恢复同一快照。若冷启动或内存限制影响演示，可在 Render 升级服务计划。当前还没有实际部署 URL，完成上述步骤后才能验证公网访问。

本机有 Docker 时，可先验证同一镜像：

```powershell
docker build -t dataagent .
docker run --rm --env-file .env -e PORT=8000 -p 8000:8000 dataagent
```

本机验证地址是 `http://127.0.0.1:8000/health` 和 `http://127.0.0.1:8000/agent.html`。`.env` 已被 Git 和 Docker 构建上下文排除；`run_dev.cmd` 仅用于 Windows 本地开发。
