# 本地开发启动

在 PyCharm 的 PowerShell 终端运行：

```powershell
Set-Location D:\DataAgent
.\run_dev.cmd
```

打开 http://127.0.0.1:8001/docs 。统一使用 8001，不要另外启动 Uvicorn。
端口被占用时，启动脚本会退出，不会关闭其他程序；先使用现有服务或停止原服务再启动。

保存 backend 内的 Python 文件后，启动器会等待写入稳定，停止自己创建的服务子进程并启动新进程。
看到新的 `Application startup complete` 后刷新浏览器文档。浏览器不会自动刷新。
代码有语法或导入错误时，查看终端提示，修正并保存后会重试。
在启动脚本的终端按 Ctrl+C，可以停止监听器和服务子进程。

## 实现与限制

此 Windows 环境中的 Uvicorn 内置重载会停留在 `Reloading...`，旧进程仍然响应。独立控制台也曾复现。
其重启代码依赖 Windows 控制台信号并等待旧进程退出。
run_dev.py 改用 multiprocessing 启动服务并直接终止自有子进程，绕过该信号重载流程。
仅监视 backend 下的 .py 文件，不监视数据或虚拟环境，不修改第三方库。

此脚本仅用于本地开发：保存代码触发重启时，在途请求可能被中断。生产环境不使用该启动方式。
