"""开发服务器：监控 backend、agent、tools，代码变化后重启服务。"""

import multiprocessing
from pathlib import Path
import signal
import socket
import time


PROJECT_DIR = Path(__file__).resolve().parent

WATCH_DIRS = [
    PROJECT_DIR / "backend",
    PROJECT_DIR / "agent",
    PROJECT_DIR / "tools",
]

HOST = "127.0.0.1"
PORT = 8001


def snapshot():
    """记录被监控目录中 Python 文件的修改时间和大小。"""
    files = {}

    for directory in WATCH_DIRS:
        for path in directory.rglob("*.py"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue

            files[str(path)] = (
                stat.st_mtime_ns,
                stat.st_size,
            )

    return files


def serve():
    """启动接口服务，由父进程负责处理停止操作。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


def start_worker(context):
    worker = context.Process(
        target=serve,
        name="dataagent-dev-server",
        daemon=True,
    )
    worker.start()

    print(
        f"[dev] 已启动服务进程，PID={worker.pid}",
        flush=True,
    )
    return worker


def stop_worker(worker):
    """只停止当前启动器创建的服务进程。"""
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)

    if worker.is_alive():
        worker.kill()
        worker.join(timeout=5)

    if worker.is_alive():
        raise RuntimeError("服务进程未能停止")

    worker.join()
    worker.close()


def main():
    # 端口被占用时退出，避免重复启动。
    try:
        with socket.socket() as probe:
            probe.bind((HOST, PORT))
    except OSError:
        print(
            f"[dev] 端口 {PORT} 已被占用，请先停止原来的服务。",
            flush=True,
        )
        return 1

    context = multiprocessing.get_context("spawn")
    state = snapshot()
    worker = None
    failure_reported = False

    try:
        worker = start_worker(context)

        print(
            "[dev] 正在监控：backend、agent、tools",
            flush=True,
        )
        print(
            f"[dev] 分析网页：http://{HOST}:{PORT}/agent.html",
            flush=True,
        )
        print(
            f"[dev] 接口文档：http://{HOST}:{PORT}/docs",
            flush=True,
        )

        while True:
            time.sleep(0.5)
            current = snapshot()

            if current != state:
                # 等待编辑器完成文件保存。
                time.sleep(0.5)
                state = snapshot()

                print(
                    "[dev] 检测到代码变化，正在重新启动服务……",
                    flush=True,
                )

                stop_worker(worker)
                worker = None
                worker = start_worker(context)
                failure_reported = False

            elif not worker.is_alive() and not failure_reported:
                print(
                    "[dev] 服务已退出。请修复错误，并保存 "
                    "backend、agent 或 tools 中的 Python 文件以重试。",
                    flush=True,
                )
                failure_reported = True

    except KeyboardInterrupt:
        print(
            "\n[dev] 正在停止服务……",
            flush=True,
        )

    finally:
        if worker is not None:
            stop_worker(worker)

    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())