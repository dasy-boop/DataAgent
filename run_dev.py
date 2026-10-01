"""Local development only: restart an owned worker when backend Python files change."""

import multiprocessing
from pathlib import Path
import signal
import socket
import time


PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "backend"
HOST = "127.0.0.1"
PORT = 8001


def snapshot():
    files = {}
    for path in BACKEND_DIR.rglob("*.py"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        files[str(path)] = (stat.st_mtime_ns, stat.st_size)
    return files


def serve():
    # Parent owns Ctrl+C handling and cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import uvicorn

    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)


def start_worker(context):
    worker = context.Process(target=serve, name="dataagent-dev-server", daemon=True)
    worker.start()
    print(f"[dev] Started worker PID={worker.pid}", flush=True)
    return worker


def stop_worker(worker):
    if worker.is_alive():
        # Windows: terminate this owned process directly; no console broadcast.
        worker.terminate()
        worker.join(timeout=5)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=5)
    if worker.is_alive():
        raise RuntimeError("Owned server process could not be stopped")
    worker.join()
    worker.close()


def main():
    # Refuse a duplicate launch instead of picking another port or killing its owner.
    try:
        with socket.socket() as probe:
            probe.bind((HOST, PORT))
    except OSError:
        print(f"[dev] Port {PORT} is occupied. Stop the existing server first.", flush=True)
        return 1

    context = multiprocessing.get_context("spawn")
    state = snapshot()
    worker = None
    failure_reported = False
    try:
        worker = start_worker(context)
        print(f"[dev] Watching {BACKEND_DIR}. Docs: http://{HOST}:{PORT}/docs", flush=True)
        while True:
            time.sleep(0.5)
            current = snapshot()
            if current != state:
                # Wait for editor writes to settle before importing the new code.
                time.sleep(0.5)
                state = snapshot()
                print("[dev] Backend changed; restarting owned worker...", flush=True)
                stop_worker(worker)
                worker = None
                worker = start_worker(context)
                failure_reported = False
            elif not worker.is_alive() and not failure_reported:
                print("[dev] Server exited. Fix the error and save a backend file to retry.", flush=True)
                failure_reported = True
    except KeyboardInterrupt:
        print("\n[dev] Stopping...", flush=True)
    finally:
        if worker is not None:
            stop_worker(worker)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
