"""Run a live, three-turn regression check through the browser's streaming API."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen


QUESTIONS = (
    "西部数据有哪些岗位？请结合岗位信息做简要分析。",
    "这些岗位主要需要什么技能？",
    "其中哪些岗位可能适合应届生？请根据经验要求说明依据。",
)
REPORT_DIR = Path(__file__).resolve().parent / "reports"


def ask_stream(base_url, question, history):
    payload = json.dumps({"question": question, "history": history}, ensure_ascii=False).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + "/agent/ask/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deltas = []
    result = None
    with urlopen(request, timeout=180) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line.decode("utf-8"))
            if event.get("type") == "delta":
                deltas.append(event["data"])
            elif event.get("type") == "result":
                result = event["data"]
            elif event.get("type") == "error":
                raise RuntimeError(str(event.get("data")))
    if not isinstance(result, dict):
        raise RuntimeError("流式接口没有返回最终结果")
    return result, deltas


def check_turn(index, result, deltas):
    """Check scope and response behavior, without relying on exact model wording."""
    checks = {}
    checks["completed"] = result.get("status") == "completed"
    checks["model_answer_streamed"] = (
        result.get("answer_source") == "model"
        and bool(deltas)
        and "".join(deltas) == result.get("answer")
    )
    calls = result.get("plan", {}).get("tool_calls", [])
    checks["company_scope_kept"] = any(
        call.get("name") == "filter_rows"
        and call.get("arguments", {}).get("column") == "company_name"
        and "western digital" in str(call.get("arguments", {}).get("keyword", "")).lower()
        for call in calls if isinstance(call, dict)
    )
    evidence = result.get("evidence") or []
    company_rows = [item for item in evidence if item.get("tool") == "filter_rows"]
    scoped = company_rows[-1] if company_rows else {}
    rows = scoped.get("result") or []
    checks["eight_scoped_records"] = scoped.get("total_rows") == 8 and len(rows) == 8
    checks["records_match_company"] = bool(rows) and all(
        "western digital" in str(row.get("company_name", "")).lower() for row in rows
    )
    answer = str(result.get("answer") or "")
    if index == 1:
        checks["skills_explained"] = any(
            term in answer.lower() for term in ("技能", "能力", "要求", "skill")
        ) and len(answer) >= 60
    if index == 2:
        checks["experience_uncertainty_explained"] = any(
            term in answer for term in ("无法判断", "不能判断", "未说明", "未明确", "不确定", "缺失")
        )
        checks["not_all_fresh_graduate_suitable"] = not any(
            term in answer for term in ("8条都适合", "8个岗位都适合", "全部适合应届生", "均适合应届生")
        )
    return checks


def run(base_url):
    history = []
    turns = []
    for index, question in enumerate(QUESTIONS):
        started = time.perf_counter()
        try:
            result, deltas = ask_stream(base_url, question, history[-6:])
            checks = check_turn(index, result, deltas)
            answer = str(result.get("answer") or "")
            turn = {
                "question": question,
                "status": result.get("status"),
                "answer_source": result.get("answer_source"),
                "answer": answer,
                "checks": checks,
                "passed": all(checks.values()),
                "seconds": round(time.perf_counter() - started, 2),
            }
            # Match the browser: only completed answers become conversation context.
            if result.get("status") == "completed" and answer:
                history.append({"question": question, "answer": answer})
        except Exception as exc:
            turn = {"question": question, "passed": False, "error": str(exc),
                    "seconds": round(time.perf_counter() - started, 2)}
        turns.append(turn)
        outcome = "通过" if turn["passed"] else "失败"
        print(f"[{outcome}] 第 {index + 1} 轮：{question}", flush=True)
        for name, passed in turn.get("checks", {}).items():
            if not passed:
                print(f"  未通过：{name}", flush=True)
        if "error" in turn:
            print(f"  错误：{turn['error']}", flush=True)
        if not turn["passed"]:
            break
    return {"mode": "live_chat_stream", "generated_at": datetime.now().astimezone().isoformat(),
            "base_url": base_url, "passed": len(turns) == len(QUESTIONS) and all(t["passed"] for t in turns),
            "turns": turns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    report = run(args.base_url)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = "chat_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    path = REPORT_DIR / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"通过：{sum(t['passed'] for t in report['turns'])}/{len(QUESTIONS)}")
    print(f"报告已保存：{path}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
