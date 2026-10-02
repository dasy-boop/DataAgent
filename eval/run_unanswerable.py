import json
import re
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EVAL_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8001/agent/ask"


def ask_agent(question):
    body = json.dumps(
        {"question": question},
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        return json.load(response)


def evaluate_case(case):
    result = {
        "id": case["id"],
        "question": case["question"],
        "expected_status": case["expected_status"],
        "expected_reason_code": case["expected_reason_code"],
        "passed": False,
    }

    try:
        response = ask_agent(case["question"])
        plan = response.get("plan")
        message = response.get("message")

        checks = {
            "状态正确": (
                response.get("status") == case["expected_status"]
            ),
            "原因正确": (
                response.get("reason_code")
                == case["expected_reason_code"]
            ),
            "包含中文说明": (
                isinstance(message, str)
                and bool(message.strip())
                and bool(re.search(r"[\u4e00-\u9fff]", message))
            ),
            "没有执行步骤": response.get("steps") == [],
            "计划一致且没有工具调用": (
                isinstance(plan, dict)
                and plan.get("status") == case["expected_status"]
                and plan.get("reason_code")
                == case["expected_reason_code"]
                and plan.get("tool_calls") == []
            ),
        }

        result["checks"] = checks
        result["actual_response"] = response
        result["passed"] = all(checks.values())

    except HTTPError as exc:
        result["error"] = (
            f"HTTP {exc.code}: "
            + exc.read().decode("utf-8", errors="replace")[:2000]
        )
    except URLError as exc:
        result["error"] = f"连接失败：{exc.reason}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def main():
    cases_path = EVAL_DIR / "unanswerable_cases.json"

    if not cases_path.is_file():
        print(f"找不到题目文件：{cases_path}")
        return 1

    cases = json.loads(
        cases_path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(cases, list) or not cases:
        raise ValueError("题目文件必须包含非空列表")

    for case in cases:
        for key in (
            "id",
            "question",
            "expected_status",
            "expected_reason_code",
        ):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"评测题缺少有效字段：{key}")

    print(
        f"无法回答与追问评测：共 {len(cases)} 题，每题调用一次模型。",
        flush=True,
    )

    results = []

    for case in cases:
        result = evaluate_case(case)
        results.append(result)

        status = "通过" if result["passed"] else "未通过"
        print(f"[{status}] {case['id']}", flush=True)

        if "error" in result:
            print(result["error"], flush=True)
        else:
            for name, passed in result["checks"].items():
                if not passed:
                    print(f"  未通过检查：{name}", flush=True)

            response = result["actual_response"]
            print(
                f"  实际状态：{response.get('status')}；"
                f"实际原因：{response.get('reason_code')}",
                flush=True,
            )
            print(
                f"  中文说明：{response.get('message')}",
                flush=True,
            )

    passed_count = sum(item["passed"] for item in results)

    report = {
        "mode": "answerability",
        "created_at": datetime.now().astimezone().isoformat(),
        "total": len(results),
        "passed": passed_count,
        "pass_rate": passed_count / len(results),
        "results": results,
    }

    report_dir = EVAL_DIR / "reports"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = (
        report_dir / f"unanswerable_{timestamp}.json"
    )

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n通过：{passed_count}/{len(results)}")
    print(f"报告已保存：{report_path}")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())