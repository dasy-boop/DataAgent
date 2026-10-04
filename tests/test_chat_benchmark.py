import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "chat_benchmark", Path(__file__).resolve().parents[1] / "eval" / "run_chat_benchmark.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_result(answer="有一条未说明经验要求，无法判断是否适合应届生。"):
    return {
        "status": "completed", "answer_source": "model", "answer": answer,
        "plan": {"tool_calls": [{"name": "filter_rows", "arguments": {
            "column": "company_name", "keyword": "Western Digital"}}]},
        "evidence": [{"tool": "filter_rows", "total_rows": 8,
                      "result": [{"company_name": "Western Digital"}] * 8}],
    }


def test_three_turns_send_previous_completed_answers(monkeypatch):
    received = []

    def fake_ask(base_url, question, history):
        received.append((question, list(history)))
        answer = "这些岗位需要技能和经验。" * 5 if len(received) == 2 else (
            "有一条未说明经验要求，无法判断是否适合应届生。" if len(received) == 3 else
            "西部数据共有八条招聘记录。")
        return make_result(answer), [answer]

    monkeypatch.setattr(runner, "ask_stream", fake_ask)
    report = runner.run("http://test")
    assert report["passed"]
    assert len(received[0][1]) == 0
    assert received[1][1] == [{"question": runner.QUESTIONS[0],
                                "answer": "西部数据共有八条招聘记录。"}]
    assert len(received[2][1]) == 2


def test_missing_company_scope_and_uncertain_experience_fail():
    result = make_result("8条都适合应届生。")
    result["plan"]["tool_calls"] = []
    checks = runner.check_turn(2, result, [result["answer"]])
    assert not checks["company_scope_kept"]
    assert not checks["experience_uncertainty_explained"]
    assert not checks["not_all_fresh_graduate_suitable"]


def test_fallback_is_not_counted_as_streamed_model_answer():
    result = make_result()
    result["answer_source"] = "fallback"
    assert not runner.check_turn(2, result, [result["answer"]])["model_answer_streamed"]
