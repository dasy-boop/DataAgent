import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.error import URLError

import pytest


spec = importlib.util.spec_from_file_location(
    'benchmark_runner', Path(__file__).resolve().parents[1] / 'eval' / 'run_benchmark.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def expected():
    return {'kind': 'counts', 'column': 'country_clean', 'row_count': 2, 'cutoff': 1,
            'candidates': {'United States': 2, 'Canada': 1, 'France': 1}}


def test_counts_allow_boundary_tie(expected):
    rows = [{'country_clean': 'United States', 'count': 2},
            {'country_clean': 'France', 'count': 1}]
    assert runner.check_result(rows, expected)[0]


@pytest.mark.parametrize('rows', [
    [{'country_clean': 'Canada', 'count': 4}, {'country_clean': 'United States', 'count': 2}],
    [{'country_clean': 'Canada', 'count': 1}, {'country_clean': 'France', 'count': 1}],
    [{'country_clean': 'United States', 'count': 2}, {'country_clean': 'United States', 'count': 2}],
    [{'country_clean': 'Canada', 'count': 1}, {'country_clean': 'United States', 'count': 2}],
    [{'country_clean': 'United States', 'count': 2}, {'country_clean': 'Canada', 'count': True}],
])
def test_reject_wrong_counts_omitted_leader_duplicates_unsorted_or_boolean(rows, expected):
    assert not runner.check_result(rows, expected)[0]


def test_projection_checks_duplicates_and_values_not_just_length():
    rows = [{'title': 'Data Analyst', 'country_clean': 'Canada'},
            {'title': 'Data Analyst', 'country_clean': 'United States'}]
    values = sorted(json.dumps([r['title'], r['country_clean']], ensure_ascii=False,
                               separators=(',', ':')) for r in rows)
    expected = {'kind': 'projection', 'columns': ['title', 'country_clean'], 'row_count': 2,
                'sha256': hashlib.sha256('\n'.join(values).encode()).hexdigest()}
    assert runner.check_result(list(reversed(rows)), expected)[0]
    assert not runner.check_result([rows[0], rows[0]], expected)[0]


def test_plan_requires_step_order_but_accepts_case_and_default():
    expected = [{'name': 'filter_rows', 'arguments': {'column': 'title', 'keyword': 'Data Analyst'}},
                {'name': 'count_values', 'arguments': {'column': 'country_clean', 'top_n': 10}}]
    actual = [{'name': 'filter_rows', 'arguments': {'column': 'title', 'keyword': ' data analyst '}},
              {'name': 'count_values', 'arguments': {'column': 'country_clean'}}]
    assert runner.match_calls(actual, expected)
    assert not runner.match_calls(list(reversed(actual)), expected)
    assert not runner.match_calls(actual[:1], expected)


def test_execution_only_does_not_claim_model_success(monkeypatch, expected):
    requests = []

    def fake_request(url, payload):
        requests.append(url)
        return {'steps': [{'result': [{'country_clean': 'United States', 'count': 2},
                                      {'country_clean': 'Canada', 'count': 1}]}]}

    monkeypatch.setattr(runner, 'request_json', fake_request)
    case = {'id': 'demo', 'question': '示例', 'expected_calls': [], 'expected_result': expected}
    result = runner.evaluate_case(case, 'http://test', execution_only=True)
    assert result['passed']
    assert result['plan_passed'] is None
    assert requests == ['http://test/agent/execute']


def test_connection_failure_is_not_successful_empty_result(monkeypatch):
    case = {'id': 'empty', 'question': '空结果', 'expected_calls': [],
            'expected_result': {'kind': 'counts', 'column': 'country_clean',
                                'row_count': 0, 'cutoff': 0, 'candidates': {}}}

    def fail(*args):
        raise URLError('offline')

    monkeypatch.setattr(runner, 'request_json', fail)
    result = runner.evaluate_case(case, 'http://test', execution_only=True)
    assert not result['passed']
    assert not result['execution_succeeded']
    assert 'error' in result
    monkeypatch.setattr(runner, 'request_json', lambda *args: {'steps': [{'result': []}]})
    assert runner.evaluate_case(case, 'http://test', execution_only=True)['passed']


def test_correct_plan_with_wrong_answer_fails(monkeypatch, expected):
    calls = [{'name': 'count_values', 'arguments': {'column': 'country_clean', 'top_n': 2}}]
    monkeypatch.setattr(runner, 'request_json', lambda *args: {
        'plan': {'tool_calls': calls}, 'steps': [{'result': []}]})
    case = {'id': 'wrong', 'question': '示例', 'expected_calls': calls, 'expected_result': expected}
    result = runner.evaluate_case(case, 'http://test')
    assert result['plan_passed']
    assert result['execution_succeeded']
    assert not result['result_passed']
    assert not result['passed']
