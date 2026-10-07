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


def test_skill_result_checks_independent_counts_and_rejects_duplicates():
    expected = {'kind': 'skill_counts', 'matched_records': 3,
                'valid_skill_records': 2, 'candidates': {'sql': 2, 'python': 1}}
    actual = {'matched_records': 3, 'valid_skill_records': 2,
              'skills': [{'skill': 'sql', '岗位记录数': 2},
                         {'skill': 'python', '岗位记录数': 1}]}
    assert runner.check_result(actual, expected)[0]
    assert not runner.check_result({**actual, 'valid_skill_records': 3}, expected)[0]
    assert not runner.check_result({**actual, 'skills': [actual['skills'][0]] * 2}, expected)[0]
    assert not runner.check_result({**actual, 'skills': [
        actual['skills'][0], {'skill': 'python', '岗位记录数': 2}]}, expected)[0]


def test_skill_plan_accepts_enough_top_n_but_rejects_too_small_or_extra_arguments():
    expected = [{'name': 'filter_rows', 'arguments': {'column': 'title', 'keyword': 'Data Analyst'}},
                {'name': 'count_skills', 'arguments': {'top_n': 10}}]
    assert runner.match_calls(expected, expected)
    assert runner.match_calls([expected[0], {'name': 'count_skills', 'arguments': {}}], expected)
    assert runner.match_calls([expected[0], {'name': 'count_skills', 'arguments': {'top_n': 50}}], expected)
    assert not runner.match_calls([expected[0], {'name': 'count_skills', 'arguments': {'top_n': 1}}], expected)
    assert not runner.match_calls([expected[0], {'name': 'count_skills', 'arguments': {'column': 'skills_required'}}], expected)


def test_compare_skills_checks_each_group_and_its_own_denominator():
    expected = {'kind': 'compare_skill_counts', 'column': 'title', 'groups': [
        {'keyword': 'Data Analyst', 'matched_records': 3, 'valid_skill_records': 2,
         'candidates': {'sql': 2}},
        {'keyword': 'Software Engineer', 'matched_records': 4, 'valid_skill_records': 4,
         'candidates': {'python': 2}},
    ]}
    actual = {'column': 'title', 'groups': [
        {'keyword': 'Data Analyst', 'matched_records': 3, 'valid_skill_records': 2,
         'skills': [{'skill': 'sql', '岗位记录数': 2, '占可解析记录比例(%)': 100.0}]},
        {'keyword': 'Software Engineer', 'matched_records': 4, 'valid_skill_records': 4,
         'skills': [{'skill': 'python', '岗位记录数': 2, '占可解析记录比例(%)': 50.0}]},
    ]}
    assert runner.check_result(actual, expected)[0]
    wrong_denominator = {**actual, 'groups': [actual['groups'][0],
        {**actual['groups'][1], 'skills': [{'skill': 'python', '岗位记录数': 2,
                                          '占可解析记录比例(%)': 100.0}]}]}
    assert not runner.check_result(wrong_denominator, expected)[0]
    assert not runner.check_result({**actual, 'groups': list(reversed(actual['groups']))}, expected)[0]


def test_compare_skills_plan_requires_independent_groups():
    expected = [{'name': 'compare_skills', 'arguments': {'column': 'title',
        'keyword_a': 'Data Analyst', 'keyword_b': 'Software Engineer', 'top_n': 5}}]
    assert runner.match_calls(expected, expected)
    assert runner.match_calls([{'name': 'compare_skills', 'arguments': {'column': 'title',
        'keyword_a': ' data analyst ', 'keyword_b': 'software engineer'}}], expected)
    assert not runner.match_calls([{'name': 'compare_skills', 'arguments': {'column': 'title',
        'keyword_a': 'Software Engineer', 'keyword_b': 'Data Analyst', 'top_n': 5}}], expected)


def test_compare_skills_plan_accepts_equivalent_group_filters_only():
    expected = [{'name': 'compare_skills', 'arguments': {'column': 'title',
        'keyword_a': 'Data Analyst', 'keyword_b': 'Software Engineer', 'top_n': 5}}]
    groups = [{'label': keyword, 'conditions': [
        {'column': 'title', 'operator': 'contains', 'value': keyword}]}
        for keyword in ('Data Analyst', 'Software Engineer')]
    actual = [{'name': 'compare_skills', 'arguments': {'groups': groups, 'top_n': 10}}]
    assert runner.match_calls(actual, expected)
    bad_groups = [groups[0], {**groups[1], 'conditions': [
        {'column': 'country_clean', 'operator': 'contains', 'value': 'Software Engineer'}]}]
    assert not runner.match_calls([{'name': 'compare_skills', 'arguments': {'groups': bad_groups}}], expected)
    assert not runner.match_calls([{'name': 'compare_skills', 'arguments': {
        'groups': groups, 'column': 'title'}}], expected)


def test_compare_skills_result_checks_group_filters_and_counts():
    expected = {'kind': 'compare_skill_counts', 'column': 'title', 'groups': [
        {'keyword': 'Data Analyst', 'matched_records': 3, 'valid_skill_records': 2,
         'candidates': {'sql': 2}},
        {'keyword': 'Software Engineer', 'matched_records': 4, 'valid_skill_records': 4,
         'candidates': {'python': 2}}]}
    groups = [
        {'keyword': 'Data Analyst', 'matched_records': 3, 'valid_skill_records': 2,
         'conditions': [{'column': 'title', 'operator': 'contains', 'value': 'Data Analyst'}],
         'skills': [{'skill': 'sql', '岗位记录数': 2, '占可解析记录比例(%)': 100.0}]},
        {'keyword': 'Software Engineer', 'matched_records': 4, 'valid_skill_records': 4,
         'conditions': [{'column': 'title', 'operator': 'contains', 'value': 'Software Engineer'}],
         'skills': [{'skill': 'python', '岗位记录数': 2, '占可解析记录比例(%)': 50.0}]}]
    assert runner.check_result({'column': None, 'groups': groups}, expected)[0]
    bad_filter = [{**groups[0], 'conditions': [{'column': 'country_clean',
        'operator': 'contains', 'value': 'Data Analyst'}]}, groups[1]]
    assert not runner.check_result({'column': None, 'groups': bad_filter}, expected)[0]
    bad_count = [groups[0], {**groups[1], 'matched_records': 5}]
    assert not runner.check_result({'column': None, 'groups': bad_count}, expected)[0]
