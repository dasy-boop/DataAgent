"""运行中文问题评测；--execution-only 使用固定计划，不调用模型。"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(url, body):
    request = Request(url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def match_calls(actual, expected):
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for got, want in zip(actual, expected):
        if not isinstance(got, dict) or got.get('name') != want['name']:
            return False
        args = got.get('arguments')
        if not isinstance(args, dict):
            return False
        args = dict(args)
        target = dict(want['arguments'])
        if want['name'] == 'count_values':
            args.setdefault('top_n', 10)
            target.setdefault('top_n', 10)
            if type(args['top_n']) is not int:
                return False
        if want['name'] == 'filter_rows':
            for values in (args, target):
                conditions = values.pop('conditions', None)
                if conditions is not None:
                    if (not isinstance(conditions, list) or len(conditions) != 1
                            or not isinstance(conditions[0], dict)
                            or set(conditions[0]) != {'column', 'operator', 'value'}
                            or conditions[0]['operator'] != 'contains'
                            or values):
                        return False
                    values['column'] = conditions[0]['column']
                    values['keyword'] = conditions[0]['value']
                if isinstance(values.get('keyword'), str):
                    values['keyword'] = values['keyword'].strip().lower()
        if want['name'] == 'count_skills':
            # Both compared skills are in the top two; a larger valid limit is equivalent.
            top_n = args.get('top_n', 10)
            if set(args) - {'top_n'} or type(top_n) is not int or not 2 <= top_n <= 50:
                return False
            continue
        if want['name'] == 'compare_skills':
            if set(args) - {'column', 'keyword_a', 'keyword_b', 'top_n'}:
                return False
            top_n = args.get('top_n', 10)
            if type(top_n) is not int or not target.get('top_n', 10) <= top_n <= 50:
                return False
            if args.get('column') != target['column']:
                return False
            for key in ('keyword_a', 'keyword_b'):
                if not isinstance(args.get(key), str) or args[key].strip().lower() != target[key].strip().lower():
                    return False
            continue
        if args != target:
            return False
    return True


def check_result(actual, expected):
    if expected['kind'] == 'compare_skill_counts':
        if not isinstance(actual, dict) or actual.get('column') != expected['column']:
            return False, '技能对比字段不符'
        groups = actual.get('groups')
        if not isinstance(groups, list) or len(groups) != len(expected['groups']):
            return False, '技能对比分组数量不符'
        for group, baseline in zip(groups, expected['groups']):
            if (not isinstance(group, dict) or not isinstance(group.get('keyword'), str)
                    or group['keyword'].strip().lower() != baseline['keyword'].lower()):
                return False, '技能对比分组名称或顺序不符'
            passed, message = check_result(group, {**baseline, 'kind': 'skill_counts'})
            if not passed:
                return False, f"{baseline['keyword']}：{message}"
            denominator = group['valid_skill_records']
            for row in group['skills']:
                percentage = row.get('占可解析记录比例(%)')
                expected_percentage = round(row['岗位记录数'] / denominator * 100, 2) if denominator else 0
                if type(percentage) not in (int, float) or abs(percentage - expected_percentage) > 0.011:
                    return False, f"{baseline['keyword']}：技能占比口径不符"
        return True, '两组匹配数、可解析数、技能频次和组内占比正确'
    if expected['kind'] == 'skill_counts':
        if not isinstance(actual, dict):
            return False, '技能统计结果不是对象'
        for key in ('matched_records', 'valid_skill_records'):
            if type(actual.get(key)) is not int or actual[key] != expected[key]:
                return False, f'{key} 与独立计算的标准答案不符'
        rows = actual.get('skills')
        if not isinstance(rows, list):
            return False, '技能排名不是列表'
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('skill'), str):
                return False, '技能排名缺少技能名称'
            skill = row['skill']
            count = row.get('岗位记录数')
            if skill in seen or type(count) is not int or count < 0:
                return False, f'技能重复或计数无效：{skill}'
            seen.add(skill)
        found = {row['skill']: row['岗位记录数'] for row in rows}
        for skill, count in expected['candidates'].items():
            if found.get(skill) != count:
                return False, f'{skill} 岗位记录数不符：预期 {count}，实际 {found.get(skill)}'
        return True, '匹配数、可解析数及目标技能频次正确'
    if not isinstance(actual, list):
        return False, '最终结果不是记录列表'
    if len(actual) != expected['row_count']:
        return False, f"结果行数不符：预期 {expected['row_count']}，实际 {len(actual)}"
    if expected['kind'] == 'projection':
        columns = expected['columns']
        if not all(isinstance(row, dict) and all(c in row for c in columns) for row in actual):
            return False, '结果缺少待核对字段'
        encoded = sorted(json.dumps([row[c] for c in columns], ensure_ascii=False,
                                    separators=(',', ':')) for row in actual)
        digest = hashlib.sha256('\n'.join(encoded).encode('utf-8')).hexdigest()
        return digest == expected['sha256'], '核对筛选行数、岗位名称及清洗国家'
    if expected['kind'] != 'counts':
        return False, '不支持的标准答案类型'
    column = expected['column']
    candidates = expected['candidates']
    seen, previous = set(), float('inf')
    for row in actual:
        if not isinstance(row, dict) or column not in row or 'count' not in row:
            return False, '统计结果缺少字段'
        label, count = row[column], row['count']
        if not isinstance(label, str) or type(count) is not int:
            return False, '统计类别或数量类型错误'
        if label in seen or label not in candidates or candidates[label] != count:
            return False, f'类别重复或数量不符：{label}'
        if count > previous:
            return False, '结果未按数量降序排列'
        seen.add(label)
        previous = count
    mandatory = {label for label, count in candidates.items() if count > expected['cutoff']}
    if not mandatory.issubset(seen):
        return False, '遗漏了数量高于末位的类别'
    return True, '类别与数量正确（允许末位并列类别互换）'


def evaluate_case(case, base_url, execution_only=False):
    result = {'id': case['id'], 'question': case['question'],
              'expected_calls': case['expected_calls'], 'expected_result': case['expected_result'],
              'plan_passed': None if execution_only else False, 'execution_succeeded': False,
              'result_passed': False, 'passed': False}
    started = time.perf_counter()
    try:
        if execution_only:
            payload = {'plan': {'question': case['question'], 'reasoning': '使用固定计划核对计算结果',
                                'tool_calls': case['expected_calls']}}
            data = request_json(base_url + '/agent/execute', payload)
        else:
            data = request_json(base_url + '/agent/ask', {'question': case['question']})
            result['actual_plan'] = data.get('plan')
            plan = data.get('plan') or {}
            result['plan_passed'] = match_calls(plan.get('tool_calls'), case['expected_calls'])
        steps = data.get('steps')
        if not isinstance(steps, list) or not steps or 'result' not in steps[-1]:
            raise ValueError('接口未返回最终执行结果')
        result['execution_succeeded'] = True
        actual = steps[-1]['result']
        result['result_passed'], result['result_check'] = check_result(actual, case['expected_result'])
        result['actual_row_count'] = len(actual) if isinstance(actual, list) else None
        # 只保存核对字段预览，避免报告塞入完整岗位描述。
        columns = case['expected_result'].get('columns', [case['expected_result'].get('column'), 'count'])
        result['actual_preview'] = [
            {c: row.get(c) for c in columns if c is not None} if isinstance(row, dict) else row
            for row in actual[:10]
        ] if isinstance(actual, list) else (
            {**actual, 'groups': [{**group, 'skills': group.get('skills', [])[:10]}
                                    for group in actual['groups']]}
            if isinstance(actual, dict) and isinstance(actual.get('groups'), list)
            else ({**actual, 'skills': actual.get('skills', [])[:10]} if isinstance(actual, dict) else actual)
        )
        result['passed'] = result['result_passed'] and (execution_only or result['plan_passed'])
    except HTTPError as exc:
        result['error'] = f'HTTP {exc.code}: ' + exc.read().decode('utf-8', errors='replace')[:2000]
    except URLError as exc:
        result['error'] = f'连接失败：{exc.reason}'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description='核对模型计划与最终计算结果')
    parser.add_argument('--execution-only', action='store_true', help='仅执行固定计划，不调用模型')
    parser.add_argument('--base-url', default='http://127.0.0.1:8001')
    args = parser.parse_args()
    baseline_path = EVAL_DIR / 'benchmark.json'
    benchmark = json.loads(baseline_path.read_text(encoding='utf-8-sig'))
    if not isinstance(benchmark, dict) or benchmark.get('version') != 2 or not benchmark.get('cases'):
        raise ValueError('需要第二版评测文件及非空题目列表')
    dataset = PROJECT_DIR / benchmark['dataset']['path']
    if file_digest(dataset) != benchmark['dataset']['sha256']:
        raise ValueError('数据文件与固定标准答案不一致，请先复核数据和标准答案，停止评测')
    print('固定计划计算校验（不调用模型）' if args.execution_only else '完整评测（每题调用一次模型）', flush=True)
    results = []
    for case in benchmark['cases']:
        item = evaluate_case(case, args.base_url.rstrip('/'), args.execution_only)
        results.append(item)
        status = '通过' if item['passed'] else '未通过'
        plan = '未评测' if args.execution_only else ('通过' if item['plan_passed'] else '未通过')
        value = '通过' if item['result_passed'] else '未通过'
        print(f"[{status}] {case['id']}｜计划：{plan}｜结果：{value}", flush=True)
        if item.get('error'):
            print(item['error'], flush=True)
        elif not item['result_passed']:
            print(item.get('result_check', ''), flush=True)
    total = len(results)
    passed = sum(item['passed'] for item in results)
    report = {
        'mode': 'execution_only' if args.execution_only else 'end_to_end',
        'created_at': datetime.now().astimezone().isoformat(),
        'dataset': benchmark['dataset'], 'benchmark_sha256': file_digest(baseline_path),
        'total': total, 'passed': passed,
        'plan_accuracy': None if args.execution_only else sum(x['plan_passed'] for x in results) / total,
        'execution_success_rate': sum(x['execution_succeeded'] for x in results) / total,
        'result_accuracy': sum(x['result_passed'] for x in results) / total,
        'results': results,
    }
    report_dir = EVAL_DIR / 'reports'
    report_dir.mkdir(exist_ok=True)
    path = report_dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n通过：{passed}/{total}')
    if args.execution_only:
        print('本次未评测模型理解能力。')
    print(f'报告已保存：{path}')
    return 0 if passed == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
