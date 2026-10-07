import pandas as pd

from agent.answers import build_evidence
from agent.llm_client import LLMClient, SYSTEM_PROMPT
from tools.analysis_tools import sample_text


def test_planner_keeps_responsibilities_and_qualification_sources_separate():
    assert '职责对应 responsibilities' in SYSTEM_PROMPT
    assert '最低要求对应 minimum_qualifications' in SYSTEM_PROMPT
    assert '优先条件对应 preferred_qualifications' in SYSTEM_PROMPT
    assert '不要用一个文本字段代替另一个字段' in SYSTEM_PROMPT


def test_answer_prompt_does_not_merge_job_text_sources():
    client = object.__new__(LLMClient)
    client.model = 'test-model'
    options = client._answer_options('岗位的职责和任职要求是什么？', [])
    system = options['messages'][0]['content']
    assert '回答职责只依据 responsibilities 样本' in system
    assert '最低要求只依据 minimum_qualifications 样本' in system
    assert '优先条件只依据 preferred_qualifications 样本' in system
    assert '不能把优先条件说成必需条件' in system
    assert '没有对应字段证据时明确说数据未提供' in system


def test_each_job_text_field_is_sampled_and_labelled_independently():
    jobs = pd.DataFrame({
        'title': ['Analyst', 'Analyst'],
        'company_name': ['Acme', 'Acme'],
        'responsibilities': ['Build reports', 'Present findings'],
        'minimum_qualifications': ['Bachelor degree', 'Three years experience'],
        'preferred_qualifications': ['SQL preferred', 'Tableau preferred'],
        'job_description': ['Full description A', 'Full description B'],
    })
    columns = ('responsibilities', 'minimum_qualifications',
               'preferred_qualifications', 'job_description')
    steps = []
    for column in columns:
        result = sample_text(jobs, column=column, limit=8)
        assert result['column'] == column
        assert result['matched_records'] == 2
        assert result['available_records'] == 2
        assert {row['text'] for row in result['records']} == set(jobs[column])
        steps.append({'tool': 'sample_text', 'arguments': {'column': column},
                      'result': result})

    evidence = build_evidence(steps)
    assert [item['arguments']['column'] for item in evidence] == list(columns)
    for item in evidence:
        assert item['result']['column'] == item['arguments']['column']
