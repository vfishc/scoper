import copy

import pytest

from scoper.benchmark.common.generic_qa import build_generic_qa_question
from scoper.benchmark.common.generic_qa_extension import append_generic_questions


@pytest.mark.parametrize('backend', ['behavior', 'procthor'])
@pytest.mark.parametrize('kind,indices', [
    ('target_state_choice', [0]),
    ('target_relation_choice', [1]),
    ('mid_execution_progress_choice', [0, 1]),
])
def test_all_added_qa_types_can_be_generated(backend, kind, indices):
    task = {'id': 'example', 'backend': backend, 'instruction': 'Clean and store the cup in the cabinet.',
            'roles': {'cup': {'object_type': 'Cup'}, 'cabinet': {'object_type': 'Cabinet'}},
            'goal': {'predicates': [
                {'type': 'object_clean', 'object_role': 'cup'},
                {'type': 'object_in_receptacle', 'object_role': 'cup', 'receptacle_role': 'cabinet'}]}}
    original = copy.deepcopy(task)
    question = build_generic_qa_question(task, question_type=kind, clause_indices=indices,
                                        activation={'type': 'before_stop'})
    assert question['question_type'] == kind
    assert 'cup' in question['question'].lower()
    assert len(question['resolver']['clause_ids']) == len(indices)
    assert task == original


def test_extension_preserves_composite_without_processing_children():
    task = {'id': 'composite', 'family_spec': {'composite': {'subtasks': [{'subtask_id': 'one'}]}}}
    result, report = append_generic_questions(task, {})
    assert result == task
    assert result is not task
    assert report['skipped'] == ['composite_scope_preserved']
