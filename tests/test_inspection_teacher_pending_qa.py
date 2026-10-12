from types import SimpleNamespace

import pytest

from scoper.benchmark.common.container_inspection import teacher as module


@pytest.mark.parametrize('backend', ['behavior', 'procthor'])
def test_qa_after_stop_does_not_pop_empty_action_list(monkeypatch, backend):
    state = {'contents': [], 'is_open': True, 'has_dishware': False}
    monkeypatch.setattr(module, 'inspect_contents', lambda *args: state)
    teacher = module.teacher_class(backend)()
    teacher.env = SimpleNamespace(_human_qa_state={
        'pending': {'q1': {}}, 'evaluator_keys': {'q1': 'open'}})
    teacher.initial = state
    teacher.events = []
    teacher.calls = []
    teacher.last_call = 'stop'
    brain_input = SimpleNamespace(last_transition={'success': True}, observation={})
    call = teacher.act(brain_input)
    assert call.tool_name == 'answer'
    assert call.arguments == {'question_id': 'q1', 'value': 'open'}
    teacher.env._human_qa_state['pending'] = {}
    with pytest.raises(RuntimeError, match='ORACLE_CALLED_AFTER_COMPLETION'):
        teacher.act(brain_input)
