from scoper.harness import QAOverlayRuntime, build_qa_overlay_spec, normalize_task_for_qa
import pytest

def _overlay():
    return build_qa_overlay_spec(normalize_task_for_qa({'id': 't', 'instruction': 'Turn off the faucet.', 'roles': {'faucet': {'object_type': 'Faucet'}}, 'goal': {'require_model_stop': True, 'predicates': [{'type': 'object_toggled_off', 'object_role': 'faucet'}]}}), max_questions=2, seed=0)

def test_runtime_posts_one_question_and_queues_the_next():
    overlay = _overlay()
    runtime = QAOverlayRuntime(overlay)
    state = {'pending': {}, 'answers': {}}
    runtime.reset(state)

    def resolver(question):
        if question['question_type'] == 'post_action_outcome_choice':
            return {'last_tool_outcome': 'success'}
        return {'clause_incomplete': {'goal_clause_0': False}}
    post_question = next((question for question in overlay['questions'] if question['question_type'] == 'post_action_outcome_choice'))
    events = runtime.consider(state, trigger='after_tool_result', total_step=1, task_action_step=1, active_subtask_id=None, resolve_state=resolver, transition={'tool_name': post_question['activation']['tool_name'], 'arguments': {'target': post_question['activation']['target_role_or_null']}})
    assert events[0]['name'] == 'human_qa_posted'
    assert len(state['pending']) == 1
    assert runtime.consider(state, trigger='before_stop', total_step=2, task_action_step=1, active_subtask_id=None, resolve_state=resolver) == []
    state['pending'].clear()
    assert runtime.post_next(state, total_step=3, resolve_state=resolver)[0]['name'] == 'human_qa_posted'

def test_runtime_resolver_failure_is_explicit_and_fail_closed():
    runtime = QAOverlayRuntime(_overlay())
    state = {}
    runtime.reset(state)
    events = runtime.consider(state, trigger='before_stop', total_step=1, task_action_step=0, active_subtask_id=None, resolve_state=lambda question: {})
    assert events == [{'name': 'qa_overlay_resolver_failed', 'payload': {'id': events[0]['payload']['id'], 'question_type': 'remaining_requirement_choice'}}]
    assert state['qa_overlay_resolver_failures']
    assert state['pending'] == {}

@pytest.mark.parametrize('original_outcome', ['success', 'failure', 'ambiguous', None])
def test_delayed_outcome_keeps_trigger_state_and_action_identity(original_outcome):
    overlay = _overlay()
    question = next((item for item in overlay['questions'] if item['question_type'] == 'post_action_outcome_choice'))
    runtime = QAOverlayRuntime({'schema_version': overlay['schema_version'], 'questions': [question]})
    state = {}
    runtime.reset(state)
    state['pending']['earlier_question'] = {}
    trigger_state = {'last_tool_outcome': original_outcome}
    assert runtime.consider(state, trigger='after_tool_result', total_step=4, task_action_step=3, active_subtask_id=None, resolve_state=lambda selected: trigger_state, transition={'tool_name': question['activation']['tool_name'], 'arguments': {'target': question['activation']['target_role_or_null']}}) == []
    trigger_state['last_tool_outcome'] = 'failure' if original_outcome == 'success' else 'success'
    state['pending'].clear()
    events = runtime.consider(state, trigger='task_step', total_step=7, task_action_step=4, active_subtask_id=None, resolve_state=lambda selected: trigger_state)
    if original_outcome is None:
        assert events[0]['name'] == 'qa_overlay_resolver_failed'
        assert question['id'] not in state['evaluator_keys']
    else:
        assert state['evaluator_keys'][question['id']] == question['resolver']['answer_by_outcome'][original_outcome]
        assert 'step 4' in state['pending'][question['id']]['question']
    state['pending'].clear()
    assert runtime.post_next(state, total_step=8, resolve_state=lambda selected: trigger_state) == []
    assert runtime.complete

def test_delayed_current_progress_uses_publication_state():
    overlay = _overlay()
    question = next((item for item in overlay['questions'] if item['question_type'] == 'remaining_requirement_choice'))
    runtime = QAOverlayRuntime({'schema_version': overlay['schema_version'], 'questions': [question]})
    state = {}
    runtime.reset(state)
    state['pending']['earlier_question'] = {}
    runtime.consider(state, trigger='before_stop', total_step=2, task_action_step=1, active_subtask_id=None, resolve_state=lambda selected: {'clause_incomplete': {'goal_clause_0': True}})
    state['pending'].clear()
    runtime.post_next(state, total_step=3, resolve_state=lambda selected: {'clause_incomplete': {'goal_clause_0': False}})
    assert state['evaluator_keys'][question['id']] == question['resolver']['answer_by_incomplete_mask']['0']

def test_child_before_stop_question_can_post_at_handoff_without_blocking_handoff():
    overlay = _overlay()
    question = next((item for item in overlay['questions'] if item['question_type'] == 'remaining_requirement_choice'))
    question['owner_subtask_id'] = 'subtask_0'
    runtime = QAOverlayRuntime({'schema_version': overlay['schema_version'], 'questions': [question]})
    state = {}
    runtime.reset(state)
    events = runtime.consider(state, trigger='subtask_handoff', total_step=2, task_action_step=1, active_subtask_id='subtask_0', handoff_subtask_id='subtask_1', resolve_state=lambda item: {'clause_incomplete': {'goal_clause_0': False}})
    assert events[0]['name'] == 'human_qa_posted'
    assert state['pending']

def test_runtime_delegates_namespaced_target_role_matching():
    overlay = _overlay()
    question = next((item for item in overlay['questions'] if item['question_type'] == 'post_action_outcome_choice'))
    question['activation']['target_role_or_null'] = 'task0__faucet'
    runtime = QAOverlayRuntime({'schema_version': overlay['schema_version'], 'questions': [question]})
    state = {}
    runtime.reset(state)
    transition = {'tool_name': question['activation']['tool_name'], 'arguments': {'target': 'faucet'}}
    assert runtime.consider(state, trigger='after_tool_result', total_step=1, task_action_step=1, active_subtask_id=None, resolve_state=lambda item: {'last_tool_outcome': 'success'}, transition=transition, target_role_matches=lambda role, call: role == 'task0__faucet' and call['arguments']['target'] == 'faucet')[0]['name'] == 'human_qa_posted'

@pytest.mark.parametrize(('activation', 'kwargs'), [({'type': 'task_step', 'after_step': 2}, {'trigger': 'task_step', 'task_action_step': 2}), ({'type': 'before_stop'}, {'trigger': 'before_stop'}), ({'type': 'after_human_insert', 'insert_id': 'insert_0'}, {'trigger': 'after_human_insert', 'insert_id': 'insert_0'}), ({'type': 'subtask_handoff', 'subtask_id': 'subtask_1'}, {'trigger': 'subtask_handoff', 'handoff_subtask_id': 'subtask_1'}), ({'type': 'after_tool_result', 'tool_name': 'open', 'target_role_or_null': 'cabinet'}, {'trigger': 'after_tool_result', 'transition': {'tool_name': 'open', 'arguments': {'target': 'cabinet'}}}), ({'type': 'after_tool_result', 'tool_name': 'wash', 'target_role_or_null': 'dish'}, {'trigger': 'after_tool_result', 'transition': {'tool_name': 'soak_under', 'arguments': {'target': 'faucet'}}}), ({'type': 'after_tool_result', 'tool_name': 'slice', 'target_role_or_null': 'food'}, {'trigger': 'after_tool_result', 'transition': {'tool_name': 'cut', 'arguments': {'target': 'food'}}}), ({'type': 'after_tool_result', 'tool_name': 'heat', 'target_role_or_null': 'food'}, {'trigger': 'after_tool_result', 'transition': {'tool_name': 'heat', 'arguments': {'target': 'microwave'}}})])
def test_all_public_activation_kinds_match_without_private_state(activation, kwargs):
    defaults = {'trigger': 'task_step', 'task_action_step': 0, 'transition': None, 'insert_id': None, 'handoff_subtask_id': None}
    defaults.update(kwargs)
    assert QAOverlayRuntime._activation_matches(activation, **defaults)
