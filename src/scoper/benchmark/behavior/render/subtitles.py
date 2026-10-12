from __future__ import annotations
from typing import Any, Mapping, Optional, Tuple

def episode_start_subtitle(task_spec: Mapping[str, Any]) -> Tuple[str, str]:
    instruction = task_spec.get('instruction', '')
    short = 'Task: {}'.format(instruction)
    lines = ['Task: {}'.format(instruction), 'Category: {}'.format(task_spec.get('category', 'unknown'))]
    predicates = list((task_spec.get('goal') or {}).get('predicates') or [])
    if predicates:
        lines.append('Goals: {}'.format('; '.join((_predicate_text(p) for p in predicates[:4]))))
    return (short, '\n'.join(lines))

def tool_start_subtitle(tool_name: str, arguments: Mapping[str, Any], high_level_step: int, reasoning: str='') -> Tuple[str, str]:
    short = 'Step {}: execute {}{}'.format(high_level_step, tool_name, _format_arguments(arguments))
    lines = ['Action: {}{}'.format(tool_name, _format_arguments(arguments))]
    if reasoning:
        short += ' | {}'.format(reasoning)
        lines.append('Why: {}'.format(reasoning))
    return (short, '\n'.join(lines))

def tool_result_subtitle(tool_name: str, transition: Mapping[str, Any]) -> Tuple[str, str]:
    status = 'succeeded' if transition.get('success') else 'failed'
    short = '{} {}.'.format(tool_name, status)
    lines = ['Result: {} {}'.format(tool_name, status)]
    if tool_name == 'answer':
        answer_short, answer_lines = _human_answer_text(transition)
        if answer_short:
            short = answer_short
        lines.extend(answer_lines)
    if tool_name == 'navigate_to':
        lines.extend(_navigation_result_lines(transition))
    feedback = _feedback_text(transition.get('env_feedback'))
    if feedback:
        short += ' {}'.format(feedback)
        lines.append('Feedback: {}'.format(feedback))
    if transition.get('done'):
        short += ' Done: {}.'.format(transition.get('done_reason'))
        lines.append('Done reason: {}'.format(transition.get('done_reason')))
    goal = transition.get('goal_result') if isinstance(transition.get('goal_result'), Mapping) else {}
    if goal:
        lines.append('Progress: task_success={}, task_progress={}'.format(goal.get('task_success'), goal.get('task_progress')))
    return (short, '\n'.join(lines))

def scheduler_event_subtitle(event: Mapping[str, Any]) -> Tuple[str, str]:
    name = str(event.get('name') or '')
    payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
    if name in {'close_object', 'open_object', 'toggle_on', 'toggle_off', 'dirty_object', 'relocate_object', 'restore_initial_state'}:
        target = payload.get('target_role') or payload.get('target') or payload.get('target_object_id') or 'target'
        before = payload.get('before') if isinstance(payload.get('before'), Mapping) else {}
        after = payload.get('after') if isinstance(payload.get('after'), Mapping) else {}
        visibility = payload.get('visibility') if isinstance(payload.get('visibility'), Mapping) else {}
        state_text = _disturbance_state_text(name, before, after, payload)
        short = 'Disturbance: {} on {}'.format(name, target)
        lines = ['Disturbance: {}'.format(name), 'Target: {}'.format(target)]
        if state_text:
            short += ' ({})'.format(state_text)
            lines.append('State: {}'.format(state_text))
        if 'target_visible' in visibility:
            lines.append('Target visible: {}'.format(visibility.get('target_visible')))
        mutation = payload.get('mutation') if isinstance(payload.get('mutation'), Mapping) else {}
        if 'renderer_synced' in mutation:
            lines.append('Renderer synced: {}'.format(mutation.get('renderer_synced')))
        return (short, '\n'.join(lines))
    if name == 'human_qa_posted':
        question_id = payload.get('id')
        question = str(payload.get('question') or '')
        expected = payload.get('evaluator_answer')
        short = 'Human QA {}: {}'.format(question_id, question)
        lines = ['Human QA posted: {}'.format(question_id)]
        if question:
            lines.append('Question: {}'.format(question))
        answer_set = payload.get('answer_set')
        if isinstance(answer_set, list) and answer_set:
            lines.append('Allowed answers: {}'.format('; '.join((str(item) for item in answer_set))))
        if expected is not None:
            lines.append('Expected answer: {}'.format(expected))
        return (short, '\n'.join(lines))
    if name == 'human_insert_posted':
        insert_id = payload.get('id')
        instruction = str(payload.get('instruction') or '')
        short = 'Human insert {}: {}'.format(insert_id, instruction)
        lines = ['Human inserted task: {}'.format(insert_id)]
        if instruction:
            lines.append('Instruction: {}'.format(instruction))
        predicates = payload.get('goal_predicates')
        if isinstance(predicates, list) and predicates:
            lines.append('Added goals: {}'.format('; '.join((_predicate_text(p) for p in predicates if isinstance(p, Mapping)))))
        return (short, '\n'.join(lines))
    return ('', '')

def episode_end_subtitle(final_goal: Mapping[str, Any], status: str) -> Tuple[str, str]:
    short = 'Episode {}. task_success={}.'.format(status, final_goal.get('task_success'))
    lines = ['Episode: {}'.format(status), 'Final task_success: {}'.format(final_goal.get('task_success')), 'Final task_progress: {}'.format(final_goal.get('task_progress'))]
    return (short, '\n'.join(lines))

def format_state_change(name: str, state: str, before: Any, after: Any) -> str:
    return '{}: {}={}->{}'.format(name, state, before, after)

def diff_catalog_states(before_catalog, after_catalog) -> list[str]:
    before_by_name = {item['name']: dict(item.get('states') or {}) for item in before_catalog or []}
    changes: list[str] = []
    for item in after_catalog or []:
        name = item['name']
        after_states = dict(item.get('states') or {})
        before_states = before_by_name.get(name, {})
        for key in sorted(set(after_states) | set(before_states)):
            b, a = (before_states.get(key), after_states.get(key))
            if b != a:
                changes.append(format_state_change(name, key, b, a))
    return changes

def _predicate_text(predicate: Mapping[str, Any]) -> str:
    ptype = predicate.get('type') or 'predicate'
    args = predicate.get('args') if isinstance(predicate.get('args'), Mapping) else {}
    obj = args.get('object')
    rec = args.get('receptacle') or args.get('heatsource')
    if obj and rec:
        return '{} {} -> {}'.format(ptype, obj, rec)
    if obj:
        return '{} {}'.format(ptype, obj)
    return str(ptype)

def _feedback_text(env_feedback: Any) -> str:
    if isinstance(env_feedback, Mapping):
        msg = env_feedback.get('error_message') or env_feedback.get('error_type')
        return str(msg) if msg else ''
    return str(env_feedback or '').strip()

def _disturbance_state_text(name: str, before: Mapping[str, Any], after: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    state_keys = {'close_object': ('open',), 'open_object': ('open',), 'toggle_on': ('toggled_on',), 'toggle_off': ('toggled_on',), 'dirty_object': ('dirty',), 'restore_initial_state': ('open', 'toggled_on', 'dirty')}.get(name, ())
    for key in state_keys:
        if key in before or key in after:
            return '{}={}->{}'.format(key, before.get(key), after.get(key))
    if name == 'relocate_object':
        mutation = payload.get('mutation') if isinstance(payload.get('mutation'), Mapping) else {}
        support = mutation.get('support')
        return 'moved to {}'.format(support) if support else 'relocated'
    return ''

def _human_answer_text(transition: Mapping[str, Any]) -> tuple[str, list[str]]:
    state_delta = transition.get('state_delta') if isinstance(transition.get('state_delta'), Mapping) else {}
    answer = state_delta.get('human_answer') if isinstance(state_delta.get('human_answer'), Mapping) else {}
    question = state_delta.get('question') if isinstance(state_delta.get('question'), Mapping) else {}
    if not answer and (not question):
        return ('', [])
    question_id = answer.get('question_id') or question.get('id')
    value = answer.get('value') if answer.get('value') is not None else question.get('answer')
    short = 'Answered QA {}: {}'.format(question_id, value)
    lines = ['Human QA answer recorded: {}'.format(question_id)]
    question_text = question.get('question')
    if question_text:
        lines.append('Question: {}'.format(question_text))
    if value is not None:
        lines.append('Model answer: {}'.format(value))
    answer_set = question.get('answer_set')
    if isinstance(answer_set, list) and answer_set:
        lines.append('Allowed answers: {}'.format('; '.join((str(item) for item in answer_set))))
    return (short, lines)

def _navigation_result_lines(transition: Mapping[str, Any]) -> list[str]:
    state_delta = transition.get('state_delta') if isinstance(transition.get('state_delta'), Mapping) else {}
    nav = state_delta.get('navigate_to') if isinstance(state_delta.get('navigate_to'), Mapping) else {}
    raw = transition.get('raw_execution') if isinstance(transition.get('raw_execution'), Mapping) else {}
    if not nav and (not raw):
        return []
    navigation_mode = nav.get('navigation_mode') or raw.get('navigation_mode')
    belief_status = nav.get('belief_status_before') or raw.get('belief_status_before')
    target_visible = nav.get('target_visible_after_navigation')
    if target_visible is None:
        target_visible = raw.get('target_visible_after_navigation')
    missing = nav.get('target_missing_at_last_seen')
    if missing is None:
        missing = raw.get('target_missing_at_last_seen')
    lines: list[str] = []
    if navigation_mode:
        lines.append('Navigation: {}'.format(navigation_mode))
    if belief_status:
        lines.append('Belief before: {}'.format(belief_status))
    if target_visible is not None:
        lines.append('Target visible after navigation: {}'.format(bool(target_visible)))
    if missing is not None:
        lines.append('Missing at last-seen: {}'.format(bool(missing)))
    return lines

def _format_arguments(arguments: Optional[Mapping[str, Any]]) -> str:
    if not arguments:
        return '()'
    parts = ['{}={}'.format(k, v) for k, v in sorted(arguments.items())]
    text = ', '.join(parts)
    if len(text) > 90:
        text = text[:87] + '...'
    return '({})'.format(text)
__all__ = ['episode_start_subtitle', 'tool_start_subtitle', 'tool_result_subtitle', 'scheduler_event_subtitle', 'episode_end_subtitle', 'format_state_change', 'diff_catalog_states']
