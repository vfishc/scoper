from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence
from scoper.benchmark.common.evaluation_binding_policy import terminal_failed_tool
from scoper.benchmark.behavior.goal_predicates import format_goal_predicate

def build_failure_report(*, transitions: Sequence[Mapping[str, Any]], final_goal: Mapping[str, Any], failure_type: Optional[str]=None, failure_reason: Optional[str]=None, scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None, task_spec: Optional[Mapping[str, Any]]=None, model_stopped: bool=False, initial_goal: Optional[Mapping[str, Any]]=None) -> dict[str, Any]:
    task_success = float(final_goal.get('task_success', 0.0) or 0.0)
    if task_success == 1.0 and (not failure_type) and (not failure_reason):
        return {'failure_code': None, 'failure_reason': None, 'failure_details': {}}
    transitions = list(transitions or [])
    scheduler_events = list(scheduler_events or [])
    last_transition = transitions[-1] if transitions else {}
    failed_tools = [t for t in transitions if t.get('success') is False]
    last_failed_tool = failed_tools[-1] if failed_tools else None
    unsatisfied = _unsatisfied_predicates(final_goal, task_spec=task_spec)
    done_reason = last_transition.get('done_reason')
    model_stopped = bool(model_stopped or final_goal.get('model_stopped', False))
    reason_text = str(failure_reason or '')
    code = _classify_failure_code(failure_reason=reason_text, done_reason=done_reason, final_goal=final_goal, last_failed_tool=terminal_failed_tool(transitions), scheduler_events=scheduler_events, model_stopped=model_stopped, initial_goal=initial_goal)
    readable = reason_text
    if _is_generic_final_goal_reason(readable) or _is_legacy_stage6_tool_code(readable):
        readable = ''
    readable = readable or _build_readable_reason(code=code, final_goal=final_goal, last_failed_tool=terminal_failed_tool(transitions), unsatisfied=unsatisfied, model_stopped=model_stopped)
    return {'failure_code': code, 'failure_reason': readable, 'failure_details': {'done_reason': done_reason, 'task_success': task_success, 'task_progress': float(final_goal.get('task_progress', 0.0) or 0.0), 'predicate_success': float(final_goal.get('predicate_success', 0.0) or 0.0), 'require_model_stop': bool(final_goal.get('require_model_stop', False)), 'model_stopped': model_stopped, 'last_tool': _transition_brief(last_transition) if last_transition else None, 'last_failed_tool': _transition_brief(last_failed_tool) if last_failed_tool else None, 'unsatisfied_predicates': unsatisfied, 'scheduler_events': [_scheduler_event_brief(ev) for ev in scheduler_events], 'task_id': task_spec.get('id') if isinstance(task_spec, Mapping) else None, 'category': task_spec.get('category') if isinstance(task_spec, Mapping) else None, 'failure_type': failure_type}}

def classify_failure_code_from_record(record: Mapping[str, Any]) -> Optional[str]:
    if record.get('status') == 'passed' or record.get('execution_ok') is True:
        return None
    report = build_failure_report(transitions=list(record.get('transitions') or record.get('tool_calls') or []), final_goal=record.get('final_goal') or {}, failure_reason=record.get('failure_reason'), failure_type=record.get('failure_type'), scheduler_events=record.get('scheduler_events') or [], task_spec=record.get('task_spec'), model_stopped=bool(record.get('model_stopped', False)))
    return report.get('failure_code') or 'closed_loop_runtime_error'

def _classify_failure_code(*, failure_reason: str, done_reason: Any, final_goal: Mapping[str, Any], last_failed_tool: Optional[Mapping[str, Any]], scheduler_events: Sequence[Mapping[str, Any]], model_stopped: bool, initial_goal: Optional[Mapping[str, Any]]) -> str:
    reason = str(failure_reason or '')
    failed_name = str((last_failed_tool or {}).get('tool_name') or '')
    if initial_goal and float(initial_goal.get('task_success', 0.0) or 0.0) == 1.0:
        return 'goal_true_at_reset'
    if 'Task is already successful at reset' in reason:
        return 'goal_true_at_reset'
    if 'Exceeded max' in reason or done_reason in {'max_high_level_steps', 'max_total_low_level_steps'}:
        return 'exceeded_step_budget'
    if bool(final_goal.get('require_model_stop', False)) and (not bool(model_stopped)):
        if float(final_goal.get('predicate_success', 0.0) or 0.0) >= 1.0:
            return 'missing_required_stop'
    if any((_is_failure_injection_event(ev) for ev in scheduler_events)):
        if failed_name:
            return 'injected_failure_not_recovered'
        if float(final_goal.get('task_success', 0.0) or 0.0) < 1.0:
            return 'injected_failure_goal_not_met'
    if failed_name:
        return _tool_failure_code(failed_name)
    if _unsatisfied_predicates(final_goal):
        return 'unsatisfied_goal_predicates'
    if 'Final task_success is not 1.0' in reason:
        return 'final_goal_not_success'
    if reason:
        return 'closed_loop_runtime_error'
    return 'final_goal_not_success'

def _build_readable_reason(*, code: str, final_goal: Mapping[str, Any], last_failed_tool: Optional[Mapping[str, Any]], unsatisfied: Sequence[Mapping[str, Any]], model_stopped: bool) -> str:
    if code == 'missing_required_stop':
        return 'Goal predicates were satisfied, but the task required a model stop and no stop was recorded.'
    if last_failed_tool is not None:
        tool = last_failed_tool.get('tool_name') or 'unknown'
        feedback = _feedback_text(last_failed_tool)
        if feedback:
            return "Task failed after tool '{}' failed: {}".format(tool, feedback)
        return "Task failed after tool '{}' failed.".format(tool)
    if unsatisfied:
        return 'Task ended with unsatisfied goal predicates: {}.'.format('; '.join((format_goal_predicate(item) for item in unsatisfied[:5])))
    if bool(final_goal.get('require_model_stop', False)) and (not bool(model_stopped)):
        return 'Task required a model stop, but the model did not stop.'
    return 'Final task_success is not 1.0.'

def _is_failure_injection_event(event: Mapping[str, Any]) -> bool:
    name = str(event.get('name') or '')
    return name.startswith('failure_') and name != 'failure_armed'

def _unsatisfied_predicates(final_goal: Mapping[str, Any], *, task_spec: Optional[Mapping[str, Any]]=None) -> list[dict[str, Any]]:
    predicates = final_goal.get('predicate_records') or final_goal.get('predicates') or []
    specs = []
    if isinstance(task_spec, Mapping):
        specs = list((task_spec.get('goal') or {}).get('predicates') or [])
    unsatisfied: list[dict[str, Any]] = []
    if not isinstance(predicates, Sequence) or isinstance(predicates, (str, bytes)):
        return unsatisfied
    for index, predicate in enumerate(predicates):
        if not isinstance(predicate, Mapping) or predicate.get('satisfied') is not False:
            continue
        item = dict(predicate)
        if specs and index < len(specs):
            spec = specs[index]
            if isinstance(spec, Mapping):
                merged = dict(spec)
                merged.update(item)
                if 'args' not in item and 'args' in spec:
                    merged['args'] = spec['args']
                item = merged
        unsatisfied.append(item)
    return unsatisfied

def _transition_brief(transition: Mapping[str, Any]) -> dict[str, Any]:
    feedback = transition.get('env_feedback')
    if not isinstance(feedback, Mapping):
        feedback = {'error_type': transition.get('error_type'), 'error_message': transition.get('error_message')}
    return {'high_level_step': transition.get('high_level_step') or transition.get('step'), 'tool_name': transition.get('tool_name'), 'arguments': transition.get('arguments'), 'success': transition.get('success'), 'env_feedback': dict(feedback or {}), 'done': transition.get('done'), 'done_reason': transition.get('done_reason')}

def _scheduler_event_brief(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
    return {'name': event.get('name'), 'phase': event.get('phase'), 'step': payload.get('step'), 'target_role': payload.get('target_role'), 'target': payload.get('target'), 'tool': payload.get('tool')}

def _feedback_text(transition: Mapping[str, Any]) -> str:
    feedback = transition.get('env_feedback')
    if isinstance(feedback, Mapping):
        error_type = feedback.get('error_type')
        message = feedback.get('error_message')
        if error_type and message:
            return '{}: {}'.format(error_type, message)
        if error_type or message:
            return str(error_type or message)
    error_type = transition.get('error_type')
    message = transition.get('error_message')
    if error_type and message:
        return '{}: {}'.format(error_type, message)
    return str(message or error_type or '')

def _tool_failure_code(tool_name: str) -> str:
    if tool_name == 'navigate_to':
        return 'navigation_failed'
    if tool_name in {'open', 'close'}:
        return 'open_close_failed'
    if tool_name in {'toggle_on', 'toggle_off'}:
        return 'toggle_failed'
    return '{}_failed'.format(tool_name or 'tool')

def _is_generic_final_goal_reason(reason: str) -> bool:
    return str(reason or '').strip() in {'Final task_success is not 1.0.', 'Final task_success is not 1.0'}

def _is_legacy_stage6_tool_code(reason: str) -> bool:
    return str(reason or '').startswith('TOOL_EXECUTION_FAILED:')
__all__ = ['build_failure_report', 'classify_failure_code_from_record']
