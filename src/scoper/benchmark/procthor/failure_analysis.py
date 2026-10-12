from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence
from scoper.benchmark.common.evaluation_binding_policy import terminal_failed_tool

def build_failure_report(*, transitions: Sequence[Mapping[str, Any]], final_goal: Mapping[str, Any], failure_type: Optional[str]=None, failure_reason: Optional[str]=None, scheduler_events: Optional[Sequence[Mapping[str, Any]]]=None, task_spec: Optional[Mapping[str, Any]]=None, model_stopped: bool=False) -> dict:
    task_success = float(final_goal.get('task_success', 0.0) or 0.0)
    if task_success == 1.0 and (not failure_type) and (not failure_reason):
        return {'failure_code': None, 'failure_reason': None, 'failure_details': {}}
    transitions = list(transitions or [])
    scheduler_events = list(scheduler_events or [])
    last_transition = transitions[-1] if transitions else {}
    failed_tools = [t for t in transitions if not bool(t.get('success', False))]
    last_failed_tool = failed_tools[-1] if failed_tools else None
    unsatisfied = _unsatisfied_predicates(final_goal)
    done_reason = last_transition.get('done_reason')
    failure_reason = failure_reason or ''
    code = _classify_failure_code(failure_reason=failure_reason, done_reason=done_reason, final_goal=final_goal, last_failed_tool=terminal_failed_tool(transitions), model_stopped=model_stopped, scheduler_events=scheduler_events)
    readable = failure_reason
    if _is_generic_final_goal_reason(readable):
        readable = ''
    readable = readable or _build_readable_reason(code=code, final_goal=final_goal, last_failed_tool=terminal_failed_tool(transitions), unsatisfied=unsatisfied, model_stopped=model_stopped)
    return {'failure_code': code, 'failure_reason': readable, 'failure_details': {'done_reason': done_reason, 'task_success': task_success, 'task_progress': float(final_goal.get('task_progress', 0.0) or 0.0), 'predicate_success': float(final_goal.get('predicate_success', 0.0) or 0.0), 'require_model_stop': bool(final_goal.get('require_model_stop', False)), 'model_stopped': bool(model_stopped), 'last_tool': _transition_brief(last_transition) if last_transition else None, 'last_failed_tool': _transition_brief(last_failed_tool) if last_failed_tool else None, 'unsatisfied_predicates': unsatisfied, 'scheduler_events': [_scheduler_event_brief(ev) for ev in scheduler_events], 'task_id': task_spec.get('id') if isinstance(task_spec, Mapping) else None, 'category': task_spec.get('category') if isinstance(task_spec, Mapping) else None}}

def classify_failure_code_from_record(record: Mapping[str, Any]) -> Optional[str]:
    if record.get('status') == 'passed':
        return None
    report = build_failure_report(transitions=list(record.get('tool_calls') or []), final_goal=record.get('final_goal') or {}, failure_reason=record.get('failure_reason'), failure_type=record.get('failure_type'), scheduler_events=record.get('scheduler_events') or [], model_stopped=bool(record.get('model_stopped', False)))
    return report.get('failure_code') or 'closed_loop_tool_error'

def _classify_failure_code(*, failure_reason: str, done_reason: Any, final_goal: Mapping[str, Any], last_failed_tool: Optional[Mapping[str, Any]], model_stopped: bool, scheduler_events: Sequence[Mapping[str, Any]]) -> str:
    reason = str(failure_reason or '')
    feedback = str((last_failed_tool or {}).get('env_feedback') or '')
    failed_name = str((last_failed_tool or {}).get('tool_name') or '')
    if 'Task is already successful at reset' in reason:
        return 'goal_true_at_reset'
    if 'No candidate pose made target' in reason or 'No candidate pose made target' in feedback:
        return 'navigation_no_visible_pose'
    if 'No candidate pose produced a successful' in feedback:
        return 'interaction_pose_unreachable'
    if 'Exceeded max' in reason or done_reason in {'max_high_level_steps', 'max_total_low_level_steps'}:
        return 'exceeded_step_budget'
    if done_reason == 'max_invalid_tool_calls':
        return 'exceeded_invalid_tool_budget'
    if bool(final_goal.get('require_model_stop', False)) and (not bool(model_stopped)):
        if float(final_goal.get('predicate_success', 0.0) or 0.0) >= 1.0:
            return 'missing_required_stop'
    if any((str(ev.get('name', '')).startswith('failure_') for ev in scheduler_events)):
        if failed_name:
            return 'injected_failure_not_recovered'
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
        feedback = str(last_failed_tool.get('env_feedback') or '').strip()
        tool = last_failed_tool.get('tool_name') or 'unknown'
        if feedback:
            return "Task failed after tool '{}' failed: {}".format(tool, feedback)
        return "Task failed after tool '{}' failed.".format(tool)
    if unsatisfied:
        return 'Task ended with unsatisfied goal predicates: {}.'.format('; '.join((_format_predicate(p) for p in unsatisfied[:5])))
    if bool(final_goal.get('require_model_stop', False)) and (not bool(model_stopped)):
        return 'Task required a model stop, but the model did not stop.'
    return 'Final task_success is not 1.0.'

def _unsatisfied_predicates(final_goal: Mapping[str, Any]) -> list:
    predicates = final_goal.get('predicates') or final_goal.get('predicate_records') or []
    unsatisfied = []
    if isinstance(predicates, Sequence) and (not isinstance(predicates, (str, bytes))):
        for predicate in predicates:
            if not isinstance(predicate, Mapping):
                continue
            if predicate.get('satisfied') is False:
                unsatisfied.append(dict(predicate))
    return unsatisfied

def _transition_brief(transition: Mapping[str, Any]) -> dict:
    return {'high_level_step': transition.get('high_level_step'), 'tool_name': transition.get('tool_name'), 'arguments': transition.get('arguments'), 'success': transition.get('success'), 'env_feedback': transition.get('env_feedback'), 'done': transition.get('done'), 'done_reason': transition.get('done_reason'), 'raw_status': (transition.get('raw_execution') or {}).get('status') if isinstance(transition.get('raw_execution'), Mapping) else None}

def _scheduler_event_brief(event: Mapping[str, Any]) -> dict:
    payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
    return {'name': event.get('name'), 'phase': event.get('phase'), 'step': payload.get('step'), 'target_role': payload.get('target_role'), 'object_id': payload.get('object_id'), 'tool': payload.get('tool'), 'config_mode': payload.get('config_mode')}

def _format_predicate(predicate: Mapping[str, Any]) -> str:
    pred_type = predicate.get('type') or 'predicate'
    object_role = predicate.get('object_role')
    receptacle_role = predicate.get('receptacle_role')
    if object_role and receptacle_role:
        return '{}({} -> {})'.format(pred_type, object_role, receptacle_role)
    if object_role:
        return '{}({})'.format(pred_type, object_role)
    return str(pred_type)

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
__all__ = ['build_failure_report', 'classify_failure_code_from_record']
