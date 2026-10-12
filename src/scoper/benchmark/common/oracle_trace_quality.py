from __future__ import annotations
import copy
from typing import Any, Mapping, Sequence
NAVIGATION_TOOLS = {'navigate_to', 'navigate_to_room'}
EXPLORATION_TOOLS = {'look_for', 'scan'}
PHYSICAL_TOOLS = {'close', 'cool', 'cut', 'drop', 'heat', 'open', 'pick_up', 'put_down', 'slice', 'soak_under', 'toggle_off', 'toggle_on'}
ORACLE_TRACE_QUALITY_SCHEMA_VERSION = 'oracle-trace-quality.v1'

class OracleTraceQualityError(RuntimeError):

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = copy.deepcopy(dict(report))
        self.code = str(report.get('failure_code') or 'TRACE_QUALITY_FAILED')
        super().__init__(self.code)

class OracleTraceQualityGate:

    def __init__(self, task_spec: Mapping[str, Any]) -> None:
        self._allowances = _failure_allowances(task_spec)
        self._pending: set[int] = set()
        self._navigation_ready: bool | None = None
        self._soft_streak = 0
        self._max_soft_streak = 0
        self._checked = 0
        self._fallbacks = 0
        self._failure_code: str | None = None
        self._failure_step: int | None = None
        self._failure_tool: str | None = None
        disturbances = task_spec.get('disturbance_spec') or []
        if isinstance(disturbances, Mapping):
            disturbances = disturbances.get('disturbances') or []
        self._close_configs = {index: dict(spec) for index, spec in enumerate(disturbances) if isinstance(spec, Mapping) and spec.get('type') == 'close_object' and (spec.get('once', True) is True) and spec.get('target_role')}
        self._close_recoveries: dict[int, dict[str, Any]] = {}

    def observe_transition(self, transition: Mapping[str, Any], *, scheduler_events: Sequence[Mapping[str, Any]] | None=None) -> dict[str, Any]:
        if self._failure_code is not None:
            return self.report()
        self._checked += 1
        tool = str(transition.get('tool_name') or transition.get('tool') or '')
        step = int(transition.get('high_level_step') or self._checked)
        self._observe_close_events(scheduler_events or (), step)
        fallback_error, fallback_count = _fallback_quality(transition)
        self._fallbacks += fallback_count
        if fallback_error:
            return self._reject(fallback_error, step, tool)
        success = transition.get('success') is True
        if success:
            self._soft_streak = 0
            if tool in NAVIGATION_TOOLS:
                self._navigation_ready = True
            if tool in PHYSICAL_TOOLS and self._navigation_ready is False:
                return self._reject('PHYSICAL_SUCCESS_AFTER_FAILED_NAVIGATION', step, tool)
            self._recover(tool, transition)
            self._recover_close(tool, transition)
            return self.report()
        if tool in EXPLORATION_TOOLS:
            self._soft_streak += 1
            self._max_soft_streak = max(self._max_soft_streak, self._soft_streak)
            if self._soft_streak > 2:
                return self._reject('EXPLORATION_SOFT_FAILURE_LIMIT', step, tool)
            return self.report()
        if any((isinstance(event, Mapping) and isinstance(event.get('payload'), Mapping) and (event['payload'].get('injection_applied') is False) and (event['payload'].get('terminal_unavailable') is False) for event in scheduler_events or ())):
            return self._reject('EXPECTED_FAILURE_NOT_INJECTED', step, tool)
        self._soft_streak = 0
        if self._consume_close(tool, transition, step):
            return self.report()
        allowance = self._consume_allowance(tool, transition, step, scheduler_events=scheduler_events)
        if allowance is None:
            return self._reject('UNEXPECTED_TOOL_FAILURE', step, tool)
        if tool in NAVIGATION_TOOLS:
            self._navigation_ready = False
        self._pending.add(allowance)
        return self.report()

    def finish(self) -> dict[str, Any]:
        if self._failure_code is None and any((item['used'] and (not item['recovered']) for item in self._close_recoveries.values())):
            self._reject('EXPECTED_FAILURE_NOT_RECOVERED', self._checked, 'put_down')
        if self._failure_code is None and self._pending:
            index = min(self._pending)
            spec = self._allowances[index]
            self._reject('EXPECTED_FAILURE_NOT_RECOVERED', self._checked, str(spec['tool']))
        if self._failure_code is None:
            for spec in self._allowances:
                required = int(spec['max_count']) if float(spec['probability']) >= 1.0 else 0
                if int(spec['used_count']) < required:
                    self._reject('EXPECTED_FAILURE_NOT_OBSERVED', self._checked, str(spec['tool']))
                    break
        return self.report()

    def report(self) -> dict[str, Any]:
        return {'expected_failures': copy.deepcopy(self._allowances), 'failure_code': self._failure_code, 'failure_step': self._failure_step, 'failure_tool': self._failure_tool, 'max_consecutive_exploration_failures': self._max_soft_streak, 'navigation_fallback_count': self._fallbacks, 'schema_version': ORACLE_TRACE_QUALITY_SCHEMA_VERSION, 'status': 'failed' if self._failure_code else 'passed', 'transitions_checked': self._checked, 'close_disturbance_recoveries': copy.deepcopy(list(self._close_recoveries.values()))}

    def _observe_close_events(self, events: Sequence[Mapping[str, Any]], step: int) -> None:
        for event in events:
            if event.get('name') != 'close_object':
                continue
            payload = event.get('payload') or {}
            index = payload.get('config_index')
            spec = self._close_configs.get(index)
            event_step = payload.get('step')
            if spec is None or index in self._close_recoveries or (not isinstance(event_step, int)):
                continue
            if event_step > step or (event_step == step and event.get('phase') != 'pre_step'):
                continue
            if payload.get('target_role') != spec['target_role'] or not payload.get('target'):
                continue
            if (payload.get('before') or {}).get('open') is not True:
                continue
            if (payload.get('after') or {}).get('open') is not False:
                continue
            if (payload.get('mutation') or {}).get('success') is not True:
                continue
            self._close_recoveries[index] = {'config_index': index, 'event_step': event_step, 'target_role': spec['target_role'], 'target': payload['target'], 'used': False, 'opened': False, 'recovered': False}

    def _consume_close(self, tool: str, transition: Mapping[str, Any], step: int) -> bool:
        if tool != 'put_down' or (transition.get('env_feedback') or {}).get('error_type') != 'container_closed':
            return False
        target = (transition.get('arguments') or {}).get('target')
        raw_target = (transition.get('raw_execution') or {}).get('target')
        for item in self._close_recoveries.values():
            if item['used'] or item['opened'] or step < item['event_step']:
                continue
            if target != item['target_role'] or (raw_target and raw_target != item['target']):
                continue
            item['used'] = True
            return True
        return False

    def _recover_close(self, tool: str, transition: Mapping[str, Any]) -> None:
        target = (transition.get('arguments') or {}).get('target')
        raw_target = (transition.get('raw_execution') or {}).get('target')
        for item in self._close_recoveries.values():
            if target != item['target_role'] or (raw_target and raw_target != item['target']):
                continue
            if tool == 'open':
                item['opened'] = True
            elif tool == 'put_down' and item['used'] and item['opened']:
                item['recovered'] = True

    def _consume_allowance(self, tool: str, transition: Mapping[str, Any], step: int, *, scheduler_events: Sequence[Mapping[str, Any]] | None) -> int | None:
        modes = _scheduler_failure_modes(scheduler_events) if scheduler_events is not None else None
        target = str((transition.get('arguments') or {}).get('target') or '')
        for index, spec in enumerate(self._allowances):
            if spec['tool'] not in {'any', tool} or int(spec['used_count']) >= int(spec['max_count']):
                continue
            if step < int(spec['after_step']) or step >= int(spec['before_step']):
                continue
            if spec['target_role'] and target and (target != spec['target_role']):
                continue
            if modes is not None and spec['mode'] not in modes:
                continue
            spec['used_count'] = int(spec['used_count']) + 1
            return index
        return None

    def _recover(self, tool: str, transition: Mapping[str, Any]) -> None:
        target = str((transition.get('arguments') or {}).get('target') or '')
        for index in sorted(self._pending):
            spec = self._allowances[index]
            if spec['tool'] not in {'any', tool}:
                continue
            if spec['target_role'] and target and (target != spec['target_role']):
                continue
            spec['recovered_count'] = int(spec['recovered_count']) + 1
            self._pending.remove(index)
            break

    def _reject(self, code: str, step: int, tool: str) -> dict[str, Any]:
        self._failure_code = 'TRACE_QUALITY_FAILED:{}'.format(code)
        self._failure_step = int(step)
        self._failure_tool = tool or None
        return self.report()

def evaluate_oracle_trace_quality(task_spec: Mapping[str, Any], trace_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate = OracleTraceQualityGate(task_spec)
    for record in trace_records:
        if not isinstance(record, Mapping) or record.get('record_type') != 'step':
            continue
        transition = record.get('transition')
        if not isinstance(transition, Mapping):
            continue
        report = gate.observe_transition(transition, scheduler_events=[event for event in record.get('scheduler_events') or [] if isinstance(event, Mapping)])
        if report['status'] == 'failed':
            return report
    return gate.finish()

def _failure_allowances(task_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw: list[Any] = []
    failure_spec = task_spec.get('failure_spec')
    if isinstance(failure_spec, Mapping):
        raw.extend(failure_spec.get('failures') or failure_spec.get('configs') or [failure_spec])
    elif isinstance(failure_spec, Sequence) and (not isinstance(failure_spec, (str, bytes))):
        raw.extend(failure_spec)
    disturbance_spec = task_spec.get('disturbance_spec')
    if isinstance(disturbance_spec, Mapping):
        raw.extend(disturbance_spec.get('failures') or [])
    allowances = []
    for item in raw:
        if not isinstance(item, Mapping) or not item.get('tool') or (not item.get('mode')):
            continue
        allowances.append({'after_step': int(item.get('after_step', 0)), 'before_step': int(item.get('before_step', 10 ** 9)), 'max_count': max(0, int(item.get('max_count', 1))), 'mode': str(item['mode']), 'probability': float(item.get('probability', 1.0)), 'recovered_count': 0, 'target_role': str(item.get('target_role') or ''), 'tool': str(item['tool']), 'used_count': 0})
    return allowances

def _scheduler_failure_modes(events: Sequence[Mapping[str, Any]]) -> set[str]:
    modes = set()
    for event in events:
        name = str(event.get('name') or '')
        if not name.startswith('failure_'):
            continue
        payload = event.get('payload') if isinstance(event.get('payload'), Mapping) else {}
        mode = str(payload.get('config_mode') or name.removeprefix('failure_'))
        if mode:
            modes.add(mode)
    return modes

def _fallback_quality(transition: Mapping[str, Any]) -> tuple[str | None, int]:
    count = 0
    for record in transition.get('low_level_actions') or []:
        if not isinstance(record, Mapping):
            continue
        if not (record.get('visual_attach_navigation_fallback') is True or 'visual_navigation_teleport' in str(record.get('label') or '')):
            continue
        count += 1
        source = record.get('from_waypoint')
        target = record.get('to_waypoint')
        if record.get('adjacent_waypoint_fallback') is not True:
            return ('NON_ADJACENT_NAVIGATION_FALLBACK', count)
        if not (isinstance(source, Sequence) and isinstance(target, Sequence) and (len(source) == len(target)) and (sum((abs(int(a) - int(b)) for a, b in zip(source, target))) == 1)):
            return ('NON_ADJACENT_NAVIGATION_FALLBACK', count)
    return (None, count)
__all__ = ['EXPLORATION_TOOLS', 'NAVIGATION_TOOLS', 'ORACLE_TRACE_QUALITY_SCHEMA_VERSION', 'PHYSICAL_TOOLS', 'OracleTraceQualityError', 'OracleTraceQualityGate', 'evaluate_oracle_trace_quality']
