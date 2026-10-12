from __future__ import annotations
import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping
from scoper.benchmark.procthor.export.schema import EXECUTION_AUDIT_SCHEMA_VERSION
AUDIT_PHASES = ('reset_init', 'decision', 'agent_action', 'scheduler', 'capture_settle', 'teardown')
FORBIDDEN_AGENT_CLASSIFICATIONS = {'forceAction', 'force_arm_open_close_collision_fallback', 'force_navigation_fallback', 'force_open_container_fallback', 'force_viewpoint_fallback', 'force_put_object_fallback', 'Teleport', 'TeleportFull', 'TeleportObject', 'visual_attach_navigation_fallback', 'SetObjectPoses', 'SetObjectStates', 'logical_put_down_fallback', 'sticky_parent_set', 'sticky_parent_clear', 'sticky_property_write', 'visual_attachment_pickup', 'visual_attachment_follow', 'visual_attachment_release'}
KNOWN_FALLBACK_CLASSIFICATIONS = frozenset(FORBIDDEN_AGENT_CLASSIFICATIONS)

class ExecutionAuditError(RuntimeError):
    pass

class NativeProfileViolation(ExecutionAuditError):
    pass

@dataclass(frozen=True)
class AuditContext:
    episode_id: str
    execution_profile: str
    phase: str = 'reset_init'
    step_id: int = 0
    action_id: str | None = None
    intervention_id: str | None = None
    trigger: str | None = None
    allowed_agent_fallbacks: tuple[str, ...] = ()
    controller_classifications: tuple[str, ...] = ()
    controller_details: Mapping[str, Any] | None = None

def classify_controller_step(kwargs: Mapping[str, Any]) -> list[str]:
    action = str(kwargs.get('action') or '')
    classifications: set[str] = set()
    if kwargs.get('forceAction') is True:
        classifications.add('forceAction')
        if action in ('MoveAhead', 'RotateLeft', 'RotateRight', 'TeleportFull'):
            classifications.add('force_navigation_fallback')
        elif action == 'OpenObject':
            classifications.add('force_open_container_fallback')
        elif action in ('PutObject', 'DropHandObject'):
            classifications.add('force_put_object_fallback')
    if action in ('Teleport', 'TeleportFull', 'TeleportObject'):
        classifications.add(action)
    if action in ('SetObjectPoses', 'SetObjectStates'):
        classifications.add(action)
    for key in ('force_arm_open_close_collision_fallback', 'force_navigation_fallback', 'force_open_container_fallback', 'force_put_object_fallback', 'force_viewpoint_fallback', 'visual_attach_navigation_fallback', 'logical_put_down_fallback'):
        if kwargs.get(key):
            classifications.add(key)
    return sorted(classifications)

def _controller_classifications(kwargs: Mapping[str, Any], declared: set[str]) -> list[str]:
    classifications = set(classify_controller_step(kwargs)) | declared
    if 'force_arm_open_close_collision_fallback' in declared:
        classifications.discard('force_open_container_fallback')
    if 'force_viewpoint_fallback' in declared:
        classifications.discard('force_navigation_fallback')
    return sorted(classifications)

def classify_overlay(kind: str, details: Mapping[str, Any]) -> list[str]:
    if kind == 'sticky_write':
        fields = set(details.get('fields', []) or [])
        values = details.get('values', {}) or {}
        output = set()
        if 'parentReceptacles' in fields:
            output.add('sticky_parent_clear' if not values.get('parentReceptacles') else 'sticky_parent_set')
        if fields - {'parentReceptacles'}:
            output.add('sticky_property_write')
        return sorted(output)
    mapping = {'visual_attachment_pickup': 'visual_attachment_pickup', 'visual_attachment_follow': 'visual_attachment_follow', 'visual_attachment_release': 'visual_attachment_release', 'logical_relation': 'logical_put_down_fallback'}
    return [mapping[kind]] if kind in mapping else []

def allowed_by_profile(context: AuditContext, classifications: list[str], *, requested_action: str | None=None) -> bool:
    if context.phase in ('reset_init', 'scheduler'):
        return True
    if context.phase == 'capture_settle':
        return requested_action == 'Pass' and (not classifications)
    if context.phase == 'agent_action':
        forbidden = FORBIDDEN_AGENT_CLASSIFICATIONS.intersection(classifications)
        allowed = set() if context.execution_profile == 'native_ai2thor' else set(context.allowed_agent_fallbacks)
        return forbidden.issubset(allowed)
    return True

class AuditedController:

    def __init__(self, controller: Any, context_provider: Callable[[], AuditContext], sink: Callable[[Mapping[str, Any]], None], event_ordinal_sink: Callable[[int], None] | None=None):
        self._controller = controller
        self._context_provider = context_provider
        self._sink = sink
        self._event_ordinal_sink = event_ordinal_sink
        self._event_ordinal = 0
        self._stopped = False

    @property
    def last_event(self) -> Any:
        return self._controller.last_event

    @property
    def event_ordinal(self) -> int:
        return self._event_ordinal

    def step(self, **kwargs: Any) -> Any:
        context = self._context_provider()
        if context.phase not in AUDIT_PHASES:
            raise ExecutionAuditError('unsupported audit phase: {}'.format(context.phase))
        declared = set(context.controller_classifications)
        if not declared.issubset(KNOWN_FALLBACK_CLASSIFICATIONS):
            raise ExecutionAuditError('unknown declared controller classification')
        classifications = _controller_classifications(kwargs, declared)
        allowed = allowed_by_profile(context, classifications, requested_action=kwargs.get('action'))
        event = None
        error = None
        try:
            event = self._controller.step(**kwargs)
            return event
        except Exception as exc:
            error = exc
            raise
        finally:
            self._event_ordinal += 1
            if self._event_ordinal_sink is not None:
                self._event_ordinal_sink(self._event_ordinal)
            metadata = getattr(event, 'metadata', {}) if event is not None else {}
            details = {'frame_available': getattr(event, 'frame', None) is not None, 'metadata_available': bool(metadata), 'error_message': metadata.get('errorMessage') or None, 'declared_classifications': sorted(declared)}
            if context.controller_details is not None:
                details.update(copy.deepcopy(dict(context.controller_details)))
            applied = event is not None
            success = bool(metadata.get('lastActionSuccess', event is not None))
            if 'force_arm_open_close_collision_fallback' in declared:
                desired = details.get('desired_is_open')
                target_id = details.get('target_object_id')
                post_is_open = _unique_object_open_state(metadata, target_id)
                details.update({'forced_attempt': True, 'post_is_open': post_is_open, 'forced_last_action_success': success})
                postcondition_satisfied = isinstance(desired, bool) and post_is_open is desired
                details['postcondition_satisfied'] = postcondition_satisfied
                applied = applied and success and postcondition_satisfied
                success = success and postcondition_satisfied
            record = _audit_record(context=context, event_ordinal=self._event_ordinal, kind='controller_step', requested_action=kwargs.get('action'), executed_action=metadata.get('lastAction') or kwargs.get('action'), kwargs_summary=summarize_kwargs(kwargs), classifications=classifications, allowed=allowed, attempted=True, applied=applied, success=success, details=details, error=error)
            self._sink(record)
            if not allowed and error is None:
                raise NativeProfileViolation('{} profile forbids agent fallback: {}'.format(context.execution_profile, ', '.join(classifications)))

    def stop(self) -> Any:
        if self._stopped:
            return None
        self._stopped = True
        return self._controller.stop()
    close = stop

    def __getattr__(self, name: str) -> Any:
        return getattr(self._controller, name)

def emit_overlay_audit(env: Any, *, kind: str, details: Mapping[str, Any], attempted: bool=True, applied: bool=True, success: bool=True) -> None:
    sink = getattr(env, '_trajectory_audit_sink', None)
    capture_enabled = bool(getattr(env, '_trajectory_capture_enabled', False))
    if sink is None:
        if capture_enabled:
            raise ExecutionAuditError('capture overlay audit sink is missing')
        return
    context = current_audit_context(env)
    classifications = classify_overlay(kind, details)
    allowed = allowed_by_profile(context, classifications, requested_action=None)
    record = _audit_record(context=context, event_ordinal=None, kind=kind, requested_action=None, executed_action=None, kwargs_summary={}, classifications=classifications, allowed=allowed, attempted=attempted, applied=applied, success=success, details=copy.deepcopy(dict(details)), error=None)
    sink(record)
    if not allowed:
        raise NativeProfileViolation('{} profile forbids agent overlay: {}'.format(context.execution_profile, ', '.join(classifications)))

def configure_trajectory_capture(env: Any, *, episode_context: Any, execution_profile: str, sink: Callable[[Mapping[str, Any]], None], allowed_agent_fallbacks: tuple[str, ...]=()) -> None:
    env._trajectory_capture_enabled = True
    env._trajectory_episode_context = episode_context
    env._trajectory_audit_sink = sink
    env._trajectory_event_ordinal = 0
    env._trajectory_audit_context = AuditContext(episode_id=episode_context.episode_id, execution_profile=execution_profile, allowed_agent_fallbacks=tuple(allowed_agent_fallbacks))

def install_audited_controller(env: Any) -> None:
    controller = getattr(env, 'controller', None)
    if controller is None or isinstance(controller, AuditedController):
        return
    if not getattr(env, '_trajectory_capture_enabled', False):
        return
    sink = getattr(env, '_trajectory_audit_sink', None)
    if sink is None:
        raise ExecutionAuditError('capture controller audit sink is missing')
    env.controller = AuditedController(controller, lambda: current_audit_context(env), sink, lambda ordinal: setattr(env, '_trajectory_event_ordinal', ordinal))

def current_audit_context(env: Any) -> AuditContext:
    context = getattr(env, '_trajectory_audit_context', None)
    if not isinstance(context, AuditContext):
        raise ExecutionAuditError('trajectory audit context is not configured')
    return context

@contextmanager
def audit_phase(env: Any, phase: str, *, step_id: int=0, action_id: str | None=None, intervention_id: str | None=None, trigger: str | None=None):
    if not getattr(env, '_trajectory_capture_enabled', False):
        yield
        return
    if phase not in AUDIT_PHASES:
        raise ExecutionAuditError('unsupported audit phase: {}'.format(phase))
    previous = current_audit_context(env)
    if previous.phase != 'reset_init' and previous.phase != phase and (not (previous.phase == 'scheduler' and phase == 'agent_action')):
        raise ExecutionAuditError('nested audit phases are forbidden: {} -> {}'.format(previous.phase, phase))
    env._trajectory_audit_context = replace(previous, phase=phase, step_id=int(step_id), action_id=action_id, intervention_id=intervention_id, trigger=trigger)
    try:
        yield
    finally:
        env._trajectory_audit_context = previous

@contextmanager
def audit_controller_classifications(env: Any, *classifications: str, details: Mapping[str, Any] | None=None):
    if not getattr(env, '_trajectory_capture_enabled', False):
        yield
        return
    declared = tuple(sorted(set(classifications)))
    if not set(declared).issubset(KNOWN_FALLBACK_CLASSIFICATIONS):
        raise ExecutionAuditError('unknown declared controller classification')
    previous = current_audit_context(env)
    if details is not None and (not isinstance(details, Mapping)):
        raise ExecutionAuditError('controller audit details must be an object')
    merged_details = copy.deepcopy(dict(previous.controller_details or {}))
    if details is not None:
        merged_details.update(copy.deepcopy(dict(details)))
    env._trajectory_audit_context = replace(previous, controller_classifications=tuple(sorted(set(previous.controller_classifications) | set(declared))), controller_details=merged_details)
    try:
        yield
    finally:
        env._trajectory_audit_context = previous

def summarize_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {'action', 'objectId', 'receptacleObjectId', 'position', 'rotation', 'horizon', 'standing', 'forceAction', 'openness', 'objectPoses', 'SetObjectStates', 'agentRadius'}
    result = {}
    for key in sorted(set(kwargs).intersection(allowed)):
        value = kwargs[key]
        if key == 'objectPoses' and isinstance(value, list):
            result[key] = {'count': len(value), 'object_names': sorted((str(item.get('objectName')) for item in value if isinstance(item, Mapping) and item.get('objectName')))}
        else:
            result[key] = copy.deepcopy(value)
    return result

def validate_execution_audit_records(records: list[Mapping[str, Any]], *, execution_profile: str, allowed_agent_fallbacks: tuple[str, ...]=()) -> dict[str, Any]:
    controller_records = []
    previous_sequence = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ExecutionAuditError('execution audit record must be an object')
        if record.get('schema_version') != EXECUTION_AUDIT_SCHEMA_VERSION:
            raise ExecutionAuditError('execution audit schema version is invalid')
        sequence = record.get('sequence_index')
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise ExecutionAuditError('execution audit sequence must be integer')
        if sequence != previous_sequence + 1:
            raise ExecutionAuditError('execution audit sequence is not contiguous')
        previous_sequence = sequence
        if record.get('kind') == 'controller_step':
            controller_records.append(record)
    ordinals = [record.get('event_ordinal') for record in controller_records]
    if ordinals != list(range(1, len(ordinals) + 1)):
        raise ExecutionAuditError('controller event ordinals are not contiguous')
    allowed = set(allowed_agent_fallbacks)
    if not allowed.issubset(KNOWN_FALLBACK_CLASSIFICATIONS):
        raise ExecutionAuditError('audit validator allowlist contains unknown fallback')
    if execution_profile == 'native_ai2thor' and allowed:
        raise ExecutionAuditError('native profile fallback allowlist must be empty')
    violations = []
    for record in records:
        classifications = set(record.get('classification', []) or [])
        if record.get('kind') == 'controller_step':
            declared = record.get('details', {}).get('declared_classifications', [])
            if not isinstance(declared, list) or not set(declared).issubset(KNOWN_FALLBACK_CLASSIFICATIONS):
                raise ExecutionAuditError('audit declared classification is invalid')
            recomputed = set(_controller_classifications(record.get('kwargs_summary', {}), set(declared)))
        else:
            recomputed = set(classify_overlay(str(record.get('kind') or ''), record.get('details', {})))
        if classifications != recomputed:
            raise ExecutionAuditError('audit classification differs from recomputed value')
        if 'force_arm_open_close_collision_fallback' in classifications:
            _validate_arm_open_close_fallback_record(record)
        context = AuditContext(episode_id=str(record.get('episode_id') or ''), execution_profile=execution_profile, phase=str(record.get('phase') or ''), allowed_agent_fallbacks=tuple(allowed_agent_fallbacks))
        expected_allowed = allowed_by_profile(context, sorted(classifications), requested_action=record.get('requested_action'))
        if record.get('allowed_by_profile') is not expected_allowed:
            raise ExecutionAuditError('audit allowed_by_profile differs from policy')
        if record.get('phase') != 'agent_action':
            if not expected_allowed:
                violations.append(record)
            continue
        forbidden = FORBIDDEN_AGENT_CLASSIFICATIONS.intersection(classifications)
        if forbidden and (not expected_allowed):
            violations.append(record)
    if violations:
        raise NativeProfileViolation('{} profile contains non-allowlisted agent fallback'.format(execution_profile))
    return {'audit_records': len(records), 'controller_steps': len(controller_records), 'profile_violations': len(violations)}

def _unique_object_open_state(metadata: Any, object_id: Any) -> bool | None:
    if not isinstance(metadata, Mapping) or not isinstance(object_id, str):
        return None
    matches = [item for item in metadata.get('objects', []) or [] if isinstance(item, Mapping) and item.get('objectId') == object_id]
    if len(matches) != 1 or type(matches[0].get('isOpen')) is not bool:
        return None
    return matches[0]['isOpen']

def _validate_arm_open_close_fallback_record(record: Mapping[str, Any]) -> None:
    details = record.get('details')
    kwargs = record.get('kwargs_summary')
    if not isinstance(details, Mapping) or not isinstance(kwargs, Mapping):
        raise ExecutionAuditError('arm open/close fallback audit details are missing')
    action = record.get('requested_action')
    target = details.get('target_object_id')
    desired = details.get('desired_is_open')
    normal_error = details.get('normal_attempt_error')
    used = details.get('fallback_budget_used')
    limit = details.get('fallback_budget_limit')
    forced_success = details.get('forced_last_action_success')
    post_is_open = details.get('post_is_open')
    postcondition = details.get('postcondition_satisfied')
    normalized_error = normal_error.casefold().replace('-', '_').replace(' ', '_') if isinstance(normal_error, str) else ''
    valid = action in {'OpenObject', 'CloseObject'} and kwargs.get('action') == action and (kwargs.get('forceAction') is True) and isinstance(target, str) and bool(target) and (kwargs.get('objectId') == target) and (type(desired) is bool) and (desired is (action == 'OpenObject')) and (details.get('arm_mode') in {'official', 'visual_attach'}) and any((token in normalized_error for token in ('robot_arm', 'robot_wrist', 'robot_shoulder', 'robot_hand', 'gripper'))) and (details.get('forced_attempt') is True) and (type(forced_success) is bool) and (type(post_is_open) is bool) and (type(postcondition) is bool) and (postcondition is (post_is_open is desired)) and (type(used) is int) and (type(limit) is int) and (1 <= used <= limit)
    expected_success = bool(forced_success and postcondition)
    if not valid or record.get('applied') is not expected_success or record.get('success') is not expected_success:
        raise ExecutionAuditError('arm open/close fallback audit is invalid')

def _audit_record(*, context: AuditContext, event_ordinal: int | None, kind: str, requested_action: Any, executed_action: Any, kwargs_summary: Mapping[str, Any], classifications: list[str], allowed: bool, attempted: bool, applied: bool, success: bool, details: Mapping[str, Any], error: Exception | None) -> dict[str, Any]:
    return {'schema_version': EXECUTION_AUDIT_SCHEMA_VERSION, 'episode_id': context.episode_id, 'step_id': context.step_id, 'action_id': context.action_id, 'intervention_id': context.intervention_id, 'event_ordinal': event_ordinal, 'phase': context.phase, 'kind': kind, 'requested_action': requested_action, 'executed_action': executed_action, 'kwargs_summary': copy.deepcopy(dict(kwargs_summary)), 'classification': list(classifications), 'trigger': context.trigger, 'allowed_by_profile': allowed, 'attempted': attempted, 'applied': applied, 'success': success, 'native_state_ref': None, 'effective_state_ref': None, 'details': copy.deepcopy(dict(details)), 'error': None if error is None else {'type': type(error).__name__, 'message': str(error)}}
__all__ = ['AUDIT_PHASES', 'AuditContext', 'AuditedController', 'ExecutionAuditError', 'NativeProfileViolation', 'KNOWN_FALLBACK_CLASSIFICATIONS', 'allowed_by_profile', 'audit_controller_classifications', 'audit_phase', 'classify_controller_step', 'classify_overlay', 'configure_trajectory_capture', 'emit_overlay_audit', 'install_audited_controller', 'summarize_kwargs', 'validate_execution_audit_records']
