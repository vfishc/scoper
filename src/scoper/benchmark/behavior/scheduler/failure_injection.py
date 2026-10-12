from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional
from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.execution_core.navigation import NavigationTarget, plan_navigation
from scoper.benchmark.behavior.execution_core.trace import to_jsonable
from scoper.benchmark.behavior.scheduler.base import Scheduler, SchedulerEvent
from scoper.benchmark.behavior.scheduler.disturbance import _relocate_object_to_recoverable_ground
KNOWN_FAILURE_MODES = frozenset({'block', 'report_fail', 'delayed_success', 'partial_execution', 'wrong_target', 'grasp_slip', 'nav_blocked_partway', 'drop_beside', 'partial_open', 'partial_close', 'nav_wrong_target'})
TOOL_RESTRICTED_MODES = {'grasp_slip': frozenset({'pick_up'}), 'nav_blocked_partway': frozenset({'navigate_to'}), 'drop_beside': frozenset({'put_down'}), 'partial_open': frozenset({'open'}), 'partial_close': frozenset({'close'}), 'nav_wrong_target': frozenset({'navigate_to'})}
TARGETED_TOOLS = frozenset({'navigate_to', 'pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'look_for', 'wipe', 'cut', 'soak_under', 'fill', 'pour', 'heat', 'cool'})
_DROP_BESIDE_COMMIT_ATTEMPT_LIMIT = 4
_DROP_BESIDE_TOTAL_COMMIT_ATTEMPT_LIMIT = 12

@dataclass
class FailureConfig:
    mode: str
    after_step: int = 0
    tool: str | None = None
    once: bool = True
    before_step: int = 10 ** 9
    probability: float = 1.0
    max_count: int = 10 ** 9
    cooldown: int = 0
    targets: List[str] | None = None
    target_role: str | None = None
    feedback_message: str | None = None
    extra: dict | None = None
    activation: dict | None = None

    def __post_init__(self):
        if self.mode not in KNOWN_FAILURE_MODES:
            raise ValueError("Unknown failure mode '{}'.".format(self.mode))
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError('FailureConfig.probability must be in [0, 1].')

class FailureInjectionScheduler(Scheduler):
    name = 'failure_injection'

    def __init__(self, configs: list, seed: int=0) -> None:
        self._configs = [FailureConfig(**_failure_config_kwargs(cfg)) if isinstance(cfg, Mapping) else cfg for cfg in configs]
        self._fired: set = set()
        self._rng = random.Random(seed)
        self._fire_counts: dict[int, int] = {i: 0 for i in range(len(self._configs))}
        self._last_fire_step: dict[int, int] = {}
        self._active_config: FailureConfig | None = None
        self._active_config_index: int | None = None
        self._active_target_id: str | None = None
        self._unavailable_configs: set[int] = set()
        self._delayed_notices: list[dict[str, Any]] = []
        self._drop_beside_rejected_centers: dict[tuple[int, str, str], set[tuple[float, float, float]]] = {}
        self._drop_beside_commit_attempts: dict[tuple[int, str, str], int] = {}

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> Optional[List[SchedulerEvent]]:
        self._fired = set()
        self._fire_counts = {i: 0 for i in range(len(self._configs))}
        self._last_fire_step = {}
        self._delayed_notices = []
        self._active_config_index = None
        self._active_target_id = None
        self._unavailable_configs = set()
        self._drop_beside_rejected_centers = {}
        self._drop_beside_commit_attempts = {}
        return [SchedulerEvent('failure_armed', 'episode_start', {'num_configs': len(self._configs)})]

    def pre_step(self, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]) -> Optional[List[SchedulerEvent]]:
        if (getattr(env, '_human_qa_state', {}) or {}).get('pending'):
            return None
        if not self._delayed_notices:
            return None
        events = [SchedulerEvent('delayed_success_notice', 'pre_step', dict(notice)) for notice in self._delayed_notices]
        self._delayed_notices = []
        return events

    def intercept_tool_call(self, env: Any, tool_call: Any, high_level_step: int, default_call: Callable[[], dict]) -> Optional[dict]:
        if getattr(tool_call, 'tool_name', None) == 'answer':
            return None
        cfg = self._due_config(env, high_level_step, tool_call)
        if cfg is None:
            return None
        transition: dict | None = None
        if cfg.mode == 'block':
            transition = self._blocked_transition(env, tool_call)
        elif cfg.mode == 'report_fail':
            transition = default_call()
            underlying_success = bool(transition.get('success', False))
            transition['success'] = False
            _record_rewritten_failure(env, underlying_success)
            transition['env_feedback'] = {'error_type': 'injected_report_fail', 'error_message': 'Failure injected (report_fail).'}
            transition['scheduler_intercept'] = [SchedulerEvent('failure_report_fail', 'intercept', {'tool': tool_call.tool_name})]
            transition = _refresh_transition(env, transition)
        elif cfg.mode == 'delayed_success':
            transition = default_call()
            underlying_success = bool(transition.get('success', False))
            transition['success'] = False
            _record_rewritten_failure(env, underlying_success)
            transition['env_feedback'] = {'error_type': 'injected_delayed', 'error_message': 'Delayed success injected.'}
            if underlying_success:
                self._delayed_notices.append({'tool': tool_call.tool_name, 'step': high_level_step, 'completed_at_step': high_level_step + 1})
            transition['scheduler_intercept'] = [SchedulerEvent('failure_delayed_success', 'intercept', {'tool': tool_call.tool_name, 'step': high_level_step, 'underlying_success': underlying_success, 'config_mode': cfg.mode, 'config_tool': cfg.tool or 'any'})]
            transition = _refresh_transition(env, transition)
        elif cfg.mode == 'partial_execution':
            transition = self._partial_execution(env, tool_call, default_call)
        elif cfg.mode == 'wrong_target':
            transition = self._wrong_target(env, tool_call, event_name='failure_wrong_target')
        elif cfg.mode == 'grasp_slip':
            transition = default_call()
            underlying_success = bool(transition.get('success', False))
            held = env.core.ego_state.held_object_name
            obj = env.core.objects.get(held) if held is not None else None
            relocation = _relocate_object_to_recoverable_ground(obj, env, {'min_distance': 0.25, 'max_distance': 0.8, 'prefer_different_room': False}) if underlying_success and obj is not None else None
            injection_applied = relocation is not None
            if injection_applied:
                env.core.ego_state.held_object_name = None
                env.core.ego_state.held_visual_name = None
                if hasattr(env.core, '_clear_held_pose_anchor'):
                    env.core._clear_held_pose_anchor()
                transition['success'] = False
                _record_rewritten_failure(env, underlying_success)
                transition['env_feedback'] = {'error_type': 'grasp_slipped', 'error_message': 'Object slipped from grasp.'}
            state_delta = dict(transition.get('state_delta') or {})
            state_delta['grasp_slip'] = {'dropped': held if injection_applied else None, 'held_object_after': env.core.ego_state.held_object_name, 'relocation': relocation}
            transition['state_delta'] = state_delta
            transition['scheduler_intercept'] = [SchedulerEvent('failure_grasp_slip', 'intercept', {'tool': tool_call.tool_name, 'dropped': held if injection_applied else None, 'underlying_success': underlying_success, 'injection_applied': injection_applied, 'terminal_unavailable': underlying_success and (not injection_applied)})]
            transition = _refresh_transition(env, transition)
        elif cfg.mode == 'drop_beside':
            transition = self._drop_beside(env, tool_call, default_call)
        elif cfg.mode == 'partial_open':
            transition = self._partial_open_close(env, tool_call, default_call, desired_open=False, target_openness=0.05, event_name='failure_partial_open')
        elif cfg.mode == 'partial_close':
            transition = self._partial_open_close(env, tool_call, default_call, desired_open=True, target_openness=0.6, event_name='failure_partial_close')
        elif cfg.mode == 'nav_wrong_target':
            transition = self._wrong_target(env, tool_call, event_name='failure_nav_wrong_target')
        elif cfg.mode == 'nav_blocked_partway':
            transition = self._nav_blocked_partway(env, tool_call, cfg)
        if transition is not None:
            _annotate_failure_transition(transition, cfg)
            for event in transition.get('scheduler_intercept') or []:
                payload = event.payload if isinstance(event, SchedulerEvent) else event.get('payload', {})
                if isinstance(payload, dict):
                    payload.setdefault('config_index', self._active_config_index)
            if self._active_config_index is not None:
                if _failure_transition_applied(transition):
                    self._mark_fired(self._active_config_index, high_level_step)
                elif _failure_transition_terminal_unavailable(transition):
                    self._unavailable_configs.add(self._active_config_index)
        return transition

    def _drop_beside(self, env: Any, tool_call: Any, default_call: Callable[[], dict]) -> dict:
        held = env.core.ego_state.held_object_name
        target_ref = (tool_call.arguments or {}).get('target')
        resolved_target = self._active_target_id or _resolve_object_name(env, target_ref)
        receptacle = env.core.objects.get(resolved_target) if resolved_target else None
        obj = env.core.objects.get(held) if held is not None else None
        before_center = [float(value) for value in obj.center] if obj is not None else None
        transition = default_call()
        underlying_success = bool(transition.get('success', False))
        if not underlying_success:
            transition['scheduler_intercept'] = [SchedulerEvent('failure_drop_beside', 'intercept', {'tool': tool_call.tool_name, 'target': target_ref, 'receptacle_id': resolved_target, 'object_id': held, 'underlying_success': False, 'injection_applied': False, 'terminal_unavailable': False})]
            return transition
        drop_report: dict[str, Any] = {'source': 'invalid_inputs', 'drop_success': False}
        if obj is not None and receptacle is not None:
            config_index = int(self._active_config_index or 0)
            attempt_key = (config_index, str(held), str(resolved_target))
            excluded_centers = self._drop_beside_rejected_centers.setdefault(attempt_key, set())
            total_attempts = int(self._drop_beside_commit_attempts.get(attempt_key, 0))
            remaining_attempts = max(0, _DROP_BESIDE_TOTAL_COMMIT_ATTEMPT_LIMIT - total_attempts)
            commit_attempts: list[dict[str, Any]] = []
            for _attempt in range(min(_DROP_BESIDE_COMMIT_ATTEMPT_LIMIT, remaining_attempts)):
                drop_center, candidate_report = _choose_drop_beside_center(env, obj, receptacle, excluded_centers=excluded_centers)
                drop_report = candidate_report
                excluded_centers.update(candidate_report.pop('rejected_center_keys', []))
                if drop_center is None:
                    break
                commit_report = _commit_drop_beside(env, obj, drop_center)
                total_attempts += 1
                self._drop_beside_commit_attempts[attempt_key] = total_attempts
                commit_attempts.append({'center': list(drop_center), 'source': candidate_report.get('source'), 'success': bool(commit_report.get('drop_success')), 'navigation_selection': commit_report.get('navigation_selection'), 'rendered_visible_at_interaction_pose': commit_report.get('rendered_visible_at_interaction_pose')})
                drop_report.update(commit_report)
                if commit_report.get('drop_success'):
                    break
                excluded_centers.add(_center_key(drop_center))
            if remaining_attempts <= 0:
                drop_report = {'source': 'attempt_budget_exhausted', 'drop_success': False, 'reachable': False, 'reachability_reason': 'drop_beside_total_attempt_budget_exhausted'}
            drop_report['commit_attempts'] = commit_attempts
            drop_report['total_commit_attempts'] = total_attempts
            drop_report['total_commit_attempt_limit'] = _DROP_BESIDE_TOTAL_COMMIT_ATTEMPT_LIMIT
            drop_report['rejected_center_count'] = len(excluded_centers)
        after_center = [float(value) for value in obj.center] if obj is not None else None
        relation_after = _relation_to_receptacle(obj, receptacle) if obj is not None and receptacle is not None else {}
        state_delta = {'held_object': {'before': held, 'after': getattr(env.core.ego_state, 'held_object_name', None)}, 'dropped_beside': {'target': target_ref, 'receptacle': resolved_target, 'object': held, 'before_center': before_center, 'after_center': after_center, 'drop_position': after_center, 'drop_success': bool(drop_report.get('drop_success', False)), 'drop_source': drop_report.get('source'), 'relation_to_receptacle_after': relation_after}}
        event_payload = {'tool': tool_call.tool_name, 'target': target_ref, 'receptacle_id': resolved_target, 'dropped': held, 'object_id': held, 'drop_position': after_center, 'drop_success': bool(drop_report.get('drop_success', False)), 'drop_source': drop_report.get('source'), 'underlying_success': underlying_success, 'injection_applied': bool(drop_report.get('drop_success', False)), 'terminal_unavailable': bool(not drop_report.get('drop_success', False) and (str(drop_report.get('source') or '') in {'invalid_inputs', 'no_safe_candidate', 'attempt_budget_exhausted'} or int(drop_report.get('total_commit_attempts', 0) or 0) >= _DROP_BESIDE_TOTAL_COMMIT_ATTEMPT_LIMIT))}
        for key in ('candidate_count', 'rejected_candidates', 'reachable', 'reachability_reason', 'occupancy_rebuilt', 'renderer_synced', 'renderer_sync_error', 'rollback_success', 'public_navigation_ready', 'rendered_visible_at_interaction_pose', 'stable_visibility_passes', 'stable_visibility_required', 'navigation_selection', 'commit_attempts', 'total_commit_attempts', 'total_commit_attempt_limit', 'rejected_center_count'):
            if key in drop_report:
                event_payload[key] = drop_report[key]
        injection_applied = bool(drop_report.get('drop_success'))
        if injection_applied:
            transition['success'] = False
            _record_rewritten_failure(env, underlying_success)
            transition['feedback'] = 'Object was dropped beside the target receptacle.'
            transition['env_feedback'] = {'error_type': 'drop_beside', 'error_message': 'Object dropped beside target instead of being placed.'}
        merged_delta = dict(transition.get('state_delta') or {})
        merged_delta.update(state_delta)
        transition['state_delta'] = merged_delta
        transition['scheduler_intercept'] = [SchedulerEvent('failure_drop_beside', 'intercept', to_jsonable(event_payload))]
        return _refresh_transition(env, transition)

    def _partial_execution(self, env: Any, tool_call: Any, default_call: Callable[[], dict]) -> dict:
        if tool_call.tool_name in {'open', 'close'}:
            desired_open = tool_call.tool_name == 'close'
            target_openness = 0.05 if tool_call.tool_name == 'open' else 0.6
            transition = self._partial_open_close(env, tool_call, default_call, desired_open=desired_open, target_openness=target_openness, event_name='failure_partial_execution')
            transition['env_feedback'] = {'error_type': 'partial_execution', 'error_message': 'Tool partially executed.'}
            return transition
        transition = default_call()
        underlying_success = bool(transition.get('success', False))
        transition['success'] = False
        _record_rewritten_failure(env, underlying_success)
        transition['feedback'] = 'Tool partially executed.'
        transition['env_feedback'] = {'error_type': 'partial_execution', 'error_message': 'Tool partially executed.'}
        transition['goal_result'] = env.check_goal(model_stopped=env.model_stopped)
        transition['scheduler_intercept'] = [SchedulerEvent('failure_partial_execution', 'intercept', {'tool': tool_call.tool_name, 'target': (tool_call.arguments or {}).get('target'), 'underlying_success': underlying_success})]
        return _refresh_transition(env, transition)

    def _partial_open_close(self, env: Any, tool_call: Any, default_call: Callable[[], dict], *, desired_open: bool, target_openness: float, event_name: str) -> dict:
        target_ref = (tool_call.arguments or {}).get('target')
        resolved = self._active_target_id or _resolve_object_name(env, target_ref)
        obj = env.core.objects.get(resolved) if resolved else None
        states = getattr(obj, 'states', None) if obj is not None else None
        before = states.get('open') if isinstance(states, dict) else None
        before_openness = states.get('openness') if isinstance(states, dict) else None
        transition = default_call()
        normal_success = bool(transition.get('success', False))
        if not normal_success:
            transition['scheduler_intercept'] = [SchedulerEvent(event_name, 'intercept', {'tool': tool_call.tool_name, 'target': target_ref, 'object_id': resolved, 'underlying_success': False, 'injection_applied': False, 'terminal_unavailable': False})]
            return transition
        normal_after = states.get('open') if isinstance(states, dict) else None
        normal_after_openness = states.get('openness') if isinstance(states, dict) else None
        after = normal_after
        after_openness = normal_after_openness
        renderer_report: dict[str, Any] = {'success': False, 'reason': 'renderer_unavailable'}
        if isinstance(states, dict):
            states['open'] = bool(desired_open)
            states['openness'] = float(target_openness)
            after = states.get('open')
            after_openness = states.get('openness')
            renderer_report = _sync_partial_open_close_renderer(env, resolved, float(target_openness), symbolic_open=bool(desired_open), reason=event_name)
        injection_applied = bool(isinstance(states, dict) and (getattr(env, 'renderer', None) is None or renderer_report.get('success') is True))
        if not injection_applied and isinstance(states, dict):
            states['open'] = normal_after
            if normal_after_openness is None:
                states.pop('openness', None)
            else:
                states['openness'] = normal_after_openness
            after = states.get('open')
            after_openness = states.get('openness')
        state_delta = {'partial_open_close': {'target': resolved, 'before': before, 'after': after, 'before_openness': before_openness, 'after_openness': after_openness, 'renderer_sync': renderer_report}}
        event_payload = {'tool': tool_call.tool_name, 'target': target_ref, 'object_id': resolved, 'before_open': before, 'after_open': after, 'before_openness': before_openness, 'openness': after_openness, 'underlying_success': normal_success, 'renderer_synced': bool(renderer_report.get('success', False)), 'renderer_sync': renderer_report, 'injection_applied': injection_applied, 'terminal_unavailable': not injection_applied}
        if injection_applied:
            transition['success'] = False
            _record_rewritten_failure(env, normal_success)
            transition['feedback'] = '{} injected.'.format(event_name)
            transition['env_feedback'] = {'error_type': event_name.replace('failure_', ''), 'error_message': '{} injected; target left at openness={}.'.format(event_name, target_openness)}
        merged_delta = dict(transition.get('state_delta') or {})
        merged_delta.update(state_delta)
        transition['state_delta'] = merged_delta
        transition['scheduler_intercept'] = [SchedulerEvent(event_name, 'intercept', event_payload)]
        return _refresh_transition(env, transition)

    def _nav_wrong_target(self, env: Any, tool_call: Any) -> dict:
        return self._wrong_target(env, tool_call, event_name='failure_nav_wrong_target')

    def _nav_blocked_partway(self, env: Any, tool_call: Any, cfg: FailureConfig) -> dict | None:
        target_name = self._active_target_id
        if target_name is None or target_name not in env.core.objects:
            return None
        if env.core.occupancy is None:
            env.core.build_occupancy('failure_nav_blocked_partway')
        start_pose = env.core.ego_state.pose.copy()
        plan = plan_navigation(start_pose, NavigationTarget.from_aabb_object(env.core.objects[target_name]), env.core.occupancy, env.core.config.navigation)
        poses = list(getattr(plan, 'frame_poses', ()) or ())
        if not plan.success or len(poses) < 2:
            return None
        partial_index = min(len(poses) - 2, max(0, len(poses) // 2))
        partial_pose = poses[partial_index]
        env._set_agent_pose_from_mapping(env._pose_to_dict(partial_pose))
        if hasattr(env.core, '_update_held_pose_anchor'):
            env.core._update_held_pose_anchor(env.core.ego_state.pose)
        if getattr(env, 'renderer', None) is not None:
            env._sync_renderer_for_observation_pose(env._pose_to_dict(env.core.ego_state.pose))
        frames = partial_index + 1
        return _synthetic_failure_transition(env, tool_call, feedback='Navigation blocked partway.', frames=frames, env_feedback={'error_type': 'nav_blocked_partway', 'error_message': 'Path blocked mid-navigation.'}, state_delta={'nav_blocked_partway': {'requested': target_name, 'start_pose': to_jsonable(start_pose), 'partial_pose': to_jsonable(env.core.ego_state.pose.copy()), 'planned_frame_count': len(poses), 'executed_frame_count': frames}}, scheduler_event=SchedulerEvent('failure_nav_blocked_partway', 'intercept', {'tool': tool_call.tool_name, 'target': (tool_call.arguments or {}).get('target'), 'object_id': target_name, 'step': env.high_level_step + 1, 'config_mode': cfg.mode, 'config_tool': cfg.tool or 'any', 'injection_applied': True, 'terminal_unavailable': False}))

    def _wrong_target(self, env: Any, tool_call: Any, *, event_name: str) -> dict:
        args = dict(tool_call.arguments or {})
        requested = args.get('target')
        requested_resolved = self._active_target_id or _resolve_object_name(env, requested)
        cfg = self._active_config
        wrong_target = self._pick_wrong_target(env, tool_call.tool_name, requested, cfg)
        if wrong_target is None:
            return _synthetic_failure_transition(env, tool_call, feedback='Wrong-target failure could not find an alternative target.', frames=0, env_feedback={'error_type': event_name.replace('failure_', ''), 'error_message': 'No alternative target available.'}, scheduler_event=SchedulerEvent(event_name, 'intercept', {'tool': tool_call.tool_name, 'requested': requested, 'actual': None, 'injection_applied': False, 'terminal_unavailable': True}))
        swapped_args = dict(args)
        swapped_args['target'] = wrong_target
        transition = env.call_tool(tool_call.tool_name, swapped_args, reasoning=tool_call.reasoning)
        underlying_success = bool(transition.get('success', False))
        transition['arguments'] = args
        transition['success'] = False
        _record_rewritten_failure(env, underlying_success)
        transition['feedback'] = 'Tool executed on the wrong target.'
        error_type = event_name.replace('failure_', '')
        transition['env_feedback'] = {'error_type': error_type, 'error_message': 'Tool was redirected to another object.'}
        transition['goal_result'] = env.check_goal(model_stopped=env.model_stopped)
        state_delta = dict(transition.get('state_delta') or {})
        state_delta['wrong_target'] = {'requested': requested_resolved, 'actual': env.role_map.get(wrong_target, wrong_target), 'actual_ref': wrong_target, 'underlying_success': underlying_success}
        if event_name == 'failure_nav_wrong_target':
            state_delta['nav_wrong_target'] = {'requested': requested_resolved, 'actual': env.role_map.get(wrong_target, wrong_target), 'actual_ref': wrong_target, 'underlying_success': underlying_success}
        transition['state_delta'] = state_delta
        transition['scheduler_intercept'] = [SchedulerEvent(event_name, 'intercept', {'tool': tool_call.tool_name, 'requested': requested, 'requested_object_id': requested_resolved, 'actual': wrong_target, 'actual_object_id': env.role_map.get(wrong_target, wrong_target), 'underlying_success': underlying_success, 'injection_applied': underlying_success, 'terminal_unavailable': not underlying_success})]
        return _refresh_transition(env, transition)

    def _pick_wrong_target(self, env: Any, tool_name: str, requested: Any, cfg: Optional[FailureConfig]) -> Optional[str]:
        if tool_name not in TARGETED_TOOLS:
            return None
        requested_text = str(requested) if requested is not None else None
        requested_resolved = env.role_map.get(requested_text, requested_text) if requested_text else None
        original = env.core.objects.get(requested_resolved) if requested_resolved else None
        original_category = getattr(original, 'category', None) if original is not None else None
        configured = [str(item) for item in getattr(cfg, 'targets', None) or [] if item]
        configured_candidates = [target for target in configured if target != requested_text and env.role_map.get(target, target) != requested_resolved and self._tool_candidate_allowed(env, tool_name, env.role_map.get(target, target))]
        if configured_candidates:
            return self._rng.choice(configured_candidates)
        candidates: list[tuple[int, str, str]] = []
        for role, object_name in sorted(env.role_map.items()):
            if requested_text is not None and role == requested_text:
                continue
            if requested_resolved is not None and object_name == requested_resolved:
                continue
            obj = env.core.objects.get(object_name)
            if obj is None:
                continue
            if not self._tool_candidate_allowed(env, tool_name, object_name):
                continue
            score = 0 if original_category and getattr(obj, 'category', None) == original_category else 1
            candidates.append((score, str(role), str(object_name)))
        if not candidates:
            for object_name, obj in sorted(env.core.objects.items()):
                if requested_resolved is not None and object_name == requested_resolved:
                    continue
                if not self._tool_candidate_allowed(env, tool_name, object_name):
                    continue
                score = 0 if original_category and getattr(obj, 'category', None) == original_category else 1
                candidates.append((score, str(object_name), str(object_name)))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def _tool_candidate_allowed(self, env: Any, tool_name: str, object_name: str) -> bool:
        obj = env.core.objects.get(object_name)
        if obj is None:
            return False
        if tool_name in {'navigate_to', 'look_for'}:
            return True
        if tool_name in {'open', 'close'}:
            states = getattr(obj, 'states', {}) or {}
            return 'open' in states or 'openable' in tuple(getattr(obj, 'abilities', ()) or ())
        if tool_name in {'toggle_on', 'toggle_off'}:
            states = getattr(obj, 'states', {}) or {}
            return 'toggled_on' in states or 'toggleable' in tuple(getattr(obj, 'abilities', ()) or ())
        if tool_name == 'pick_up':
            return actions.supports_pick_action(obj)
        if tool_name == 'put_down':
            return actions.supports_place_on_surface(obj) or actions.supports_place_inside_container(obj)
        if tool_name == 'wipe':
            return actions.supports_wipe_target(obj)
        if tool_name == 'cut':
            return actions.supports_cut_target(obj)
        if tool_name in {'soak_under'}:
            return actions.supports_fluid_source(obj)
        if tool_name == 'fill':
            return actions.supports_fillable_container(obj)
        if tool_name == 'pour':
            return actions.supports_pourable_source(obj)
        if tool_name == 'heat':
            return actions.supports_heatsource(obj)
        if tool_name == 'cool':
            return actions.supports_coldsource(obj)
        return True

    def _due_config(self, env: Any, high_level_step: int, tool_call: Any) -> Optional[FailureConfig]:
        tool_name = tool_call.tool_name if hasattr(tool_call, 'tool_name') else str(tool_call)
        for i, cfg in enumerate(self._configs):
            if i in self._unavailable_configs:
                continue
            if tool_name == 'stop':
                continue
            if not _subtask_gate_matches(env, cfg.activation):
                continue
            count = int(self._fire_counts.get(i, 0))
            if cfg.once and count > 0:
                continue
            if count >= int(cfg.max_count):
                continue
            if high_level_step < cfg.after_step:
                continue
            if high_level_step >= int(cfg.before_step):
                continue
            last = self._last_fire_step.get(i)
            if last is not None and int(cfg.cooldown) > 0 and (high_level_step - int(last) < int(cfg.cooldown)):
                continue
            restricted = TOOL_RESTRICTED_MODES.get(cfg.mode)
            if restricted and tool_name not in restricted:
                continue
            cfg_tool = str(cfg.tool) if cfg.tool is not None else None
            if cfg_tool == 'any':
                cfg_tool = None
            if cfg.mode == 'wrong_target' and (tool_name not in TARGETED_TOOLS or 'target' not in dict(getattr(tool_call, 'arguments', {}) or {})):
                continue
            if cfg_tool and tool_name != cfg_tool:
                continue
            target_id = None
            if cfg.target_role or cfg.mode in {'grasp_slip', 'drop_beside', 'partial_open', 'partial_close', 'nav_blocked_partway', 'nav_wrong_target'}:
                target_id = _prebind_target(env, tool_call)
                if target_id is None:
                    continue
            target_role = cfg.target_role
            if target_role and str(target_id) != str(env.role_map.get(str(target_role), '')):
                continue
            if cfg.mode == 'drop_beside' and (not _drop_beside_inputs_valid(env, target_id)):
                continue
            if float(cfg.probability) < 1.0 and self._rng.random() > float(cfg.probability):
                continue
            self._active_config = cfg
            self._active_config_index = i
            self._active_target_id = target_id
            return cfg
        return None

    def _mark_fired(self, index: int, high_level_step: int) -> None:
        self._fire_counts[index] = int(self._fire_counts.get(index, 0)) + 1
        self._last_fire_step[index] = int(high_level_step)
        if self._configs[index].once:
            self._fired.add(index)

    def _blocked_transition(self, env: Any, tool_call: Any) -> dict:
        return _synthetic_failure_transition(env, tool_call, feedback='Tool call blocked by scheduler.', frames=0, env_feedback={'error_type': 'blocked_by_scheduler', 'error_message': 'Blocked.'}, scheduler_event=SchedulerEvent('failure_block', 'intercept', {'tool': tool_call.tool_name}))

def _failure_config_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {'mode': cfg.get('mode'), 'after_step': int(cfg.get('after_step', 0) or 0), 'tool': cfg.get('tool'), 'once': bool(cfg.get('once', True)), 'before_step': int(cfg.get('before_step', 10 ** 9) or 10 ** 9), 'probability': float(cfg.get('probability', 1.0) if cfg.get('probability', 1.0) is not None else 1.0), 'max_count': int(cfg.get('max_count', 10 ** 9) or 10 ** 9), 'cooldown': int(cfg.get('cooldown', 0) or 0), 'targets': [str(item) for item in cfg.get('targets') or [] if item], 'target_role': cfg.get('target_role'), 'feedback_message': cfg.get('feedback_message'), 'extra': dict(cfg.get('extra') or {}) if isinstance(cfg.get('extra'), Mapping) else {}, 'activation': dict(cfg.get('activation') or {}) if isinstance(cfg.get('activation'), Mapping) else None}

def _subtask_gate_matches(env: Any, activation: Mapping[str, Any] | None) -> bool:
    if not isinstance(activation, Mapping):
        return True
    expected = activation.get('subtask_id') or activation.get('owner_subtask_id')
    if expected is None:
        return True
    current = getattr(env, 'current_composite_subtask_id', None)
    return str(current) == str(expected)

def _annotate_failure_transition(transition: Mapping[str, Any], cfg: FailureConfig) -> None:
    owner = _activation_subtask_id(cfg.activation)
    if owner is None:
        owner = getattr(cfg, 'owner_subtask_id', None)
    if owner is None:
        return
    intercepts = transition.get('scheduler_intercept') if isinstance(transition, Mapping) else None
    if not isinstance(intercepts, list):
        return
    for event in intercepts:
        payload = getattr(event, 'payload', None)
        if isinstance(payload, dict):
            payload.setdefault('owner_subtask_id', str(owner))

def _activation_subtask_id(activation: Mapping[str, Any] | None) -> str | None:
    if not isinstance(activation, Mapping):
        return None
    value = activation.get('subtask_id') or activation.get('owner_subtask_id')
    return str(value) if value is not None else None

def _record_rewritten_failure(env: Any, underlying_success: bool) -> None:
    if underlying_success:
        try:
            env.invalid_tool_calls += 1
        except Exception:
            pass

def _refresh_transition(env: Any, transition: dict) -> dict:
    goal_result = None
    try:
        transition['agent_pose'] = env._agent_pose()
    except Exception:
        pass
    try:
        transition['held_objects'] = env._held_objects()
    except Exception:
        pass
    try:
        goal_result = env.check_goal(model_stopped=env.model_stopped)
        transition['goal_result'] = goal_result
    except Exception:
        pass
    try:
        transition['observation'] = env._observation()
    except Exception:
        pass
    try:
        if hasattr(env, '_done_status'):
            done, done_reason = env._done_status(goal_result or transition.get('goal_result') or {})
        else:
            done = bool(env.model_stopped) or env.high_level_step >= env.max_high_level_steps
            done_reason = 'model_stop' if env.model_stopped else 'max_high_level_steps' if done else None
        transition['done'] = done
        transition['done_reason'] = done_reason
    except Exception:
        pass
    try:
        if hasattr(env, '_episode_metrics'):
            transition['episode_metrics'] = env._episode_metrics()
    except Exception:
        pass
    return transition

def _synthetic_failure_transition(env: Any, tool_call: Any, *, feedback: str, frames: int, env_feedback: Mapping[str, Any], scheduler_event: SchedulerEvent, state_delta: Optional[Mapping[str, Any]]=None) -> dict:
    env.high_level_step += 1
    env.invalid_tool_calls += 1
    env.total_frames += int(frames)
    if hasattr(env, 'total_low_level_steps'):
        env.total_low_level_steps += int(frames)
    transition = {'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments or {}), 'reasoning': tool_call.reasoning, 'success': False, 'feedback': feedback, 'frames': int(frames), 'num_low_level_steps': int(frames), 'total_low_level_steps': getattr(env, 'total_low_level_steps', env.total_frames), 'env_feedback': dict(env_feedback), 'high_level_step': env.high_level_step, 'agent_pose': {}, 'held_objects': [], 'state_delta': dict(state_delta or {}), 'goal_result': {}, 'done': False, 'done_reason': None, 'scheduler_intercept': [scheduler_event]}
    _refresh_transition(env, transition)
    env.tool_history.append(transition)
    return transition

def _resolve_object_name(env: Any, ref: Any) -> str | None:
    if ref is None:
        return None
    text = str(ref)
    return env.role_map.get(text, text)

def _prebind_target(env: Any, tool_call: Any) -> str | None:
    prebind = getattr(env, 'prebind_synthetic_operation', None)
    if not callable(prebind):
        return _resolve_object_name(env, (getattr(tool_call, 'arguments', {}) or {}).get('target'))
    try:
        result = prebind(tool_call.tool_name, dict(getattr(tool_call, 'arguments', {}) or {}))
    except Exception:
        return None
    return str(result.selected_object) if getattr(result, 'success', False) else None

def _failure_transition_payload(transition: Mapping[str, Any]) -> Mapping[str, Any]:
    events = transition.get('scheduler_intercept') if isinstance(transition, Mapping) else None
    if not isinstance(events, list) or not events:
        return {}
    event = events[0]
    payload = event.payload if isinstance(event, SchedulerEvent) else event.get('payload', {})
    return payload if isinstance(payload, Mapping) else {}

def _failure_transition_applied(transition: Mapping[str, Any]) -> bool:
    payload = _failure_transition_payload(transition)
    return bool(payload.get('injection_applied', True))

def _failure_transition_terminal_unavailable(transition: Mapping[str, Any]) -> bool:
    return bool(_failure_transition_payload(transition).get('terminal_unavailable', False))

def _drop_beside_inputs_valid(env: Any, target: Any) -> bool:
    held = getattr(getattr(env.core, 'ego_state', None), 'held_object_name', None)
    return bool(held in env.core.objects and target in env.core.objects)

def _choose_drop_beside_center(env: Any, obj: Any, receptacle: Any | None, *, excluded_centers: set[tuple[float, float, float]] | None=None) -> tuple[tuple[float, float, float] | None, dict[str, Any]]:
    candidates: list[tuple[str, tuple[float, float, float]]] = []
    if receptacle is not None:
        candidates.extend(_receptacle_beside_candidates(env, obj, receptacle))
    objects = list(getattr(env.core, 'objects', {}).values())
    ignore_names = {str(getattr(obj, 'name', ''))}
    room_query = getattr(env, '_private_room_id_for_xy', None)
    receptacle_room_id = room_query(_center3(receptacle)[:2]) if receptacle is not None and callable(room_query) else None
    rejected: list[dict[str, Any]] = []
    rejected_center_keys: set[tuple[float, float, float]] = set()
    for source, center in _unique_center_candidates(candidates):
        record: dict[str, Any] = {'source': source, 'center': center}
        if _center_key(center) in set(excluded_centers or set()):
            record['reason'] = 'commit_rejected'
            rejected.append(record)
            continue
        candidate_room_id = room_query(center[:2]) if callable(room_query) else None
        record['candidate_room_id'] = candidate_room_id
        record['receptacle_room_id'] = receptacle_room_id
        if receptacle_room_id and candidate_room_id != receptacle_room_id:
            record['reason'] = 'different_room_from_receptacle'
            rejected.append(record)
            rejected_center_keys.add(_center_key(center))
            continue
        if receptacle is not None:
            relation = _relation_to_receptacle(_AABBProxy(obj, center), receptacle)
            record['relation_to_receptacle'] = relation
            if relation.get('inside') or relation.get('on_top'):
                record['reason'] = 'would_satisfy_place_relation'
                rejected.append(record)
                rejected_center_keys.add(_center_key(center))
                continue
        if _center_collides(obj, center, objects, ignore_names=ignore_names):
            record['reason'] = 'aabb_collision'
            rejected.append(record)
            rejected_center_keys.add(_center_key(center))
            continue
        reachable, reachability_reason = _drop_center_reachable(env, obj, center)
        record['reachable'] = reachable
        record['reachability_reason'] = reachability_reason
        if not reachable:
            record['reason'] = 'not_reachable_after_drop'
            rejected.append(record)
            rejected_center_keys.add(_center_key(center))
            continue
        return (center, {'source': source, 'candidate_count': len(candidates), 'rejected_candidates': rejected[:8], 'reachable': True, 'reachability_reason': reachability_reason, 'candidate_room_id': candidate_room_id, 'receptacle_room_id': receptacle_room_id, 'rejected_center_keys': rejected_center_keys})
    return (None, {'source': 'no_safe_candidate', 'candidate_count': len(candidates), 'rejected_candidates': rejected[:8], 'reachable': False, 'reachability_reason': 'no_collision_free_reachable_candidate', 'rejected_center_keys': rejected_center_keys})

def _receptacle_beside_candidates(env: Any, obj: Any, receptacle: Any) -> list[tuple[str, tuple[float, float, float]]]:
    pose = env.core.ego_state.pose
    obj_extent = _extent3(obj)
    rec_center = _center3(receptacle)
    rec_extent = _extent3(receptacle)
    z = _floor_drop_z(env, obj)
    margin = 0.25
    x_clear = rec_extent[0] * 0.5 + obj_extent[0] * 0.5 + margin
    y_clear = rec_extent[1] * 0.5 + obj_extent[1] * 0.5 + margin
    x_offsets = _lateral_offsets(rec_extent[0], obj_extent[0])
    y_offsets = _lateral_offsets(rec_extent[1], obj_extent[1])
    primary_candidates: list[tuple[str, tuple[float, float, float]]] = []
    secondary_candidates: list[tuple[str, tuple[float, float, float]]] = []
    dx = float(getattr(pose, 'x', rec_center[0])) - rec_center[0]
    dy = float(getattr(pose, 'y', rec_center[1])) - rec_center[1]
    distance = math.hypot(dx, dy)
    if distance > 1e-06:
        ux = dx / distance
        uy = dy / distance
        axis_clearances = []
        if abs(ux) > 1e-06:
            axis_clearances.append(x_clear / abs(ux))
        if abs(uy) > 1e-06:
            axis_clearances.append(y_clear / abs(uy))
        if axis_clearances:
            radial_clearance = min(axis_clearances)
            secondary_candidates.append(('receptacle_agent_ray', (rec_center[0] + ux * radial_clearance, rec_center[1] + uy * radial_clearance, z)))

    def add_x(sign: float, source: str) -> None:
        for offset in y_offsets:
            secondary_candidates.append((source, (rec_center[0] + sign * x_clear, rec_center[1] + offset, z)))

    def add_y(sign: float, source: str) -> None:
        for offset in x_offsets:
            secondary_candidates.append((source, (rec_center[0] + offset, rec_center[1] + sign * y_clear, z)))
    if abs(dx) >= abs(dy):
        facing_sign = 1.0 if dx >= 0.0 else -1.0
        primary_candidates.extend([('receptacle_agent_facing_side', (rec_center[0] + facing_sign * x_clear, rec_center[1], z)), ('receptacle_agent_opposite_side', (rec_center[0] - facing_sign * x_clear, rec_center[1], z)), ('receptacle_agent_left_side', (rec_center[0], rec_center[1] + y_clear, z)), ('receptacle_agent_right_side', (rec_center[0], rec_center[1] - y_clear, z))])
    else:
        facing_sign = 1.0 if dy >= 0.0 else -1.0
        primary_candidates.extend([('receptacle_agent_facing_side', (rec_center[0], rec_center[1] + facing_sign * y_clear, z)), ('receptacle_agent_opposite_side', (rec_center[0], rec_center[1] - facing_sign * y_clear, z)), ('receptacle_agent_left_side', (rec_center[0] + x_clear, rec_center[1], z)), ('receptacle_agent_right_side', (rec_center[0] - x_clear, rec_center[1], z))])
    for sign in (1.0, -1.0):
        add_x(sign, 'receptacle_x_side')
        add_y(sign, 'receptacle_y_side')
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            secondary_candidates.append(('receptacle_corner', (rec_center[0] + sx * x_clear, rec_center[1] + sy * y_clear, z)))
    return primary_candidates + secondary_candidates

def _unique_center_candidates(candidates: list[tuple[str, tuple[float, float, float]]]) -> list[tuple[str, tuple[float, float, float]]]:
    seen: set[tuple[float, float, float]] = set()
    unique: list[tuple[str, tuple[float, float, float]]] = []
    for source, center in candidates:
        key = (round(float(center[0]), 4), round(float(center[1]), 4), round(float(center[2]), 4))
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, (float(center[0]), float(center[1]), float(center[2]))))
    return unique

def _lateral_offsets(target_extent: float, obj_extent: float) -> tuple[float, ...]:
    half = max(float(target_extent) * 0.5 - float(obj_extent) * 0.5 - 0.05, 0.0)
    near = min(half, max(float(obj_extent) + 0.08, 0.2))
    far = min(half, max(near * 2.0, 0.4))
    values = [0.0]
    for value in (near, far):
        if value > 1e-06:
            values.extend([value, -value])
    deduped: list[float] = []
    for value in values:
        if all((abs(value - existing) > 1e-06 for existing in deduped)):
            deduped.append(float(value))
    return tuple(deduped)

def _floor_drop_z(env: Any, obj: Any) -> float:
    core_config = getattr(env.core, 'config', None)
    occupancy_config = getattr(core_config, 'occupancy', None)
    floor_z = float(getattr(occupancy_config, 'floor_height', 0.0))
    try:
        floor_tops = [float(other.center[2]) + float(other.extent[2]) * 0.5 for other in env.core.objects.values() if str(getattr(other, 'category', '')).lower() == 'floors']
        if floor_tops:
            floor_z = max([floor_z, *floor_tops])
    except Exception:
        pass
    return floor_z + _extent3(obj)[2] * 0.5 + 0.005

def _drop_center_reachable(env: Any, obj: Any, center: tuple[float, float, float]) -> tuple[bool, str]:
    original_center = getattr(obj, 'center', None)
    ego_state = getattr(env.core, 'ego_state', None)
    original_held = getattr(ego_state, 'held_object_name', None)
    original_visual = getattr(ego_state, 'held_visual_name', None)
    try:
        obj.center = center
        if ego_state is not None and original_held == getattr(obj, 'name', None):
            ego_state.held_object_name = None
            ego_state.held_visual_name = None
        occupancy = env.core._build_occupancy_snapshot()
        plan = plan_navigation(env.core.ego_state.pose.copy(), NavigationTarget.from_aabb_object(obj), occupancy, env.core.config.navigation)
        if plan.success:
            return (True, 'reachable')
        reasons = list(getattr(plan.check, 'reasons', []) or [])
        return (False, ','.join((str(reason) for reason in reasons)) or 'navigation_failed')
    except Exception as exc:
        return (False, 'reachability_error:{}'.format(type(exc).__name__))
    finally:
        if original_center is not None:
            obj.center = original_center
        if ego_state is not None:
            ego_state.held_object_name = original_held
            ego_state.held_visual_name = original_visual

def _commit_drop_beside(env: Any, obj: Any, center: tuple[float, float, float]) -> dict[str, Any]:
    ego_state = env.core.ego_state
    before_pose = {name: getattr(obj, name, None) for name in ('center', 'extent', 'root_position', 'root_orientation_xyzw', 'root_yaw', 'pose_source')}
    before_center = tuple((float(value) for value in obj.center))
    before_held = getattr(ego_state, 'held_object_name', None)
    before_visual = getattr(ego_state, 'held_visual_name', None)
    report: dict[str, Any] = {'drop_success': False, 'occupancy_rebuilt': False, 'renderer_synced': False, 'rendered_visible_at_interaction_pose': None, 'public_navigation_ready': False, 'rollback_success': False}
    renderer = getattr(env, 'renderer', None)
    obj.center = center
    try:
        if renderer is not None:
            setter = getattr(renderer, 'set_object_center', None)
            if not callable(setter):
                raise RuntimeError('renderer_set_object_center_unavailable')
            diagnostic = setter(str(getattr(obj, 'name', '')), center, verify_readback=True)
            if not isinstance(diagnostic, Mapping) or not bool(diagnostic.get('success')):
                raise RuntimeError('renderer_readback_failed:{!r}'.format(diagnostic))
            pose_mutation = dict(diagnostic)
            pose_mutation['core_object_name'] = str(getattr(obj, 'name', ''))
            env._update_core_object_pose_from_visual_mutation(pose_mutation)
            report['renderer_synced'] = True
        ego_state.held_object_name = None
        ego_state.held_visual_name = None
        env.core.build_occupancy('failure_drop_beside')
        report['occupancy_rebuilt'] = True
        if renderer is not None:
            selection = env._validate_default_pickup_route(str(getattr(obj, 'name', '')))
            report['navigation_selection'] = _navigation_selection_summary(selection)
            report['stable_visibility_passes'] = selection.get('stable_visibility_passes')
            report['stable_visibility_required'] = selection.get('stable_visibility_required')
            if not bool(selection.get('success')):
                raise RuntimeError('dropped_object_not_publicly_navigable')
            report['public_navigation_ready'] = True
            report['rendered_visible_at_interaction_pose'] = bool(selection.get('rendered_visible'))
            env._sync_renderer_for_observation_pose(env._pose_to_dict(env.core.ego_state.pose))
        else:
            report['navigation_validation_mode'] = 'symbolic_prefilter_without_renderer'
        if hasattr(env, 'invalidate_private_parent_placement'):
            env.invalidate_private_parent_placement(str(getattr(obj, 'name', '')))
        staged = getattr(env, '_staged_object_ledger', None)
        if isinstance(staged, dict):
            staged.pop(str(getattr(obj, 'name', '')), None)
        if hasattr(env.core, '_clear_held_pose_anchor'):
            env.core._clear_held_pose_anchor()
        report['drop_success'] = True
        return report
    except Exception as exc:
        report['renderer_sync_error'] = repr(exc)
        for name, value in before_pose.items():
            setattr(obj, name, value)
        ego_state.held_object_name = before_held
        ego_state.held_visual_name = before_visual
        try:
            if renderer is not None and callable(getattr(renderer, 'set_object_center', None)):
                renderer.set_object_center(str(getattr(obj, 'name', '')), before_center, verify_readback=True)
            env.core.build_occupancy('failure_drop_beside_rollback')
            report['rollback_success'] = True
        except Exception as rollback_exc:
            report['rollback_error'] = repr(rollback_exc)
        return report

def _navigation_selection_summary(selection: Any) -> dict[str, Any]:
    if not isinstance(selection, Mapping):
        return {'success': False, 'error_type': 'invalid_navigation_selection'}
    return {'success': bool(selection.get('success')), 'error_type': selection.get('error_type'), 'selected_goal_xy': selection.get('selected_goal_xy'), 'reachable_candidate_count': selection.get('reachable_candidate_count'), 'line_of_sight_clear_candidate_count': selection.get('line_of_sight_clear_candidate_count')}

def _center_key(center: Any) -> tuple[float, float, float]:
    return tuple((round(float(value), 4) for value in center[:3]))

def _sync_partial_open_close_renderer(env: Any, object_name: str | None, openness: float, *, symbolic_open: bool, reason: str) -> dict[str, Any]:
    report: dict[str, Any] = {'success': False, 'renderer_synced': False, 'object_id': object_name, 'openness': float(openness), 'symbolic_open': bool(symbolic_open)}
    renderer = getattr(env, 'renderer', None)
    if renderer is None:
        report['reason'] = 'renderer_unavailable'
        return report
    if object_name is None:
        report['reason'] = 'target_unresolved'
        return report
    sync = getattr(renderer, 'set_open_openness', None)
    if not callable(sync):
        report['reason'] = 'renderer_missing_set_open_openness'
        return report
    try:
        payload = dict(sync(str(object_name), float(openness), symbolic_open=bool(symbolic_open), reason=reason) or {})
        report.update(payload)
        report['renderer_synced'] = bool(payload.get('success', False))
    except Exception as exc:
        report['reason'] = 'renderer_sync_error'
        report['error'] = repr(exc)
    return report

def _center_collides(obj: Any, center: tuple[float, float, float], objects: list[Any], *, ignore_names: set[str]) -> bool:
    proxy = _AABBProxy(obj, center)
    for other in objects:
        other_name = str(getattr(other, 'name', ''))
        if other_name in ignore_names or _skip_collision_object(other):
            continue
        if _aabb_overlap(proxy, other, tolerance=0.01):
            return True
    return False

class _AABBProxy:

    def __init__(self, obj: Any, center: tuple[float, float, float]) -> None:
        self.name = getattr(obj, 'name', '')
        self.category = getattr(obj, 'category', '')
        self.center = center
        self.extent = _extent3(obj)

def _aabb_overlap(a: Any, b: Any, *, tolerance: float) -> bool:
    a_center = _center3(a)
    b_center = _center3(b)
    a_extent = _extent3(a)
    b_extent = _extent3(b)
    for axis in range(3):
        delta = abs(a_center[axis] - b_center[axis])
        limit = a_extent[axis] * 0.5 + b_extent[axis] * 0.5 - float(tolerance)
        if delta >= limit:
            return False
    return True

def _skip_collision_object(obj: Any) -> bool:
    return str(getattr(obj, 'category', '')).lower() in {'floors', 'ceilings', 'walls'}

def _relation_to_receptacle(obj: Any, receptacle: Any) -> dict[str, bool]:
    return {'inside': _is_inside_aabb(obj, receptacle), 'on_top': _is_ontop_aabb(obj, receptacle)}

def _is_inside_aabb(a: Any, b: Any) -> bool:
    a_center = _center3(a)
    b_center = _center3(b)
    b_extent = _extent3(b)
    return abs(a_center[0] - b_center[0]) <= b_extent[0] * 0.5 and abs(a_center[1] - b_center[1]) <= b_extent[1] * 0.5 and (b_center[2] - b_extent[2] * 0.5 <= a_center[2] <= b_center[2] + b_extent[2] * 0.5)

def _is_ontop_aabb(a: Any, b: Any) -> bool:
    a_center = _center3(a)
    b_center = _center3(b)
    a_extent = _extent3(a)
    b_extent = _extent3(b)
    if not (abs(a_center[0] - b_center[0]) <= b_extent[0] * 0.5 and abs(a_center[1] - b_center[1]) <= b_extent[1] * 0.5):
        return False
    a_bottom = a_center[2] - a_extent[2] * 0.5
    b_top = b_center[2] + b_extent[2] * 0.5
    return abs(a_bottom - b_top) <= 0.06

def _center3(obj: Any) -> tuple[float, float, float]:
    value = tuple(getattr(obj, 'center', (0.0, 0.0, 0.0)))
    return (float(value[0]), float(value[1]), float(value[2]))

def _extent3(obj: Any) -> tuple[float, float, float]:
    value = tuple(getattr(obj, 'extent', (0.2, 0.2, 0.2)))
    return (float(value[0]), float(value[1]), float(value[2]))
__all__ = ['FailureConfig', 'FailureInjectionScheduler', 'KNOWN_FAILURE_MODES']
