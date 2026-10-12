from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from scoper.benchmark.procthor.closed_loop_tool_env import MODEL_VISIBLE_TOOLS, TOOL_TO_AI2THOR_ACTION
from scoper.benchmark.procthor.scheduler.base import Scheduler, SchedulerEvent
KNOWN_FAILURE_MODES = ('block', 'report_fail', 'delayed_success', 'partial_execution', 'wrong_target', 'grasp_slip', 'drop_beside', 'partial_open', 'partial_close', 'nav_blocked_partway', 'nav_wrong_target')
_MODE_TOOL_CONSTRAINTS = {'grasp_slip': ('pick_up',), 'drop_beside': ('put_down',), 'partial_open': ('open',), 'partial_close': ('close',), 'nav_blocked_partway': ('navigate_to',), 'nav_wrong_target': ('navigate_to',)}
_REALISTIC_FAILURE_MODES = frozenset(_MODE_TOOL_CONSTRAINTS)
ANY_TOOL = 'any'
INTERCEPTABLE_TOOLS = tuple((t for t in MODEL_VISIBLE_TOOLS if t != 'stop'))

@dataclass
class FailureConfig:
    tool: str = ANY_TOOL
    mode: str = 'block'
    after_step: int = 0
    before_step: int = 10 ** 9
    probability: float = 1.0
    max_count: int = 10 ** 9
    cooldown: int = 0
    targets: List[str] = field(default_factory=list)
    target_role: Optional[str] = None
    feedback_message: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in KNOWN_FAILURE_MODES:
            raise ValueError("Unknown failure mode '{}'. Known: {}.".format(self.mode, ', '.join(KNOWN_FAILURE_MODES)))
        if self.tool != ANY_TOOL and self.tool not in INTERCEPTABLE_TOOLS:
            raise ValueError("Failure config tool '{}' is not interceptable. Choose one of {} or '{}'.".format(self.tool, ', '.join(INTERCEPTABLE_TOOLS), ANY_TOOL))
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError('FailureConfig.probability must be in [0, 1].')
        if self.mode in {'wrong_target', 'nav_wrong_target'} and self.tool == ANY_TOOL:
            raise ValueError('{} failure requires a specific tool.'.format(self.mode))
        if self.mode in _MODE_TOOL_CONSTRAINTS:
            allowed = _MODE_TOOL_CONSTRAINTS[self.mode]
            if self.tool == ANY_TOOL or self.tool not in allowed:
                raise ValueError("Failure mode '{}' requires tool to be one of {}, got '{}'.".format(self.mode, ', '.join(allowed), self.tool))

class FailureScheduler(Scheduler):
    name = 'failure_injection'

    def __init__(self, configs: Sequence[Mapping[str, Any] | FailureConfig], seed: int=0) -> None:
        self._configs: List[FailureConfig] = [cfg if isinstance(cfg, FailureConfig) else FailureConfig(**dict(cfg)) for cfg in configs]
        self._rng = random.Random(seed)
        self._fire_counts: Dict[int, int] = {i: 0 for i in range(len(self._configs))}
        self._last_fire_step: Dict[int, int] = {}
        self._delayed_notices: List[dict] = []
        self._active_object_id: Optional[str] = None
        self._active_binding_public_record: Optional[dict] = None
        self._unavailable_configs: set[int] = set()

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> Optional[List[SchedulerEvent]]:
        self._fire_counts = {i: 0 for i in range(len(self._configs))}
        self._last_fire_step = {}
        self._delayed_notices = []
        self._active_object_id = None
        self._active_binding_public_record = None
        self._unavailable_configs = set()
        return [SchedulerEvent('failure_armed', 'episode_start', {'num_configs': len(self._configs)})]

    def pre_step(self, env: Any, high_level_step: int, last_transition: Optional[Mapping[str, Any]]) -> Optional[List[SchedulerEvent]]:
        if (getattr(env, '_human_qa_state', {}) or {}).get('pending'):
            return None
        if not self._delayed_notices:
            return None
        events = [SchedulerEvent('delayed_success_notice', 'pre_step', note) for note in self._delayed_notices]
        self._delayed_notices = []
        return events

    def intercept_tool_call(self, env: Any, tool_call: Any, high_level_step: int, default_call: Callable[[], dict]) -> Optional[dict]:
        if getattr(tool_call, 'tool_name', None) == 'answer':
            return None
        cfg, cfg_index = self._select_config(env, tool_call, high_level_step)
        if cfg is None:
            return None
        if cfg.probability < 1.0 and self._rng.random() > cfg.probability:
            return None
        if cfg.mode == 'block':
            transition = self._build_block(env, tool_call, cfg, high_level_step)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'report_fail':
            transition = self._build_report_fail(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'delayed_success':
            transition = self._build_delayed_success(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'partial_execution':
            transition = self._build_partial(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'wrong_target':
            transition = self._build_wrong_target(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'nav_blocked_partway':
            transition = self._build_nav_blocked_partway(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'nav_wrong_target':
            transition = self._build_nav_wrong_target(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'grasp_slip':
            transition = self._build_grasp_slip(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'drop_beside':
            transition = self._build_drop_beside(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'partial_open':
            transition = self._build_partial_open(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        if cfg.mode == 'partial_close':
            transition = self._build_partial_close(env, tool_call, cfg, high_level_step, default_call)
            return self._record_transition(cfg_index, high_level_step, transition)
        return None

    def _record_transition(self, config_index: int, high_level_step: int, transition: dict) -> dict:
        payload = _failure_transition_payload(transition)
        if payload.get('injection_applied', True):
            self._fire_counts[config_index] += 1
            self._last_fire_step[config_index] = high_level_step
        elif payload.get('terminal_unavailable', False):
            self._unavailable_configs.add(config_index)
        return transition

    def _select_config(self, env: Any, tool_call: Any, high_level_step: int) -> tuple[Optional[FailureConfig], int]:
        tool_name = tool_call.tool_name
        if tool_name == 'stop':
            return (None, -1)
        for i, cfg in enumerate(self._configs):
            if i in self._unavailable_configs:
                continue
            if cfg.tool != ANY_TOOL and cfg.tool != tool_name:
                continue
            if not _active_subtask_matches(env, cfg.extra.get('active_subtask_id')):
                continue
            if high_level_step < cfg.after_step:
                continue
            if high_level_step >= cfg.before_step:
                continue
            if self._fire_counts.get(i, 0) >= cfg.max_count:
                continue
            last = self._last_fire_step.get(i)
            if last is not None and high_level_step - last < cfg.cooldown:
                continue
            self._active_object_id = None
            self._active_binding_public_record = None
            if cfg.mode in _REALISTIC_FAILURE_MODES or cfg.target_role:
                resolved = self._prebind_target(env, tool_call)
                if resolved is None:
                    continue
                self._active_object_id = resolved.object_id
                if isinstance(getattr(resolved, 'binding_public_record', None), Mapping):
                    self._active_binding_public_record = dict(resolved.binding_public_record)
            if cfg.target_role and (not self._tool_targets_role(env, cfg.target_role)):
                continue
            return (cfg, i)
        return (None, -1)

    def _tool_targets_role(self, env: Any, role_name: str) -> bool:
        role_object_id = getattr(env, 'role_object_ids', {}).get(role_name)
        return role_object_id is not None and self._active_object_id == role_object_id

    def _prebind_target(self, env: Any, tool_call: Any) -> Any:
        prebind = getattr(env, 'prebind_synthetic_operation', None)
        if callable(prebind):
            try:
                resolved = prebind(tool_call.tool_name, dict(tool_call.arguments or {}))
            except Exception:
                return None
            return resolved if getattr(resolved, 'object_id', None) else None
        object_id = self._resolve_object_id(env, (tool_call.arguments or {}).get('target'))
        return SimpleNamespace(object_id=object_id, binding_public_record=None) if object_id else None

    def _build_block(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int) -> dict:
        feedback = cfg.feedback_message or "Tool '{}' was blocked by failure scheduler.".format(tool_call.tool_name)
        transition = self._synthetic_transition(env=env, tool_call=tool_call, high_level_step=high_level_step, success=False, feedback=feedback, raw_status='failed', raw_extra={'injected_failure': 'block'})
        transition['scheduler_intercept'] = [SchedulerEvent('failure_block', 'intercept', {'tool': tool_call.tool_name, 'step': high_level_step, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
        return transition

    def _build_report_fail(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        actual = default_call()
        original_success = bool(actual.get('success', False))
        actual['success'] = False
        actual['env_feedback'] = cfg.feedback_message or "Tool '{}' reported failure (actual underlying success={}).".format(tool_call.tool_name, original_success)
        actual['scheduler_intercept'] = [SchedulerEvent('failure_report_fail', 'intercept', {'tool': tool_call.tool_name, 'step': high_level_step, 'underlying_success': original_success, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
        return actual

    def _build_delayed_success(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        actual = default_call()
        original_success = bool(actual.get('success', False))
        actual['success'] = False
        actual['env_feedback'] = cfg.feedback_message or "Tool '{}' has not completed yet (delayed_success).".format(tool_call.tool_name)
        if original_success:
            self._delayed_notices.append({'tool': tool_call.tool_name, 'step': high_level_step, 'completed_at_step': high_level_step + 1})
        actual['scheduler_intercept'] = [SchedulerEvent('failure_delayed_success', 'intercept', {'tool': tool_call.tool_name, 'step': high_level_step, 'underlying_success': original_success, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
        return actual

    def _build_partial(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        tool = tool_call.tool_name
        details: Dict[str, Any]
        if tool in ('toggle_on', 'toggle_off'):
            return self._build_report_fail(env, tool_call, cfg, high_level_step, default_call)
        if tool == 'navigate_to':
            transition = self._partial_navigate(env, tool_call, default_call)
            details = transition.get('partial_details', {})
            transition.pop('partial_details', None)
        elif tool in ('open', 'close'):
            transition = self._partial_open_close(env, tool_call, default_call)
            details = transition.get('partial_details', {})
            transition.pop('partial_details', None)
        elif tool in ('pick_up', 'put_down'):
            transition = self._partial_pick_or_put(env, tool_call, default_call)
            details = transition.get('partial_details', {})
            transition.pop('partial_details', None)
        else:
            transition = self._build_report_fail(env, tool_call, cfg, high_level_step, default_call)
            details = {'fallback': 'report_fail'}
        transition['scheduler_intercept'] = [SchedulerEvent('failure_partial_execution', 'intercept', {'tool': tool, 'step': high_level_step, 'details': details, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
        return transition

    def _build_grasp_slip(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        transition = default_call()
        underlying_success = bool(transition.get('success', False))
        if not underlying_success:
            transition['env_feedback'] = "Tool 'pick_up' failed during grasp."
            transition['scheduler_intercept'] = [SchedulerEvent('failure_grasp_slip', 'intercept', {'tool': 'pick_up', 'step': high_level_step, 'underlying_success': False, 'injection_applied': False, 'terminal_unavailable': False, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
            return transition
        target_ref = (tool_call.arguments or {}).get('target')
        object_id = self._active_object_id or self._resolve_object_id(env, target_ref)
        drop_position = self._floor_position_near_agent(env)
        drop_success = False
        visual_drop = False
        try:
            ev = env.controller.step(action='DropHandObject', forceAction=True)
            drop_success = bool(ev.metadata.get('lastActionSuccess', False))
            if not drop_success and object_id and (drop_position is not None) and (getattr(env, '_visual_attached_object_id', None) == object_id):
                drop_success = self._drop_visual_attachment(env, object_id, drop_position)
                visual_drop = drop_success
            if drop_success and object_id and (drop_position is not None):
                if not visual_drop:
                    env.controller.step(action='TeleportObject', objectId=object_id, position=drop_position, rotation={'x': 0.0, 'y': 0.0, 'z': 0.0}, forceAction=True)
        except Exception:
            pass
        observation = env._observation() if hasattr(env, '_observation') else {}
        try:
            goal = env.check_goal(model_stopped=getattr(env, 'model_stopped', False))
        except Exception:
            goal = {}
        transition['success'] = False
        transition['env_feedback'] = "Tool 'pick_up' grasp slipped — the object slipped out and fell to the floor near you."
        transition['observation'] = observation
        transition['goal_result'] = goal
        transition['scheduler_intercept'] = [SchedulerEvent('failure_grasp_slip', 'intercept', {'tool': 'pick_up', 'step': high_level_step, 'underlying_success': True, 'object_id': object_id, 'drop_position': drop_position, 'drop_success': drop_success, 'injection_applied': drop_success, 'terminal_unavailable': not drop_success, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
        return transition

    @staticmethod
    def _drop_visual_attachment(env: Any, object_id: str, drop_position: Mapping[str, Any], *, rotation: Optional[Mapping[str, Any]]=None) -> bool:
        teleport = getattr(env, '_teleport_visual_object', None)
        clear = getattr(env, '_clear_visual_attachment_state', None)
        sticky = getattr(env, '_set_sticky_object_state', None)
        if not callable(teleport) or not callable(clear) or (not callable(sticky)):
            return False
        record = teleport(object_id=object_id, position=drop_position, rotation=dict(rotation) if rotation is not None else {'x': 0.0, 'y': 0.0, 'z': 0.0}, label='failure grasp_slip [visual_attach_release]', record_action='VisualFailureDrop', append_record=True)
        if record.get('success') is not True:
            return False
        sticky(object_id, parentReceptacles=[])
        clear(update_metadata=True)
        return True

    def _build_drop_beside(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        held = self._held_object_id(env)
        target_ref = (tool_call.arguments or {}).get('target')
        receptacle_id = self._active_object_id or self._resolve_object_id(env, target_ref)
        if not held:
            transition = self._synthetic_transition(env=env, tool_call=tool_call, high_level_step=high_level_step, success=False, feedback="Tool 'put_down' failed — agent is not holding any object.", raw_status='failed', raw_extra={'injected_failure': 'drop_beside', 'no_held_object': True})
            transition['scheduler_intercept'] = [SchedulerEvent('failure_drop_beside', 'intercept', {'tool': 'put_down', 'step': high_level_step, 'underlying_success': False, 'injection_applied': False, 'terminal_unavailable': False, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
            return transition
        from scoper.benchmark.procthor.scheduler.drop_placement import select_drop_position
        drop_position, placement_report = select_drop_position(env, held, receptacle_id)
        drop_rotation = placement_report.get('rotation') or {'x': 0.0, 'y': 0.0, 'z': 0.0}
        drop_success = False
        try:
            if drop_position is None:
                pass
            elif getattr(env, '_visual_attached_object_id', None) == held:
                drop_success = self._drop_visual_attachment(env, held, drop_position, rotation=drop_rotation)
            else:
                ev = env.controller.step(action='DropHandObject', forceAction=True)
                drop_success = bool(ev.metadata.get('lastActionSuccess', False))
                if drop_success and drop_position is not None:
                    ev = env.controller.step(action='TeleportObject', objectId=held, position=drop_position, rotation=drop_rotation, forceAction=True)
                    drop_success = bool(ev.metadata.get('lastActionSuccess', False))
        except Exception as exc:
            placement_report['execution_error'] = str(exc)
        observation = env._observation() if hasattr(env, '_observation') else {}
        try:
            goal = env.check_goal(model_stopped=getattr(env, 'model_stopped', False))
        except Exception:
            goal = {}
        transition = self._synthetic_transition(env=env, tool_call=tool_call, high_level_step=high_level_step, success=False, feedback="Tool 'put_down' missed — the object slipped off and landed on the floor next to the {}.".format(target_ref or 'receptacle') if drop_success else "Tool 'put_down' failed — no safe drop could be applied.", raw_status='failed', raw_extra={'injected_failure': 'drop_beside'})
        transition['observation'] = observation
        transition['goal_result'] = goal
        transition['scheduler_intercept'] = [SchedulerEvent('failure_drop_beside', 'intercept', {'tool': 'put_down', 'step': high_level_step, 'underlying_success': True, 'object_id': held, 'receptacle_id': receptacle_id, 'drop_position': drop_position, 'placement_report': placement_report, 'drop_success': drop_success, 'injection_applied': drop_success, 'terminal_unavailable': not drop_success, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
        return transition

    def _build_partial_open(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        return self._build_partial_open_close_named(env, tool_call, cfg, high_level_step, default_call, target_openness=0.05, name='failure_partial_open')

    def _build_partial_close(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        return self._build_partial_open_close_named(env, tool_call, cfg, high_level_step, default_call, target_openness=0.6, name='failure_partial_close')

    def _build_nav_blocked_partway(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        transition = self._partial_navigate(env, tool_call, default_call)
        details = transition.get('partial_details', {})
        transition.pop('partial_details', None)
        transition['scheduler_intercept'] = [SchedulerEvent('failure_nav_blocked_partway', 'intercept', {'tool': 'navigate_to', 'step': high_level_step, 'details': details, 'injection_applied': bool(details.get('underlying_success', False)), 'terminal_unavailable': False, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
        return transition

    def _build_nav_wrong_target(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        return self._build_wrong_target_named(env, tool_call, cfg, high_level_step, event_name='failure_nav_wrong_target')

    def _build_partial_open_close_named(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict], target_openness: float, name: str) -> dict:
        target_ref = (tool_call.arguments or {}).get('target')
        binding_public_record = self._active_binding_public_record
        object_id = self._active_object_id
        if not object_id:
            transition = self._synthetic_transition(env=env, tool_call=tool_call, high_level_step=high_level_step, success=False, feedback="Tool '{}' partially executed; target unresolved.".format(tool_call.tool_name), raw_status='failed', raw_extra={'injected_failure': name})
            transition['binding'] = binding_public_record
            transition['scheduler_intercept'] = [SchedulerEvent(name, 'intercept', {'tool': tool_call.tool_name, 'step': high_level_step, 'underlying_success': False, 'injection_applied': False, 'terminal_unavailable': False, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
            return transition
        applied = False
        try:
            event = env.controller.step(action='OpenObject', objectId=object_id, openness=target_openness, forceAction=True)
            applied = bool(event.metadata.get('lastActionSuccess', False))
        except Exception:
            pass
        observation = env._observation() if hasattr(env, '_observation') else {}
        try:
            goal = env.check_goal(model_stopped=getattr(env, 'model_stopped', False))
        except Exception:
            goal = {}
        transition = self._synthetic_transition(env=env, tool_call=tool_call, high_level_step=high_level_step, success=False, feedback="Tool '{}' jammed — the {} ended up only {}% open.".format(tool_call.tool_name, target_ref or 'object', int(target_openness * 100)), raw_status='failed', raw_extra={'injected_failure': name})
        transition['binding'] = binding_public_record
        transition['observation'] = observation
        transition['goal_result'] = goal
        transition['scheduler_intercept'] = [SchedulerEvent(name, 'intercept', {'tool': tool_call.tool_name, 'step': high_level_step, 'object_id': object_id, 'openness': target_openness, 'injection_applied': applied, 'terminal_unavailable': not applied, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
        return transition

    def _held_object_id(self, env: Any) -> Optional[str]:
        getter = getattr(env, '_held_inventory_object_id', None)
        if callable(getter):
            object_id = getter()
            if object_id:
                return object_id
        observe = getattr(env, '_observation', None)
        if callable(observe):
            try:
                for entry in (observe() or {}).get('held_objects', []) or []:
                    if isinstance(entry, Mapping) and entry.get('objectId'):
                        return str(entry['objectId'])
            except Exception:
                pass
        meta = env.controller.last_event.metadata
        inventory = meta.get('inventoryObjects') or []
        for entry in inventory:
            if isinstance(entry, Mapping) and entry.get('objectId'):
                return entry['objectId']
        return None

    def _floor_position_near_agent(self, env: Any) -> Optional[Dict[str, float]]:
        agent = env.controller.last_event.metadata.get('agent') or {}
        pos = agent.get('position') or {}
        rot = agent.get('rotation') or {}
        try:
            yaw_deg = float(rot.get('y', 0.0))
        except (TypeError, ValueError):
            yaw_deg = 0.0
        yaw_rad = math.radians(yaw_deg)
        offset = 0.4
        return {'x': float(pos.get('x', 0.0)) + offset * math.sin(yaw_rad), 'y': 0.05, 'z': float(pos.get('z', 0.0)) + offset * math.cos(yaw_rad)}

    def _floor_position_near_object(self, env: Any, object_id: Optional[str]) -> Optional[Dict[str, float]]:
        if not object_id:
            return None
        objs = env._objects_by_id() if hasattr(env, '_objects_by_id') else {}
        obj = objs.get(object_id)
        if not obj:
            return None
        pos = obj.get('position') or {}
        agent_pos = (env.controller.last_event.metadata.get('agent') or {}).get('position') or {}
        dx = float(agent_pos.get('x', 0.0)) - float(pos.get('x', 0.0))
        dz = float(agent_pos.get('z', 0.0)) - float(pos.get('z', 0.0))
        norm = math.hypot(dx, dz) or 1.0
        offset = 0.3
        return {'x': float(pos.get('x', 0.0)) + offset * dx / norm, 'y': 0.05, 'z': float(pos.get('z', 0.0)) + offset * dz / norm}

    def _partial_navigate(self, env: Any, tool_call: Any, default_call: Callable[[], dict]) -> dict:
        before_pose = self._agent_position(env)
        transition = default_call()
        if not bool(transition.get('success', False)):
            transition['env_feedback'] = "Tool 'navigate_to' partially executed and failed to reach the target."
            return transition
        after_pose = self._agent_position(env)
        fraction = self._rng.uniform(0.5, 0.8)
        target_x = before_pose['x'] + (after_pose['x'] - before_pose['x']) * fraction
        target_z = before_pose['z'] + (after_pose['z'] - before_pose['z']) * fraction
        try:
            self._teleport_to(env, x=target_x, y=after_pose.get('y', before_pose.get('y', 0.9)), z=target_z)
        except Exception:
            pass
        transition['success'] = False
        transition['env_feedback'] = "Tool 'navigate_to' executed partially; agent stopped before reaching the target."
        transition['partial_details'] = {'underlying_success': True, 'fraction_of_route': fraction, 'before_position': before_pose, 'after_position_target': {'x': target_x, 'y': after_pose.get('y'), 'z': target_z}}
        return transition

    def _partial_open_close(self, env: Any, tool_call: Any, default_call: Callable[[], dict]) -> dict:
        transition = default_call()
        target_ref = (tool_call.arguments or {}).get('target')
        object_id = self._resolve_object_id(env, target_ref)
        if not object_id:
            transition['success'] = False
            transition['env_feedback'] = "Tool '{}' partially executed; target unresolved.".format(tool_call.tool_name)
            transition['partial_details'] = {'resolved': False}
            return transition
        target_openness = 0.4 if tool_call.tool_name == 'open' else 0.6
        try:
            env.controller.step(action='OpenObject', objectId=object_id, openness=target_openness, forceAction=True)
        except Exception:
            pass
        transition['success'] = False
        transition['env_feedback'] = "Tool '{}' partially executed; target left at openness={}.".format(tool_call.tool_name, target_openness)
        transition['partial_details'] = {'object_id': object_id, 'openness': target_openness}
        return transition

    def _partial_pick_or_put(self, env: Any, tool_call: Any, default_call: Callable[[], dict]) -> dict:
        transition = default_call()
        underlying_success = bool(transition.get('success', False))
        transition['success'] = False
        if not underlying_success:
            transition['env_feedback'] = "Tool '{}' partially executed and failed.".format(tool_call.tool_name)
            transition['partial_details'] = {'underlying_success': False, 'note': 'underlying tool failed; partial mode unchanged it'}
            return transition
        transition['env_feedback'] = "Tool '{}' partially executed; object position may have shifted.".format(tool_call.tool_name)
        transition['partial_details'] = {'underlying_success': True, 'tool': tool_call.tool_name}
        return transition

    def _build_wrong_target(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, default_call: Callable[[], dict]) -> dict:
        return self._build_wrong_target_named(env, tool_call, cfg, high_level_step, event_name='failure_wrong_target')

    def _build_wrong_target_named(self, env: Any, tool_call: Any, cfg: FailureConfig, high_level_step: int, event_name: str) -> dict:
        original_args = dict(tool_call.arguments or {})
        original_target = original_args.get('target')
        new_target = self._pick_wrong_target(env, tool_call, cfg, original_target)
        if new_target is None:
            transition = self._build_block(env, tool_call, cfg, high_level_step)
            payload = transition['scheduler_intercept'][0].payload
            payload.update({'injection_applied': False, 'terminal_unavailable': True})
            return transition
        feedback_args = dict(original_args)
        feedback_args['target'] = new_target
        before_position = self._agent_position(env)
        internal_navigation = getattr(env, '_call_scheduler_navigation', None)
        if tool_call.tool_name == 'navigate_to' and callable(internal_navigation):
            result = internal_navigation(new_target, original_args, reasoning=tool_call.reasoning)
        else:
            result = env.call_tool(tool_call.tool_name, feedback_args, reasoning=tool_call.reasoning)
        after_position = self._agent_position(env)
        displacement = ((after_position['x'] - before_position['x']) ** 2 + (after_position['z'] - before_position['z']) ** 2) ** 0.5
        applied = bool(result.get('success', False))
        underlying_success = applied
        if tool_call.tool_name == 'navigate_to' and callable(internal_navigation):
            applied = applied and displacement > 0.05
            result['success'] = False
            result['env_feedback'] = 'Navigation did not reach the requested target. Check the scene and navigate again.'
        result['arguments'] = dict(original_args)
        tool_history = getattr(env, 'tool_history', None)
        if isinstance(tool_history, list) and tool_history:
            latest = tool_history[-1]
            if isinstance(latest, dict):
                latest['arguments'] = dict(original_args)
        result['scheduler_intercept'] = [SchedulerEvent(event_name, 'intercept', {'tool': tool_call.tool_name, 'step': high_level_step, 'original_target': original_target, 'swapped_target': new_target, 'underlying_success': underlying_success, 'injection_applied': applied, 'terminal_unavailable': not applied, 'before_position': before_position, 'after_position': after_position, 'displacement_m': displacement, 'config_mode': cfg.mode, 'config_tool': cfg.tool})]
        return result

    def _pick_wrong_target(self, env: Any, tool_call: Any, cfg: FailureConfig, original_target: Optional[str]) -> Optional[str]:
        if cfg.targets:
            original_id = self._resolve_object_id(env, original_target)
            candidates = [resolved for target in cfg.targets if (resolved := self._resolve_object_id(env, target)) and resolved != original_id]
            return self._rng.choice(candidates) if candidates else None
        original_object_id = self._active_object_id or self._resolve_object_id(env, original_target)
        objects = env._objects_by_id()
        original_obj = objects.get(original_object_id) if original_object_id else None
        target_type = original_obj.get('objectType') if original_obj else None
        if not target_type:
            return None
        same_type = [obj['objectId'] for obj in objects.values() if obj.get('objectType') == target_type and obj['objectId'] != original_object_id]
        if not same_type:
            return None
        return self._rng.choice(same_type)

    def _resolve_object_id(self, env: Any, target_ref: Optional[str]) -> Optional[str]:
        if not target_ref:
            return None
        roles = getattr(env, 'role_object_ids', {})
        if target_ref in roles:
            return roles[target_ref]
        return target_ref if target_ref in env._objects_by_id() else None

    def _agent_position(self, env: Any) -> Dict[str, float]:
        meta = env.controller.last_event.metadata.get('agent') or {}
        pos = meta.get('position') or {}
        return {'x': float(pos.get('x', 0.0)), 'y': float(pos.get('y', 0.9)), 'z': float(pos.get('z', 0.0))}

    def _teleport_to(self, env: Any, *, x: float, y: float, z: float) -> None:
        env.controller.step(action='TeleportFull', position={'x': x, 'y': y, 'z': z}, rotation=env.controller.last_event.metadata.get('agent', {}).get('rotation'), horizon=env.controller.last_event.metadata.get('agent', {}).get('cameraHorizon', 0.0), standing=env.controller.last_event.metadata.get('agent', {}).get('isStanding', True), forceAction=True)

    def _synthetic_transition(self, env: Any, tool_call: Any, high_level_step: int, success: bool, feedback: str, raw_status: str, raw_extra: Mapping[str, Any]) -> dict:
        observation = env._observation() if hasattr(env, '_observation') else {}
        try:
            goal = env.check_goal(model_stopped=getattr(env, 'model_stopped', False))
        except Exception:
            goal = {}
        return {'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments or {}), 'reasoning': tool_call.reasoning, 'high_level_step': high_level_step, 'success': success, 'env_feedback': feedback, 'done': False, 'done_reason': None, 'observation': observation, 'goal_result': goal, 'raw_execution': {'status': raw_status, **dict(raw_extra)}, 'low_level_actions': [], 'num_low_level_steps': 0, 'episode_metrics': {'high_level_steps': getattr(env, 'high_level_step', 0), 'total_low_level_steps': getattr(env, 'total_low_level_steps', 0), 'invalid_tool_calls': getattr(env, 'invalid_tool_calls', 0), 'model_stopped': getattr(env, 'model_stopped', False)}, 'ready_target': None, 'elapsed_seconds': 0.0}

def _failure_transition_payload(transition: Mapping[str, Any]) -> Mapping[str, Any]:
    events = transition.get('scheduler_intercept') if isinstance(transition, Mapping) else None
    if not isinstance(events, list) or not events:
        return {}
    event = events[0]
    payload = event.payload if isinstance(event, SchedulerEvent) else event.get('payload', {})
    return payload if isinstance(payload, Mapping) else {}

def _active_subtask_matches(env: Any, expected: Any) -> bool:
    if not expected:
        return True
    active = getattr(env, '_evaluation_active_subtask_spec', getattr(env, '_composite_active_task_spec', None))
    if not isinstance(active, Mapping):
        return False
    subtask_id = active.get('subtask_id')
    if not isinstance(subtask_id, str) or not subtask_id:
        task_id = active.get('id')
        if isinstance(task_id, str) and '::' in task_id:
            subtask_id = task_id.rsplit('::', 1)[1]
    return subtask_id == str(expected)
__all__ = ['ANY_TOOL', 'FailureConfig', 'FailureScheduler', 'INTERCEPTABLE_TOOLS', 'KNOWN_FAILURE_MODES']
