from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.execution_core.trace import ActionResult, CheckResult

@dataclass
class ExecutionResult:
    success: bool
    frames: int
    seconds: float
    error_type: Optional[str]
    error_message: Optional[str]
    action_result: ActionResult

def _failed_action(action: str, target: Optional[str], reason: str) -> ActionResult:
    return ActionResult(action=action, args={'target': target}, success=False, precondition=CheckResult.fail(reason, target=target), postcondition=None, trace_id='heat_wrapper', frames=0)

def _heat(core, target, **kwargs) -> ActionResult:
    held = core.ego_state.held_object_name
    if held is None:
        return _failed_action('HEAT', target, 'not_holding_object')
    if target not in core.objects:
        return _failed_action('HEAT', target, 'target_not_found')
    if not actions.supports_heatsource(core.objects[target]):
        return _failed_action('HEAT', target, 'target_not_heatsource')
    place_res = core.place(target)
    if not place_res.success:
        return place_res
    toggle_res = core.toggle_on(target)
    if not toggle_res.success:
        return toggle_res
    food = core.objects[held]
    mutation = actions.set_cooked_state(food, True)
    return ActionResult(action='HEAT', args={'target': target, 'food': held}, success=True, precondition=place_res.precondition, postcondition=toggle_res.postcondition, trace_id='heat_wrapper', frames=int(place_res.frames) + int(toggle_res.frames), start_pose=place_res.start_pose, end_pose=toggle_res.end_pose, state_delta={'place': place_res.state_delta, 'toggle': toggle_res.state_delta, 'object_state': {'target': held, 'state': 'cooked', 'before': mutation.before, 'after': mutation.after, 'changed': mutation.changed}})

def _cool(core, target, **kwargs) -> ActionResult:
    held = core.ego_state.held_object_name
    if held is None:
        return _failed_action('COOL', target, 'not_holding_object')
    if target not in core.objects:
        return _failed_action('COOL', target, 'target_not_found')
    if not actions.supports_coldsource(core.objects[target]):
        return _failed_action('COOL', target, 'target_not_coldsource')
    place_res = core.place(target)
    placed_despite_postcondition = core.ego_state.held_object_name is None and held in core.objects
    if not place_res.success and (not placed_despite_postcondition):
        return place_res
    food = core.objects[held]
    mutation = actions.set_cooled_state(food, True)
    return ActionResult(action='COOL', args={'target': target, 'food': held}, success=True, precondition=place_res.precondition, postcondition=place_res.postcondition if place_res.success else CheckResult.pass_(target=target, food=held, recovered_from='place_postcondition_failed'), trace_id='cool_wrapper', frames=int(place_res.frames), start_pose=place_res.start_pose, end_pose=place_res.end_pose, state_delta={'place': place_res.state_delta, 'object_state': {'target': held, 'state': 'cooled', 'before': mutation.before, 'after': mutation.after, 'changed': mutation.changed}})

def _soak_under(core, target, **kwargs) -> ActionResult:
    held = core.ego_state.held_object_name
    if held is None:
        return _soak_staged_dirty_object(core, target, staged_target=kwargs.get('staged_target'), trace_id=str(kwargs.get('trace_id') or 'soak_staged_wrapper'))
    result = core.soak(target)
    if not result.success:
        return result
    obj = core.objects.get(held)
    if obj is None:
        return result
    before_dirty = actions.get_dirty_state(obj)
    if before_dirty is None:
        return result
    mutation = actions.set_dirty_state(obj, False)
    state_delta = dict(result.state_delta or {})
    state_delta['cleaning'] = {'target': held, 'state': 'dirty', 'before': mutation.before, 'after': mutation.after, 'changed': mutation.changed, 'source': target}
    result.state_delta = state_delta
    return result

def _soak_staged_dirty_object(core, source_name: str, *, staged_target: Any, trace_id: str) -> ActionResult:
    source_obj = core.objects.get(source_name)
    if source_obj is None:
        return _failed_action('SOAK', source_name, 'target_not_found')
    candidate_name = str(staged_target or '')
    candidate = core.objects.get(candidate_name)
    if candidate is None:
        return _failed_action('SOAK', source_name, 'staged_target_unresolved')
    if actions.get_dirty_state(candidate) is not True:
        return _failed_action('SOAK', source_name, 'staged_target_not_dirty')
    if not bool(getattr(candidate, 'visible', True)):
        return _failed_action('SOAK', source_name, 'staged_target_not_visible')
    if _xy_distance(getattr(candidate, 'center', (0.0, 0.0, 0.0)), getattr(source_obj, 'center', (0.0, 0.0, 0.0))) > 1.6:
        return _failed_action('SOAK', source_name, 'staged_target_out_of_range')
    if getattr(core, 'occupancy', None) is None:
        core.build_occupancy(trace_id=f'{trace_id}_occupancy')
    source_ready, readiness_failure = _staged_soak_source_ready(core, source_name)
    if readiness_failure is not None:
        return ActionResult(action='SOAK', args={'source': source_name, 'soak_relation': 'under'}, success=False, precondition=readiness_failure, postcondition=None, trace_id=trace_id, frames=0)
    assert source_ready is not None
    start_pose = core.ego_state.pose.copy()
    substance = str(source_ready.get('substance') or actions.get_source_substance(source_obj) or 'water')
    before_saturated = actions.get_substances(candidate, 'saturated')
    before_dirty = actions.get_dirty_state(candidate)
    core.trace.emit(trace_id, 'action_requested', action='SOAK', source=source_name, soak_relation='under', mode='staged_sink_wash')
    saturation = actions.set_saturated_substance(candidate, substance, True)
    cleaning = actions.set_dirty_state(candidate, False)
    after_saturated = actions.get_substances(candidate, 'saturated')
    after_dirty = actions.get_dirty_state(candidate)
    post = CheckResult.pass_(source=source_name, soak_relation='under', substance=substance, mode='staged_sink_wash', saturated_substances=after_saturated, dirty_before=before_dirty, dirty_after=after_dirty, held_object_name=core.ego_state.held_object_name)
    result = ActionResult(action='SOAK', args={'source': source_name, 'soak_relation': 'under', 'substance': substance}, success=True, precondition=CheckResult.pass_(source=source_name, source_category=getattr(source_obj, 'category', None), substance=substance, mode='staged_sink_wash'), postcondition=post, trace_id=trace_id, frames=0, start_pose=start_pose, end_pose=core.ego_state.pose.copy(), state_delta={'object_state': {'target': candidate.name, 'state': 'saturated', 'substance': substance, 'before': before_saturated, 'after': after_saturated, 'changed': saturation.changed}, 'cleaning': {'target': candidate.name, 'state': 'dirty', 'before': before_dirty, 'after': after_dirty, 'changed': cleaning.changed, 'source': source_name, 'mode': 'staged_sink_wash'}, 'held_object': {'held_object_name': core.ego_state.held_object_name, 'held_visual_name': core.ego_state.held_visual_name}})
    core.action_results.append(result)
    core.trace.emit(trace_id, 'action_succeeded', mode='staged_sink_wash')
    return result

def _staged_soak_source_ready(core, source_name: str) -> tuple[dict[str, Any] | None, CheckResult | None]:
    checker = core.make_precondition_checker()
    try:
        interaction = checker.check_interaction(source_name, core._interaction_precondition_config())
    except Exception as exc:
        return (None, CheckResult.fail('staged_source_readiness_check_failed', exception_type=type(exc).__name__))
    affordance = checker.check_fluid_source_affordance(source_name)
    active = checker.check_fluid_source_active(source_name)
    for check in (interaction, affordance, active):
        if not check.allowed:
            return (None, check)
    return ({'substance': affordance.metrics.get('substance'), 'interaction': interaction.metrics, 'affordance': affordance.metrics, 'active': active.metrics}, None)

def _xy_distance(a: Any, b: Any) -> float:
    try:
        return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5
    except Exception:
        return float('inf')

def _drop(core, target=None, **kwargs) -> ActionResult:
    if core.ego_state.held_object_name is None:
        return _failed_action('DROP', None, 'not_holding_object')
    result = core.drop()
    hand_cleared = core.ego_state.held_object_name is None
    if hand_cleared:
        result.success = True
    return result
ACTION_ROUTING: dict[str, Callable[..., ActionResult]] = {'navigate_to': lambda core, t, **kw: core.navigate_to(t, **kw), 'pick_up': lambda core, t, **kw: core.pick_object(t), 'put_down': lambda core, t, **kw: core.place(t), 'drop': _drop, 'open': lambda core, t, **kw: core.open_object(t), 'close': lambda core, t, **kw: core.close_object(t), 'toggle_on': lambda core, t, **kw: core.toggle_on(t), 'toggle_off': lambda core, t, **kw: core.toggle_off(t), 'wipe': lambda core, t, **kw: core.wipe(t), 'cut': lambda core, t, **kw: core.cut(t), 'look_for': lambda core, t, **kw: core.look_for(t), 'soak_under': _soak_under, 'fill': lambda core, t, **kw: core.fill(t, kw['source']), 'pour': lambda core, t, **kw: core.pour(t), 'heat': _heat, 'cool': _cool}

class BehaviorExecutorError(ValueError):
    pass

class BehaviorExecutor:

    def __init__(self, core, stream_callback: Optional[Callable[[Any, int], None]]=None):
        self.core = core
        self.stream_callback = stream_callback

    def execute_action(self, tool_name: str, target: Optional[str]=None, **kwargs) -> ExecutionResult:
        method = ACTION_ROUTING.get(tool_name)
        if method is None:
            raise BehaviorExecutorError("Unknown action tool '{}'.".format(tool_name))
        started = time.time()
        action_result = method(self.core, target, **kwargs)
        elapsed = time.time() - started
        if self.stream_callback is not None:
            for frame_index in range(int(action_result.frames)):
                self.stream_callback(self.core.ego_state.pose, frame_index)
        error_type, error_message = (None, None)
        if not action_result.success:
            reasons = list(action_result.precondition.reasons or [])
            if not reasons and action_result.postcondition is not None:
                reasons = list(action_result.postcondition.reasons or [])
            error_type = reasons[0] if reasons else 'action_failed'
            error_message = '; '.join(reasons) if reasons else 'Action did not succeed.'
        return ExecutionResult(success=bool(action_result.success), frames=int(action_result.frames), seconds=float(elapsed), error_type=error_type, error_message=error_message, action_result=action_result)
__all__ = ['BehaviorExecutor', 'BehaviorExecutorError', 'ExecutionResult', 'ACTION_ROUTING']
