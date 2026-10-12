from __future__ import annotations
import copy
from typing import Any, List, Mapping
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainError, BrainInput, ToolCall

class PrepareTeacher(BaseBrain):

    def __init__(self) -> None:
        self._queue: List[ToolCall] = []
        self._env: Any = None

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        self._env = env
        family = task_spec.get('family_spec') or {}
        prepare = family.get('prepare')
        if not isinstance(prepare, Mapping):
            raise BrainError('PrepareTeacher requires family_spec.prepare.')
        normalized = self._normalize_prepare(prepare, task_spec)
        if not normalized.get('stages'):
            raise BrainError('PrepareTeacher requires family_spec.prepare.stages.')
        self._queue = self._multi_stage_plan(normalized)
        self._queue.extend(self._goal_cleanup_plan(task_spec))
        self._queue.append(ToolCall('stop', {}, reasoning='prepare plan complete'))

    def act(self, brain_input: BrainInput) -> ToolCall:
        if self._queue:
            return self._queue.pop(0)
        return ToolCall('stop', {}, reasoning='prepare plan complete')

    def _multi_stage_plan(self, prepare: Mapping[str, Any]) -> List[ToolCall]:
        target = prepare.get('target_role', 'target_object')
        calls: List[ToolCall] = []
        for stage in prepare['stages']:
            stype = stage.get('type')
            if stype == 'heat':
                calls += self._heat_stage(target, stage)
            elif stype == 'cool':
                calls += self._cool_stage(target, stage)
            elif stype == 'slice':
                calls += self._slice_stage(target, stage)
            elif stype == 'wash':
                calls += self._wash_stage(target, stage)
            elif stype in ('store', 'place'):
                calls += self._store_stage(target, stage)
            else:
                raise BrainError("Unknown prepare stage '{}'.".format(stype))
        return calls

    def _normalize_prepare(self, prepare: Mapping[str, Any], task_spec: Mapping[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(dict(prepare))
        if out.get('stages'):
            return out
        out['stages'] = self._stages_from_goal(out, task_spec)
        return out

    def _stages_from_goal(self, prepare: Mapping[str, Any], task_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
        stages: list[dict[str, Any]] = []
        target = prepare.get('target_role', 'target_object')
        tool_roles = list(prepare.get('tool_roles') or [])
        device_roles = list(prepare.get('device_roles') or [])
        final_receptacle = prepare.get('final_receptacle_role', 'target_receptacle')
        for predicate in (task_spec.get('goal') or {}).get('predicates') or []:
            ptype = predicate.get('type')
            args = predicate.get('args') or {}
            if args.get('object') != target:
                continue
            if ptype in {'object_hot', 'object_cooked'}:
                stages.append({'type': 'heat', 'heatsource': args.get('heatsource') or _first_role(device_roles, ('microwave', 'oven', 'stove'))})
            elif ptype == 'object_cool':
                stages.append({'type': 'cool', 'coldsource': args.get('coldsource') or _first_role(device_roles, ('fridge', 'freezer'))})
            elif ptype == 'object_sliced':
                stages.append({'type': 'slice', 'tool': args.get('tool') or _first_role(tool_roles, ('knife', 'cutting_tool', 'tool'))})
            elif ptype == 'object_clean':
                stages.append({'type': 'wash', 'method': 'wipe' if tool_roles else 'soak', 'tool': _first_role(tool_roles, ('rag', 'sponge', 'cloth', 'tool')), 'source': args.get('source') or _first_role(device_roles, ('faucet', 'sink', 'sink_basin'))})
            elif ptype in {'object_in_receptacle', 'object_on_receptacle'}:
                final_receptacle = args.get('receptacle') or final_receptacle
        if final_receptacle:
            stages.append({'type': 'store', 'receptacle': final_receptacle, 'open_receptacle': self._stage_needs_open(str(final_receptacle))})
        return stages

    def _heat_stage(self, target: str, stage: Mapping[str, Any]) -> List[ToolCall]:
        heatsource = stage.get('heatsource') or stage.get('receptacle_role') or 'heatsource'
        calls = [ToolCall('navigate_to', {'target': target}), ToolCall('pick_up', {'target': target})]
        if self._stage_needs_open(str(heatsource)):
            calls += [ToolCall('navigate_to', {'target': heatsource}), ToolCall('open', {'target': heatsource})]
        calls += [ToolCall('navigate_to', {'target': heatsource}), ToolCall('heat', {'target': heatsource})]
        return calls

    def _cool_stage(self, target: str, stage: Mapping[str, Any]) -> List[ToolCall]:
        coldsource = stage.get('coldsource') or stage.get('receptacle_role') or 'fridge'
        calls = [ToolCall('navigate_to', {'target': target}), ToolCall('pick_up', {'target': target})]
        if self._stage_needs_open(str(coldsource)):
            calls += [ToolCall('navigate_to', {'target': coldsource}), ToolCall('open', {'target': coldsource})]
        calls += [ToolCall('navigate_to', {'target': coldsource}), ToolCall('cool', {'target': coldsource})]
        return calls

    def _slice_stage(self, target: str, stage: Mapping[str, Any]) -> List[ToolCall]:
        tool = stage.get('tool') or stage.get('tool_role') or 'knife'
        return [ToolCall('navigate_to', {'target': tool}), ToolCall('pick_up', {'target': tool}), ToolCall('navigate_to', {'target': target}), ToolCall('cut', {'target': target}), ToolCall('drop', {})]

    def _wash_stage(self, target: str, stage: Mapping[str, Any]) -> List[ToolCall]:
        source = stage.get('source') or stage.get('faucet_role') or stage.get('receptacle_role') or 'sink'
        method = stage.get('method', 'soak')
        if method == 'wipe':
            tool = stage.get('tool') or stage.get('tool_role') or 'rag'
            return [ToolCall('navigate_to', {'target': tool}), ToolCall('pick_up', {'target': tool}), ToolCall('navigate_to', {'target': target}), ToolCall('wipe', {'target': target}), ToolCall('drop', {})]
        return [ToolCall('navigate_to', {'target': target}), ToolCall('pick_up', {'target': target}), ToolCall('navigate_to', {'target': source}), ToolCall('toggle_on', {'target': source}), ToolCall('soak_under', {'target': source}), ToolCall('toggle_off', {'target': source}), ToolCall('drop', {})]

    def _store_stage(self, target: str, stage: Mapping[str, Any]) -> List[ToolCall]:
        receptacle = stage.get('receptacle') or stage.get('receptacle_role') or 'target_receptacle'
        open_receptacle = stage.get('open_receptacle')
        if open_receptacle is None:
            open_receptacle = self._stage_needs_open(str(receptacle))
        calls = [ToolCall('navigate_to', {'target': target}), ToolCall('pick_up', {'target': target})]
        if open_receptacle:
            calls += [ToolCall('navigate_to', {'target': receptacle}), ToolCall('open', {'target': receptacle})]
        calls += [ToolCall('navigate_to', {'target': receptacle}), ToolCall('put_down', {'target': receptacle})]
        return calls

    def _goal_cleanup_plan(self, task_spec: Mapping[str, Any]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for predicate in (task_spec.get('goal') or {}).get('predicates') or []:
            ptype = predicate.get('type')
            args = predicate.get('args') or {}
            role = args.get('object')
            if not role:
                continue
            if ptype == 'object_closed':
                calls += [ToolCall('navigate_to', {'target': role}), ToolCall('close', {'target': role})]
            elif ptype == 'object_open':
                calls += [ToolCall('navigate_to', {'target': role}), ToolCall('open', {'target': role})]
            elif ptype == 'object_toggled_off':
                calls += [ToolCall('navigate_to', {'target': role}), ToolCall('toggle_off', {'target': role})]
            elif ptype == 'object_toggled_on':
                calls += [ToolCall('navigate_to', {'target': role}), ToolCall('toggle_on', {'target': role})]
        return calls

    def _stage_needs_open(self, role_name: str) -> bool:
        if self._env is None:
            return False
        obj_name = self._env.role_map.get(role_name, role_name)
        obj = self._env.core.objects.get(obj_name)
        if obj is None:
            return False
        abilities = tuple(getattr(obj, 'abilities', ()) or ())
        states = getattr(obj, 'states', {}) or {}
        return bool(('openable' in abilities or 'open' in states) and states.get('open') is not True)

def _first_role(role_names: list[Any], preferred_tokens: tuple[str, ...]) -> str | None:
    normalized = [str(role) for role in role_names if role]
    for token in preferred_tokens:
        for role in normalized:
            if token in role:
                return role
    return normalized[0] if normalized else None
__all__ = ['PrepareTeacher']
