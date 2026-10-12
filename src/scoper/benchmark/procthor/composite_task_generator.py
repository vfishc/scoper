from __future__ import annotations
import copy
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from scoper.benchmark.common import attach_compiled_task_binding_spec, merge_qa_overlays, rewrite_qa_overlay_roles
from scoper.benchmark.common.oracle_bank_supervisor import redundant_composite_child_indices
from scoper.benchmark.procthor.online_task_spec import ALLOWED_CATEGORIES, FAMILY_CATEGORIES, TaskSpecValidationError, validate_online_task_spec
_ROLE_LIST_KEYS = frozenset({'target_objects'})
_INIT_STATE_FIELDS = {'set_open_state': ('open', 'is_open'), 'set_toggle_state': ('toggled', 'is_toggled'), 'set_dirty_state': ('dirty', 'is_dirty')}
_GOAL_STATE_VALUES = {'object_open': ('open', True), 'object_closed': ('open', False), 'object_toggled_on': ('toggled', True), 'object_toggled_off': ('toggled', False), 'object_clean': ('dirty', False), 'object_dirty': ('dirty', True), 'object_sliced': ('sliced', True), 'object_cool': ('cooled', True), 'object_cooked': ('cooked', True), 'object_hot': ('heated', True), 'object_filled': ('filled', True), 'object_soaked': ('saturated', True)}
_PLACEMENT_GOALS = frozenset({'object_in_receptacle', 'object_on_receptacle'})

@dataclass(frozen=True)
class CompositeBuildResult:
    task: dict[str, Any] | None
    rejected: list[dict[str, Any]]
    summary: dict[str, Any]

def build_composite_task_spec(task_specs: Sequence[Mapping[str, Any]], *, composite_id: str | None=None, instruction: str | None=None, shared_roles: Mapping[str, str] | None=None, auto_share_scene_roles: bool=True, require_independent_goals: bool=False, max_subtasks: int | None=None, qa_overlay: Mapping[str, Any] | None=None) -> CompositeBuildResult:
    tasks = [copy.deepcopy(dict(task)) for task in task_specs]
    if max_subtasks is not None:
        tasks = tasks[:max(0, int(max_subtasks))]
    rejected: list[dict[str, Any]] = []
    if len(tasks) < 2:
        return _reject(tasks, rejected, 'NOT_ENOUGH_SUBTASKS', 'At least two child tasks are required.')
    source_key = _task_group_key(tasks[0])
    agent_pose = _canonical(tasks[0].get('init', {}).get('agent_pose'))
    for index, task in enumerate(tasks):
        category = str(task.get('category') or '')
        if category == 'composite':
            rejected.append(_row(index, task, 'RECURSIVE_COMPOSITE', 'Recursive composite tasks are not supported.'))
            continue
        if category not in ALLOWED_CATEGORIES:
            rejected.append(_row(index, task, 'UNSUPPORTED_CHILD_CATEGORY', "Unknown child category '{}'.".format(category)))
            continue
        try:
            validate_online_task_spec(task)
        except Exception as exc:
            rejected.append(_row(index, task, 'TASK_SPEC_VALIDATION_FAILED', str(exc)))
            continue
        if _task_group_key(task) != source_key:
            rejected.append(_row(index, task, 'SCENE_LAYOUT_MISMATCH', 'All children must use the same house and layout source.'))
        if _canonical(task.get('init', {}).get('agent_pose')) != agent_pose:
            rejected.append(_row(index, task, 'AGENT_POSE_MISMATCH', 'All children must use the same agent pose.'))
    if rejected:
        return CompositeBuildResult(None, rejected, _summary(None, tasks, rejected))
    expanded_shared, sharing_errors = _expanded_shared_roles(tasks, shared_roles or {}, auto_share_scene_roles=auto_share_scene_roles)
    rejected.extend(sharing_errors)
    role_maps = _role_maps(tasks, expanded_shared)
    rejected.extend(_conflict_rows(tasks, role_maps))
    if rejected:
        return CompositeBuildResult(None, rejected, _summary(None, tasks, rejected))
    rewritten_child_goals = [[_rewrite_predicate(predicate, role_maps[index], owner_subtask_id='subtask_{}'.format(index)) for predicate in task.get('goal', {}).get('predicates') or []] for index, task in enumerate(tasks)]
    redundant_children = redundant_composite_child_indices(rewritten_child_goals) if require_independent_goals else []
    if redundant_children:
        rejected.extend((_row(index, tasks[index], 'REDUNDANT_CHILD_GOAL', 'Child adds no independent physical goal.') for index in redundant_children))
        return CompositeBuildResult(None, rejected, _summary(None, tasks, rejected))
    roles = _merged_roles(tasks, role_maps)
    init = _merged_init(tasks, role_maps)
    parent_predicates: list[dict[str, Any]] = []
    subtasks: list[dict[str, Any]] = []
    disturbances: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    child_qa_overlays: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        subtask_id = 'subtask_{}'.format(index)
        mapping = role_maps[index]
        child_predicates = rewritten_child_goals[index]
        parent_predicates.extend(copy.deepcopy(child_predicates))
        subtask: dict[str, Any] = {'subtask_id': subtask_id, 'source_task_id': str(task.get('id') or ''), 'category': str(task.get('category') or ''), 'instruction_fragment': str(task.get('instruction') or ''), 'goal_predicates': child_predicates}
        family_spec = task.get('family_spec')
        if isinstance(family_spec, Mapping):
            subtask['family_spec'] = _rewrite_family_spec(family_spec, mapping, subtask_id=subtask_id)
        child_overlay = task.get('qa_overlay')
        if isinstance(child_overlay, Mapping) and child_overlay.get('questions'):
            rewritten_overlay = rewrite_qa_overlay_roles(child_overlay, mapping, subtask_id=subtask_id)
            rewritten_overlay['questions'] = rewritten_overlay['questions'][:1]
            child_qa_overlays.append(rewritten_overlay)
        child_disturbance = _rewrite_disturbance_spec(task.get('disturbance_spec'), mapping, subtask_id=subtask_id)
        if child_disturbance:
            subtask['disturbance_spec'] = copy.deepcopy(child_disturbance)
            disturbances.extend(copy.deepcopy(child_disturbance.get('disturbances') or []))
            failures.extend(copy.deepcopy(child_disturbance.get('failures') or []))
        subtasks.append(subtask)
    parent_id = composite_id or _default_composite_id(tasks)
    parent = {'id': parent_id, 'version': '0.1', 'backend': 'procthor', 'category': 'composite', 'source': copy.deepcopy(dict(tasks[0]['source'])), 'template': {'template_id': 'composite_sequential', 'template_version': '0.1'}, 'instruction': instruction or _combined_instruction(tasks), 'roles': roles, 'init': init, 'goal': {'type': 'predicate', 'predicates': parent_predicates, 'require_model_stop': any((bool(task.get('goal', {}).get('require_model_stop')) for task in tasks))}, 'online_requirements': {'expected_events': _stable_union(tasks, 'expected_events', extra='sequential_composite'), 'expected_brain_behaviors': _stable_union(tasks, 'expected_brain_behaviors', extra='handoff_between_subtasks')}, 'validation': {'status': 'candidate', 'checks': {}, 'failure_reason': None}, 'tags': sorted({'composite', *(str(task.get('category') or '') for task in tasks)}), 'family_spec': {'family': 'composite', 'scope': {'type': 'whole_house', 'label': 'house', 'room_ids': []}, 'composite': {'ordering': 'sequential', 'subtasks': subtasks}}, 'metadata': {'composite_generation': {'source_task_ids': [str(task.get('id') or '') for task in tasks], 'role_namespace': role_maps, 'explicit_shared_roles': dict(shared_roles or {}), 'auto_shared_roles': {key: value for key, value in expanded_shared.items() if key not in dict(shared_roles or {})}}}}
    if disturbances or failures:
        parent['disturbance_spec'] = {'seed': 0}
        if disturbances:
            parent['disturbance_spec']['disturbances'] = disturbances
        if failures:
            parent['disturbance_spec']['failures'] = failures
    merged_overlay = merge_qa_overlays(child_qa_overlays, parent_overlay=qa_overlay, max_questions=3)
    if merged_overlay is not None:
        parent['qa_overlay'] = merged_overlay
    for key in ('binding_spec', 'task_binding_spec', 'task_binding_spec_sha256'):
        parent.pop(key, None)
    try:
        validate_online_task_spec(parent)
        parent = attach_compiled_task_binding_spec(parent)
        validate_online_task_spec(parent)
        from scoper.benchmark.common.task_identity_policy import enabled, assert_distinct
        if enabled():
            assert_distinct(parent)
    except Exception as exc:
        rejected.append(_row(None, parent, 'COMPOSITE_TASK_SPEC_VALIDATION_FAILED', str(exc)))
        return CompositeBuildResult(None, rejected, _summary(None, tasks, rejected))
    return CompositeBuildResult(parent, [], _summary(parent, tasks, []))

def build_composite_task_bank(task_specs: Sequence[Mapping[str, Any]], *, max_subtasks: int=3, max_composites: int=20, seed: int=0, auto_share_scene_roles: bool=True) -> dict[str, Any]:
    if int(max_subtasks) < 2:
        raise ValueError('max_subtasks must be at least 2.')
    if int(max_composites) < 0:
        raise ValueError('max_composites must be non-negative.')
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for task in task_specs:
        groups[_task_group_key(task)].append(task)
    rng = random.Random(int(seed))
    generated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        candidates = sorted(groups[group_key], key=lambda task: str(task.get('id') or ''))
        rng.shuffle(candidates)
        while len(candidates) >= 2 and len(generated) < int(max_composites):
            selected = _select_diverse(candidates, max_subtasks=int(max_subtasks))
            result = build_composite_task_spec(selected, composite_id=_default_composite_id(selected, index=len(generated)), auto_share_scene_roles=auto_share_scene_roles)
            if result.task is not None:
                generated.append(result.task)
                selected_ids = {id(task) for task in selected}
                candidates = [task for task in candidates if id(task) not in selected_ids]
            else:
                rejected.extend(result.rejected)
                candidates = candidates[1:]
        if len(generated) >= int(max_composites):
            break
    return {'tasks': generated, 'rejected': rejected, 'summary': {'generated_total': len(generated), 'rejected_total': len(rejected)}}

def _task_group_key(task: Mapping[str, Any]) -> tuple[str, ...]:
    source = task.get('source') or {}
    return tuple((str(source.get(key) or '') for key in ('house_id', 'house_path', 'split', 'source_json', 'layout_variant_id')))

def _expanded_shared_roles(tasks: Sequence[Mapping[str, Any]], shared_roles: Mapping[str, str], *, auto_share_scene_roles: bool) -> tuple[dict[str, str], list[dict[str, Any]]]:
    expanded = {str(key): str(value) for key, value in shared_roles.items()}
    errors: list[dict[str, Any]] = []
    by_object_id: dict[str, list[tuple[int, str, Mapping[str, Any]]]] = defaultdict(list)
    for index, task in enumerate(tasks):
        for role_name, role in (task.get('roles') or {}).items():
            if isinstance(role, Mapping) and role.get('object_id'):
                by_object_id[str(role['object_id'])].append((index, str(role_name), role))
    if auto_share_scene_roles:
        for object_id, entries in sorted(by_object_id.items()):
            if len(entries) < 2:
                continue
            object_types = {str(role.get('object_type') or '') for _, _, role in entries}
            if len(object_types) > 1:
                index, _, _ = entries[1]
                errors.append(_row(index, tasks[index], 'SHARED_ROLE_TYPE_CONFLICT', "Object '{}' has conflicting object types.".format(object_id)))
                continue
            canonical = 'task{}.{}'.format(entries[0][0], entries[0][1])
            for index, role_name, _ in entries[1:]:
                expanded.setdefault('task{}.{}'.format(index, role_name), canonical)
    return (expanded, errors)

def _role_maps(tasks: Sequence[Mapping[str, Any]], shared_roles: Mapping[str, str]) -> list[dict[str, str]]:

    def resolve(ref: str) -> str:
        seen: set[str] = set()
        while ref in shared_roles and ref not in seen:
            seen.add(ref)
            ref = str(shared_roles[ref])
        return ref
    maps: list[dict[str, str]] = []
    for index, task in enumerate(tasks):
        mapping: dict[str, str] = {}
        for role_name in task.get('roles') or {}:
            fq = 'task{}.{}'.format(index, role_name)
            target = resolve(fq)
            if '.' in target and target.partition('.')[0].startswith('task'):
                target_index, _, target_role = target.partition('.')
                mapped = '{}__{}'.format(target_index, target_role)
            elif target == fq:
                mapped = 'task{}__{}'.format(index, role_name)
            else:
                mapped = target.replace('.', '__')
            mapping[str(role_name)] = mapped
        maps.append(mapping)
    return maps

def _conflict_rows(tasks: Sequence[Mapping[str, Any]], role_maps: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role_values: dict[str, tuple[str, str]] = {}
    init_states: dict[tuple[str, str], Any] = {}
    relocations: dict[str, str] = {}
    goal_states: dict[tuple[str, str], Any] = {}
    placements: dict[str, tuple[str, str]] = {}
    all_object_ids = {str(role.get('object_id')) for task in tasks for role in (task.get('roles') or {}).values() if isinstance(role, Mapping) and role.get('object_id')}
    for index, task in enumerate(tasks):
        mapping = role_maps[index]
        for role_name, role in (task.get('roles') or {}).items():
            mapped = mapping[str(role_name)]
            identity = (str(role.get('object_id') or ''), str(role.get('object_type') or ''))
            existing = role_values.get(mapped)
            if existing is not None and existing[0] != identity[0]:
                rows.append(_row(index, task, 'SHARED_ROLE_OBJECT_ID_CONFLICT', mapped))
            elif existing is not None and existing[1] != identity[1]:
                rows.append(_row(index, task, 'SHARED_ROLE_TYPE_CONFLICT', mapped))
            role_values[mapped] = identity
        init = task.get('init') or {}
        for override in init.get('object_state_overrides') or []:
            if not isinstance(override, Mapping):
                continue
            role = override.get('object_role')
            state_info = _INIT_STATE_FIELDS.get(str(override.get('type') or ''))
            if not isinstance(role, str) or state_info is None:
                continue
            state_key, value_key = state_info
            mapped = mapping.get(role, role)
            key = (mapped, state_key)
            value = override.get(value_key)
            if key in init_states and init_states[key] != value:
                rows.append(_row(index, task, 'INIT_STATE_CONFLICT', '{}.{}'.format(mapped, state_key)))
            init_states[key] = copy.deepcopy(value)
        for override in init.get('object_pose_overrides') or []:
            if not isinstance(override, Mapping):
                continue
            override_type = str(override.get('type') or '')
            if override_type == 'relocate_object' and isinstance(override.get('object_role'), str):
                mapped = mapping.get(str(override['object_role']), str(override['object_role']))
                signature = _canonical(_rewrite_nested(copy.deepcopy(dict(override)), mapping))
                if mapped in relocations and relocations[mapped] != signature:
                    rows.append(_row(index, task, 'INIT_POSE_CONFLICT', mapped))
                relocations[mapped] = signature
            if override_type == 'clear_receptacle_clutter':
                deleted = set((str(value) for value in override.get('object_ids') or []))
                required = sorted(deleted & all_object_ids)
                if required:
                    rows.append(_row(index, task, 'CLUTTER_REMOVAL_CONFLICT', 'Required objects would be removed: {}'.format(', '.join(required))))
        for predicate in task.get('goal', {}).get('predicates') or []:
            if not isinstance(predicate, Mapping):
                continue
            predicate_type = str(predicate.get('type') or '')
            object_role = predicate.get('object_role')
            if not isinstance(object_role, str):
                continue
            mapped_object = mapping.get(object_role, object_role)
            if predicate_type in _GOAL_STATE_VALUES:
                state_key, value = _GOAL_STATE_VALUES[predicate_type]
                key = (mapped_object, state_key)
                if key in goal_states and goal_states[key] != value:
                    rows.append(_row(index, task, 'FINAL_STATE_GOAL_CONFLICT', '{}.{}'.format(mapped_object, state_key)))
                goal_states[key] = value
            if predicate_type in _PLACEMENT_GOALS and isinstance(predicate.get('receptacle_role'), str):
                target = mapping.get(str(predicate['receptacle_role']), str(predicate['receptacle_role']))
                placement = (predicate_type, target)
                if mapped_object in placements and placements[mapped_object] != placement:
                    rows.append(_row(index, task, 'FINAL_PLACEMENT_GOAL_CONFLICT', mapped_object))
                placements[mapped_object] = placement
    return rows

def _merged_roles(tasks: Sequence[Mapping[str, Any]], role_maps: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for index, task in enumerate(tasks):
        for role_name, role in (task.get('roles') or {}).items():
            mapped = role_maps[index][str(role_name)]
            if mapped not in roles:
                roles[mapped] = copy.deepcopy(dict(role))
    return roles

def _merged_init(tasks: Sequence[Mapping[str, Any]], role_maps: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    state_overrides: list[dict[str, Any]] = []
    pose_overrides: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    seen_poses: set[str] = set()
    for index, task in enumerate(tasks):
        mapping = role_maps[index]
        init = task.get('init') or {}
        for override in init.get('object_state_overrides') or []:
            rewritten = _rewrite_nested(copy.deepcopy(dict(override)), mapping)
            signature = _canonical(rewritten)
            if signature not in seen_states:
                seen_states.add(signature)
                state_overrides.append(rewritten)
        for override in init.get('object_pose_overrides') or []:
            rewritten = _rewrite_nested(copy.deepcopy(dict(override)), mapping)
            signature = _canonical(rewritten)
            if signature not in seen_poses:
                seen_poses.add(signature)
                pose_overrides.append(rewritten)
    return {'agent_pose': copy.deepcopy(tasks[0].get('init', {}).get('agent_pose') or {}), 'object_state_overrides': state_overrides, 'object_pose_overrides': pose_overrides}

def _rewrite_predicate(predicate: Mapping[str, Any], mapping: Mapping[str, str], *, owner_subtask_id: str) -> dict[str, Any]:
    result = _rewrite_nested(copy.deepcopy(dict(predicate)), mapping)
    result['owner_subtask_id'] = owner_subtask_id
    return result

def _rewrite_family_spec(family_spec: Mapping[str, Any], mapping: Mapping[str, str], *, subtask_id: str) -> dict[str, Any]:
    result = _rewrite_nested(copy.deepcopy(dict(family_spec)), mapping)
    human = result.get('human')
    if not isinstance(human, dict):
        return result
    events = human.get('human_events')
    if not isinstance(events, list):
        return result
    for event in events:
        if not isinstance(event, dict):
            continue
        event['id'] = '{}__{}'.format(subtask_id, str(event.get('id') or 'human_event'))
        event['owner_subtask_id'] = subtask_id
    return result

def _rewrite_disturbance_spec(value: Any, mapping: Mapping[str, str], *, subtask_id: str) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {'seed': int(value.get('seed', 0) or 0)}
    for key in ('disturbances', 'failures'):
        rewritten: list[dict[str, Any]] = []
        for config in value.get(key) or []:
            if not isinstance(config, Mapping):
                continue
            item = _rewrite_nested(copy.deepcopy(dict(config)), mapping)
            extra = copy.deepcopy(dict(item.get('extra') or {}))
            extra['active_subtask_id'] = subtask_id
            item['extra'] = extra
            rewritten.append(item)
        if rewritten:
            result[key] = rewritten
    return result if len(result) > 1 else None

def _rewrite_nested(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, list):
        return [_rewrite_nested(item, mapping) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(key, str) and (key == 'role' or key.endswith('_role')) and isinstance(item, str):
            result[key] = mapping.get(item, item)
        elif isinstance(key, str) and (key == 'roles' or key.endswith('_roles') or key in _ROLE_LIST_KEYS):
            result[key] = _rewrite_role_collection(item, mapping)
        else:
            result[key] = _rewrite_nested(item, mapping)
    return result

def _rewrite_role_collection(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [mapping.get(item, item) if isinstance(item, str) else _rewrite_nested(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: mapping.get(item, item) if isinstance(item, str) else _rewrite_nested(item, mapping) for key, item in value.items()}
    return copy.deepcopy(value)

def _select_diverse(candidates: Sequence[Mapping[str, Any]], *, max_subtasks: int) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    categories: set[str] = set()
    for task in candidates:
        category = str(task.get('category') or '')
        if category in categories:
            continue
        selected.append(task)
        categories.add(category)
        if len(selected) >= max_subtasks:
            return selected
    for task in candidates:
        if task not in selected:
            selected.append(task)
            if len(selected) >= max_subtasks:
                break
    return selected

def _stable_union(tasks: Sequence[Mapping[str, Any]], key: str, *, extra: str) -> list[str]:
    values = {str(value) for task in tasks for value in (task.get('online_requirements') or {}).get(key) or [] if str(value)}
    values.add(extra)
    return sorted(values)

def _combined_instruction(tasks: Sequence[Mapping[str, Any]]) -> str:
    parts = [str(task.get('instruction') or '').strip() for task in tasks]
    return 'Complete these tasks in order: {}'.format(' '.join(('({}) {}'.format(index + 1, part) for index, part in enumerate(parts) if part)))

def _default_composite_id(tasks: Sequence[Mapping[str, Any]], index: int | None=None) -> str:
    house_id = str(tasks[0].get('source', {}).get('house_id') or 'house')
    if index is not None:
        return 'procthor_composite_{}_{:06d}'.format(house_id, index + 1)
    digest = hashlib.sha1('\x00'.join((str(task.get('id') or '') for task in tasks)).encode('utf-8')).hexdigest()[:10]
    return 'procthor_composite_{}_{}'.format(house_id, digest)

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)

def _row(index: int | None, task: Mapping[str, Any], code: str, reason: str) -> dict[str, Any]:
    return {'task_index': index, 'task_id': task.get('id'), 'code': code, 'reason': reason}

def _reject(tasks: Sequence[Mapping[str, Any]], rejected: list[dict[str, Any]], code: str, reason: str) -> CompositeBuildResult:
    rows = list(rejected)
    rows.append(_row(None, tasks[0] if tasks else {}, code, reason))
    return CompositeBuildResult(None, rows, _summary(None, tasks, rows))

def _summary(task: Mapping[str, Any] | None, tasks: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {'status': 'generated' if task is not None else 'rejected', 'task_id': task.get('id') if task else None, 'child_count': len(tasks), 'rejected_total': len(rejected)}
__all__ = ['CompositeBuildResult', 'build_composite_task_bank', 'build_composite_task_spec']
