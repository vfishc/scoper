from __future__ import annotations
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from scoper.benchmark.common import merge_qa_overlays, rewrite_qa_overlay_roles
from scoper.benchmark.common.oracle_bank_supervisor import redundant_composite_child_indices
from scoper.benchmark.behavior.brains.oracle import can_solve_goal_predicates, unsupported_goal_predicates
from scoper.benchmark.behavior.brains.teachers.dispatcher import default_dispatcher
from scoper.benchmark.behavior.task_spec import TaskSpecValidationError, validate_behavior_task_spec
ROLE_REF_KEYS = frozenset({'object', 'role', 'receptacle', 'target', 'source', 'tool', 'heatsource', 'coldsource', 'object_role', 'receptacle_role', 'target_role', 'source_role', 'tool_role', 'heatsource', 'coldsource', 'coldsource_role', 'faucet_role', 'sink_role', 'final_receptacle_role', 'receptacle', 'receptacle_role', 'support_role', 'staging_receptacle_role', 'interrupt_role', 'required_parent_role', 'state_delta_object_role', 'put_away_ref'})
ROLE_REF_LIST_KEYS = frozenset({'roles', 'object_roles', 'receptacle_roles', 'target_roles', 'storage_roles', 'support_roles', 'device_roles', 'tool_roles', 'protected_distractor_roles', 'exclude_roles', 'exclude_target_roles', 'target_objects', 'requires_visible_roles'})
LITERAL_ROLE_REF_KEYS = frozenset({'room_id', 'room_ids', 'semantic_category', 'substance', 'threshold', 'liquid'})
INIT_STATE_PRECEDENCE = {'open': ('open', True), 'closed': ('open', False), 'toggled_on': ('toggled_on', True), 'toggled_off': ('toggled_on', False), 'dirty': ('dirty', True), 'clean': ('dirty', False), 'sliced': ('sliced', True), 'cooled': ('cooled', True)}

@dataclass(frozen=True)
class CompositeBuildResult:
    task: dict[str, Any] | None
    rejected: list[dict[str, Any]]
    summary: dict[str, Any]

def build_composite_task_spec(task_specs: Sequence[Mapping[str, Any]], *, composite_id: str | None=None, instruction: str | None=None, shared_roles: Mapping[str, str] | None=None, auto_share_scene_roles: bool=True, require_independent_goals: bool=False, qa_overlay: Mapping[str, Any] | None=None, max_subtasks: int | None=None, reject_scheduler_specs: bool=True, allow_scheduler_specs: bool=False) -> CompositeBuildResult:
    tasks = [copy.deepcopy(dict(task)) for task in task_specs]
    if max_subtasks is not None:
        tasks = tasks[:int(max_subtasks)]
    rejected: list[dict[str, Any]] = []
    if len(tasks) < 2:
        return _reject(None, rejected, 'NOT_ENOUGH_SUBTASKS', 'Composite tasks require at least two children.')
    group_key = _task_group_key(tasks[0])
    for index, task in enumerate(tasks):
        try:
            validate_behavior_task_spec(task)
        except Exception as exc:
            rejected.append(_row(index, task, 'TASK_SPEC_VALIDATION_FAILED', str(exc)))
            continue
        if _task_group_key(task) != group_key:
            rejected.append(_row(index, task, 'SCENE_SOURCE_MISMATCH', 'All subtasks must share scene_model/source_json/layout_variant_id.'))
        if reject_scheduler_specs and (not allow_scheduler_specs) and (task.get('disturbance_spec') or task.get('failure_spec')):
            rejected.append(_row(index, task, 'SCHEDULER_SPEC_DEFERRED', 'V1 composite materializer skips child disturbance/failure specs.'))
        route_error = _teacher_route_error(task)
        if route_error:
            rejected.append(_row(index, task, 'TEACHER_ROUTE_UNAVAILABLE', route_error))
    if rejected:
        return CompositeBuildResult(task=None, rejected=rejected, summary=_summary(None, tasks, rejected))
    expanded_shared_roles = _expanded_shared_roles(tasks, shared_roles or {}, auto_share_scene_roles=auto_share_scene_roles)
    role_maps = _role_maps(tasks, expanded_shared_roles)
    conflict_rows = _conflict_rows(tasks, role_maps)
    if conflict_rows:
        rejected.extend(conflict_rows)
        return CompositeBuildResult(task=None, rejected=rejected, summary=_summary(None, tasks, rejected))
    rewritten_child_goals = [[_rewrite_goal_predicate(predicate, role_maps[index], owner_subtask_id='subtask_{}'.format(index)) for predicate in (task.get('goal') or {}).get('predicates') or []] for index, task in enumerate(tasks)]
    redundant_children = redundant_composite_child_indices(rewritten_child_goals) if require_independent_goals else []
    if redundant_children:
        rejected.extend((_row(index, tasks[index], 'REDUNDANT_CHILD_GOAL', 'Child adds no independent physical goal.') for index in redundant_children))
        return CompositeBuildResult(task=None, rejected=rejected, summary=_summary(None, tasks, rejected))
    roles: dict[str, Any] = {}
    init: list[dict[str, Any]] = []
    predicates: list[dict[str, Any]] = []
    subtasks: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    parent_disturbance_spec: list[dict[str, Any]] = []
    parent_failure_spec: list[dict[str, Any]] = []
    parent_human_events: list[dict[str, Any]] = []
    child_qa_overlays: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        subtask_id = 'subtask_{}'.format(index)
        mapping = role_maps[index]
        for role_name, role in (task.get('roles') or {}).items():
            mapped = mapping[str(role_name)]
            rewritten_role = _rewrite_role(copy.deepcopy(dict(role)), mapping)
            rewritten_role['bddl_inst'] = mapped
            if mapped in roles:
                _merge_shared_role(roles[mapped], rewritten_role, mapped)
            else:
                roles[mapped] = rewritten_role
            operations.append({'type': 'role', 'subtask': subtask_id, 'from': role_name, 'to': mapped})
        for entry in task.get('init') or []:
            init.append(_rewrite_init(entry, mapping))
        child_predicates = rewritten_child_goals[index]
        predicates.extend(child_predicates)
        child_family = task.get('family_spec')
        subtask = {'subtask_id': subtask_id, 'category': str(task.get('category') or ''), 'instruction_fragment': str(task.get('instruction') or ''), 'goal_predicates': child_predicates, 'source_task_id': task.get('id')}
        if isinstance(child_family, Mapping):
            subtask['family_spec'] = _rewrite_nested(copy.deepcopy(dict(child_family)), mapping)
        child_disturbances, child_failures = _rewrite_child_scheduler_specs(task, mapping, subtask_id=subtask_id)
        child_human_events = _rewrite_child_human_events(task, mapping, subtask_id=subtask_id)
        if child_disturbances:
            subtask['disturbance_spec'] = copy.deepcopy(child_disturbances)
            parent_disturbance_spec.extend(child_disturbances)
        if child_failures:
            subtask['failure_spec'] = copy.deepcopy(child_failures)
            parent_failure_spec.extend(child_failures)
        if child_human_events:
            subtask.setdefault('family_spec', {})
            if isinstance(subtask['family_spec'], dict):
                human = subtask['family_spec'].setdefault('human', {})
                if isinstance(human, dict):
                    human['human_events'] = copy.deepcopy(child_human_events)
            parent_human_events.extend(child_human_events)
        child_overlay = task.get('qa_overlay')
        if isinstance(child_overlay, Mapping) and child_overlay.get('questions'):
            rewritten_overlay = rewrite_qa_overlay_roles(child_overlay, mapping, subtask_id=subtask_id)
            rewritten_overlay['questions'] = rewritten_overlay['questions'][:1]
            child_qa_overlays.append(rewritten_overlay)
        subtasks.append(subtask)
    first = tasks[0]
    parent_source = copy.deepcopy(dict(first.get('source') or {}))
    parent_source.setdefault('type', 'template')
    parent_source['composite_child_task_ids'] = [str(task.get('id')) for task in tasks]
    task_id = composite_id or _default_composite_id(tasks)
    parent = {'id': task_id, 'version': 'stage4.0-composite', 'backend': 'behavior', 'category': 'composite', 'source': parent_source, 'instruction': instruction or _combined_instruction(tasks), 'roles': roles, 'init': init, 'goal': {'type': 'predicates', 'require_model_stop': any((bool((task.get('goal') or {}).get('require_model_stop')) for task in tasks)), 'bddl_goal': None, 'predicates': predicates}, 'family_spec': {'family': 'composite', 'composite': {'ordering': 'sequential', 'subtasks': subtasks}}, 'generation': {'stage': 'composite_materialized', 'source_task_ids': [task.get('id') for task in tasks], 'role_namespace': role_maps, 'explicit_shared_roles': dict(shared_roles or {}), 'auto_shared_scene_roles': {key: value for key, value in expanded_shared_roles.items() if key not in dict(shared_roles or {})}, 'operations': operations, 'stage4_readiness': 'composite_materialized', 'stage4_readiness_debt': sorted(_merge_readiness_debt(tasks))}, 'reset_plan': {'state_and_relation_init': init, 'assertions': _rewrite_assertions(tasks, role_maps), 'required_init_mutations': sorted(_merge_required_init_mutations(tasks)), 'runtime_capabilities_required': sorted(_merge_runtime_capabilities(tasks))}, 'oracle_teacher_hint': {'family': 'composite', 'expected_events': ['sequential_composite'], 'expected_brain_behaviors': ['sequential_composition', 'handoff_between_subtasks']}, 'metrics_config': {'track_predicate_progress': True, 'track_tool_success_rate': True}, 'validation': {'status': 'generated_pending_reset_validation', 'checks': []}, 'tags': ['composite_materialized']}
    if parent_disturbance_spec:
        parent['disturbance_spec'] = parent_disturbance_spec
    if parent_failure_spec:
        parent['failure_spec'] = parent_failure_spec
    if parent_human_events:
        parent['family_spec']['human'] = {'human_events': parent_human_events}
    merged_overlay = merge_qa_overlays(child_qa_overlays, parent_overlay=qa_overlay, max_questions=3)
    if merged_overlay is not None:
        parent['qa_overlay'] = merged_overlay
    try:
        from scoper.benchmark.behavior.scene.concrete_categories import apply_concrete_public_names
        apply_concrete_public_names(parent)
        validate_behavior_task_spec(parent)
        from scoper.benchmark.common.task_identity_policy import enabled, assert_distinct
        if enabled():
            assert_distinct(parent)
    except Exception as exc:
        rejected.append(_row(None, parent, 'COMPOSITE_TASK_SPEC_VALIDATION_FAILED', str(exc)))
        return CompositeBuildResult(task=None, rejected=rejected, summary=_summary(None, tasks, rejected))
    return CompositeBuildResult(task=parent, rejected=[], summary=_summary(parent, tasks, []))

def build_composite_task_bank(task_specs: Sequence[Mapping[str, Any]], *, max_subtasks: int=3, max_composites: int=20, seed: int=0, allow_scheduler_specs: bool=False, auto_share_scene_roles: bool=True) -> dict[str, Any]:
    del seed
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for task in task_specs:
        groups[_task_group_key(task)].append(task)
    tasks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda row: row[0]):
        candidates = sorted(items, key=lambda task: (str(task.get('category') or ''), str(task.get('id') or '')))
        while len(candidates) >= 2 and len(tasks) < int(max_composites):
            selected = _select_diverse(candidates, max_subtasks=max_subtasks)
            result = build_composite_task_spec(selected, composite_id=_default_composite_id(selected, index=len(tasks)), max_subtasks=max_subtasks, allow_scheduler_specs=allow_scheduler_specs, reject_scheduler_specs=not allow_scheduler_specs, auto_share_scene_roles=auto_share_scene_roles)
            if result.task is not None:
                tasks.append(result.task)
                selected_ids = {id(task) for task in selected}
                candidates = [task for task in candidates if id(task) not in selected_ids]
            else:
                rejected.extend(result.rejected)
                candidates = candidates[1:]
        if len(tasks) >= int(max_composites):
            break
    return {'tasks': tasks, 'rejected': rejected, 'summary': _bank_summary(tasks, rejected)}

def _select_diverse(candidates: Sequence[Mapping[str, Any]], *, max_subtasks: int) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    categories = set()
    for task in candidates:
        category = str(task.get('category') or '')
        if category in categories and len(selected) >= 2:
            continue
        selected.append(task)
        categories.add(category)
        if len(selected) >= int(max_subtasks):
            break
    if len(selected) < 2:
        selected = list(candidates[:2])
    return selected

def _task_group_key(task: Mapping[str, Any]) -> tuple[str, str, str]:
    source = task.get('source') or {}
    return (str(source.get('scene_model') or ''), str(source.get('source_json') or ''), str(source.get('layout_variant_id') or ''))

def _teacher_route_error(task: Mapping[str, Any]) -> str | None:
    try:
        default_dispatcher().select(task)
    except Exception as exc:
        if can_solve_goal_predicates(task):
            return None
        unsupported = unsupported_goal_predicates(task)
        return '{} unsupported={}'.format(str(exc), ','.join(unsupported))
    return None

def _expanded_shared_roles(tasks: Sequence[Mapping[str, Any]], shared_roles: Mapping[str, str], *, auto_share_scene_roles: bool) -> dict[str, str]:
    expanded = {str(ref): str(target) for ref, target in shared_roles.items()}
    if not auto_share_scene_roles:
        return expanded
    groups: dict[str, list[tuple[int, str, Mapping[str, Any]]]] = defaultdict(list)
    for index, task in enumerate(tasks):
        for role_name, role in (task.get('roles') or {}).items():
            if not isinstance(role, Mapping):
                continue
            object_id = _shareable_scene_object_id(role)
            if object_id is None:
                continue
            groups[object_id].append((index, str(role_name), role))
    for _object_id, entries in groups.items():
        if len(entries) < 2 or not _can_auto_share_entries(entries):
            continue
        canonical_index, canonical_role, _canonical = entries[0]
        canonical_ref = 'task{}.{}'.format(canonical_index, canonical_role)
        canonical_target = expanded.get(canonical_ref, canonical_ref)
        for index, role_name, _role in entries:
            ref = 'task{}.{}'.format(index, role_name)
            if ref == canonical_ref:
                continue
            expanded.setdefault(ref, canonical_target)
    return expanded

def _shareable_scene_object_id(role: Mapping[str, Any]) -> str | None:
    object_id = role.get('object_id')
    if not isinstance(object_id, str) or not object_id:
        return None
    if object_id.startswith('spawnable:'):
        return None
    if role.get('requires_spawn') is True:
        return None
    return object_id

def _can_auto_share_entries(entries: Sequence[tuple[int, str, Mapping[str, Any]]]) -> bool:
    first = entries[0][2]
    first_synset = str(first.get('synset') or '')
    first_category = str(first.get('category') or first.get('object_type') or '')
    first_object_type = str(first.get('object_type') or '')
    first_center = _roundish_tuple(first.get('center'))
    first_extent = _roundish_tuple(first.get('extent'))
    state_by_key: dict[str, Any] = {}
    for _index, _role_name, role in entries:
        synset = str(role.get('synset') or '')
        category = str(role.get('category') or role.get('object_type') or '')
        object_type = str(role.get('object_type') or '')
        if first_synset and synset and (synset != first_synset):
            return False
        if first_category and category and (category != first_category):
            return False
        if first_object_type and object_type and (object_type != first_object_type):
            return False
        center = _roundish_tuple(role.get('center'))
        extent = _roundish_tuple(role.get('extent'))
        if first_center and center and (center != first_center):
            return False
        if first_extent and extent and (extent != first_extent):
            return False
        for key, value in (role.get('init_states') or {}).items():
            existing = state_by_key.get(str(key))
            if existing is not None and existing != value:
                return False
            state_by_key[str(key)] = copy.deepcopy(value)
    return True

def _roundish_tuple(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple((round(float(item), 4) for item in value))
    except Exception:
        return None

def _role_maps(tasks: Sequence[Mapping[str, Any]], shared_roles: Mapping[str, str]) -> list[dict[str, str]]:
    canonical: dict[str, str] = {}
    for ref, target in shared_roles.items():
        canonical[str(ref)] = str(target)
    out: list[dict[str, str]] = []
    for index, task in enumerate(tasks):
        mapping: dict[str, str] = {}
        for role_name in task.get('roles') or {}:
            fq = 'task{}.{}'.format(index, role_name)
            target = canonical.get(fq, fq)
            if '.' in target:
                target_index, _, target_role = target.partition('.')
                if target_index.startswith('task') and target_role:
                    mapped = '{}__{}'.format(target_index, target_role)
                else:
                    mapped = target.replace('.', '__')
            else:
                mapped = target
            if target == fq:
                mapped = 'task{}__{}'.format(index, role_name)
            mapping[str(role_name)] = mapped
        out.append(mapping)
    return out

def _conflict_rows(tasks: Sequence[Mapping[str, Any]], role_maps: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    state_by_role: dict[str, dict[str, Any]] = defaultdict(dict)
    goal_state_by_role: dict[str, dict[str, Any]] = defaultdict(dict)
    goal_relation_by_object: dict[str, set[tuple[str, str]]] = defaultdict(set)
    relation_by_object: dict[str, set[tuple[str, str]]] = defaultdict(set)
    object_id_by_role: dict[str, str] = {}
    for index, task in enumerate(tasks):
        mapping = role_maps[index]
        for role_name, role in (task.get('roles') or {}).items():
            mapped = mapping[str(role_name)]
            object_id = role.get('object_id')
            if object_id:
                existing = object_id_by_role.get(mapped)
                if existing is not None and existing != str(object_id):
                    rows.append(_row(index, task, 'SHARED_ROLE_OBJECT_ID_CONFLICT', mapped))
                object_id_by_role[mapped] = str(object_id)
            for key, value in (role.get('init_states') or {}).items():
                _add_state_conflict(rows, state_by_role, mapped, str(key), value, index, task)
        for entry in task.get('init') or []:
            predicate = str(entry.get('predicate') or '')
            args = list(entry.get('args') or [])
            if not args:
                continue
            obj = mapping.get(str(args[0]), str(args[0]))
            if predicate in INIT_STATE_PRECEDENCE:
                key, value = INIT_STATE_PRECEDENCE[predicate]
                _add_state_conflict(rows, state_by_role, obj, key, value, index, task)
            elif predicate in {'ontop', 'inside'} and len(args) >= 2:
                target = mapping.get(str(args[1]), str(args[1]))
                relation_by_object[obj].add((predicate, target))
        for predicate in (task.get('goal') or {}).get('predicates') or []:
            state_constraint = _goal_state_constraint(predicate, mapping)
            if state_constraint is not None:
                obj, key, value = state_constraint
                _add_state_conflict(rows, goal_state_by_role, obj, key, value, index, task, code='GOAL_STATE_CONFLICT')
            relation_constraint = _goal_relation_constraint(predicate, mapping)
            if relation_constraint is not None:
                obj, relation = relation_constraint
                goal_relation_by_object[obj].add(relation)
    for role, relations in relation_by_object.items():
        targets = {target for _predicate, target in relations}
        if len(targets) > 1:
            rows.append({'task_index': None, 'task_id': None, 'code': 'SPATIAL_INIT_CONFLICT', 'reason': role})
    for role, relations in goal_relation_by_object.items():
        positive_targets = {target for predicate, target in relations if predicate in {'object_on_receptacle', 'object_in_receptacle'}}
        if len(positive_targets) > 1:
            rows.append({'task_index': None, 'task_id': None, 'code': 'GOAL_SPATIAL_CONFLICT', 'reason': role})
    return rows

def _add_state_conflict(rows: list[dict[str, Any]], state_by_role: dict[str, dict[str, Any]], role: str, key: str, value: Any, index: int, task: Mapping[str, Any], *, code: str='INIT_STATE_CONFLICT') -> None:
    existing = state_by_role[role].get(key)
    if existing is not None and existing != value:
        rows.append(_row(index, task, code, '{}.{}'.format(role, key)))
    state_by_role[role][key] = copy.deepcopy(value)

def _goal_state_constraint(predicate: Mapping[str, Any], mapping: Mapping[str, str]) -> tuple[str, str, Any] | None:
    ptype = str(predicate.get('type') or '')
    args = predicate.get('args') if isinstance(predicate.get('args'), Mapping) else {}
    obj = args.get('object')
    if not isinstance(obj, str):
        return None
    role = mapping.get(obj, obj)
    if ptype == 'object_open':
        return (role, 'open', True)
    if ptype == 'object_closed':
        return (role, 'open', False)
    if ptype == 'object_toggled_on':
        return (role, 'toggled_on', True)
    if ptype == 'object_toggled_off':
        return (role, 'toggled_on', False)
    if ptype == 'object_clean':
        return (role, 'dirty', False)
    if ptype == 'object_dirty':
        return (role, 'dirty', True)
    if ptype == 'object_sliced':
        return (role, 'sliced', True)
    if ptype == 'object_filled':
        return (role, 'filled', True)
    if ptype == 'object_soaked':
        return (role, 'saturated', True)
    if ptype in {'object_cooked', 'object_hot'}:
        return (role, 'cooked', True)
    if ptype == 'object_cool':
        return (role, 'cooled', True)
    return None

def _goal_relation_constraint(predicate: Mapping[str, Any], mapping: Mapping[str, str]) -> tuple[str, tuple[str, str]] | None:
    ptype = str(predicate.get('type') or '')
    if ptype not in {'object_on_receptacle', 'object_in_receptacle'}:
        return None
    args = predicate.get('args') if isinstance(predicate.get('args'), Mapping) else {}
    obj = args.get('object')
    receptacle = args.get('receptacle')
    if not isinstance(obj, str) or not isinstance(receptacle, str):
        return None
    return (mapping.get(obj, obj), (ptype, mapping.get(receptacle, receptacle)))

def _merge_shared_role(target: dict[str, Any], role: Mapping[str, Any], mapped: str) -> None:
    for key in ('synset', 'category', 'object_id', 'object_type', 'bddl_inst'):
        value = role.get(key)
        if value and target.get(key) and (target.get(key) != value):
            raise TaskSpecValidationError("Shared role '{}' has conflicting {}.".format(mapped, key))
        if value and key not in target:
            target[key] = copy.deepcopy(value)
    target.setdefault('abilities', [])
    target['abilities'] = sorted(set(target.get('abilities') or []) | set(role.get('abilities') or []))
    states = dict(target.get('init_states') or {})
    for key, value in (role.get('init_states') or {}).items():
        if key in states and states[key] != value:
            raise TaskSpecValidationError("Shared role '{}' has conflicting init state '{}'.".format(mapped, key))
        states[key] = copy.deepcopy(value)
    if states:
        target['init_states'] = states

def _rewrite_role(role: dict[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    if isinstance(role.get('spawn_placement'), Mapping):
        role['spawn_placement'] = _rewrite_nested(copy.deepcopy(dict(role['spawn_placement'])), mapping)
    return role

def _rewrite_init(entry: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    out = copy.deepcopy(dict(entry))
    args = []
    for idx, arg in enumerate(out.get('args') or []):
        if idx == 0 or (out.get('predicate') in {'ontop', 'inside'} and idx == 1):
            args.append(mapping.get(str(arg), str(arg)))
        else:
            args.append(arg)
    out['args'] = args
    return out

def _rewrite_goal_predicate(predicate: Mapping[str, Any], mapping: Mapping[str, str], *, owner_subtask_id: str) -> dict[str, Any]:
    out = _rewrite_nested(copy.deepcopy(dict(predicate)), mapping)
    out['owner_subtask_id'] = owner_subtask_id
    return out

def _rewrite_nested(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, list):
        return [_rewrite_nested(item, mapping) for item in value]
    if not isinstance(value, dict):
        return mapping.get(value, value) if isinstance(value, str) else value
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key in LITERAL_ROLE_REF_KEYS:
            out[key] = copy.deepcopy(item)
        elif key in ROLE_REF_KEYS and isinstance(item, str):
            out[key] = mapping.get(item, item)
        elif key in ROLE_REF_LIST_KEYS and isinstance(item, list):
            out[key] = [mapping.get(str(role), str(role)) if isinstance(role, str) else role for role in item]
        elif key in ROLE_REF_LIST_KEYS and isinstance(item, dict):
            out[key] = {k: mapping.get(str(v), str(v)) if isinstance(v, str) else v for k, v in item.items()}
        else:
            out[key] = _rewrite_nested(item, mapping)
    return out

def _rewrite_assertions(tasks: Sequence[Mapping[str, Any]], role_maps: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        mapping = role_maps[index]
        for assertion in (task.get('reset_plan') or {}).get('assertions') or []:
            assertions.append(_rewrite_nested(copy.deepcopy(dict(assertion)), mapping))
        for assertion in (task.get('generation') or {}).get('deferred_init_assertions') or []:
            assertions.append(_rewrite_nested(copy.deepcopy(dict(assertion)), mapping))
    return assertions

def _rewrite_child_scheduler_specs(task: Mapping[str, Any], mapping: Mapping[str, str], *, subtask_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    disturbances, failures = _normalized_child_scheduler_specs(task)
    child_task_id = str(task.get('id') or '')
    rewritten_disturbances = [_rewrite_scheduler_spec(item, mapping, subtask_id=subtask_id, child_task_id=child_task_id) for item in disturbances]
    rewritten_failures = [_rewrite_scheduler_spec(item, mapping, subtask_id=subtask_id, child_task_id=child_task_id) for item in failures]
    return (rewritten_disturbances, rewritten_failures)

def _rewrite_child_human_events(task: Mapping[str, Any], mapping: Mapping[str, str], *, subtask_id: str) -> list[dict[str, Any]]:
    family = task.get('family_spec')
    human = family.get('human') if isinstance(family, Mapping) else None
    if not isinstance(human, Mapping):
        return []
    events = human.get('human_events')
    if not isinstance(events, list):
        events = human.get('human_events_template')
    if not isinstance(events, list):
        return []
    rewritten: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        out = _rewrite_nested(copy.deepcopy(dict(event)), mapping)
        event_id = str(out.get('id') or 'human_event')
        out['id'] = '{}__{}'.format(subtask_id, event_id)
        out['owner_subtask_id'] = subtask_id
        activation = out.get('activation')
        if isinstance(activation, Mapping):
            activation_out = copy.deepcopy(dict(activation))
        else:
            activation_out = {}
        activation_out.setdefault('subtask_id', subtask_id)
        out['activation'] = activation_out
        rewritten.append(out)
    return rewritten

def _normalized_child_scheduler_specs(task: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    disturbances: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    dist_spec = task.get('disturbance_spec')
    if isinstance(dist_spec, Mapping):
        if isinstance(dist_spec.get('disturbances'), list):
            disturbances.extend((dict(item) for item in dist_spec.get('disturbances') or [] if isinstance(item, Mapping)))
        elif dist_spec.get('type') is not None:
            disturbances.append(dict(dist_spec))
        if isinstance(dist_spec.get('failures'), list):
            failures.extend((dict(item) for item in dist_spec.get('failures') or [] if isinstance(item, Mapping)))
    elif isinstance(dist_spec, list):
        disturbances.extend((dict(item) for item in dist_spec if isinstance(item, Mapping)))
    fail_spec = task.get('failure_spec')
    if isinstance(fail_spec, Mapping):
        if isinstance(fail_spec.get('failures'), list):
            failures.extend((dict(item) for item in fail_spec.get('failures') or [] if isinstance(item, Mapping)))
        elif isinstance(fail_spec.get('configs'), list):
            failures.extend((dict(item) for item in fail_spec.get('configs') or [] if isinstance(item, Mapping)))
        elif fail_spec.get('mode') is not None:
            failures.append(dict(fail_spec))
    elif isinstance(fail_spec, list):
        failures.extend((dict(item) for item in fail_spec if isinstance(item, Mapping)))
    return (disturbances, failures)

def _rewrite_scheduler_spec(spec: Mapping[str, Any], mapping: Mapping[str, str], *, subtask_id: str, child_task_id: str) -> dict[str, Any]:
    out = _rewrite_nested(copy.deepcopy(dict(spec)), mapping)
    out['owner_subtask_id'] = subtask_id
    if child_task_id:
        out['owner_child_task_id'] = child_task_id
    activation = out.get('activation')
    if isinstance(activation, Mapping):
        activation_out = copy.deepcopy(dict(activation))
    else:
        activation_out = {}
    activation_out.setdefault('subtask_id', subtask_id)
    out['activation'] = activation_out
    return out

def _merge_readiness_debt(tasks: Sequence[Mapping[str, Any]]) -> set[str]:
    debt: set[str] = set()
    for task in tasks:
        debt.update((task.get('generation') or {}).get('stage4_readiness_debt') or [])
    debt.discard('composite_executor')
    return debt

def _merge_required_init_mutations(tasks: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for task in tasks:
        out.update((task.get('reset_plan') or {}).get('required_init_mutations') or [])
        out.update((task.get('generation') or {}).get('required_init_mutations') or [])
    return out

def _merge_runtime_capabilities(tasks: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for task in tasks:
        out.update((task.get('reset_plan') or {}).get('runtime_capabilities_required') or [])
        out.update((task.get('generation') or {}).get('runtime_capabilities_required') or [])
    return out

def _combined_instruction(tasks: Sequence[Mapping[str, Any]]) -> str:
    parts = [str(task.get('instruction') or '').strip() for task in tasks if str(task.get('instruction') or '').strip()]
    return ' Then '.join(parts) if parts else 'Complete the subtasks in order.'

def _default_composite_id(tasks: Sequence[Mapping[str, Any]], index: int=0) -> str:
    source = tasks[0].get('source') or {}
    scene_model = str(source.get('scene_model') or 'scene')
    variant = str(source.get('layout_variant_id') or source.get('source_json') or 'base').split('/')[-1].replace('.json', '')
    return 'composite_{}_{}_{:04d}'.format(scene_model, variant, index)

def _reject(task: Mapping[str, Any] | None, rejected: list[dict[str, Any]], code: str, reason: str) -> CompositeBuildResult:
    rows = list(rejected)
    rows.append(_row(None, task or {}, code, reason))
    return CompositeBuildResult(task=None, rejected=rows, summary={'status': 'rejected', 'rejected_total': len(rows)})

def _row(index: int | None, task: Mapping[str, Any], code: str, reason: str) -> dict[str, Any]:
    return {'task_index': index, 'task_id': task.get('id'), 'code': code, 'reason': reason}

def _summary(task: Mapping[str, Any] | None, tasks: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {'status': 'generated' if task is not None else 'rejected', 'child_count': len(tasks), 'rejected_total': len(rejected), 'task_id': task.get('id') if task else None}

def _bank_summary(tasks: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {'generated_total': len(tasks), 'rejected_total': len(rejected)}
__all__ = ['CompositeBuildResult', 'build_composite_task_bank', 'build_composite_task_spec']
