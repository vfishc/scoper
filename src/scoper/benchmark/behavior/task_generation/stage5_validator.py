from __future__ import annotations
import copy
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence
from scoper.benchmark.behavior.execution_core import actions
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose
from scoper.benchmark.behavior.execution_core.navigation import NavigationFace, NavigationTarget, plan_navigation
from scoper.benchmark.behavior.execution_core.trace import to_jsonable
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.benchmark.behavior.brains.base import ToolCall
from scoper.benchmark.behavior.goal_checker import _eval_inside, _eval_ontop
from scoper.benchmark.behavior.scheduler import normalize_scheduler_specs
from scoper.benchmark.behavior.scheduler.disturbance import KNOWN_DISTURBANCE_TYPES, DisturbanceScheduler
from scoper.benchmark.behavior.scheduler.failure_injection import KNOWN_FAILURE_MODES, FailureInjectionScheduler
from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec
RESET_RUNTIME_DEBT_OK = frozenset({'spawn_object', 'spawn_placement', 'state_override'})
ORACLE_DEFER_DEBT_PREFIXES = ('family_policy:', 'family_scheduler:', 'family_teacher:recover', 'scheduler:unsupported:', 'relocation:')
ORACLE_DEFER_DEBT_EXACT = frozenset({'composite_executor', 'exploration_policy', 'human_interaction', 'object_pose_relocation', 'reachable_map', 'scheduler:disturbance', 'scheduler:failure_injection', 'visibility_query_or_controlled_spawn_visibility', 'visual_discoverability'})
SKIP_REACHABILITY_CATEGORIES = frozenset({'floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'})
DIRECTIONAL_OPENABLE_CATEGORIES = frozenset({'fridge', 'refrigerator', 'freezer', 'microwave', 'microwave_oven'})
DIRECTIONAL_INTERACTION_ABILITIES = frozenset({'openable'})

def _stage5_scheduler_specs(task_spec: Mapping[str, Any], *, rng_seed: int=0) -> tuple[list[dict], list[dict], int]:
    disturbances, failures, _human_events, seed = normalize_scheduler_specs(task_spec, default_seed=rng_seed)
    return (disturbances, failures, seed)

def validate_stage5_reset_mutation(task_spec: Mapping[str, Any], *, rng_seed: int=0, run_reachability: bool=True, scene_layout_mode: str='task_roles', load_doors: bool=False) -> dict[str, Any]:
    result: dict[str, Any] = {'task_id': task_spec.get('id'), 'template_id': (task_spec.get('template') or {}).get('template_id'), 'scene_model': (task_spec.get('source') or {}).get('scene_model'), 'structure_ok': False, 'reset_ok': False, 'deterministic_reset': False, 'init_ok': False, 'initial_goal_false': False, 'geometry_ok': False, 'reachability_ok': None, 'reachability_skipped': False, 'mutation_spec_ok': False, 'mutation_replay_ok': False, 'mutation_runtime_complete': False, 'stage5_reset_mutation_ok': False, 'stage5_bucket': 'reset_validation_failed', 'failure_code': None, 'failure_reason': None, 'deferred_codes': [], 'oracle_gate': None, 'oracle_deferred_reasons': [], 'readiness_debt': list((task_spec.get('generation') or {}).get('stage4_readiness_debt') or []), 'reset_before_after': {}, 'state_predicate_checks': [], 'deferred_assertion_checks': [], 'geometry_report': {}, 'reachability_report': {}, 'mutation_diff': [], 'mutation_replay': [], 'scene_layout_mode': str(scene_layout_mode or 'task_roles'), 'load_doors': bool(load_doors)}
    try:
        validate_behavior_task_spec(task_spec)
        result['structure_ok'] = True
        env = BehaviorClosedLoopToolEnv(scene_layout_mode=scene_layout_mode, load_doors=load_doors)
        env.reset_to_task(task_spec, rng_seed=rng_seed)
        result['reset_ok'] = True
        result['placement_report'] = copy.deepcopy(env.scene_result.placement_report)
        result['scene_layout_report'] = copy.deepcopy((env.scene_result.placement_report or {}).get('source_json_full_layout') or {})
        result['reset_before_after'] = build_reset_before_after(task_spec, env)
        env2 = BehaviorClosedLoopToolEnv(scene_layout_mode=scene_layout_mode, load_doors=load_doors)
        env2.reset_to_task(task_spec, rng_seed=rng_seed)
        first_reset_snapshot = reset_snapshot(env)
        second_reset_snapshot = reset_snapshot(env2)
        result['reset_snapshots'] = {'first': first_reset_snapshot, 'second': second_reset_snapshot, 'first_sha256': _snapshot_sha256(first_reset_snapshot), 'second_sha256': _snapshot_sha256(second_reset_snapshot), 'diff': _snapshot_diff(first_reset_snapshot, second_reset_snapshot)}
        result['deterministic_reset'] = first_reset_snapshot == second_reset_snapshot
        if not result['deterministic_reset']:
            result['determinism_diff'] = {'first': first_reset_snapshot, 'second': second_reset_snapshot}
        init_checks = check_init_predicates(env, task_spec)
        assertion_checks = check_deferred_init_assertions(env, task_spec)
        result['state_predicate_checks'] = init_checks
        result['deferred_assertion_checks'] = assertion_checks
        result['init_ok'] = all((item.get('satisfied') for item in init_checks + assertion_checks))
        initial_goal = env.check_goal(model_stopped=False)
        result['initial_goal'] = initial_goal
        predicate_success = initial_goal.get('predicate_success')
        result['initial_goal_false'] = isinstance(predicate_success, (int, float)) and (not isinstance(predicate_success, bool)) and (float(predicate_success) != 1.0)
        policy = task_spec.get('conditional_storage_policy') or {}
        allowed_restoration = (task_spec.get('template') or {}).get('template_id') == 'inspect_and_restore_storage_container_001' and policy.get('schema_version') == 'conditional-storage-policy.v1' and (policy.get('expected_branch') == 'dishware') and (policy.get('initial_state') == 'closed') and ((task_spec.get('goal') or {}).get('predicates') == [{'type': 'object_closed', 'args': {'object': 'cabinet'}}])
        result['initial_goal_allowed_restoration'] = allowed_restoration
        result['initial_goal_admissible'] = result['initial_goal_false'] or allowed_restoration
        geometry = validate_geometry(env, task_spec)
        result['geometry_report'] = geometry
        result['geometry_ok'] = bool(geometry.get('ok'))
        room_constraints = validate_observation_conditioned_room_constraints(env, task_spec)
        result['observation_conditioned_room_constraints'] = room_constraints
        if not room_constraints.get('ok', True):
            result['geometry_ok'] = False
            result['geometry_report'] = dict(result['geometry_report'] or {})
            issues = list(result['geometry_report'].get('issues') or [])
            issues.extend(room_constraints.get('issues') or [])
            result['geometry_report']['issues'] = issues[:50]
            result['geometry_report']['issue_count'] = len(issues)
        if run_reachability:
            reachability = validate_reachability(env, task_spec)
            result['reachability_report'] = reachability
            result['reachability_ok'] = bool(reachability.get('ok'))
        else:
            result['reachability_ok'] = None
            result['reachability_skipped'] = True
            result['reachability_report'] = {'ok': True, 'skipped': True}
        mutation = validate_mutation_plan(task_spec, rng_seed=rng_seed, scene_layout_mode=scene_layout_mode, load_doors=load_doors)
        result['mutation_spec_ok'] = bool(mutation.get('spec_ok'))
        result['mutation_replay_ok'] = bool(mutation.get('replay_ok'))
        result['mutation_runtime_complete'] = bool(result['mutation_spec_ok'] and result['mutation_replay_ok'] and (not mutation.get('deferred_codes')))
        result['mutation_diff'] = mutation.get('diff', [])
        result['mutation_replay'] = mutation.get('replay', [])
        result['mutation_deferred_codes'] = mutation.get('deferred_codes', [])
        result['deferred_codes'] = sorted(set(result['deferred_codes']) | set(mutation.get('deferred_codes', [])))
        result['oracle_deferred_reasons'] = oracle_deferred_reasons(task_spec)
        result['oracle_gate'] = 'ready_for_oracle_precheck' if not result['oracle_deferred_reasons'] else 'deferred_runtime_debt'
        reachability_gate_ok = result.get('reachability_ok') is True or result.get('reachability_skipped') is True
        gates_ok = all((bool(result.get(key)) for key in ('structure_ok', 'reset_ok', 'deterministic_reset', 'init_ok', 'initial_goal_admissible', 'geometry_ok', 'mutation_spec_ok', 'mutation_replay_ok'))) and reachability_gate_ok
        result['stage5_reset_mutation_ok'] = gates_ok
        if gates_ok and result['deferred_codes']:
            result['stage5_bucket'] = 'reset_validated_with_deferred_mutation_runtime'
        elif gates_ok:
            result['stage5_bucket'] = 'reset_mutation_validated'
        else:
            result['stage5_bucket'] = 'reset_validation_failed'
    except Exception as exc:
        if not result['structure_ok']:
            result['failure_code'] = 'TASK_SPEC_VALIDATION_FAILED'
        elif not result['reset_ok']:
            result['failure_code'] = 'RESET_FAILED'
        else:
            result['failure_code'] = 'STAGE5_VALIDATION_EXCEPTION'
        result['failure_reason'] = str(exc)
    if result['failure_code'] is None:
        result['failure_code'] = first_failure_code(result)
    return result

def first_failure_code(result: Mapping[str, Any]) -> str | None:
    checks = (('structure_ok', 'TASK_SPEC_VALIDATION_FAILED'), ('reset_ok', 'RESET_FAILED'), ('deterministic_reset', 'RESET_NOT_DETERMINISTIC'), ('init_ok', 'INIT_PREDICATES_UNSATISFIED'), ('initial_goal_false', 'INITIAL_GOAL_ALREADY_TRUE'), ('geometry_ok', 'GEOMETRY_VALIDATION_FAILED'), ('reachability_ok', 'REACHABILITY_VALIDATION_FAILED'), ('mutation_spec_ok', 'MUTATION_SPEC_INVALID'), ('mutation_replay_ok', 'MUTATION_REPLAY_FAILED'))
    for key, code in checks:
        if key == 'initial_goal_false' and result.get('initial_goal_allowed_restoration') is True:
            continue
        if result.get(key) is False:
            return code
    return None

def build_reset_before_after(task_spec: Mapping[str, Any], env: BehaviorClosedLoopToolEnv) -> dict[str, Any]:
    before_roles = {}
    for role_name, role in sorted((task_spec.get('roles') or {}).items()):
        before_roles[role_name] = {'category': role.get('category'), 'object_type': role.get('object_type'), 'center': _round_vector(role.get('center')), 'extent': _round_vector(role.get('extent')), 'init_states': copy.deepcopy(role.get('init_states') or {}), 'requires_spawn': bool(role.get('requires_spawn'))}
    return {'before_role_spec': before_roles, 'after_reset': reset_snapshot(env), 'placement_report': copy.deepcopy(env.scene_result.placement_report)}

def reset_snapshots_equal(env1: BehaviorClosedLoopToolEnv, env2: BehaviorClosedLoopToolEnv) -> bool:
    return _stable_json(reset_snapshot(env1)) == _stable_json(reset_snapshot(env2))

def _snapshot_sha256(value: Mapping[str, Any]) -> str:
    import hashlib
    payload = json.dumps(_jsonable_sorted(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def _snapshot_diff(first: Any, second: Any, pointer: str='') -> list[dict[str, Any]]:
    if type(first) is not type(second):
        return [{'json_pointer': pointer or '/', 'first': first, 'second': second}]
    if isinstance(first, Mapping):
        rows = []
        for key in sorted(set(first) | set(second)):
            escaped = str(key).replace('~', '~0').replace('/', '~1')
            child = f'{pointer}/{escaped}'
            if key not in first:
                rows.append({'json_pointer': child, 'first': None, 'second': second[key]})
            elif key not in second:
                rows.append({'json_pointer': child, 'first': first[key], 'second': None})
            else:
                rows.extend(_snapshot_diff(first[key], second[key], child))
        return rows
    if isinstance(first, list):
        if len(first) != len(second):
            return [{'json_pointer': pointer or '/', 'first': first, 'second': second}]
        rows = []
        for index, (left, right) in enumerate(zip(first, second)):
            rows.extend(_snapshot_diff(left, right, f'{pointer}/{index}'))
        return rows
    return [] if first == second else [{'json_pointer': pointer or '/', 'first': first, 'second': second}]

def reset_snapshot(env: BehaviorClosedLoopToolEnv) -> dict[str, Any]:
    objects = {}
    for name, obj in sorted(env.core.objects.items()):
        objects[name] = {'category': obj.category, 'center': _round_vector(obj.center), 'extent': _round_vector(obj.extent), 'abilities': sorted(getattr(obj, 'abilities', ()) or ()), 'states': _jsonable_sorted(getattr(obj, 'states', {}) or {})}
    return {'role_map': dict(sorted(env.role_map.items())), 'agent_pose': _jsonable_sorted(env._agent_pose()), 'held_objects': list(env._held_objects()), 'objects': objects}

def check_init_predicates(env: BehaviorClosedLoopToolEnv, task_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for entry in task_spec.get('init') or []:
        predicate = str(entry.get('predicate'))
        args = list(entry.get('args') or [])
        record = {'predicate': predicate, 'args': args, 'satisfied': False, 'actual': None}
        if not args:
            record['actual'] = 'missing_args'
            checks.append(record)
            continue
        obj = _role_object(env, args[0])
        if obj is None:
            record['actual'] = 'missing_object'
            checks.append(record)
            continue
        states = getattr(obj, 'states', {}) or {}
        if predicate == 'open':
            record.update(satisfied=states.get('open') is True, actual=states.get('open'))
        elif predicate == 'closed':
            record.update(satisfied=states.get('open') is False, actual=states.get('open'))
        elif predicate == 'toggled_on':
            record.update(satisfied=states.get('toggled_on') is True, actual=states.get('toggled_on'))
        elif predicate == 'toggled_off':
            record.update(satisfied=states.get('toggled_on') is False, actual=states.get('toggled_on'))
        elif predicate == 'dirty':
            record.update(satisfied=states.get('dirty') is True, actual=states.get('dirty'))
        elif predicate == 'clean':
            record.update(satisfied=states.get('dirty') is False, actual=states.get('dirty'))
        elif predicate == 'sliced':
            record.update(satisfied=states.get('sliced') is True, actual=states.get('sliced'))
        elif predicate == 'cooled':
            record.update(satisfied=states.get('cooled') is True, actual=states.get('cooled'))
        elif predicate in {'filled', 'saturated'}:
            substance = args[1] if len(args) > 1 else 'water'
            actual = states.get(predicate) or {}
            record.update(satisfied=bool(isinstance(actual, Mapping) and actual.get(substance)), actual=actual)
        elif predicate in {'ontop', 'inside'}:
            target = _role_object(env, args[1]) if len(args) > 1 else None
            if target is None:
                record['actual'] = 'missing_target'
            elif predicate == 'ontop':
                record.update(satisfied=bool(_eval_ontop(env.core, [obj, target])), actual=_relation_actual(obj, target))
            else:
                record.update(satisfied=bool(_eval_inside(env.core, [obj, target])), actual=_relation_actual(obj, target))
        elif predicate == 'inroom':
            record.update(satisfied=True, actual='room_membership_deferred')
        else:
            record['actual'] = 'unsupported_init_predicate'
        checks.append(record)
    return checks

def check_deferred_init_assertions(env: BehaviorClosedLoopToolEnv, task_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    assertions = (task_spec.get('generation') or {}).get('deferred_init_assertions') or []
    for assertion in assertions:
        atype = str(assertion.get('type'))
        args = dict(assertion.get('args') or {})
        record = {'assertion': atype, 'args': args, 'satisfied': False, 'actual': None}
        if atype != 'object_not_in_receptacle':
            record['actual'] = 'unsupported_assertion'
            checks.append(record)
            continue
        obj = _role_object(env, args.get('object'))
        rec = _role_object(env, args.get('receptacle'))
        if obj is None or rec is None:
            record['actual'] = 'missing_object_or_receptacle'
        else:
            inside = bool(_eval_inside(env.core, [obj, rec]))
            ontop = bool(_eval_ontop(env.core, [obj, rec]))
            relation = str(assertion.get('relation') or '')
            if relation == 'inside':
                satisfied = not inside
            elif relation == 'ontop':
                satisfied = not ontop
            else:
                satisfied = not inside and (not ontop)
            record.update(satisfied=satisfied, actual={'inside': inside, 'ontop': ontop, 'relation': relation or 'any'})
        checks.append(record)
    return checks

def validate_geometry(env: BehaviorClosedLoopToolEnv, task_spec: Mapping[str, Any]) -> dict[str, Any]:
    bounds = (task_spec.get('source') or {}).get('occupancy_bounds') or [-100.0, -100.0, 100.0, 100.0]
    min_x, min_y, max_x, max_y = [float(v) for v in bounds]
    issues: list[dict[str, Any]] = []
    objects = list(env.core.objects.values())
    role_object_names = {str(name) for name in env.role_map.values()}
    relation_pairs = spatial_relation_pairs(task_spec, env)
    role_by_name = roles_by_object_name(task_spec, env)
    forbidden_pairs = forbidden_not_in_receptacle_pairs(task_spec, env)
    for obj in objects:
        if not _finite_vector(obj.center) or not _finite_vector(obj.extent):
            issues.append({'type': 'non_finite_geometry', 'object': obj.name, 'center': obj.center, 'extent': obj.extent})
        if any((float(v) <= 0.0 for v in obj.extent)):
            issues.append({'type': 'non_positive_extent', 'object': obj.name, 'extent': obj.extent})
        if not (min_x - 0.5 <= obj.center[0] <= max_x + 0.5 and min_y - 0.5 <= obj.center[1] <= max_y + 0.5):
            issues.append({'type': 'outside_occupancy_bounds', 'object': obj.name, 'center': _round_vector(obj.center), 'bounds': bounds})
    overlaps = []
    background_overlaps = []
    for idx, a in enumerate(objects):
        if _skip_geometry_object(a):
            continue
        for b in objects[idx + 1:]:
            if _skip_geometry_object(b):
                continue
            pair = frozenset((a.name, b.name))
            if pair in relation_pairs:
                continue
            if _allowed_fixture_overlap(a, b):
                continue
            if _allowed_support_overlap(a, b, role_by_name, forbidden_pairs):
                continue
            if _aabb_overlap(a, b, tolerance=0.01):
                row = {'a': a.name, 'b': b.name, 'a_category': a.category, 'b_category': b.category}
                if str(a.name) in role_object_names and str(b.name) in role_object_names:
                    overlaps.append(row)
                else:
                    background_overlaps.append(row)
    issues.extend(({'type': 'aabb_overlap', **item} for item in overlaps))
    return {'ok': not issues, 'issue_count': len(issues), 'issues': issues[:50], 'overlap_count': len(overlaps), 'background_overlap_count': len(background_overlaps), 'background_overlaps': background_overlaps[:50], 'bounds': bounds, 'object_count': len(objects)}

def validate_reachability(env: BehaviorClosedLoopToolEnv, task_spec: Mapping[str, Any]) -> dict[str, Any]:
    if env.core.occupancy is None:
        env.core.build_occupancy('stage5_reachability')
    contained_roles = roles_contained_at_reset(task_spec)
    key_roles = key_roles_for_validation(task_spec)
    probes = []
    issues = []
    interaction_roles = interaction_required_roles(env, task_spec)
    for role_name, obj_name in sorted(env.role_map.items()):
        if key_roles and role_name not in key_roles:
            continue
        role = (task_spec.get('roles') or {}).get(role_name) or {}
        obj = env.core.objects.get(obj_name)
        category = str(role.get('category') or role.get('object_type') or getattr(obj, 'category', '')).lower()
        if category in SKIP_REACHABILITY_CATEGORIES:
            continue
        if role_name in contained_roles:
            probes.append({'role': role_name, 'object': obj_name, 'status': 'skipped_contained_init_role'})
            continue
        if obj is None:
            issues.append({'role': role_name, 'object': obj_name, 'reason': 'missing_object'})
            continue
        plan = plan_navigation(env.core.ego_state.pose.copy(), NavigationTarget.from_aabb_object(obj), env.core.occupancy, env.core.config.navigation)
        plan_reason = None if plan.success else plan.check.reasons[0] if plan.check.reasons else 'target_not_reachable'
        record = {'role': role_name, 'object': obj_name, 'success': bool(plan.success), 'reason': plan_reason, 'selected_goal_xy': _round_vector(plan.selected_goal_xy), 'rejected_candidates': len(plan.rejected_candidates)}
        probes.append(record)
        if role_name in interaction_roles:
            opening_probe = _directional_openable_reachability(env, obj)
            record['interaction_required'] = True
            record['base_navigation'] = {'success': bool(plan.success), 'reason': plan_reason}
            record['interaction_face_probe'] = opening_probe
            record['directional_openable'] = opening_probe
            if opening_probe.get('success'):
                record['success'] = True
                record['reason'] = None
                selected = opening_probe.get('selected') or {}
                if selected.get('selected_goal_xy') is not None:
                    record['selected_goal_xy'] = selected['selected_goal_xy']
            else:
                issue = {**record, 'success': False, 'reason': 'interaction_face_unreachable'}
                issues.append(issue)
        else:
            record['interaction_required'] = False
            if not plan.success:
                issues.append(record)
    sequence_probe = validate_sequential_interaction_reachability(env, task_spec, key_roles=key_roles, interaction_roles=interaction_roles)
    if not sequence_probe.get('ok'):
        issues.extend(sequence_probe.get('issues') or [])
    return {'ok': not issues, 'probe_count': len(probes), 'probes': probes, 'issues': issues, 'key_roles': sorted(key_roles), 'interaction_required_roles': sorted(interaction_roles), 'sequential': sequence_probe}

def interaction_required_roles(env: BehaviorClosedLoopToolEnv, task_spec: Mapping[str, Any]) -> set[str]:
    required: set[str] = set()
    for role_name in key_roles_for_validation(task_spec):
        obj_name = env.role_map.get(str(role_name))
        obj = env.core.objects.get(obj_name) if obj_name and env.core is not None else None
        role = (task_spec.get('roles') or {}).get(role_name) or {}
        category = str(role.get('category') or role.get('object_type') or getattr(obj, 'category', '')).lower()
        if obj is None:
            continue
        states = getattr(obj, 'states', {}) or {}
        explicit_faces = states.get('interaction_faces') if isinstance(states, Mapping) else None
        if _has_interaction_faces(explicit_faces) or _has_interaction_faces(role.get('interaction_faces')):
            required.add(str(role_name))
            continue
        abilities = {str(value).lower() for value in getattr(obj, 'abilities', ()) or []}
        abilities.update((str(value).lower() for value in role.get('abilities') or []))
        if category in DIRECTIONAL_OPENABLE_CATEGORIES or abilities.intersection(DIRECTIONAL_INTERACTION_ABILITIES):
            required.add(str(role_name))
            continue
        if isinstance(states, Mapping) and 'open' in states:
            required.add(str(role_name))
    return required

def validate_sequential_interaction_reachability(env: BehaviorClosedLoopToolEnv, task_spec: Mapping[str, Any], *, key_roles: set[str] | None=None, interaction_roles: set[str] | None=None) -> dict[str, Any]:
    if env.core is None or env.core.occupancy is None:
        return {'ok': False, 'roles': [], 'issues': [{'reason': 'occupancy_unavailable'}]}
    key_roles = set(key_roles if key_roles is not None else key_roles_for_validation(task_spec))
    interaction_roles = set(interaction_roles if interaction_roles is not None else interaction_required_roles(env, task_spec))
    ordered = ordered_navigation_roles(task_spec, key_roles)
    contained = roles_contained_at_reset(task_spec)
    current_pose = env.core.ego_state.pose.copy()
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for role_name in ordered:
        if role_name in contained:
            records.append({'role': role_name, 'status': 'skipped_contained_init_role'})
            continue
        object_name = env.role_map.get(role_name)
        obj = env.core.objects.get(object_name) if object_name else None
        if obj is None:
            issues.append({'role': role_name, 'reason': 'missing_object'})
            continue
        if role_name in interaction_roles:
            probe = _directional_reachability_from_pose(env, obj, current_pose)
        else:
            plan = plan_navigation(current_pose, NavigationTarget.from_aabb_object(obj), env.core.occupancy, env.core.config.navigation)
            probe = {'success': bool(plan.success), 'selected_goal_xy': _round_vector(plan.selected_goal_xy), 'rejected_candidates': len(plan.rejected_candidates), 'reasons': list(plan.check.reasons or []) if not plan.success else [], 'face_attempts': []}
            if plan.success and plan.frame_poses:
                current_pose = plan.frame_poses[-1].copy()
        record = {'role': role_name, 'object': object_name, 'directional': role_name in interaction_roles, **probe}
        records.append(record)
        if not probe.get('success'):
            issues.append({**record, 'reason': 'sequential_interaction_unreachable'})
            continue
        if role_name in interaction_roles and probe.get('selected_pose') is not None:
            current_pose = _pose_from_dict(probe['selected_pose'])
    return {'ok': not issues, 'roles': records, 'issues': issues, 'ordered_roles': ordered}

def ordered_navigation_roles(task_spec: Mapping[str, Any], key_roles: set[str]) -> list[str]:
    ordered: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value in key_roles and (value not in ordered):
            ordered.append(value)
    for entry in task_spec.get('init') or []:
        for value in entry.get('args') or []:
            add(value)
    for predicate in (task_spec.get('goal') or {}).get('predicates') or []:
        args = predicate.get('args') or {}
        for value in args.values() if isinstance(args, Mapping) else args:
            add(value)
    primary = ((task_spec.get('family_spec') or {}).get('mid_execution') or {}).get('primary_task') or {}
    if isinstance(primary, Mapping):
        for key in ('object_role', 'sink_role', 'faucet_role', 'staging_receptacle_role', 'receptacle_role'):
            add(primary.get(key))
        secondary = primary.get('secondary_task') or {}
        if isinstance(secondary, Mapping):
            add(secondary.get('object_role'))
            add(secondary.get('receptacle_role'))
    ordered.extend(sorted(key_roles - set(ordered)))
    return ordered

def _directional_openable_reachability(env: BehaviorClosedLoopToolEnv, obj: Any) -> dict[str, Any]:
    return _directional_reachability_from_pose(env, obj, env.core.ego_state.pose.copy())

def _directional_reachability_from_pose(env: BehaviorClosedLoopToolEnv, obj: Any, start_pose: Any) -> dict[str, Any]:
    target = NavigationTarget.from_aabb_object(obj)
    faces = list(target.interaction_faces)
    if faces and all((face.source == 'aabb_openable_face_fallback' for face in faces)):
        faces = _root_yaw_opening_faces(obj) or faces
    attempts: list[dict[str, Any]] = []
    for face in faces:
        directional_target = NavigationTarget(name=target.name, category=target.category, center=target.center, extent=target.extent, interaction_faces=(face,))
        try:
            plan = plan_navigation(start_pose.copy(), directional_target, env.core.occupancy, env.core.config.navigation)
        except Exception as exc:
            attempts.append({'face_source': face.source, 'success': False, 'error': repr(exc)})
            continue
        attempt = {'face_source': face.source, 'face_center': _round_vector(face.center), 'face_normal_xy': _round_vector(face.normal_xy), 'success': bool(plan.success), 'selected_goal_xy': _round_vector(plan.selected_goal_xy), 'rejected_candidates': len(plan.rejected_candidates), 'reasons': list(plan.check.reasons or []) if not plan.success else []}
        attempts.append(attempt)
        if plan.success:
            selected_pose = plan.frame_poses[-1].as_dict() if plan.frame_poses else None
            return {'success': True, 'attempts': attempts, 'selected': attempt, 'selected_pose': selected_pose}
    return {'success': False, 'attempts': attempts, 'selected': None, 'selected_pose': None}

def _has_interaction_faces(value: Any) -> bool:
    return isinstance(value, Sequence) and (not isinstance(value, (str, bytes))) and bool(value)

def _pose_from_dict(value: Mapping[str, Any]) -> EgoPose:
    return EgoPose(x=float(value.get('x', 0.0)), y=float(value.get('y', 0.0)), z=float(value.get('z', 1.55)), yaw=float(value.get('yaw', 0.0)), pitch=float(value.get('pitch', 0.0)))

def _root_yaw_opening_faces(obj: Any) -> list[NavigationFace]:
    yaw = getattr(obj, 'root_yaw', None)
    if yaw is None:
        return []
    try:
        yaw = float(yaw)
        center = tuple((float(value) for value in obj.center))
        extent = tuple((float(value) for value in obj.extent))
    except Exception:
        return []
    normal = (-math.sin(yaw), math.cos(yaw))
    half_depth = 0.5 * (abs(normal[0]) * extent[0] + abs(normal[1]) * extent[1])
    return [NavigationFace((center[0] + sign * normal[0] * half_depth, center[1] + sign * normal[1] * half_depth, center[2]), (sign * normal[0], sign * normal[1]), 'root_yaw_opening_side') for sign in (1.0, -1.0)]

def validate_observation_conditioned_room_constraints(env: BehaviorClosedLoopToolEnv, task_spec: Mapping[str, Any]) -> dict[str, Any]:
    family = task_spec.get('family_spec') or {}
    obs = family.get('observation_conditioned') if isinstance(family, Mapping) else None
    if not isinstance(obs, Mapping):
        return {'ok': True, 'checks': [], 'issues': [], 'skipped': True}
    constraints: dict[str, str] = {}
    raw_constraints = obs.get('latent_role_room_constraints') or (task_spec.get('generation') or {}).get('latent_role_room_constraints') or {}
    if isinstance(raw_constraints, Mapping):
        constraints.update({str(role): str(room) for role, room in raw_constraints.items() if room})
    for payload in obs.get('latent_payloads') or []:
        if not isinstance(payload, Mapping):
            continue
        room_id = str(payload.get('room_id') or '')
        object_role = payload.get('object_role')
        if object_role and room_id:
            constraints.setdefault(str(object_role), room_id)
        support_roles = payload.get('support_roles')
        if isinstance(support_roles, Mapping) and room_id:
            for role in support_roles.values():
                if role:
                    constraints.setdefault(str(role), room_id)
    checks = []
    issues = []
    for role_name, expected_room in sorted(constraints.items()):
        role = (task_spec.get('roles') or {}).get(role_name) or {}
        obj = _role_object(env, role_name)
        states = getattr(obj, 'states', {}) or {}
        actual_room = str(role.get('room_id') or states.get('room_id') or '')
        placement = role.get('spawn_placement') if isinstance(role.get('spawn_placement'), Mapping) else {}
        placement_room = str(placement.get('room_id') or '')
        satisfied = obj is not None and actual_room == expected_room and (not placement_room or placement_room == expected_room)
        check = {'role': role_name, 'expected_room': expected_room, 'actual_role_room': actual_room, 'spawn_placement_room': placement_room or None, 'object_present': obj is not None, 'center': _round_vector(getattr(obj, 'center', None)), 'satisfied': bool(satisfied)}
        checks.append(check)
        if not satisfied:
            issues.append({'type': 'observation_conditioned_room_constraint_failed', **check})
    return {'ok': not issues, 'checks': checks, 'issues': issues, 'skipped': False}

def validate_mutation_plan(task_spec: Mapping[str, Any], *, rng_seed: int=0, scene_layout_mode: str='task_roles', load_doors: bool=False) -> dict[str, Any]:
    unsupported = list((task_spec.get('generation') or {}).get('unsupported_scheduler_specs') or [])
    disturbance_specs, failure_specs, scheduler_seed = _stage5_scheduler_specs(task_spec, rng_seed=rng_seed)
    deferred_codes = []
    replay: list[dict[str, Any]] = []
    diff: list[dict[str, Any]] = []
    spec_errors = mutation_spec_errors(task_spec)
    if unsupported:
        deferred_codes.append('UNSUPPORTED_SCHEDULER_SPEC')
        replay.append({'status': 'deferred', 'reason': 'unsupported_scheduler_specs', 'specs': unsupported})
    for item in disturbance_specs:
        activation = item.get('activation')
        if isinstance(activation, Mapping) and activation.get('type') not in {None, 'after_tool_result', 'high_level_step'}:
            deferred_codes.append('DISTURBANCE_ACTIVATION_MODEL_DEFERRED')
    has_supported_scheduler = bool(disturbance_specs or failure_specs)
    if spec_errors:
        return {'spec_ok': False, 'replay_ok': False, 'diff': diff, 'replay': replay, 'deferred_codes': sorted(set(deferred_codes)), 'errors': spec_errors}
    if not has_supported_scheduler:
        return {'spec_ok': True, 'replay_ok': True, 'diff': diff, 'replay': replay or [{'status': 'not_applicable', 'reason': 'no_supported_scheduler_spec'}], 'deferred_codes': sorted(set(deferred_codes))}
    try:
        if disturbance_specs:
            dist = replay_disturbances(task_spec, rng_seed=rng_seed, disturbance_spec=disturbance_specs, scheduler_seed=scheduler_seed, scene_layout_mode=scene_layout_mode, load_doors=load_doors)
            replay.extend(dist['replay'])
            diff.extend(dist['diff'])
        if failure_specs:
            fail = replay_failures(task_spec, rng_seed=rng_seed, failure_spec=failure_specs, scheduler_seed=scheduler_seed, scene_layout_mode=scene_layout_mode, load_doors=load_doors)
            replay.extend(fail['replay'])
            diff.extend(fail['diff'])
    except Exception as exc:
        replay.append({'status': 'failed', 'error': str(exc)})
        return {'spec_ok': not unsupported, 'replay_ok': False, 'diff': diff, 'replay': replay, 'deferred_codes': sorted(set(deferred_codes))}
    replay_ok = any((item.get('events') or item.get('transition') for item in replay)) or not has_supported_scheduler
    return {'spec_ok': True, 'replay_ok': bool(replay_ok), 'diff': diff, 'replay': replay, 'deferred_codes': sorted(set(deferred_codes))}

def mutation_spec_errors(task_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    role_names = set((task_spec.get('roles') or {}).keys())
    errors: list[dict[str, Any]] = []
    disturbance_specs, failure_specs, _seed = _stage5_scheduler_specs(task_spec)
    for entry in disturbance_specs:
        dtype = entry.get('type')
        if dtype not in KNOWN_DISTURBANCE_TYPES:
            errors.append({'type': 'unknown_disturbance_type', 'value': dtype})
        target = entry.get('target_role')
        target_object_id = entry.get('target_object_id')
        if not target and (not target_object_id):
            errors.append({'type': 'missing_disturbance_target'})
        elif target and target not in role_names:
            errors.append({'type': 'unknown_disturbance_target_role', 'value': target})
        extra = entry.get('extra') or {}
        for role in extra.get('exclude_roles') or []:
            if role not in role_names:
                errors.append({'type': 'unknown_disturbance_exclude_role', 'value': role})
    for entry in failure_specs:
        mode = entry.get('mode')
        if mode not in KNOWN_FAILURE_MODES:
            errors.append({'type': 'unknown_failure_mode', 'value': mode})
        target = entry.get('target_role')
        if target is not None and target not in role_names:
            errors.append({'type': 'unknown_failure_target_role', 'value': target})
    return errors

def replay_disturbances(task_spec: Mapping[str, Any], *, rng_seed: int, disturbance_spec: Sequence[Mapping[str, Any]] | None=None, scheduler_seed: int | None=None, scene_layout_mode: str='task_roles', load_doors: bool=False) -> dict[str, Any]:
    env = BehaviorClosedLoopToolEnv(scene_layout_mode=scene_layout_mode, load_doors=load_doors)
    env.reset_to_task(task_spec, rng_seed=rng_seed)
    before = reset_snapshot(env)
    if disturbance_spec is None:
        disturbance_spec, _failure_spec, normalized_seed = _stage5_scheduler_specs(task_spec, rng_seed=rng_seed)
        if scheduler_seed is None:
            scheduler_seed = normalized_seed
    disturbance_spec = [dict(item) for item in disturbance_spec or [] if isinstance(item, Mapping)]
    scheduler = DisturbanceScheduler(disturbance_spec, seed=rng_seed if scheduler_seed is None else scheduler_seed)
    scheduler.on_episode_start(env, task_spec)
    replay: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    fixed_step_specs = [item for item in disturbance_spec if _is_fixed_step_disturbance_spec(item)]
    if fixed_step_specs:
        for item in fixed_step_specs:
            _prime_disturbance_target_state(env, item)
        max_after = max((int(item.get('after_step', 0) or 0) for item in fixed_step_specs))
        fixed_events = []
        for step in range(1, max_after + 2):
            for item in fixed_step_specs:
                _set_stage5_composite_subtask_gate(env, item)
                if item.get('requires_out_of_view') and step > int(item.get('after_step', 0) or 0):
                    _orient_for_out_of_view_gate(env, item, visible=True)
                    wait = scheduler.pre_step(env, step, None)
                    if wait:
                        fixed_events.extend(_event_dicts(wait))
                    _orient_for_out_of_view_gate(env, item, visible=False)
            fired = scheduler.pre_step(env, step, None)
            if fired:
                fixed_events.extend(_event_dicts(fired))
        events.extend(fixed_events)
        replay.append({'kind': 'disturbance', 'status': 'replayed', 'trigger': 'pre_step', 'events': fixed_events})
    for item in disturbance_spec:
        activation = item.get('activation')
        if not isinstance(activation, Mapping) or activation.get('type') != 'after_tool_result':
            continue
        _set_stage5_composite_subtask_gate(env, item)
        transition = _activation_transition(env, activation, item)
        _prime_disturbance_target_state(env, item)
        required_parent_role = activation.get('required_parent_role')
        if isinstance(required_parent_role, str) and required_parent_role:
            _place_disturbance_target_on_parent(env, item, required_parent_role)
        if item.get('requires_out_of_view'):
            _orient_for_out_of_view_gate(env, item, visible=False)
        activation_events = _activation_replay_events(env, scheduler, item, transition)
        events.extend(activation_events)
        replay.append({'kind': 'disturbance', 'status': 'replayed', 'trigger': 'pre_step_after_activation', 'activation': dict(activation), 'transition': _jsonable_sorted(transition), 'events': activation_events})
    after = reset_snapshot(env)
    return {'replay': replay or [{'kind': 'disturbance', 'status': 'replayed', 'events': events}], 'diff': object_snapshot_diff(before, after)}

def _prime_disturbance_target_state(env: BehaviorClosedLoopToolEnv, disturbance: Mapping[str, Any]) -> None:
    obj_name = _disturbance_target_name(env, disturbance)
    if not obj_name:
        return
    obj = env.core.objects.get(obj_name)
    if obj is None:
        return
    dtype = str(disturbance.get('type') or '')
    if dtype == 'close_object':
        _set_state_for_replay(obj, 'open', True, actions.set_open_state)
    elif dtype == 'open_object':
        _set_state_for_replay(obj, 'open', False, actions.set_open_state)
    elif dtype == 'toggle_on':
        _set_state_for_replay(obj, 'toggled_on', False, actions.set_toggle_state)
    elif dtype == 'toggle_off':
        _set_state_for_replay(obj, 'toggled_on', True, actions.set_toggle_state)
    elif dtype == 'dirty_object':
        _set_state_for_replay(obj, 'dirty', False, actions.set_dirty_state)
    elif dtype == 'restore_initial_state':
        states = getattr(obj, 'states', None)
        if isinstance(states, dict):
            if 'open' in states:
                states['open'] = not bool(states['open'])
            elif 'toggled_on' in states:
                states['toggled_on'] = not bool(states['toggled_on'])

def _set_state_for_replay(obj: Any, state_key: str, value: bool, setter) -> None:
    try:
        setter(obj, value)
    except Exception:
        states = getattr(obj, 'states', None)
        if isinstance(states, dict):
            states[state_key] = bool(value)

def _activation_transition(env: BehaviorClosedLoopToolEnv, activation: Mapping[str, Any], disturbance: Mapping[str, Any]) -> dict[str, Any]:
    tools = activation.get('tools')
    tool_name = str((tools[0] if isinstance(tools, list) and tools else None) or activation.get('tool') or 'scan')
    target_ref = activation.get('target_role') or disturbance.get('target_role') or disturbance.get('target_object_id') or ''
    target_ref = str(target_ref) if target_ref else ''
    if activation.get('requires_target_not_held'):
        disturbed_obj = _disturbance_target_name(env, disturbance)
        if getattr(env.core.ego_state, 'held_object_name', None) == disturbed_obj:
            env.core.ego_state.held_object_name = None
            env.core.ego_state.held_visual_name = None
    high_level_step = max(len(env.tool_history) + 1, int(disturbance.get('after_step', 0) or 0) + 1)
    transition = {'tool_name': tool_name, 'arguments': {'target': target_ref} if target_ref else {}, 'reasoning': 'stage5 disturbance activation dry-run', 'success': bool(activation.get('success', True)), 'feedback': 'stage5 dry-run activation transition', 'frames': 0, 'num_low_level_steps': 0, 'env_feedback': {'error_type': None, 'error_message': None}, 'high_level_step': high_level_step, 'agent_pose': env._agent_pose(), 'held_objects': env._held_objects(), 'goal_result': env.check_goal(model_stopped=False), 'done': False, 'done_reason': None}
    env.tool_history.append(transition)
    env.high_level_step = max(env.high_level_step, high_level_step)
    return transition

def _activation_replay_events(env: BehaviorClosedLoopToolEnv, scheduler: DisturbanceScheduler, disturbance: Mapping[str, Any], activation_transition: Mapping[str, Any]) -> list[dict[str, Any]]:
    activation = disturbance.get('activation') if isinstance(disturbance.get('activation'), Mapping) else {}
    armed_step = int(activation_transition.get('high_level_step') or 1)
    min_delay = int(activation.get('min_delay_steps', 0) or 0)
    required_count = int(activation.get('after_distinct_tool_count', activation.get('distinct_tool_count', 0)) or 0)
    events: list[dict[str, Any]] = []
    last_transition: Mapping[str, Any] = activation_transition
    max_step = armed_step + max(min_delay, required_count) + 3
    for step in range(armed_step + 1, max_step + 1):
        if disturbance.get('requires_out_of_view'):
            _orient_for_out_of_view_gate(env, disturbance, visible=False)
        fired = scheduler.pre_step(env, step, last_transition)
        event_dicts = _event_dicts(fired)
        events.extend(event_dicts)
        if any((item.get('name') != 'disturbance_out_of_view_wait' for item in event_dicts)):
            break
        synthetic = _synthetic_distinct_transition(env, activation, step)
        env.tool_history.append(synthetic)
        env.high_level_step = max(env.high_level_step, int(synthetic['high_level_step']))
        last_transition = synthetic
    return events

def _synthetic_distinct_transition(env: BehaviorClosedLoopToolEnv, activation: Mapping[str, Any], high_level_step: int) -> dict[str, Any]:
    ignore_tools = {str(tool) for tool in activation.get('ignore_tools', ['stop', 'answer']) or []}
    candidates = ['scan', 'look_for', 'navigate_to', 'open', 'close', 'toggle_on', 'toggle_off', 'pick_up']
    tool_name = next((tool for tool in candidates if tool not in ignore_tools), 'scan')
    transition = {'tool_name': tool_name, 'arguments': {}, 'reasoning': 'stage5 dry-run post-activation work', 'success': True, 'feedback': 'stage5 dry-run post-activation work', 'frames': 0, 'num_low_level_steps': 0, 'env_feedback': {'error_type': None, 'error_message': None}, 'high_level_step': int(high_level_step), 'agent_pose': env._agent_pose(), 'held_objects': env._held_objects(), 'goal_result': env.check_goal(model_stopped=False), 'done': False, 'done_reason': None}
    return transition

def _place_disturbance_target_on_parent(env: BehaviorClosedLoopToolEnv, disturbance: Mapping[str, Any], parent_role: str) -> None:
    target_name = _disturbance_target_name(env, disturbance)
    parent_name = env.role_map.get(str(parent_role), str(parent_role))
    obj = env.core.objects.get(target_name) if target_name else None
    parent = env.core.objects.get(parent_name) if parent_name else None
    if obj is None or parent is None:
        return
    try:
        obj.center = (float(parent.center[0]), float(parent.center[1]), float(parent.center[2] + parent.extent[2] * 0.5 + obj.extent[2] * 0.5 + 0.005))
        env.core.build_occupancy('stage5_required_parent_activation')
    except Exception:
        pass

def _seed_distinct_tool_history(env: BehaviorClosedLoopToolEnv, required_count: int, *, current_tool: str) -> None:
    if required_count <= 1:
        return
    candidates = ['scan', 'navigate_to', 'look_for', 'open', 'close', 'toggle_on', 'toggle_off', 'pick_up']
    existing = {str(item.get('tool_name')) for item in env.tool_history if item.get('tool_name')}
    existing.add(str(current_tool))
    for name in candidates:
        if len(existing) >= required_count:
            break
        if name in existing:
            continue
        synthetic = {'tool_name': name, 'arguments': {}, 'reasoning': 'stage5 dry-run distinct-tool seed', 'success': True, 'feedback': 'stage5 dry-run history seed', 'frames': 0, 'num_low_level_steps': 0, 'env_feedback': {'error_type': None, 'error_message': None}, 'high_level_step': len(env.tool_history) + 1, 'agent_pose': env._agent_pose(), 'held_objects': env._held_objects(), 'goal_result': env.check_goal(model_stopped=False), 'done': False, 'done_reason': None}
        env.tool_history.append(synthetic)
        env.high_level_step = max(env.high_level_step, int(synthetic['high_level_step']))
        existing.add(name)

def _orient_for_out_of_view_gate(env: BehaviorClosedLoopToolEnv, disturbance: Mapping[str, Any], *, visible: bool) -> None:
    target_name = _disturbance_target_name(env, disturbance)
    if not target_name or target_name not in env.core.objects:
        return
    if visible:
        env.core.look_for(target_name, trace_id='stage5_out_of_view_visible_probe')
    else:
        pose = env.core.ego_state.pose
        pose.yaw = _normalize_angle(float(getattr(pose, 'yaw', 0.0)) + math.pi)

def replay_failures(task_spec: Mapping[str, Any], *, rng_seed: int, failure_spec: Sequence[Mapping[str, Any]] | None=None, scheduler_seed: int | None=None, scene_layout_mode: str='task_roles', load_doors: bool=False) -> dict[str, Any]:
    env = BehaviorClosedLoopToolEnv(scene_layout_mode=scene_layout_mode, load_doors=load_doors)
    env.reset_to_task(task_spec, rng_seed=rng_seed)
    before = reset_snapshot(env)
    if failure_spec is None:
        _disturbance_spec, failure_spec, normalized_seed = _stage5_scheduler_specs(task_spec, rng_seed=rng_seed)
        if scheduler_seed is None:
            scheduler_seed = normalized_seed
    failure_spec = [dict(item) for item in failure_spec or [] if isinstance(item, Mapping)]
    scheduler = FailureInjectionScheduler(failure_spec, seed=rng_seed if scheduler_seed is None else scheduler_seed)
    replay = []
    transitions = []
    for entry in failure_spec:
        _set_stage5_composite_subtask_gate(env, entry)
        tool = str(entry.get('tool') or _default_tool_for_failure(entry.get('mode')))
        target_role = entry.get('target_role') or _first_pickupable_role(task_spec) or next(iter((task_spec.get('roles') or {}).keys()))
        if entry.get('mode') in {'grasp_slip', 'drop_beside'}:
            held_role = _first_pickupable_role(task_spec)
            if held_role in env.role_map:
                env.core.ego_state.held_object_name = env.role_map[held_role]
                env.core.ego_state.held_visual_name = env.role_map[held_role]
        call = ToolCall(tool, {'target': target_role} if tool not in {'stop', 'drop'} else {}, reasoning='stage5 failure dry-run')
        high_level_step = int(entry.get('after_step', 0) or 0) + 1

        def _default_call() -> dict[str, Any]:
            return {'tool_name': call.tool_name, 'arguments': dict(call.arguments), 'success': True, 'feedback': 'stage5 dry-run default transition', 'env_feedback': {'error_type': None, 'error_message': None}, 'high_level_step': high_level_step, 'agent_pose': env._agent_pose(), 'held_objects': env._held_objects(), 'goal_result': env.check_goal(model_stopped=False), 'done': False, 'done_reason': None}
        transition = scheduler.intercept_tool_call(env, call, high_level_step, _default_call)
        transitions.append(_jsonable_sorted(transition))
        replay.append({'kind': 'failure', 'status': 'replayed', 'spec': dict(entry), 'transition': _jsonable_sorted(transition)})
    after = reset_snapshot(env)
    return {'replay': replay, 'diff': object_snapshot_diff(before, after), 'transitions': transitions}

def object_snapshot_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    before_objects = before.get('objects') or {}
    after_objects = after.get('objects') or {}
    for name in sorted(set(before_objects) | set(after_objects)):
        b = before_objects.get(name)
        a = after_objects.get(name)
        if b is None or a is None:
            diffs.append({'object': name, 'field': 'presence', 'before': b is not None, 'after': a is not None})
            continue
        for field in ('center', 'states'):
            if b.get(field) != a.get(field):
                diffs.append({'object': name, 'field': field, 'before': b.get(field), 'after': a.get(field)})
    if before.get('agent_pose') != after.get('agent_pose'):
        diffs.append({'object': '__agent__', 'field': 'pose', 'before': before.get('agent_pose'), 'after': after.get('agent_pose')})
    if before.get('held_objects') != after.get('held_objects'):
        diffs.append({'object': '__agent__', 'field': 'held_objects', 'before': before.get('held_objects'), 'after': after.get('held_objects')})
    return diffs

def oracle_deferred_reasons(task_spec: Mapping[str, Any]) -> list[str]:
    debt = list((task_spec.get('generation') or {}).get('stage4_readiness_debt') or [])
    reasons = []
    for item in debt:
        if item in RESET_RUNTIME_DEBT_OK:
            continue
        if item in ORACLE_DEFER_DEBT_EXACT or any((str(item).startswith(prefix) for prefix in ORACLE_DEFER_DEBT_PREFIXES)):
            reasons.append(str(item))
    return sorted(set(reasons))

def summarize_stage5_results(generation: Mapping[str, Any], validation_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    generated = generation.get('tasks') or []
    rejected = generation.get('rejected') or []
    validated_total = len(validation_results)
    reset_mutation_passed = sum((1 for item in validation_results if item.get('stage5_reset_mutation_ok')))
    mutation_runtime_complete = sum((1 for item in validation_results if item.get('mutation_runtime_complete')))
    reachability_passed = sum((1 for item in validation_results if item.get('reachability_ok') is True))
    reachability_skipped = sum((1 for item in validation_results if item.get('reachability_skipped')))
    return {'generated_total': len(generated), 'rejected_total': len(rejected), 'validated_total': validated_total, 'reset_mutation_passed': reset_mutation_passed, 'reset_mutation_pass_ratio': float(reset_mutation_passed / validated_total) if validated_total else 0.0, 'structure_passed': sum((1 for item in validation_results if item.get('structure_ok'))), 'reset_passed': sum((1 for item in validation_results if item.get('reset_ok'))), 'deterministic_reset_passed': sum((1 for item in validation_results if item.get('deterministic_reset'))), 'init_passed': sum((1 for item in validation_results if item.get('init_ok'))), 'initial_goal_false_passed': sum((1 for item in validation_results if item.get('initial_goal_false'))), 'geometry_passed': sum((1 for item in validation_results if item.get('geometry_ok'))), 'reachability_passed': reachability_passed, 'reachability_skipped': reachability_skipped, 'mutation_spec_passed': sum((1 for item in validation_results if item.get('mutation_spec_ok'))), 'mutation_replay_passed': sum((1 for item in validation_results if item.get('mutation_replay_ok'))), 'mutation_runtime_complete': mutation_runtime_complete, 'mutation_runtime_complete_ratio': float(mutation_runtime_complete / validated_total) if validated_total else 0.0, 'stage5_buckets': dict(sorted(Counter((item.get('stage5_bucket') for item in validation_results)).items())), 'oracle_gates': dict(sorted(Counter((item.get('oracle_gate') for item in validation_results)).items())), 'deferred_codes': dict(sorted(Counter((code for item in validation_results for code in item.get('deferred_codes') or [])).items())), 'failure_codes': dict(sorted(Counter((item.get('failure_code') for item in validation_results if item.get('failure_code'))).items())), 'failed_by_template': dict(sorted(Counter((item.get('template_id') for item in validation_results if item.get('failure_code'))).items())), 'rejection_codes': (generation.get('summary') or {}).get('rejection_codes', {}), 'generated_by_template': (generation.get('summary') or {}).get('generated_by_template', {}), 'generated_by_scene': (generation.get('summary') or {}).get('generated_by_scene', {}), 'generated_by_readiness': (generation.get('summary') or {}).get('generated_by_readiness', {})}

def _role_object(env: BehaviorClosedLoopToolEnv, role: Any) -> Any | None:
    if role is None:
        return None
    obj_name = env.role_map.get(str(role), str(role))
    return env.core.objects.get(obj_name)

def _disturbance_target_name(env: BehaviorClosedLoopToolEnv, disturbance: Mapping[str, Any]) -> str | None:
    target_role = disturbance.get('target_role')
    if isinstance(target_role, str) and target_role:
        return env.role_map.get(target_role, target_role)
    target_object_id = disturbance.get('target_object_id')
    if isinstance(target_object_id, str) and target_object_id:
        return target_object_id
    return None

def _relation_actual(obj: Any, target: Any) -> dict[str, Any]:
    return {'object_center': _round_vector(obj.center), 'target_center': _round_vector(target.center), 'object_extent': _round_vector(obj.extent), 'target_extent': _round_vector(target.extent)}

def spatial_relation_pairs(task_spec: Mapping[str, Any], env: BehaviorClosedLoopToolEnv) -> set[frozenset[str]]:
    pairs = set()
    for entry in task_spec.get('init') or []:
        if entry.get('predicate') not in {'ontop', 'inside'}:
            continue
        args = list(entry.get('args') or [])
        if len(args) < 2:
            continue
        a = env.role_map.get(str(args[0]))
        b = env.role_map.get(str(args[1]))
        if a and b:
            pairs.add(frozenset((a, b)))
    return pairs

def roles_contained_at_reset(task_spec: Mapping[str, Any]) -> set[str]:
    return {str((entry.get('args') or [None])[0]) for entry in task_spec.get('init') or [] if entry.get('predicate') == 'inside' and entry.get('args')}

def key_roles_for_validation(task_spec: Mapping[str, Any]) -> set[str]:
    roles: set[str] = set()
    for entry in task_spec.get('init') or []:
        for arg in entry.get('args') or []:
            if isinstance(arg, str) and arg in (task_spec.get('roles') or {}):
                roles.add(arg)
    for predicate in (task_spec.get('goal') or {}).get('predicates') or []:
        for arg in (predicate.get('args') or {}).values():
            if isinstance(arg, str) and arg in (task_spec.get('roles') or {}):
                roles.add(arg)
    disturbance_specs, failure_specs, _seed = _stage5_scheduler_specs(task_spec)
    for entry in disturbance_specs:
        target = entry.get('target_role')
        if isinstance(target, str):
            roles.add(target)
        extra = entry.get('extra') or {}
        for role in extra.get('exclude_roles') or []:
            if isinstance(role, str):
                roles.add(role)
    for entry in failure_specs:
        target = entry.get('target_role')
        if isinstance(target, str):
            roles.add(target)
    primary = ((task_spec.get('family_spec') or {}).get('mid_execution') or {}).get('primary_task') or {}
    if isinstance(primary, Mapping):
        for key in ('faucet_role', 'object_role', 'receptacle_role', 'sink_role', 'staging_receptacle_role'):
            role = primary.get(key)
            if isinstance(role, str) and role in (task_spec.get('roles') or {}):
                roles.add(role)
        secondary = primary.get('secondary_task') or {}
        if isinstance(secondary, Mapping):
            for key in ('object_role', 'receptacle_role'):
                role = secondary.get(key)
                if isinstance(role, str) and role in (task_spec.get('roles') or {}):
                    roles.add(role)
    return roles

def _set_stage5_composite_subtask_gate(env: Any, spec: Mapping[str, Any]) -> None:
    activation = spec.get('activation') if isinstance(spec.get('activation'), Mapping) else {}
    subtask_id = activation.get('subtask_id') or activation.get('owner_subtask_id') or spec.get('owner_subtask_id')
    if subtask_id is not None:
        setattr(env, 'current_composite_subtask_id', str(subtask_id))

def _is_fixed_step_disturbance_spec(spec: Mapping[str, Any]) -> bool:
    activation = spec.get('activation')
    if not isinstance(activation, Mapping):
        return True
    return activation.get('type') in {None, 'high_level_step'}

def _skip_geometry_object(obj: Any) -> bool:
    cat = str(getattr(obj, 'category', '')).lower()
    return cat in SKIP_REACHABILITY_CATEGORIES

def _aabb_overlap(a: Any, b: Any, *, tolerance: float) -> bool:
    for axis in range(3):
        delta = abs(float(a.center[axis]) - float(b.center[axis]))
        limit = float(a.extent[axis]) * 0.5 + float(b.extent[axis]) * 0.5 - tolerance
        if delta >= limit:
            return False
    return True

def _allowed_fixture_overlap(a: Any, b: Any) -> bool:
    cats = {str(getattr(a, 'category', '')).lower(), str(getattr(b, 'category', '')).lower()}
    joined = ' '.join(sorted(cats))
    if 'faucet' in joined and 'sink' in joined:
        return True
    return False

def _allowed_support_overlap(a: Any, b: Any, role_by_name: Mapping[str, Mapping[str, Any]], forbidden_pairs: set[frozenset[str]]) -> bool:
    pair = frozenset((str(getattr(a, 'name', '')), str(getattr(b, 'name', ''))))
    if pair in forbidden_pairs:
        return False
    return _support_overlap_one_way(a, b, role_by_name) or _support_overlap_one_way(b, a, role_by_name)

def _support_overlap_one_way(obj: Any, support: Any, role_by_name: Mapping[str, Mapping[str, Any]]) -> bool:
    if not _is_small_pickupable(obj) or not _is_support_like(support):
        return False
    if not _looks_supported_by(obj, support):
        return False
    obj_role = role_by_name.get(str(getattr(obj, 'name', ''))) or {}
    support_role = role_by_name.get(str(getattr(support, 'name', ''))) or {}
    spawn = obj_role.get('spawn_placement') if isinstance(obj_role.get('spawn_placement'), Mapping) else None
    if isinstance(spawn, Mapping):
        support_role_name = str(spawn.get('support_role') or '')
        if support_role_name and support_role_name == str(support_role.get('_role_name') or ''):
            return True
    return bool(support_role.get('auxiliary_spawn_support'))

def _is_small_pickupable(obj: Any) -> bool:
    try:
        abilities = {str(value) for value in getattr(obj, 'abilities', None) or []}
        return (bool(getattr(obj, 'pickupable', False)) or 'pickupable' in abilities) and max((float(v) for v in obj.extent)) <= 0.35
    except Exception:
        return False

def _is_support_like(obj: Any) -> bool:
    try:
        abilities = {str(value) for value in getattr(obj, 'abilities', None) or []}
        category = str(getattr(obj, 'category', '')).lower()
        return 'support_surface' in abilities or category in {'breakfast_table', 'coffee_table', 'countertop', 'desk', 'nightstand', 'stand', 'console_table', 'pedestal_table'}
    except Exception:
        return False

def _looks_supported_by(obj: Any, support: Any) -> bool:
    try:
        obj_extent = [float(v) for v in obj.extent]
        support_extent = [float(v) for v in support.extent]
        obj_center = [float(v) for v in obj.center]
        support_center = [float(v) for v in support.center]
    except Exception:
        return False
    if obj_extent[0] > support_extent[0] * 0.65 or obj_extent[1] > support_extent[1] * 0.65:
        return False
    if abs(obj_center[0] - support_center[0]) > support_extent[0] * 0.5 + 0.05:
        return False
    if abs(obj_center[1] - support_center[1]) > support_extent[1] * 0.5 + 0.05:
        return False
    support_top = support_center[2] + support_extent[2] * 0.5
    max_embed = max(0.08, min(0.3, support_extent[2] * 0.35))
    if obj_center[2] < support_top - max_embed:
        return False
    if obj_center[2] < support_center[2]:
        return False
    return True

def roles_by_object_name(task_spec: Mapping[str, Any], env: BehaviorClosedLoopToolEnv) -> dict[str, dict[str, Any]]:
    roles = task_spec.get('roles') or {}
    out: dict[str, dict[str, Any]] = {}
    for role_name, role in roles.items():
        if not isinstance(role, Mapping):
            continue
        object_name = env.role_map.get(str(role_name)) or str(role.get('bddl_inst') or role.get('object_id') or role_name)
        entry = dict(role)
        entry['_role_name'] = str(role_name)
        out[str(object_name)] = entry
    return out

def forbidden_not_in_receptacle_pairs(task_spec: Mapping[str, Any], env: BehaviorClosedLoopToolEnv) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for entry in _not_in_receptacle_entries(task_spec):
        args = entry.get('args') or {}
        if isinstance(args, Mapping):
            obj_role = args.get('object')
            rec_role = args.get('receptacle')
        else:
            values = list(args or [])
            obj_role = values[0] if len(values) > 0 else None
            rec_role = values[1] if len(values) > 1 else None
        obj = env.role_map.get(str(obj_role)) if obj_role is not None else None
        rec = env.role_map.get(str(rec_role)) if rec_role is not None else None
        if obj and rec:
            pairs.add(frozenset((obj, rec)))
    return pairs

def _not_in_receptacle_entries(task_spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    for predicate in (task_spec.get('goal') or {}).get('predicates') or []:
        if isinstance(predicate, Mapping) and predicate.get('type') == 'object_not_in_receptacle':
            entries.append(predicate)
    for assertion in (task_spec.get('generation') or {}).get('deferred_init_assertions') or []:
        if isinstance(assertion, Mapping) and assertion.get('type') == 'object_not_in_receptacle':
            entries.append(assertion)
    for assertion in (task_spec.get('reset_plan') or {}).get('assertions') or []:
        if isinstance(assertion, Mapping) and assertion.get('type') == 'object_not_in_receptacle':
            entries.append(assertion)
    return entries

def _finite_vector(value: Any) -> bool:
    try:
        return all((math.isfinite(float(item)) for item in value))
    except Exception:
        return False

def _round_vector(value: Any, ndigits: int=4) -> Any:
    if value is None:
        return None
    try:
        return [round(float(item), ndigits) for item in value]
    except Exception:
        return value

def _jsonable_sorted(value: Any) -> Any:
    return json.loads(json.dumps(to_jsonable(value), sort_keys=True, default=str))

def _stable_json(value: Any) -> str:
    return json.dumps(_jsonable_sorted(value), sort_keys=True, separators=(',', ':'))

def _normalize_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

def _event_dicts(events: Sequence[Any]) -> list[dict[str, Any]]:
    if not events:
        return []
    out = []
    for event in events:
        out.append({'name': event.name, 'phase': event.phase, 'payload': _jsonable_sorted(event.payload)})
    return out

def _default_tool_for_failure(mode: Any) -> str:
    if mode == 'nav_blocked_partway':
        return 'navigate_to'
    if mode == 'grasp_slip':
        return 'pick_up'
    if mode == 'drop_beside':
        return 'put_down'
    return 'navigate_to'

def _first_pickupable_role(task_spec: Mapping[str, Any]) -> str | None:
    for role_name, role in sorted((task_spec.get('roles') or {}).items()):
        if 'pickupable' in set(role.get('abilities') or []):
            return str(role_name)
    return None
__all__ = ['validate_stage5_reset_mutation', 'summarize_stage5_results', 'check_init_predicates', 'check_deferred_init_assertions', 'validate_geometry', 'validate_reachability', 'interaction_required_roles', 'validate_sequential_interaction_reachability', 'ordered_navigation_roles', 'validate_observation_conditioned_room_constraints', 'validate_mutation_plan']
