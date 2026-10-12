from __future__ import annotations
import copy
import random
from collections import Counter
from typing import Any, Mapping, Sequence
from scoper.benchmark.behavior.task_generation.latent_materializers import LatentMaterializationError, materialize_latent_issue
from scoper.benchmark.behavior.task_generation.profile_task_generator import goal_plan_skeleton, occupancy_bounds_from_profile, room_catalog_from_profile, start_pose_from_profile, stage4_readiness_from_debt
from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec
DEFAULT_ROOM_TYPES = ('kitchen', 'bathroom', 'living_room', 'bedroom')
DEFAULT_ISSUE_TYPES = ('faucet_on', 'open_container', 'dirty_dish')
OBSERVATION_TEMPLATE_IDS = ('observation_conditioned_cross_room_inspection_001', 'observation_conditioned_dirty_dish_workflow_001', 'observation_conditioned_mixed_inspection_001', 'observation_conditioned_safety_inspection_001', 'observation_conditioned_storage_policy_001', 'observation_conditioned_dirty_dish_inspection_001')
_TEMPLATE_PLANS = {OBSERVATION_TEMPLATE_IDS[0]: {'rooms_per_task': 2, 'issue_types': ('faucet_on', 'open_container'), 'issues_per_room': 1}, OBSERVATION_TEMPLATE_IDS[1]: {'rooms_per_task': 1, 'issue_types': ('dirty_dish',), 'issues_per_room': 1}, OBSERVATION_TEMPLATE_IDS[2]: {'rooms_per_task': 1, 'issue_types': DEFAULT_ISSUE_TYPES, 'issues_per_room': 3}, OBSERVATION_TEMPLATE_IDS[3]: {'rooms_per_task': 1, 'issue_types': ('faucet_on', 'open_container'), 'issues_per_room': 2}, OBSERVATION_TEMPLATE_IDS[4]: {'rooms_per_task': 1, 'issue_types': ('misplaced_object',), 'issues_per_room': 1}, OBSERVATION_TEMPLATE_IDS[5]: {'rooms_per_task': 1, 'issue_types': ('dirty_dish',), 'issues_per_room': 1}}

def generate_observation_conditioned_tasks(*, profiles: Sequence[Mapping[str, Any]], seed: int=0, tasks_per_scene: int=1, rooms_per_task: int | None=None, issues_per_room: int | None=None, issue_types: Sequence[str] | None=None, template_id: str | None=None, template_ids: Sequence[str] | None=None) -> dict[str, Any]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    selected_template_ids = list(template_ids or ([template_id] if template_id else [None]))
    for selected in selected_template_ids:
        if selected is not None and selected not in OBSERVATION_TEMPLATE_IDS:
            raise ValueError("Unknown observation template '{}'".format(selected))
    for profile in profiles:
        scene_model = str(profile.get('scene_model') or 'scene')
        for variant_index in range(max(0, int(tasks_per_scene))):
            for current_template_id in selected_template_ids:
                plan = _TEMPLATE_PLANS.get(current_template_id or '', {})
                try:
                    task = build_observation_conditioned_task_from_profile(scene_profile={**dict(profile), '_selection_seed': int(seed) + int(variant_index)}, seed=seed, variant_index=variant_index, rooms_per_task=rooms_per_task if rooms_per_task is not None else plan.get('rooms_per_task', 1), issues_per_room=issues_per_room if issues_per_room is not None else plan.get('issues_per_room', 1), issue_types=issue_types if issue_types is not None else plan.get('issue_types', DEFAULT_ISSUE_TYPES), template_id=current_template_id, rng=rng)
                    validate_behavior_task_spec(task)
                except Exception as exc:
                    rejected.append({'scene_model': scene_model, 'template_id': current_template_id, 'category': 'observation_conditioned_room_inspection', 'failure_code': 'OBSERVATION_CONDITIONED_GENERATION_FAILED', 'error': str(exc)})
                    continue
                tasks.append(task)
    return {'tasks': tasks, 'rejected': rejected, 'summary': summarize_observation_conditioned_generation(tasks, rejected)}

def build_observation_conditioned_task_from_profile(*, scene_profile: Mapping[str, Any], seed: int=0, variant_index: int=0, rooms_per_task: int=1, issues_per_room: int=1, issue_types: Sequence[str]=DEFAULT_ISSUE_TYPES, template_id: str | None='observation_conditioned_room_inspection_generated', rng: random.Random | None=None) -> dict[str, Any]:
    rng = rng or random.Random(seed + variant_index)
    template_id = template_id or 'observation_conditioned_room_inspection_generated'
    if template_id in _TEMPLATE_PLANS:
        plan = _TEMPLATE_PLANS[template_id]
        rooms_per_task = int(plan['rooms_per_task'])
        issues_per_room = int(plan['issues_per_room'])
        issue_types = tuple(plan['issue_types'])
    scene_model = str(scene_profile.get('scene_model') or 'scene')
    candidate_rooms = selectable_rooms(scene_profile)
    if not candidate_rooms:
        raise ValueError('No selectable rooms in scene profile.')
    selected_rooms = candidate_rooms[:max(1, int(rooms_per_task))]
    roles: dict[str, dict[str, Any]] = {}
    init: list[dict[str, Any]] = []
    goal_predicates: list[dict[str, Any]] = []
    latent_payloads: list[dict[str, Any]] = []
    generation_notes: list[str] = []
    room_constraints: dict[str, str] = {}
    issue_queue = list(issue_types or DEFAULT_ISSUE_TYPES)
    rng.shuffle(issue_queue)
    max_issues_per_room = max(1, int(issues_per_room))
    latent_index = 0
    errors: list[str] = []
    assignments = [(room, issue_type) for room, issue_type in zip(selected_rooms, issue_queue)] if template_id == OBSERVATION_TEMPLATE_IDS[0] else [(room, issue_type) for room in selected_rooms for issue_type in issue_queue]
    per_room_counts = Counter()
    for room, issue_type in assignments:
        room_id = str(room['room_id'])
        if per_room_counts[room_id] >= max(1, int(issues_per_room)):
            continue
        try:
            latent_spec = {'issue_type': issue_type, 'room_id': room_id, 'latent_id': 'latent_{}_{}'.format(issue_type, latent_index)}
            if issue_type in {'dirty_dish', 'misplaced_object'}:
                latent_spec['object_family'] = ('plate', 'bowl', 'drinking_vessel')[latent_index % 3]
            materialized = materialize_latent_issue(latent_spec, scene_profile=scene_profile, latent_index=latent_index)
        except LatentMaterializationError as exc:
            errors.append('{}:{}:{}'.format(room_id, issue_type, exc))
            continue
        _merge_unique_roles(roles, materialized.roles)
        init.extend(materialized.init)
        goal_predicates.extend(materialized.goal_predicates)
        latent_payloads.append(materialized.latent_payload)
        generation_notes.extend(materialized.generation_notes)
        room_constraints.update(materialized.room_constraints)
        latent_index += 1
        per_room_counts[room_id] += 1
    if not latent_payloads:
        raise ValueError('No latent payload could be materialized. Attempts: {}'.format('; '.join(errors[:8])))
    room_ids = sorted({str(payload['room_id']) for payload in latent_payloads})
    instruction = abstract_instruction_for_template(template_id, room_ids)
    task_id = 'behavior_{}_{}_seed{:04d}_{:02d}'.format(scene_model, template_id, int(seed), int(variant_index))
    readiness_debt = {'family_teacher:observation_conditioned', 'observation_conditioned_discovery'}
    if any((role.get('requires_spawn') for role in roles.values())):
        readiness_debt.update({'spawn_object', 'spawn_placement'})
    source = {'type': 'template', 'scene_model': scene_model, 'source_json': scene_profile.get('source_json'), 'template_id': template_id, 'template_version': 'observation_conditioned.2', 'stage3_match_status': 'profile_observation_conditioned', 'occupancy_bounds': occupancy_bounds_from_profile(scene_profile), 'start_pose': start_pose_from_profile(scene_profile), 'room_catalog': room_catalog_from_profile(scene_profile)}
    family_spec = {'family': 'observation_conditioned_room_inspection', 'template_id': template_id, 'observation_conditioned': {'template_id': template_id, 'rooms': room_ids, 'scan_policy': {'num_views': 4, 'require_room_entry_before_scan': True}, 'latent_payloads': latent_payloads, 'latent_role_room_constraints': room_constraints}}
    task = {'id': task_id, 'version': 'stage4.0', 'backend': 'behavior', 'category': 'observation_conditioned_room_inspection', 'source': source, 'template': {'template_id': template_id, 'template_version': 'observation_conditioned.2', 'template_file': None}, 'instruction': instruction, 'roles': roles, 'init': _dedupe_records(init), 'init_source_predicates': [], 'goal': {'type': 'predicates', 'require_model_stop': True, 'bddl_goal': None, 'predicates': goal_predicates}, 'family_spec': family_spec, 'generation': {'stage': 'stage4_observation_conditioned_profile_driven', 'seed': int(seed), 'variant_index': int(variant_index), 'stage4_readiness': stage4_readiness_from_debt(readiness_debt), 'stage4_readiness_debt': sorted(readiness_debt), 'required_init_mutations': [], 'runtime_capabilities_required': [], 'binding_notes': generation_notes, 'deferred_init_assertions': [], 'family_spec_materialized': family_spec, 'unsupported_scheduler_specs': [], 'latent_role_room_constraints': room_constraints, 'latent_generation_errors': errors[:20]}, 'reset_plan': {'state_and_relation_init': _dedupe_records(init), 'assertions': [], 'required_init_mutations': [], 'runtime_capabilities_required': []}, 'oracle_teacher_hint': {'family': 'observation_conditioned_room_inspection', 'expected_events': ['navigate_to_room', 'scan', 'issue_discovered'], 'expected_brain_behaviors': ['inspect_before_repair'], 'expected_plan_skeleton': ['navigate_to_room', 'scan'] + goal_plan_skeleton(goal_predicates) + ['stop']}, 'metrics_config': {'track_predicate_progress': True, 'track_tool_success_rate': True, 'track_observation_conditioned_discovery': True, 'require_model_stop': True}, 'online_requirements': {'expected_events': ['navigate_to_room', 'scan', 'issue_discovered'], 'expected_brain_behaviors': ['inspect_before_repair']}, 'validation': {'status': 'generated_pending_reset_validation', 'checks': []}, 'tags': ['stage4_candidate', 'profile_driven', 'observation_conditioned']}
    return task

def selectable_rooms(scene_profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    rooms = []
    for room in scene_profile.get('rooms') or []:
        if not isinstance(room, Mapping) or not room.get('room_id'):
            continue
        room_type = str(room.get('room_type') or str(room.get('room_id')).rsplit('_', 1)[0])
        if room_type not in DEFAULT_ROOM_TYPES:
            continue
        rooms.append(dict(room))
    rooms.sort(key=lambda item: (_room_priority(str(item.get('room_type') or '')), str(item.get('room_id') or '')))
    return rooms

def abstract_instruction_for_rooms(room_ids: Sequence[str]) -> str:
    labels = [str(room).rsplit('_', 1)[0].replace('_', ' ') for room in room_ids]
    if len(labels) == 1:
        room_text = 'the {}'.format(labels[0])
    elif len(labels) == 2:
        room_text = 'the {} and {}'.format(labels[0], labels[1])
    else:
        room_text = 'the ' + ', '.join(labels[:-1]) + ', and ' + labels[-1]
    return 'Inspect {} and fix any household issues you discover.'.format(room_text)

def abstract_instruction_for_template(template_id: str, room_ids: Sequence[str]) -> str:
    labels = [str(room).rsplit('_', 1)[0].replace('_', ' ') for room in room_ids]
    if len(labels) == 1:
        room_text = 'the {}'.format(labels[0])
    elif len(labels) == 2:
        room_text = 'the {} and the {}'.format(labels[0], labels[1])
    else:
        room_text = 'the ' + ', '.join(labels[:-1]) + ', and ' + labels[-1]
    instructions = {'observation_conditioned_cross_room_inspection_001': 'Inspect {}. In each room, resolve only household issues supported by what you observe, then stop.'.format(room_text), 'observation_conditioned_dirty_dish_workflow_001': 'Inspect {}. If you find an item that visibly needs cleaning, clean it and put it away, then stop.'.format(room_text), 'observation_conditioned_mixed_inspection_001': 'Inspect {} and resolve any applicable household issues supported by your observations before stopping.'.format(room_text), 'observation_conditioned_safety_inspection_001': 'Inspect {} for actionable household conditions and address only what your observations support.'.format(room_text), 'observation_conditioned_storage_policy_001': 'Inspect {} and put away any misplaced item you can identify, using an appropriate receptacle.'.format(room_text), 'observation_conditioned_dirty_dish_inspection_001': 'Inspect {} and clean and put away any visibly soiled item that requires attention.'.format(room_text)}
    return instructions.get(template_id, abstract_instruction_for_rooms(room_ids))

def summarize_observation_conditioned_generation(tasks: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {'generated_total': len(tasks), 'rejected_total': len(rejected), 'generated_by_scene': dict(sorted(Counter(((task.get('source') or {}).get('scene_model') for task in tasks)).items())), 'generated_by_readiness': dict(sorted(Counter(((task.get('generation') or {}).get('stage4_readiness') for task in tasks)).items())), 'latent_issue_types': dict(sorted(Counter((payload.get('issue_type') for task in tasks for payload in ((task.get('family_spec') or {}).get('observation_conditioned') or {}).get('latent_payloads') or [])).items())), 'rejection_codes': dict(sorted(Counter((item.get('failure_code') for item in rejected)).items()))}

def _merge_unique_roles(target: dict[str, dict[str, Any]], source: Mapping[str, Mapping[str, Any]]) -> None:
    for name, role in source.items():
        role_name = str(name)
        if role_name in target:
            raise ValueError("Duplicate generated role '{}'.".format(role_name))
        target[role_name] = copy.deepcopy(dict(role))

def _dedupe_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for record in records:
        copied = copy.deepcopy(dict(record))
        key = repr(copied)
        if key in seen:
            continue
        seen.add(key)
        out.append(copied)
    return out

def _room_priority(room_type: str) -> int:
    order = {name: idx for idx, name in enumerate(DEFAULT_ROOM_TYPES)}
    return order.get(str(room_type), 100)
__all__ = ['DEFAULT_ISSUE_TYPES', 'OBSERVATION_TEMPLATE_IDS', 'abstract_instruction_for_template', 'build_observation_conditioned_task_from_profile', 'generate_observation_conditioned_tasks', 'selectable_rooms']
