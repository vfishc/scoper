from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
EB_PROCTHOR_PROFILES_ENV_VAR = 'EB_PROCTHOR_PROFILES'
from scoper.benchmark.procthor.object_taxonomy import ObjectTaxonomy, load_object_taxonomy
from scoper.benchmark.procthor.pose_candidates import PoseCandidate, find_pose_candidates_evacuate_from_goal_receptacle, find_pose_candidates_route_near_surface, find_pose_candidates_route_visible_surface, find_pose_candidates_same_room_as, find_pose_candidates_unblock_from_closed_container
from scoper.benchmark.procthor.online_task_spec import validate_online_task_spec, validate_task_bank
from scoper.benchmark.procthor.scene_profiler import load_house_json, validate_scene_profile
from scoper.benchmark.procthor.scene_profile_utils import profile_object_room_id
from scoper.benchmark.procthor.task_templates import enumerate_role_bindings, load_task_templates, validate_task_template
from scoper.benchmark.procthor.template_enhancements import effective_disturbance_spec_for_template
from scoper.benchmark.common import RoomSemanticError, attach_compiled_task_binding_spec, build_room_type_catalog, materialize_observation_conditioned_spec, materialize_room_scope
TASK_SPEC_VERSION = '0.1'
DEFAULT_MAX_TASKS_PER_HOUSE = 20
DEFAULT_MAX_TASKS_PER_CATEGORY_PER_HOUSE = 4
DEFAULT_MAX_TASKS_PER_TEMPLATE_PER_HOUSE = 2
FLOOR_RELOCATE_MODES = frozenset({'near_primary_object_floor'})
DEFAULT_FLOOR_RELOCATE_MIN_DISTANCE = 0.4
DEFAULT_FLOOR_RELOCATE_MAX_DISTANCE = 2.0
DEFAULT_FLOOR_OBJECT_OFFSET_FROM_AGENT = 0.35
DEFAULT_FLOOR_RELOCATE_MAX_CANDIDATES = 16

class TaskGenerationError(ValueError):
    pass

@dataclass(frozen=True)
class FloorPositionCandidate:
    position: dict
    distance_to_anchor: float
    rationale: dict

def load_scene_profiles(path: str) -> List[dict]:
    profiles = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            profile = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskGenerationError("Invalid scene profile JSONL '{}' at line {}: {}.".format(path, line_number, exc)) from exc
        if not isinstance(profile, dict):
            raise TaskGenerationError("Scene profile '{}' line {} must be a JSON object.".format(path, line_number))
        _hydrate_profile_room_geometry(profile)
        validate_scene_profile(profile)
        profiles.append(profile)
    return profiles

def _hydrate_profile_room_geometry(profile: MutableMapping[str, Any]) -> None:
    rooms = profile.get('rooms')
    if not isinstance(rooms, list):
        return
    if any((isinstance(room, Mapping) and isinstance(room.get('floorPolygon') or room.get('floor_polygon'), list) and (len(room.get('floorPolygon') or room.get('floor_polygon') or []) >= 3) for room in rooms)):
        return
    house_path = profile.get('house_path')
    if not isinstance(house_path, str) or not house_path:
        return
    try:
        house = load_house_json(house_path)
    except Exception:
        return
    house_rooms = {str(room.get('id') or room.get('roomId') or room.get('room_id')): room for room in house.get('rooms') or [] if isinstance(room, Mapping) and (room.get('id') or room.get('roomId') or room.get('room_id'))}
    for room in rooms:
        if not isinstance(room, MutableMapping):
            continue
        room_id = str(room.get('room_id') or '')
        source_room = house_rooms.get(room_id)
        polygon = source_room.get('floorPolygon') or source_room.get('floor_polygon') if isinstance(source_room, Mapping) else None
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        room['floor_polygon'] = [{'x': float(vertex['x']), 'z': float(vertex['z'])} for vertex in polygon if isinstance(vertex, Mapping) and 'x' in vertex and ('z' in vertex)]
        if len(room['floor_polygon']) < 3:
            room.pop('floor_polygon', None)

def generate_candidate_tasks(scene_profiles: Sequence[Mapping[str, Any]], templates: Sequence[Mapping[str, Any]], taxonomy: ObjectTaxonomy, seed: int=0, max_tasks_per_house: int=DEFAULT_MAX_TASKS_PER_HOUSE, max_tasks_per_category_per_house: int=DEFAULT_MAX_TASKS_PER_CATEGORY_PER_HOUSE, max_tasks_per_template_per_house: int=DEFAULT_MAX_TASKS_PER_TEMPLATE_PER_HOUSE, max_tasks_per_category: Optional[int]=None, enable_visual_discoverability: bool=False, visual_discoverability_env_factory: Optional[Any]=None) -> Tuple[List[dict], dict]:
    for template in templates:
        validate_task_template(template)
    if max_tasks_per_house <= 0:
        return ([], _empty_summary(scene_profiles, templates))
    rng = random.Random(seed)
    unnumbered_tasks = []
    rejection_reasons = Counter()
    sorted_profiles = sorted(scene_profiles, key=_profile_sort_key)
    sorted_templates = sorted(templates, key=lambda template: (template['category'], template['template_id']))
    templates_by_category: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for template in sorted_templates:
        templates_by_category[template['category']].append(template)
    for profile in sorted_profiles:
        per_house_count = 0
        per_house_category_count = Counter()
        for category in sorted(templates_by_category):
            if per_house_count >= max_tasks_per_house:
                break
            category_templates = templates_by_category[category]
            selected_bindings_by_template: Dict[str, List[Mapping[str, Mapping[str, Any]]]] = {}
            selected_tasks_by_template: Dict[str, List[dict]] = {}
            next_binding_index_by_template: Counter = Counter()
            next_task_index_by_template: Counter = Counter()
            per_template_count: Counter = Counter()
            made_progress = True
            while made_progress and per_house_count < max_tasks_per_house:
                if per_house_category_count[category] >= max_tasks_per_category_per_house:
                    break
                made_progress = False
                for template in category_templates:
                    if per_house_count >= max_tasks_per_house:
                        break
                    if per_house_category_count[category] >= max_tasks_per_category_per_house:
                        break
                    template_id = template['template_id']
                    if per_template_count[template_id] >= max_tasks_per_template_per_house:
                        continue
                    if category == 'tidy_by_policy':
                        if template_id not in selected_tasks_by_template:
                            tidy_tasks, reason = _generate_tidy_tasks(profile, template, taxonomy, rng, limit=max_tasks_per_template_per_house)
                            if not tidy_tasks:
                                rejection_reasons[category, reason or 'no_binding'] += 1
                                selected_tasks_by_template[template_id] = []
                            else:
                                selected_tasks_by_template[template_id] = tidy_tasks
                        selected_tasks = selected_tasks_by_template[template_id]
                        next_index = next_task_index_by_template[template_id]
                        if next_index >= len(selected_tasks):
                            continue
                        task = selected_tasks[next_index]
                        next_task_index_by_template[template_id] += 1
                        from scoper.benchmark.common.task_identity_policy import enabled, conflicts
                        if enabled() and conflicts(task):
                            rejection_reasons[category, 'TASK_PUBLIC_IDENTITY_CONFLICT'] += 1
                            made_progress = True
                            continue
                        task['_generation_key'] = _generation_key(task)
                        unnumbered_tasks.append(task)
                        per_house_count += 1
                        per_house_category_count[category] += 1
                        per_template_count[template_id] += 1
                        made_progress = True
                        continue
                    if template_id not in selected_bindings_by_template:
                        bindings, reason = enumerate_role_bindings(template, profile, taxonomy)
                        if not bindings:
                            rejection_reasons[category, reason or 'no_binding'] += 1
                            selected_bindings_by_template[template_id] = []
                        else:
                            selected_bindings_by_template[template_id] = _sample_bindings(template, bindings, _binding_attempt_limit(template, max_tasks_per_template_per_house), rng)
                    selected_bindings = selected_bindings_by_template[template_id]
                    next_index = next_binding_index_by_template[template_id]
                    if next_index >= len(selected_bindings):
                        continue
                    binding = selected_bindings[next_index]
                    next_binding_index_by_template[template_id] += 1
                    task, reason = _instantiate_task_spec_with_relocate(profile, template, binding, taxonomy)
                    if task is None:
                        rejection_reasons[category, reason or 'relocate_rejected'] += 1
                        made_progress = True
                        continue
                    from scoper.benchmark.common.task_identity_policy import enabled, conflicts
                    if enabled() and conflicts(task):
                        rejection_reasons[category, 'TASK_PUBLIC_IDENTITY_CONFLICT'] += 1
                        made_progress = True
                        continue
                    task['_generation_key'] = _generation_key(task)
                    unnumbered_tasks.append(task)
                    per_house_count += 1
                    per_house_category_count[category] += 1
                    per_template_count[template_id] += 1
                    made_progress = True
    if enable_visual_discoverability:
        unnumbered_tasks, vdv_summary = _filter_visual_discoverability(unnumbered_tasks, visual_discoverability_env_factory)
    else:
        vdv_summary = None
    numbered_tasks = _assign_task_ids(unnumbered_tasks, max_tasks_per_category=max_tasks_per_category)
    from scoper.benchmark.common.storage_instruction import repair_storage_instructions
    numbered_tasks = [attach_compiled_task_binding_spec(repair_storage_instructions(task)) for task in numbered_tasks]
    for task in numbered_tasks:
        validate_online_task_spec(task)
    validate_task_bank(numbered_tasks)
    summary = summarize_generated_tasks(numbered_tasks, scene_profiles=scene_profiles, templates=templates, rejections=rejection_reasons)
    if vdv_summary is not None:
        summary['visual_discoverability'] = vdv_summary
    return (numbered_tasks, summary)

def write_task_bank(tasks: Iterable[Mapping[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        for task in tasks:
            validate_online_task_spec(task)
            json.dump(task, f, ensure_ascii=False, sort_keys=True)
            f.write('\n')

def summarize_generated_tasks(tasks: Sequence[Mapping[str, Any]], scene_profiles: Optional[Sequence[Mapping[str, Any]]]=None, templates: Optional[Sequence[Mapping[str, Any]]]=None, rejections: Optional[Mapping[Tuple[str, str], int]]=None) -> dict:
    category_counts = Counter((task['category'] for task in tasks))
    house_counts = Counter((task['source']['house_id'] for task in tasks))
    template_counts = Counter((task['template']['template_id'] for task in tasks))
    object_type_counts = Counter()
    role_pair_counts = Counter()
    for task in tasks:
        for role in task['roles'].values():
            object_type_counts[role['object_type']] += 1
        target = task['roles'].get('target_object')
        receptacle = task['roles'].get('target_receptacle')
        if target and receptacle:
            role_pair_counts['{}->{}'.format(target['object_type'], receptacle['object_type'])] += 1
    rejection_summary = []
    for (category, reason), count in sorted((rejections or {}).items()):
        rejection_summary.append({'category': category, 'reason': reason, 'count': count})
    return {'num_profiles': len(scene_profiles or []), 'num_templates': len(templates or []), 'num_tasks': len(tasks), 'category_counts': dict(sorted(category_counts.items())), 'house_counts': dict(sorted(house_counts.items())), 'template_counts': dict(sorted(template_counts.items())), 'object_type_counts': dict(sorted(object_type_counts.items())), 'target_to_receptacle_counts': dict(sorted(role_pair_counts.items())), 'rejections': rejection_summary}

def main(argv: Optional[Sequence[str]]=None) -> None:
    parser = argparse.ArgumentParser(description='Generate ProcTHOR online tasks.')
    parser.add_argument('--profiles', default=None, help='Path to scene profiles JSONL. If omitted, falls back to the environment variable {} (recommended: point at the 12K profile produced by data_processing.md).'.format(EB_PROCTHOR_PROFILES_ENV_VAR))
    parser.add_argument('--templates', required=True)
    parser.add_argument('--taxonomy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-tasks-per-house', type=int, default=DEFAULT_MAX_TASKS_PER_HOUSE)
    parser.add_argument('--max-tasks-per-category-per-house', type=int, default=DEFAULT_MAX_TASKS_PER_CATEGORY_PER_HOUSE)
    parser.add_argument('--max-tasks-per-template-per-house', type=int, default=DEFAULT_MAX_TASKS_PER_TEMPLATE_PER_HOUSE)
    parser.add_argument('--max-tasks-per-category', type=int, default=None)
    parser.add_argument('--enable-visual-discoverability', action='store_true')
    parser.add_argument('--gpu-device', type=int, default=None)
    parser.add_argument('--manipulathor-arm', action='store_true', help="Use AI2-THOR's official ManipulaTHOR arm agent mode for visual-discoverability filtering.")
    parser.add_argument('--manipulathor-visual-attach', action='store_true', help='Show the ManipulaTHOR arm and use the visual-attach arm pose for visual-discoverability filtering.')
    parser.add_argument('--agent-radius', type=float, default=None, help='Optional AI2-THOR SetAgentRadius override in meters. Visual-attach mode defaults to 0.3.')
    args = parser.parse_args(argv)
    profiles_path = args.profiles or os.environ.get(EB_PROCTHOR_PROFILES_ENV_VAR)
    if not profiles_path:
        parser.error('--profiles is required (or set the {} environment variable).'.format(EB_PROCTHOR_PROFILES_ENV_VAR))
    profiles = load_scene_profiles(profiles_path)
    templates = load_task_templates(args.templates)
    taxonomy = load_object_taxonomy(args.taxonomy)
    env_factory = None
    if args.enable_visual_discoverability:
        from scoper.benchmark.procthor.closed_loop_tool_env import ProcTHORClosedLoopToolEnv
        from scoper.benchmark.procthor.manipulathor_config import manipulathor_env_kwargs

        def env_factory():
            env_kwargs = {'gpu_device': args.gpu_device}
            env_kwargs.update(manipulathor_env_kwargs(manipulathor_arm=args.manipulathor_arm, manipulathor_visual_attach=args.manipulathor_visual_attach, agent_radius=args.agent_radius))
            return ProcTHORClosedLoopToolEnv(**env_kwargs)
    tasks, summary = generate_candidate_tasks(profiles, templates, taxonomy, seed=args.seed, max_tasks_per_house=args.max_tasks_per_house, max_tasks_per_category_per_house=args.max_tasks_per_category_per_house, max_tasks_per_template_per_house=args.max_tasks_per_template_per_house, max_tasks_per_category=args.max_tasks_per_category, enable_visual_discoverability=args.enable_visual_discoverability, visual_discoverability_env_factory=env_factory)
    write_task_bank(tasks, args.output)
    summary['output'] = args.output
    summary_path = str(Path(args.output).with_suffix(Path(args.output).suffix + '.summary.json'))
    Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

def _instantiate_task_spec_with_relocate(profile: Mapping[str, Any], template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], taxonomy: ObjectTaxonomy) -> Tuple[Optional[dict], Optional[str]]:
    mutable_binding: Dict[str, Mapping[str, Any]] = dict(binding)
    object_pose_overrides, reason = _apply_relocate_pipeline(template, mutable_binding, profile, taxonomy)
    if reason is not None:
        return (None, reason)
    task, reason = _instantiate_task_spec(profile, template, mutable_binding, object_pose_overrides=object_pose_overrides)
    return (task, reason)

def _instantiate_task_spec(profile: Mapping[str, Any], template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], object_pose_overrides: Optional[Sequence[Mapping[str, Any]]]=None) -> Tuple[Optional[dict], Optional[str]]:
    from scoper.benchmark.procthor.template_enhancements import effective_binding_for_template, effective_roles_for_template, effective_goal_predicates_for_template, effective_family_spec_for_template, effective_instruction_for_template
    try:
        effective_binding = effective_binding_for_template(template, binding, scene_profile=profile)
    except ValueError as exc:
        return (None, str(exc))
    try:
        effective_roles_for_template(template, effective_binding, scene_profile=profile)
    except ValueError as exc:
        return (None, str(exc))
    for role_name, role_obj in effective_binding.items():
        if isinstance(role_obj, Mapping) and role_obj.get('requires_spawn'):
            return (None, "Role '{}' requires spawn, but generated tasks only use real scene objects.".format(role_name))
    roles = {role_name: {'object_id': _object_id(obj), 'object_type': _object_type(obj)} for role_name, obj in effective_binding.items()}
    try:
        family_spec = _resolve_family_spec(template, effective_binding, profile)
    except RoomSemanticError as exc:
        return (None, '{}:{}'.format(exc.code, exc))
    instruction = _instantiate_instruction(template['instruction_templates'][0], effective_binding, scope_room_label=(family_spec or {}).get('scope', {}).get('label'), scope_room_labels=(family_spec or {}).get('scope', {}).get('room_labels'))
    instruction = effective_instruction_for_template(template, effective_binding, instruction)
    init_overrides = _initial_state_overrides(template.get('initial_state_constraints', []))
    goal_predicates = _resolve_template_goal_predicates(template, effective_binding)
    goal_predicates = effective_goal_predicates_for_template(template, effective_binding, goal_predicates)
    family_spec = effective_family_spec_for_template(template, family_spec, binding=effective_binding)
    if family_spec is not None:
        goal_predicates = _merge_human_insert_goal_predicates(goal_predicates, family_spec)
        init_overrides = _merge_human_insert_antistate_overrides(init_overrides, family_spec)
    pose_overrides, cleanup_reason = _object_pose_overrides_with_cleanup(object_pose_overrides, profile, effective_binding, goal_predicates, family_spec)
    if cleanup_reason is not None:
        return (None, cleanup_reason)
    task = {'id': 'pending', 'version': TASK_SPEC_VERSION, 'backend': 'procthor', 'category': template['category'], 'source': {'house_id': profile['house_id'], 'house_path': profile.get('house_path') or '', 'split': profile.get('split') or 'unknown'}, 'template': {'template_id': template['template_id'], 'template_version': template['template_version']}, 'instruction': instruction, 'roles': roles, 'init': {'agent_pose': _initial_agent_pose(profile), 'object_state_overrides': init_overrides, 'object_pose_overrides': [copy.deepcopy(dict(override)) for override in pose_overrides]}, 'goal': {'type': 'predicate', 'predicates': goal_predicates, 'require_model_stop': bool(template.get('goal_options', {}).get('require_model_stop', False))}, 'online_requirements': {'expected_events': list(template['expected_events']), 'expected_brain_behaviors': list(template['expected_brain_behaviors'])}, 'validation': {'status': 'candidate', 'checks': {}, 'failure_reason': None}, 'tags': _task_tags(template, effective_binding)}
    _attach_profile_source_metadata(task, profile)
    if family_spec is not None:
        task['family_spec'] = family_spec
    if 'disturbance_spec' in template:
        enhanced_disturbance = effective_disturbance_spec_for_template(template, template['disturbance_spec'], binding=effective_binding)
        if enhanced_disturbance is not None:
            task['disturbance_spec'] = enhanced_disturbance
    if template.get('category') == 'composite':
        composite_disturbance = _resolve_composite_disturbance_spec(template)
        if composite_disturbance is not None:
            task['disturbance_spec'] = _merge_disturbance_specs(task.get('disturbance_spec'), composite_disturbance)
    for role_name, role_obj in effective_binding.items():
        if not isinstance(role_obj, Mapping):
            continue
        if 'init_states' in role_obj and isinstance(role_obj['init_states'], Mapping):
            for state_key, state_value in role_obj['init_states'].items():
                if state_key == 'dirty':
                    task['init']['object_state_overrides'].append({'type': 'set_dirty_state', 'object_role': role_name, 'is_dirty': bool(state_value)})
    return (task, None)

def _apply_relocate_pipeline(template: Mapping[str, Any], binding: Dict[str, Mapping[str, Any]], profile: Mapping[str, Any], taxonomy: ObjectTaxonomy) -> Tuple[List[dict], Optional[str]]:
    overrides: List[dict] = []
    for role_name, role_constraints in template['requires'].items():
        relocate_spec = role_constraints.get('relocate_if')
        if not isinstance(relocate_spec, Mapping):
            continue
        modes = list(relocate_spec.get('modes') or [])
        fallback = relocate_spec.get('fallback_to')
        if isinstance(fallback, str) and fallback not in modes:
            modes.append(fallback)
        triggered = False
        for mode in modes:
            if not _mode_trigger_satisfied(mode, role_name, binding, profile, template):
                continue
            triggered = True
            if mode in FLOOR_RELOCATE_MODES:
                floor_candidates = _find_floor_candidates_for_mode(mode, role_name, binding, profile, template)
                if not floor_candidates:
                    continue
                anchor_role = _floor_anchor_role(template, role_name)
                if anchor_role is None:
                    return ([], 'invalid_floor_anchor_role_{}'.format(role_name))
                overrides.append({'type': 'relocate_object', 'object_role': role_name, 'mode': mode, 'to_floor_anchor_role': anchor_role, 'floor_position_candidates': [copy.deepcopy(candidate.position) for candidate in floor_candidates], 'validation': 'interactable_poses', 'open_parent_during_relocate': True, 'restore_parent_state_after_relocate': True, 'rationale': {'mode': mode, 'floor_candidate_source': 'reachable_positions_near_anchor', 'candidates': [copy.deepcopy(candidate.rationale) for candidate in floor_candidates]}})
                break
            to_receptacle_role = relocate_spec.get('to_receptacle_role')
            if not isinstance(to_receptacle_role, str) or not to_receptacle_role:
                return ([], 'invalid_relocate_target_role_{}'.format(role_name))
            candidates = _find_candidates_for_mode(mode, role_name, binding, profile, template, taxonomy)
            if not candidates:
                continue
            chosen = candidates[0]
            chosen_obj = _profile_object_by_id(profile, chosen.receptacle_object_id)
            if chosen_obj is None:
                return ([], 'relocate_candidate_missing_{}'.format(role_name))
            binding[to_receptacle_role] = chosen_obj
            override = {'type': 'relocate_object', 'object_role': role_name, 'mode': mode, 'to_receptacle_role': to_receptacle_role, 'open_parent_during_relocate': True, 'restore_parent_state_after_relocate': True, 'rationale': copy.deepcopy(chosen.rationale)}
            if mode in {'route_visible_surface', 'route_near_surface'}:
                override['geometry_estimator_already_passed'] = True
            overrides.append(override)
            break
        else:
            if triggered:
                return ([], 'no_candidate_for_relocate_{}_{}'.format(role_name, '_'.join((str(mode) for mode in modes))))
        if not any((override.get('object_role') == role_name for override in overrides)) and _relocation_required_for_binding(role_name, modes, binding, profile, template):
            return ([], 'relocate_required_but_not_triggered_{}'.format(role_name))
    return (overrides, None)

def _object_pose_overrides_with_cleanup(object_pose_overrides: Optional[Sequence[Mapping[str, Any]]], profile: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], goal_predicates: Sequence[Mapping[str, Any]], family_spec: Optional[Mapping[str, Any]]) -> Tuple[List[dict], Optional[str]]:
    overrides = [copy.deepcopy(dict(override)) for override in object_pose_overrides or [] if isinstance(override, Mapping)]
    cleanup_overrides, reason = _placement_cleanup_overrides(profile, binding, goal_predicates, family_spec)
    if reason is not None:
        return ([], reason)
    overrides.extend(cleanup_overrides)
    return (overrides, None)

def _placement_cleanup_overrides(profile: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], goal_predicates: Sequence[Mapping[str, Any]], family_spec: Optional[Mapping[str, Any]]) -> Tuple[List[dict], Optional[str]]:
    placements = _collect_required_placements(goal_predicates, family_spec)
    if not placements:
        return ([], None)
    profile_objects = [obj for obj in profile.get('objects', []) or [] if isinstance(obj, Mapping)]
    role_object_ids = {_object_id(obj) for obj in binding.values() if isinstance(obj, Mapping)}
    grouped: Dict[str, dict] = {}
    for placement in placements:
        object_role = placement.get('object_role')
        receptacle_role = placement.get('receptacle_role')
        if not isinstance(object_role, str) or object_role not in binding:
            continue
        if not isinstance(receptacle_role, str) or receptacle_role not in binding:
            continue
        receptacle_id = _object_id(binding[receptacle_role])
        clutter = _cleanup_clutter_objects(profile_objects, receptacle_id, protected_object_ids=role_object_ids)
        if not clutter:
            continue
        group = grouped.setdefault(receptacle_role, {'receptacle_object_id': receptacle_id, 'clutter_by_id': {}, 'required_for': []})
        for obj in clutter:
            group['clutter_by_id'][_object_id(obj)] = obj
        group['required_for'].append({'object_role': object_role, 'receptacle_role': receptacle_role, 'source': placement.get('source')})
    overrides: List[dict] = []
    for receptacle_role, group in sorted(grouped.items()):
        receptacle_id = group['receptacle_object_id']
        clutter_by_id = dict(group['clutter_by_id'])
        if not clutter_by_id:
            continue
        overrides.append({'type': 'clear_receptacle_clutter', 'receptacle_role': receptacle_role, 'object_ids': sorted(clutter_by_id), 'rationale': {'reason': 'remove_unrelated_pickupable_objects_before_required_put_down', 'receptacle_object_id': receptacle_id, 'required_for': list(group['required_for'])}})
    return (overrides, None)

def _collect_required_placements(goal_predicates: Sequence[Mapping[str, Any]], family_spec: Optional[Mapping[str, Any]]) -> List[dict]:
    placements: List[dict] = []
    seen = set()

    def add(object_role: Any, receptacle_role: Any, source: str) -> None:
        if not isinstance(object_role, str) or not object_role:
            return
        if not isinstance(receptacle_role, str) or not receptacle_role:
            return
        key = (object_role, receptacle_role, source)
        if key in seen:
            return
        seen.add(key)
        placements.append({'object_role': object_role, 'receptacle_role': receptacle_role, 'source': source})

    def collect_goal_predicates(predicates: Any, source: str) -> None:
        if not isinstance(predicates, Sequence) or isinstance(predicates, (str, bytes)):
            return
        for predicate in predicates:
            if not isinstance(predicate, Mapping):
                continue
            predicate_type = predicate.get('type')
            object_role = predicate.get('object_role')
            if predicate_type in {'object_in_receptacle', 'object_on_receptacle'}:
                add(object_role, predicate.get('receptacle_role'), source)
            elif predicate_type == 'object_in_any_receptacle':
                for role in predicate.get('receptacle_roles') or []:
                    add(object_role, role, source)

    def visit_family(node: Any, inherited_object_role: Optional[str]=None) -> None:
        if isinstance(node, Mapping):
            goal_items = node.get('goal_predicates')
            if goal_items is not None:
                collect_goal_predicates(goal_items, 'family_goal_predicates')
            placement_fields = ('receptacle_role', 'staging_receptacle_role', 'final_receptacle_role', 'sink_role', 'target_receptacle_role')
            object_role = node.get('object_role')
            target_role = node.get('target_role')
            current_object_role = object_role if isinstance(object_role, str) else inherited_object_role
            if current_object_role is None and any((field in node for field in placement_fields)):
                current_object_role = target_role if isinstance(target_role, str) else None
            if isinstance(current_object_role, str):
                for field in placement_fields:
                    add(current_object_role, node.get(field), 'family_spec.{}'.format(field))
            target_roles = node.get('target_roles')
            storage_roles = node.get('storage_roles')
            if isinstance(target_roles, Sequence) and (not isinstance(target_roles, (str, bytes))):
                if isinstance(storage_roles, Sequence) and (not isinstance(storage_roles, (str, bytes))):
                    for object_role_item in target_roles:
                        for receptacle_role_item in storage_roles:
                            add(object_role_item, receptacle_role_item, 'family_spec.storage_roles')
            for child in node.values():
                visit_family(child, current_object_role)
        elif isinstance(node, Sequence) and (not isinstance(node, (str, bytes))):
            for child in node:
                visit_family(child, inherited_object_role)
    collect_goal_predicates(goal_predicates, 'goal.predicates')
    if isinstance(family_spec, Mapping):
        visit_family(family_spec)
    return placements

def _cleanup_clutter_objects(profile_objects: Sequence[Mapping[str, Any]], receptacle_id: str, *, protected_object_ids: set[str]) -> List[Mapping[str, Any]]:
    clutter = []
    for obj in profile_objects:
        object_id = _object_id(obj)
        if object_id in protected_object_ids:
            continue
        if not bool(obj.get('pickupable', False)):
            continue
        if receptacle_id not in set(obj.get('parent_receptacles') or []):
            continue
        clutter.append(obj)
    return sorted(clutter, key=lambda item: (_object_type(item), _object_id(item)))

def _mode_trigger_satisfied(mode: str, role_name: str, binding: Mapping[str, Mapping[str, Any]], profile: Mapping[str, Any], template: Mapping[str, Any]) -> bool:
    target = binding.get(role_name)
    if target is None:
        return False
    if mode == 'unblock_from_closed_container':
        return bool(_closed_openable_parent_ids(target, profile))
    if mode == 'evacuate_from_goal_receptacle':
        goal_roles = _goal_receptacle_roles_for_object(template, role_name)
        parent_ids = set(target.get('parent_receptacles') or [])
        for goal_role in goal_roles:
            receptacle = binding.get(goal_role)
            if receptacle is not None and _object_id(receptacle) in parent_ids:
                return True
        return False
    if mode in {'route_visible_surface', 'route_near_surface'}:
        if target.get('visible_from_initial_pose') is True:
            return True
        return True
    if mode == 'near_primary_object_floor':
        return True
    if mode == 'same_room_as':
        anchor_role = _same_room_anchor_role(template, role_name)
        anchor = binding.get(anchor_role) if anchor_role else None
        return bool(anchor is not None and profile_object_room_id(target) != profile_object_room_id(anchor))
    return False

def _relocation_required_for_binding(role_name: str, modes: Sequence[str], binding: Mapping[str, Mapping[str, Any]], profile: Mapping[str, Any], template: Mapping[str, Any]) -> bool:
    target = binding.get(role_name)
    if target is None:
        return False
    if 'unblock_from_closed_container' in modes and _closed_openable_parent_ids(target, profile):
        return True
    if 'evacuate_from_goal_receptacle' in modes:
        parent_ids = set(target.get('parent_receptacles') or [])
        for goal_role in _goal_receptacle_roles_for_object(template, role_name):
            receptacle = binding.get(goal_role)
            if receptacle is not None and _object_id(receptacle) in parent_ids:
                return True
    if 'near_primary_object_floor' in modes:
        return True
    return False

def _find_floor_candidates_for_mode(mode: str, role_name: str, binding: Mapping[str, Mapping[str, Any]], profile: Mapping[str, Any], template: Mapping[str, Any]) -> List[FloorPositionCandidate]:
    if mode != 'near_primary_object_floor':
        return []
    anchor_role = _floor_anchor_role(template, role_name)
    anchor = binding.get(anchor_role) if anchor_role else None
    if anchor is None:
        return []
    return _floor_candidates_near_anchor(profile, anchor, _scope_room_ids(template, binding, profile), anchor_role=anchor_role or '', min_distance=DEFAULT_FLOOR_RELOCATE_MIN_DISTANCE, max_distance=DEFAULT_FLOOR_RELOCATE_MAX_DISTANCE, max_candidates=DEFAULT_FLOOR_RELOCATE_MAX_CANDIDATES)

def _find_candidates_for_mode(mode: str, role_name: str, binding: Mapping[str, Mapping[str, Any]], profile: Mapping[str, Any], template: Mapping[str, Any], taxonomy: ObjectTaxonomy) -> List[PoseCandidate]:
    target = binding[role_name]
    target_object_id = _object_id(target)
    scope_room_ids = _scope_room_ids(template, binding, profile)
    agent_pose = _initial_agent_pose(profile)
    agent_pos = agent_pose.get('position') or {}
    if mode == 'unblock_from_closed_container':
        return find_pose_candidates_unblock_from_closed_container(profile, target_object_id, scope_room_ids, taxonomy, agent_pos)
    if mode == 'evacuate_from_goal_receptacle':
        candidates: List[PoseCandidate] = []
        for goal_role in _goal_receptacle_roles_for_object(template, role_name):
            goal_receptacle = binding.get(goal_role)
            if goal_receptacle is None:
                continue
            candidates.extend(find_pose_candidates_evacuate_from_goal_receptacle(profile, target_object_id, _object_id(goal_receptacle), scope_room_ids, taxonomy, agent_pos))
        return sorted(candidates, key=lambda item: item.distance_to_anchor)
    if mode == 'route_visible_surface':
        return _route_surface_candidates(binding, profile, template, taxonomy, strict_visibility=True)
    if mode == 'route_near_surface':
        return _route_surface_candidates(binding, profile, template, taxonomy, strict_visibility=False)
    if mode == 'same_room_as':
        anchor_role = _same_room_anchor_role(template, role_name)
        anchor = binding.get(anchor_role) if anchor_role else None
        if anchor is None:
            return []
        return find_pose_candidates_same_room_as(profile, _object_id(anchor), target_object_id, taxonomy)
    return []

def _floor_candidates_near_anchor(profile: Mapping[str, Any], anchor: Mapping[str, Any], scope_room_ids: Sequence[str], *, anchor_role: str, min_distance: float, max_distance: float, max_candidates: int) -> List[FloorPositionCandidate]:
    anchor_position = anchor.get('position') or {}
    if 'x' not in anchor_position or 'z' not in anchor_position:
        return []
    anchor_room_id = profile_object_room_id(anchor)
    scope_room_filter = sorted((room_id for room_id in scope_room_ids if room_id))
    reachable = _reachable_positions_from_profile(profile)
    room_geometry_available = any((isinstance(room, Mapping) and isinstance(room.get('floorPolygon') or room.get('floor_polygon'), list) and (len(room.get('floorPolygon') or room.get('floor_polygon') or []) >= 3) for room in profile.get('rooms') or []))
    reachable_room_data_available = any((isinstance(point.get('room_id') or point.get('roomId'), str) and bool(point.get('room_id') or point.get('roomId')) for point in reachable))
    room_membership_available = room_geometry_available or reachable_room_data_available
    candidates: List[FloorPositionCandidate] = []
    seen_positions = set()
    for index, point in enumerate(reachable):
        if 'x' not in point or 'z' not in point:
            continue
        point_room_id = _reachable_room_id(point, profile)
        if point_room_id is None and len(scope_room_filter) == 1 and (len(profile.get('rooms') or []) == 1):
            point_room_id = scope_room_filter[0]
        if scope_room_filter and room_membership_available and (point_room_id is None or point_room_id not in scope_room_filter):
            continue
        distance_to_anchor = _xz_distance(point, anchor_position)
        if distance_to_anchor < min_distance or distance_to_anchor > max_distance:
            continue
        floor_position = _floor_object_position_from_reachable_point(point, anchor_position)
        floor_room_id = _reachable_room_id(floor_position, profile)
        if floor_room_id is None and len(scope_room_filter) == 1 and (len(profile.get('rooms') or []) == 1):
            floor_room_id = scope_room_filter[0]
        if floor_room_id is None and point_room_id is not None:
            floor_room_id = point_room_id
        if scope_room_filter and room_membership_available and (floor_room_id is None or floor_room_id not in scope_room_filter):
            continue
        key = (round(float(floor_position['x']), 3), round(float(floor_position['y']), 3), round(float(floor_position['z']), 3))
        if key in seen_positions:
            continue
        seen_positions.add(key)
        candidates.append(FloorPositionCandidate(position=floor_position, distance_to_anchor=distance_to_anchor, rationale={'anchor_role': anchor_role, 'anchor_object_id': _object_id(anchor), 'anchor_room_id': anchor_room_id, 'scope_room_ids': scope_room_filter, 'reachable_position_index': index, 'reachable_position': {'x': float(point.get('x', 0.0)), 'y': float(point.get('y', 0.0)), 'z': float(point.get('z', 0.0))}, 'reachable_room_id': point_room_id, 'floor_room_id': floor_room_id, 'room_membership_available': room_membership_available, 'distance_to_anchor': distance_to_anchor, 'floor_position': copy.deepcopy(floor_position)}))
    return sorted(candidates, key=lambda item: (item.rationale.get('reachable_room_id') != anchor_room_id, item.distance_to_anchor, item.position['x'], item.position['z']))[:max_candidates]

def _floor_object_position_from_reachable_point(reachable_point: Mapping[str, Any], anchor_position: Mapping[str, Any]) -> dict:
    point_x = float(reachable_point.get('x', 0.0))
    point_z = float(reachable_point.get('z', 0.0))
    anchor_x = float(anchor_position.get('x', point_x))
    anchor_z = float(anchor_position.get('z', point_z))
    dx = anchor_x - point_x
    dz = anchor_z - point_z
    norm = math.hypot(dx, dz)
    if norm <= 1e-06:
        return {'x': point_x, 'y': 0.05, 'z': point_z}
    offset = min(DEFAULT_FLOOR_OBJECT_OFFSET_FROM_AGENT, max(0.1, norm * 0.5))
    return {'x': point_x + offset * dx / norm, 'y': 0.05, 'z': point_z + offset * dz / norm}

def _reachable_positions_from_profile(profile: Mapping[str, Any]) -> List[dict]:
    reachable = profile.get('reachable') or {}
    positions = reachable.get('positions') or reachable.get('reachable_positions') or []
    if positions:
        return [dict(position) for position in positions if isinstance(position, Mapping)]
    poses = reachable.get('sampled_agent_poses') or []
    return [dict(pose.get('position') or {}) for pose in poses if isinstance(pose, Mapping) and isinstance(pose.get('position'), Mapping)]

def _reachable_room_id(point: Mapping[str, Any], profile: Mapping[str, Any]) -> Optional[str]:
    room_id = point.get('room_id') or point.get('roomId')
    if isinstance(room_id, str) and room_id:
        return room_id
    rooms = profile.get('rooms') or []
    for room in rooms:
        if not isinstance(room, Mapping):
            continue
        polygon = room.get('floorPolygon') or room.get('floor_polygon')
        if _point_in_floor_polygon(point, polygon):
            candidate = room.get('room_id') or room.get('roomId') or room.get('id')
            if isinstance(candidate, str) and candidate:
                return candidate
    return None

def _point_in_floor_polygon(point: Mapping[str, Any], polygon: Any) -> bool:
    if 'x' not in point or 'z' not in point:
        return False
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    vertices = []
    for vertex in polygon:
        if not isinstance(vertex, Mapping):
            return False
        if 'x' not in vertex or 'z' not in vertex:
            return False
        vertices.append((float(vertex['x']), float(vertex['z'])))
    x = float(point['x'])
    z = float(point['z'])
    inside = False
    previous = len(vertices) - 1
    for current in range(len(vertices)):
        xi, zi = vertices[current]
        xj, zj = vertices[previous]
        if _point_on_segment(x, z, xi, zi, xj, zj):
            return True
        crosses = (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi or 1e-12) + xi
        if crosses:
            inside = not inside
        previous = current
    return inside

def _point_on_segment(x: float, z: float, x1: float, z1: float, x2: float, z2: float) -> bool:
    cross = (x - x1) * (z2 - z1) - (z - z1) * (x2 - x1)
    if abs(cross) > 1e-09:
        return False
    dot = (x - x1) * (x2 - x1) + (z - z1) * (z2 - z1)
    if dot < -1e-09:
        return False
    squared_len = (x2 - x1) ** 2 + (z2 - z1) ** 2
    return dot <= squared_len + 1e-09

def _route_surface_candidates(binding: Mapping[str, Mapping[str, Any]], profile: Mapping[str, Any], template: Mapping[str, Any], taxonomy: ObjectTaxonomy, *, strict_visibility: bool) -> List[PoseCandidate]:
    primary_object = _primary_object_for_route(template, binding)
    primary_receptacle = _primary_receptacle_for_route(template, binding)
    incident_dish = binding.get('incident_dish') or binding.get('target_object')
    if primary_object is None or primary_receptacle is None or incident_dish is None:
        return []
    excluded_ids = {_object_id(incident_dish), _object_id(primary_object), _object_id(primary_receptacle)}
    finder = find_pose_candidates_route_visible_surface if strict_visibility else find_pose_candidates_route_near_surface
    candidates = finder(profile, primary_object.get('position') or {}, primary_receptacle.get('position') or {}, _initial_agent_pose(profile), _scope_room_ids(template, binding, profile), taxonomy)
    return [candidate for candidate in candidates if candidate.receptacle_object_id not in excluded_ids]

def _scope_room_ids(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], profile: Mapping[str, Any]) -> List[str]:
    family_template = template.get('family_spec_template') or {}
    scope = family_template.get('scope') if isinstance(family_template, Mapping) else {}
    if isinstance(scope, Mapping):
        raw_room_ids = scope.get('room_ids')
        if isinstance(raw_room_ids, list):
            return [item for item in raw_room_ids if isinstance(item, str)]
        role_names = scope.get('room_ids_from_roles')
        if isinstance(role_names, list):
            room_ids = _room_ids_from_roles(role_names, binding)
            if room_ids:
                return room_ids
        role_name = scope.get('room_ids_from_role')
        bound = binding.get(role_name) if isinstance(role_name, str) else None
        room_id = profile_object_room_id(bound) if isinstance(bound, Mapping) else None
        if isinstance(room_id, str) and room_id:
            return [room_id]
    return []

def _goal_receptacle_roles_for_object(template: Mapping[str, Any], object_role: str) -> List[str]:
    roles: List[str] = []
    for predicate in template.get('goal_predicates', []) or []:
        if predicate.get('object_role') != object_role:
            continue
        if isinstance(predicate.get('receptacle_role'), str):
            roles.append(predicate['receptacle_role'])
        for role in predicate.get('receptacle_roles') or []:
            if isinstance(role, str):
                roles.append(role)
    return roles

def _same_room_anchor_role(template: Mapping[str, Any], role_name: str) -> Optional[str]:
    role_constraints = template.get('requires', {}).get(role_name, {})
    relocate_if = role_constraints.get('relocate_if', {})
    if isinstance(relocate_if, Mapping) and isinstance(relocate_if.get('anchor_role'), str):
        return relocate_if['anchor_role']
    same_room_as = role_constraints.get('same_room_as')
    return same_room_as if isinstance(same_room_as, str) else None

def _floor_anchor_role(template: Mapping[str, Any], role_name: str) -> Optional[str]:
    role_constraints = template.get('requires', {}).get(role_name, {})
    relocate_if = role_constraints.get('relocate_if', {})
    if isinstance(relocate_if, Mapping):
        anchor_role = relocate_if.get('to_floor_anchor_role')
        if isinstance(anchor_role, str) and anchor_role:
            return anchor_role
    return None

def _primary_object_for_route(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    recover = template.get('family_spec_template', {}).get('recover', {}).get('primary_task', {})
    role = recover.get('object_role')
    if isinstance(role, str) and role in binding:
        return binding[role]
    return binding.get('primary_object') or binding.get('target_object')

def _primary_receptacle_for_route(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    recover = template.get('family_spec_template', {}).get('recover', {}).get('primary_task', {})
    role = recover.get('receptacle_role')
    if isinstance(role, str) and role in binding:
        return binding[role]
    return binding.get('primary_receptacle') or binding.get('target_receptacle')

def _closed_openable_parent_ids(obj: Mapping[str, Any], profile: Mapping[str, Any]) -> set[str]:
    objects = {_object_id(candidate): candidate for candidate in profile.get('objects', [])}
    parent_ids = set()
    for parent_id in obj.get('parent_receptacles') or []:
        parent = objects.get(parent_id, {})
        if parent.get('openable') and parent.get('is_open') is False:
            parent_ids.add(parent_id)
    return parent_ids

def _profile_object_by_id(profile: Mapping[str, Any], object_id: str) -> Optional[Mapping[str, Any]]:
    for obj in profile.get('objects', []) or []:
        if isinstance(obj, Mapping) and _object_id(obj) == object_id:
            return obj
    return None

def _resolve_family_spec(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], profile: Optional[Mapping[str, Any]]=None) -> Optional[dict]:
    family_spec_template = template.get('family_spec_template')
    if not family_spec_template:
        return None
    out = copy.deepcopy(family_spec_template)
    scope = out.get('scope')
    if isinstance(scope, Mapping):
        if not isinstance(profile, Mapping):
            raise RoomSemanticError('ROOM_TYPE_UNKNOWN', 'scene profile is missing')
        scope_roles = {str(role_name): {'room_id': profile_object_room_id(bound)} for role_name, bound in binding.items() if isinstance(bound, Mapping)}
        scope = materialize_room_scope(scope, roles=scope_roles, room_catalog=build_room_type_catalog(profile.get('rooms') or []))
        out['scope'] = scope
    put_away = out.get('put_away')
    if isinstance(put_away, dict):
        target_prefix = put_away.pop('target_roles_prefix', None)
        distractor_prefix = put_away.pop('protected_distractor_roles_prefix', None)
        relocate_only_roles = _relocate_target_only_roles(template)
        if isinstance(target_prefix, str) and target_prefix:
            put_away['target_roles'] = sorted((role_name for role_name in binding if role_name.startswith(target_prefix) and role_name not in relocate_only_roles))
        if isinstance(distractor_prefix, str) and distractor_prefix:
            put_away['protected_distractor_roles'] = sorted((role_name for role_name in binding if role_name.startswith(distractor_prefix) and role_name not in relocate_only_roles))
    human = out.get('human')
    if isinstance(human, dict):
        events_template = human.pop('human_events_template', None)
        if events_template is not None:
            human['human_events'] = _resolve_human_events(events_template, binding)
    observation_conditioned = out.get('observation_conditioned')
    if isinstance(observation_conditioned, dict):
        out['observation_conditioned'] = materialize_observation_conditioned_spec(observation_conditioned, roles={role_name: {**dict(bound), 'room_id': profile_object_room_id(bound)} for role_name, bound in binding.items() if isinstance(bound, Mapping)})
    composite = out.get('composite')
    if isinstance(composite, dict):
        _resolve_composite_subtask_family_specs(composite, binding, profile)
    return out

def _attach_profile_source_metadata(task: dict, profile: Mapping[str, Any]) -> None:
    source = task.setdefault('source', {})
    for field in ('source_json', 'layout_variant_id', 'profile_source', 'occupancy_map_path', 'bev_map_path', 'labels_json_path'):
        value = profile.get(field)
        if value is not None:
            source[field] = value
    layout = profile.get('layout_variant')
    if isinstance(layout, Mapping):
        for field in ('source_json', 'layout_variant_id', 'profile_source', 'occupancy_map_path', 'bev_map_path', 'labels_json_path'):
            value = layout.get(field)
            if value is not None:
                source.setdefault(field, value)

def _resolve_human_events(events_template: Sequence[Mapping[str, Any]], binding: Mapping[str, Mapping[str, Any]]) -> List[dict]:
    if not isinstance(events_template, list):
        raise TaskGenerationError('human_events_template must be a list.')
    events: List[dict] = []
    for index, event_template in enumerate(events_template):
        if not isinstance(event_template, Mapping):
            raise TaskGenerationError('human_events_template[{}] must be a mapping.'.format(index))
        event = copy.deepcopy(dict(event_template))
        lookup = _role_format_lookup(binding)
        if 'question_template' in event:
            event['question'] = _format_template_text(event.pop('question_template'), lookup, 'human question_template')
        if 'instruction_template' in event:
            event['instruction'] = _format_template_text(event.pop('instruction_template'), lookup, 'human instruction_template')
        resolver = event.pop('evaluator_answer_resolver', None)
        if resolver is not None:
            event['evaluator_answer'] = _resolve_human_evaluator_answer(str(resolver), event, binding)
            if resolver == 'primary_receptacle_isOpen':
                event.setdefault('evaluator_answer_predicate', {'type': 'object_open', 'object_role': 'primary_receptacle', 'expected': True})
        events.append(event)
    return events

def _resolve_composite_subtask_family_specs(composite: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]], profile: Optional[Mapping[str, Any]]) -> None:
    subtasks = composite.get('subtasks')
    if not isinstance(subtasks, list):
        return
    for subtask in subtasks:
        if not isinstance(subtask, dict):
            continue
        sub_family_spec = subtask.get('family_spec')
        if not isinstance(sub_family_spec, dict):
            continue
        resolved = _resolve_family_spec({'family_spec_template': sub_family_spec}, binding, profile)
        if resolved is not None:
            subtask['family_spec'] = resolved
        goal_predicates = subtask.get('goal_predicates')
        if isinstance(goal_predicates, list):
            subtask['goal_predicates'] = _resolve_goal_predicates(goal_predicates, binding)
        disturbance_spec = subtask.get('disturbance_spec')
        if isinstance(disturbance_spec, Mapping):
            enhanced = effective_disturbance_spec_for_template({'template_id': subtask.get('template_id') or subtask.get('subtask_id') or ''}, disturbance_spec, binding=binding)
            if enhanced is not None:
                subtask['disturbance_spec'] = enhanced

def _resolve_composite_disturbance_spec(template: Mapping[str, Any]) -> Optional[dict]:
    family_template = template.get('family_spec_template')
    if not isinstance(family_template, Mapping):
        return None
    composite = family_template.get('composite')
    if not isinstance(composite, Mapping):
        return None
    subtasks = composite.get('subtasks')
    if not isinstance(subtasks, list):
        return None
    merged: Optional[dict] = None
    for subtask in subtasks:
        if not isinstance(subtask, Mapping):
            continue
        disturbance_spec = subtask.get('disturbance_spec')
        if isinstance(disturbance_spec, Mapping):
            merged = _merge_disturbance_specs(merged, _scope_disturbance_spec_to_subtask(disturbance_spec, subtask.get('subtask_id')))
    return merged

def _scope_disturbance_spec_to_subtask(spec: Mapping[str, Any], subtask_id: Any) -> dict:
    if not isinstance(subtask_id, str) or not subtask_id:
        return copy.deepcopy(dict(spec))
    scoped = copy.deepcopy(dict(spec))
    for key in ('disturbances', 'failures'):
        items = scoped.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            extra = item.get('extra')
            if not isinstance(extra, dict):
                extra = {}
            extra.setdefault('active_subtask_id', subtask_id)
            item['extra'] = extra
    return scoped

def _merge_disturbance_specs(base: Any, extra: Any) -> Optional[dict]:
    base_mapping = base if isinstance(base, Mapping) else {}
    extra_mapping = extra if isinstance(extra, Mapping) else {}
    disturbances = [copy.deepcopy(dict(item)) for item in base_mapping.get('disturbances', []) or [] if isinstance(item, Mapping)]
    disturbances.extend((copy.deepcopy(dict(item)) for item in extra_mapping.get('disturbances', []) or [] if isinstance(item, Mapping)))
    failures = [copy.deepcopy(dict(item)) for item in base_mapping.get('failures', []) or [] if isinstance(item, Mapping)]
    failures.extend((copy.deepcopy(dict(item)) for item in extra_mapping.get('failures', []) or [] if isinstance(item, Mapping)))
    if not disturbances and (not failures):
        return None
    seed = extra_mapping.get('seed') if 'seed' in extra_mapping else base_mapping.get('seed', 0)
    merged = {'seed': int(seed or 0)}
    if disturbances:
        merged['disturbances'] = disturbances
    if failures:
        merged['failures'] = failures
    return merged

def _resolve_human_evaluator_answer(resolver: str, event: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> str:
    if resolver == 'primary_receptacle_isOpen':
        receptacle = binding.get('primary_receptacle') or {}
        return 'yes' if _profile_bool(receptacle, 'is_open', 'isOpen') else 'no'
    if resolver == 'primary_object_in_inventory':
        activation = event.get('activation') or {}
        after_step = activation.get('after_step', 0)
        return 'yes' if isinstance(after_step, int) and after_step >= 2 else 'no'
    if resolver == 'post_place_next_action':
        receptacle = binding.get('primary_receptacle') or {}
        if bool(receptacle.get('openable', False)):
            return 'close the destination'
        return 'no destination action needed'
    raise TaskGenerationError("Unknown human evaluator_answer_resolver '{}'.".format(resolver))

def _profile_bool(obj: Mapping[str, Any], snake_key: str, camel_key: str, default: bool=False) -> bool:
    if snake_key in obj:
        return bool(obj.get(snake_key))
    if camel_key in obj:
        return bool(obj.get(camel_key))
    return bool(default)

def _role_format_lookup(binding: Mapping[str, Mapping[str, Any]]) -> dict:
    lookup = {}
    for role_name, obj in binding.items():
        object_type = _object_type(obj)
        type_lower = _humanize_object_type(object_type)
        lookup[role_name] = type_lower
        lookup['{}_type'.format(role_name)] = object_type
        lookup['{}_type_lower'.format(role_name)] = type_lower
        lookup['{}_alias'.format(role_name)] = type_lower.replace(' ', '_')
    return lookup

def _format_template_text(template_text: Any, lookup: Mapping[str, Any], context: str) -> str:
    if not isinstance(template_text, str) or not template_text:
        raise TaskGenerationError('{} must be a non-empty string.'.format(context))
    try:
        return template_text.format(**lookup)
    except KeyError as exc:
        raise TaskGenerationError("{} references unknown placeholder '{}'.".format(context, exc.args[0])) from exc

def _merge_human_insert_goal_predicates(goal_predicates: Sequence[Mapping[str, Any]], family_spec: Mapping[str, Any]) -> List[dict]:
    merged = [copy.deepcopy(dict(predicate)) for predicate in goal_predicates]
    seen = {_canonical_mapping_key(predicate) for predicate in merged}
    for predicate in _human_insert_goal_predicates(family_spec):
        key = _canonical_mapping_key(predicate)
        if key in seen:
            continue
        merged.append(copy.deepcopy(dict(predicate)))
        seen.add(key)
    return merged

def _merge_human_insert_antistate_overrides(init_overrides: Sequence[Mapping[str, Any]], family_spec: Mapping[str, Any]) -> List[dict]:
    merged = [copy.deepcopy(dict(override)) for override in init_overrides]
    seen = {_canonical_mapping_key(override) for override in merged}
    for predicate in _human_insert_goal_predicates(family_spec):
        override = _insert_antistate_override(predicate)
        if override is None:
            continue
        key = _canonical_mapping_key(override)
        if key in seen:
            continue
        merged.append(override)
        seen.add(key)
    return merged

def _human_insert_goal_predicates(family_spec: Mapping[str, Any]) -> List[dict]:
    human = family_spec.get('human')
    if not isinstance(human, Mapping):
        return []
    predicates: List[dict] = []
    for event in human.get('human_events') or []:
        if not isinstance(event, Mapping) or event.get('type') != 'insert':
            continue
        for predicate in event.get('goal_predicates') or []:
            if isinstance(predicate, Mapping):
                predicates.append(copy.deepcopy(dict(predicate)))
    return predicates

def _insert_antistate_override(predicate: Mapping[str, Any]) -> Optional[dict]:
    predicate_type = predicate.get('type')
    object_role = predicate.get('object_role')
    if not isinstance(object_role, str) or not object_role:
        return None
    if predicate_type == 'object_toggled_off':
        return {'type': 'set_toggle_state', 'object_role': object_role, 'is_toggled': True}
    if predicate_type == 'object_toggled_on':
        return {'type': 'set_toggle_state', 'object_role': object_role, 'is_toggled': False}
    if predicate_type == 'object_closed':
        return {'type': 'set_open_state', 'object_role': object_role, 'is_open': True}
    if predicate_type == 'object_open':
        return {'type': 'set_open_state', 'object_role': object_role, 'is_open': False}
    if predicate_type == 'object_clean':
        return {'type': 'set_dirty_state', 'object_role': object_role, 'is_dirty': True}
    return None

def _canonical_mapping_key(value: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple(((str(key), json.dumps(child, sort_keys=True)) for key, child in sorted(value.items(), key=lambda item: str(item[0]))))

def _room_ids_from_roles(role_names: Sequence[Any], binding: Mapping[str, Mapping[str, Any]]) -> List[str]:
    room_ids: List[str] = []
    seen = set()
    for role_name in role_names:
        if not isinstance(role_name, str):
            continue
        bound = binding.get(role_name)
        room_id = profile_object_room_id(bound) if isinstance(bound, Mapping) else None
        if not isinstance(room_id, str) or not room_id or room_id in seen:
            continue
        seen.add(room_id)
        room_ids.append(room_id)
    return room_ids

def _relocate_target_only_roles(template: Mapping[str, Any]) -> set:
    requires = template.get('requires')
    if not isinstance(requires, Mapping):
        return set()
    return {role_name for role_name, spec in requires.items() if isinstance(spec, Mapping) and spec.get('relocate_target_only')}

def _resolve_goal_predicates(predicates: Sequence[Mapping[str, Any]], binding: Mapping[str, Mapping[str, Any]]) -> List[dict]:
    resolved = []
    for predicate in predicates:
        if not _predicate_include_if_satisfied(predicate, binding):
            continue
        out = copy.deepcopy(dict(predicate))
        out.pop('include_if', None)
        resolved.append(out)
    return resolved

def _resolve_template_goal_predicates(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> List[dict]:
    top_level = _resolve_goal_predicates(template['goal_predicates'], binding)
    if template.get('category') != 'composite':
        return top_level
    merged: List[dict] = []
    seen = set()
    for predicate in _composite_subtask_goal_predicates(template, binding) + top_level:
        key = json.dumps(predicate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(predicate)
    return merged

def _composite_subtask_goal_predicates(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> List[dict]:
    family_template = template.get('family_spec_template')
    if not isinstance(family_template, Mapping):
        return []
    composite = family_template.get('composite')
    if not isinstance(composite, Mapping):
        return []
    subtasks = composite.get('subtasks')
    if not isinstance(subtasks, list):
        return []
    merged: List[dict] = []
    for index, subtask in enumerate(subtasks):
        if not isinstance(subtask, Mapping):
            continue
        predicates = subtask.get('goal_predicates')
        if not isinstance(predicates, list):
            continue
        subtask_id = subtask.get('subtask_id') or 'subtask_{}'.format(index)
        for predicate in _resolve_goal_predicates(predicates, binding):
            item = copy.deepcopy(dict(predicate))
            item.setdefault('owner_subtask_id', str(subtask_id))
            merged.append(item)
    return merged

def _predicate_include_if_satisfied(predicate: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> bool:
    include_if = predicate.get('include_if')
    if include_if is None:
        return True
    if isinstance(include_if, str):
        return include_if in binding
    if not isinstance(include_if, Mapping):
        raise TaskGenerationError('Predicate include_if must be a mapping or role string.')
    role_name = include_if.get('role')
    field_name = include_if.get('field')
    if not isinstance(role_name, str) or role_name not in binding:
        return False
    if field_name is None:
        return True
    if not isinstance(field_name, str) or not field_name:
        raise TaskGenerationError('Predicate include_if.field must be a non-empty string.')
    expected = include_if.get('equals', True)
    return binding[role_name].get(field_name) == expected

def _generate_tidy_tasks(profile: Mapping[str, Any], template: Mapping[str, Any], taxonomy: ObjectTaxonomy, rng: random.Random, limit: int) -> Tuple[List[dict], Optional[str]]:
    kitchen_room_ids = [str(room['room_id']) for room in profile.get('rooms', []) or [] if isinstance(room, Mapping) and isinstance(room.get('room_id'), str) and ('kitchen' in str(room.get('room_type') or '').lower())]
    if not kitchen_room_ids:
        scope_options = []
    else:
        scope_options = [('single_room', 'kitchen', kitchen_room_ids)]
    scope_options.append(('whole_house', 'house', []))
    for scope_type, scope_label, scope_room_ids in scope_options:
        tasks = _sample_tidy_tasks_for_scope(profile, template, taxonomy, rng, limit=limit, scope_type=scope_type, scope_label=scope_label, scope_room_ids=scope_room_ids)
        if tasks:
            return (tasks, None)
    return ([], 'No compatible tidy violation sample.')

def _sample_tidy_tasks_for_scope(profile: Mapping[str, Any], template: Mapping[str, Any], taxonomy: ObjectTaxonomy, rng: random.Random, limit: int, scope_type: str, scope_label: str, scope_room_ids: Sequence[str]) -> List[dict]:
    from scoper.benchmark.procthor.tidy_violation_sampler import sample_violations
    tasks: List[dict] = []
    seen = set()
    attempts = max(10, int(limit) * 12)
    for _ in range(attempts):
        if len(tasks) >= int(limit):
            break
        sample_seed = rng.randrange(2 ** 31)
        violations = sample_violations(profile, scope_room_ids, taxonomy, seed=sample_seed)
        if not violations:
            continue
        key = tuple(sorted(((violation.v_type, violation.object_id, tuple(sorted(violation.support_roles.items()))) for violation in violations)))
        if key in seen:
            continue
        seen.add(key)
        tasks.append(_build_tidy_task_spec(profile, template, violations, scope_room_ids, scope_type=scope_type, scope_label=scope_label))
    return tasks

def _build_tidy_task_spec(profile: Mapping[str, Any], template: Mapping[str, Any], violations: Sequence[Any], scope_room_ids: Sequence[str], scope_type: str, scope_label: str) -> dict:
    objects_by_id = {_object_id(obj): obj for obj in profile.get('objects', [])}
    roles: Dict[str, dict] = {}
    init_overrides: List[dict] = []
    goal_predicates: List[dict] = []
    violation_specs: List[dict] = []
    role_counters: Counter = Counter()

    def bind_role(prefix: str, object_id: str) -> str:
        role_name = '{}_{}'.format(prefix, role_counters[prefix])
        role_counters[prefix] += 1
        roles[role_name] = _role_entry(objects_by_id[object_id])
        return role_name
    for index, violation in enumerate(violations):
        if violation.v_type == 'faucet_on':
            object_role = bind_role('faucet', violation.object_id)
            init_overrides.append({'type': 'set_toggle_state', 'object_role': object_role, 'is_toggled': True})
            goal_predicates.append({'type': 'object_toggled_off', 'object_role': object_role})
            support_roles: Dict[str, str] = {}
        elif violation.v_type == 'open_container':
            object_role = bind_role('container', violation.object_id)
            init_overrides.append({'type': 'set_open_state', 'object_role': object_role, 'is_open': True})
            goal_predicates.append({'type': 'object_closed', 'object_role': object_role})
            support_roles = {}
        elif violation.v_type == 'dirty_dish':
            object_role = bind_role('dish', violation.object_id)
            init_overrides.append({'type': 'set_dirty_state', 'object_role': object_role, 'is_dirty': True})
            sink_role = bind_role('sink_basin', violation.support_roles['sink'])
            faucet_role = bind_role('faucet', violation.support_roles['faucet'])
            storage_role = bind_role('storage', violation.support_roles['storage'])
            support_roles = {'sink': sink_role, 'faucet': faucet_role, 'storage': storage_role}
            goal_predicates.append({'type': 'object_clean', 'object_role': object_role})
            goal_predicates.append({'type': 'object_in_receptacle', 'object_role': object_role, 'receptacle_role': storage_role})
            storage_obj = objects_by_id[violation.support_roles['storage']]
            if bool(storage_obj.get('openable', False)):
                goal_predicates.append({'type': 'object_closed', 'object_role': storage_role})
        else:
            raise TaskGenerationError("Unsupported tidy violation type '{}'.".format(violation.v_type))
        violation_specs.append({'id': 'violation_{}'.format(index), 'type': violation.v_type, 'object_role': object_role, 'urgency': violation.urgency, 'support_roles': support_roles})
    instruction_template = template['instruction_templates'][0]
    if '{scope_label}' in instruction_template or '{scope_room_label}' in instruction_template:
        instruction = instruction_template.format(scope_label=scope_label, scope_room_label=scope_label)
    else:
        instruction = instruction_template.replace('the kitchen', 'the {}'.format(scope_label))
    family_template = copy.deepcopy(template['family_spec_template'])
    tidy_template = family_template['tidy']
    task = {'id': 'pending', 'version': TASK_SPEC_VERSION, 'backend': 'procthor', 'category': 'tidy_by_policy', 'source': {'house_id': profile['house_id'], 'house_path': profile.get('house_path') or '', 'split': profile.get('split') or 'unknown'}, 'template': {'template_id': template['template_id'], 'template_version': template['template_version']}, 'instruction': instruction, 'roles': roles, 'init': {'agent_pose': _initial_agent_pose(profile), 'object_state_overrides': init_overrides}, 'goal': {'type': 'predicate', 'predicates': goal_predicates, 'require_model_stop': bool(template.get('goal_options', {}).get('require_model_stop', False))}, 'online_requirements': {'expected_events': list(template['expected_events']), 'expected_brain_behaviors': list(template['expected_brain_behaviors'])}, 'validation': {'status': 'candidate', 'checks': {}, 'failure_reason': None}, 'family_spec': {'family': 'tidy_by_policy', 'scope': {'type': scope_type, 'label': scope_label, 'room_ids': list(scope_room_ids)}, 'tidy': {'policy_text': tidy_template['policy_text'], 'policy_rules': list(tidy_template['policy_rules']), 'violations': violation_specs}}, 'tags': sorted({'tidy_by_policy', template['template_id'], *(violation.v_type for violation in violations), *(_humanize_object_type(role['object_type']).replace(' ', '_') for role in roles.values())})}
    return task

def _role_entry(obj: Mapping[str, Any]) -> dict:
    return {'object_id': _object_id(obj), 'object_type': _object_type(obj)}

def _assign_task_ids(tasks: Sequence[Mapping[str, Any]], max_tasks_per_category: Optional[int]) -> List[dict]:
    template_counts = Counter()
    category_counts = Counter()
    numbered = []
    for task in sorted(tasks, key=lambda item: item['_generation_key']):
        template_id = task.get('template', {}).get('template_id', 'unknown_template')
        category = task['category']
        if max_tasks_per_category is not None and category_counts[category] >= max_tasks_per_category:
            continue
        category_counts[category] += 1
        template_counts[template_id] += 1
        numbered_task = {key: value for key, value in task.items() if key != '_generation_key'}
        numbered_task['id'] = 'procthor_{}_{:06d}'.format(template_id, template_counts[template_id])
        numbered.append(numbered_task)
    return numbered

def _filter_visual_discoverability(tasks: Sequence[Mapping[str, Any]], env_factory: Optional[Any]) -> Tuple[List[dict], dict]:
    if env_factory is None:
        return (list(tasks), {'skipped': True})
    from scoper.benchmark.procthor.visual_discoverability_validator import has_visual_discovery_incident, validate_incident_visual_discoverability
    env = env_factory()
    kept: List[dict] = []
    rejected = 0
    rejection_reasons: Counter = Counter()
    try:
        for task in tasks:
            if not has_visual_discovery_incident(task):
                kept.append(dict(task))
                continue
            task_copy = dict(task)
            result = validate_incident_visual_discoverability(task_copy, env, geometry_estimator_already_passed=_task_geometry_prefilter_passed(task_copy))
            if result.discoverability_passed:
                task_copy.setdefault('metadata', {})['visual_discoverability'] = {'passed': True, 'source': result.incident_visibility_source, 'first_visible_tool_step': result.incident_first_visible_tool_step, 'first_visible_stream_step': result.incident_first_visible_stream_step}
                kept.append(task_copy)
            else:
                rejected += 1
                rejection_reasons[result.failure_reason or 'unknown'] += 1
    finally:
        close = getattr(env, 'close', None)
        if callable(close):
            close()
    return (kept, {'skipped': False, 'num_passed': len(kept), 'num_rejected': rejected, 'rejection_reasons': dict(sorted(rejection_reasons.items()))})

def _task_geometry_prefilter_passed(task: Mapping[str, Any]) -> Optional[bool]:
    for override in task.get('init', {}).get('object_pose_overrides', []) or []:
        if not isinstance(override, Mapping):
            continue
        if override.get('mode') != 'route_visible_surface':
            continue
        if 'geometry_estimator_already_passed' in override:
            return bool(override['geometry_estimator_already_passed'])
    return None

def _sample_bindings(template: Mapping[str, Any], bindings: Sequence[Mapping[str, Mapping[str, Any]]], limit: int, rng: random.Random) -> List[Mapping[str, Mapping[str, Any]]]:
    stable_bindings = sorted(bindings, key=_binding_key)
    if len(stable_bindings) <= limit:
        return stable_bindings
    if _template_has_route_visible_relocate(template):
        route_diverse = _sample_route_diverse_bindings(stable_bindings, limit, rng)
        if route_diverse:
            return route_diverse
    selected_indices = sorted(rng.sample(range(len(stable_bindings)), limit))
    return [stable_bindings[index] for index in selected_indices]

def _sample_route_diverse_bindings(stable_bindings: Sequence[Mapping[str, Mapping[str, Any]]], limit: int, rng: random.Random) -> List[Mapping[str, Mapping[str, Any]]]:
    by_route_key: Dict[Tuple[str, str, str], List[Mapping[str, Mapping[str, Any]]]] = defaultdict(list)
    for binding in stable_bindings:
        route_key = _route_visible_binding_key(binding)
        if route_key is None:
            continue
        by_route_key[route_key].append(binding)
    if not by_route_key:
        return []
    sampled: List[Mapping[str, Mapping[str, Any]]] = []
    for route_key in sorted(by_route_key):
        options = by_route_key[route_key]
        sampled.append(options[rng.randrange(len(options))])
        if len(sampled) >= limit:
            return sorted(sampled, key=_binding_key)
    seen_keys = {_binding_key(binding) for binding in sampled}
    remaining = [binding for binding in stable_bindings if _binding_key(binding) not in seen_keys]
    if remaining and len(sampled) < limit:
        sample_count = min(limit - len(sampled), len(remaining))
        sampled.extend(rng.sample(remaining, sample_count))
    return sorted(sampled, key=_binding_key)

def _route_visible_binding_key(binding: Mapping[str, Mapping[str, Any]]) -> Optional[Tuple[str, str, str]]:
    dish = binding.get('incident_dish')
    primary_object = binding.get('primary_object') or binding.get('target_object')
    primary_receptacle = binding.get('primary_receptacle') or binding.get('target_receptacle')
    if dish is None or primary_object is None or primary_receptacle is None:
        return None
    return (_object_id(dish), _object_id(primary_object), _object_id(primary_receptacle))

def _binding_attempt_limit(template: Mapping[str, Any], output_limit: int) -> int:
    if template.get('template_id') == 'mid_execution_container_re_closed_001':
        return min(100, max(int(output_limit) * 20, int(output_limit) + 20))
    if not _template_has_relocate(template):
        return int(output_limit)
    return min(100, max(int(output_limit) * 20, int(output_limit) + 20))

def _template_has_route_visible_relocate(template: Mapping[str, Any]) -> bool:
    for role_constraints in template.get('requires', {}).values():
        if not isinstance(role_constraints, Mapping):
            continue
        relocate_if = role_constraints.get('relocate_if')
        if not isinstance(relocate_if, Mapping):
            continue
        modes = relocate_if.get('modes') or []
        if 'route_visible_surface' in modes:
            return True
    return False

def _template_has_relocate(template: Mapping[str, Any]) -> bool:
    for role_constraints in template.get('requires', {}).values():
        if isinstance(role_constraints, Mapping) and isinstance(role_constraints.get('relocate_if'), Mapping):
            return True
    return False

def _generation_key(task: Mapping[str, Any]) -> Tuple[Any, ...]:
    role_items = tuple(((role_name, role['object_id']) for role_name, role in sorted(task['roles'].items())))
    return (task['category'], task['source']['house_id'], task['template']['template_id'], role_items, task['instruction'])

def _binding_key(binding: Mapping[str, Mapping[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    return tuple(((role_name, _object_id(obj)) for role_name, obj in sorted(binding.items())))

def _profile_sort_key(profile: Mapping[str, Any]) -> Tuple[str, str]:
    return (str(profile.get('split') or ''), str(profile.get('house_id') or ''))

def _instantiate_instruction(instruction_template: str, binding: Mapping[str, Mapping[str, Any]], *, scope_room_label: Optional[str]=None, scope_room_labels: Optional[Sequence[str]]=None) -> str:
    role_names = {role_name: _humanize_object_type(_object_type(obj)) for role_name, obj in binding.items()}
    role_names.setdefault('scope_room_label', 'the specified room')
    role_names.setdefault('scope_room_labels', 'the specified room')
    if isinstance(scope_room_label, str) and scope_room_label:
        role_names['scope_room_label'] = scope_room_label
    if isinstance(scope_room_labels, Sequence) and (not isinstance(scope_room_labels, (str, bytes))):
        labels = [str(label) for label in scope_room_labels if str(label)]
        if labels:
            if len(labels) == 1:
                role_names['scope_room_labels'] = labels[0]
            elif len(labels) == 2:
                role_names['scope_room_labels'] = '{} and {}'.format(labels[0], labels[1])
            else:
                role_names['scope_room_labels'] = '{}, and {}'.format(', '.join(labels[:-1]), labels[-1])
    return instruction_template.format(**role_names)

def _humanize_object_type(object_type: str) -> str:
    words = re.sub('(?<!^)(?=[A-Z])', ' ', object_type).replace('_', ' ')
    return words.lower()

def _initial_agent_pose(profile: Mapping[str, Any]) -> dict:
    poses = profile.get('reachable', {}).get('sampled_agent_poses') or []
    if poses:
        return copy.deepcopy(poses[0])
    return {'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'rotation': {'y': 0.0}, 'horizon': 0.0}

def _initial_state_overrides(initial_state_constraints: Sequence[Mapping[str, Any]]) -> List[dict]:
    overrides = []
    for constraint in initial_state_constraints:
        constraint_type = constraint.get('type')
        if constraint_type in {'object_open', 'object_closed'}:
            overrides.append({'type': 'set_open_state', 'object_role': constraint['object_role'], 'is_open': constraint_type == 'object_open'})
        elif constraint_type in {'object_toggled_on', 'object_toggled_off'}:
            overrides.append({'type': 'set_toggle_state', 'object_role': constraint['object_role'], 'is_toggled': constraint_type == 'object_toggled_on'})
        elif constraint_type in {'object_dirty', 'object_clean'}:
            overrides.append({'type': 'set_dirty_state', 'object_role': constraint['object_role'], 'is_dirty': constraint_type == 'object_dirty'})
    return overrides

def _task_tags(template: Mapping[str, Any], binding: Mapping[str, Mapping[str, Any]]) -> List[str]:
    tags = {template['category'], template['template_id']}
    for obj in binding.values():
        tags.add(_humanize_object_type(_object_type(obj)).replace(' ', '_'))
    return sorted(tags)

def _object_id(obj: Mapping[str, Any]) -> str:
    object_id = obj.get('object_id') or obj.get('objectId')
    if not isinstance(object_id, str) or not object_id:
        raise TaskGenerationError('Profile object is missing object_id.')
    return object_id

def _object_type(obj: Mapping[str, Any]) -> str:
    object_type = obj.get('object_type') or obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    return _object_id(obj).split('|', 1)[0]

def _xz_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax = float(a.get('x', 0.0))
    az = float(a.get('z', 0.0))
    bx = float(b.get('x', 0.0))
    bz = float(b.get('z', 0.0))
    return math.hypot(ax - bx, az - bz)

def _empty_summary(scene_profiles: Sequence[Mapping[str, Any]], templates: Sequence[Mapping[str, Any]]) -> dict:
    return {'num_profiles': len(scene_profiles), 'num_templates': len(templates), 'num_tasks': 0, 'category_counts': {}, 'house_counts': {}, 'template_counts': {}, 'object_type_counts': {}, 'target_to_receptacle_counts': {}, 'rejections': []}
if __name__ == '__main__':
    main()
__all__ = ['DEFAULT_MAX_TASKS_PER_CATEGORY_PER_HOUSE', 'DEFAULT_MAX_TASKS_PER_HOUSE', 'DEFAULT_MAX_TASKS_PER_TEMPLATE_PER_HOUSE', 'TASK_SPEC_VERSION', 'TaskGenerationError', 'generate_candidate_tasks', 'load_scene_profiles', 'summarize_generated_tasks', 'write_task_bank']
