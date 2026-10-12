from __future__ import annotations
import copy
import math
import random
from collections import Counter
from functools import lru_cache
from typing import Any, Mapping, Sequence
from scoper.benchmark.behavior.behavior_closed_loop_tool_env import BehaviorClosedLoopToolEnv
from scoper.benchmark.behavior.brains.teachers.dispatcher import default_dispatcher
from scoper.benchmark.behavior.episode_runner import EpisodeRunner
from scoper.benchmark.behavior.task_generation.template_activation import TEMPLATE_DISABLED_FAILURE_CODE, load_active_template_ids
from scoper.benchmark.behavior.task_normalization import materialize_tidy_policy_initial_violations
from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec
from scoper.benchmark.behavior.scene.placement_target_policy import supports_generic_surface_placement
from scoper.benchmark.behavior.scene_geometry.rules import load_rules
from scoper.benchmark.behavior.scene_geometry.surface_constraints import support_matches_surface_constraints, support_surface_descriptor_from_values, surface_constraints_from_policy
from scoper.benchmark.common import build_room_type_catalog, materialize_observation_conditioned_spec, materialize_room_scope
STAGE4_5_GOLD_TEMPLATE_IDS = ('static_place_001', 'dynamic_state_change_toggle_001', 'failure_recovery_partial_open_container_001', 'failure_recovery_partial_close_container_001')
SUPPORTED_STAGE4_GOAL_TYPES = {'object_in_receptacle', 'object_not_in_receptacle', 'object_in_any_receptacle', 'object_not_in_any_receptacle', 'object_open', 'object_closed', 'object_toggled_on', 'object_toggled_off', 'object_clean', 'object_dirty', 'object_sliced', 'object_filled', 'object_soaked', 'object_hot', 'object_cool', 'object_cooked', 'object_held', 'object_not_held', 'agent_near'}
STATE_INIT_MAP = {'object_open': 'open', 'object_closed': 'closed', 'object_toggled_on': 'toggled_on', 'object_toggled_off': 'toggled_off', 'object_dirty': 'dirty', 'object_clean': 'clean', 'object_sliced': 'sliced', 'object_cool': 'cooled', 'object_filled': 'filled'}
DEFAULT_EXTENTS = {'Apple': [0.12, 0.12, 0.12], 'Bowl': [0.2, 0.2, 0.12], 'Cup': [0.14, 0.14, 0.16], 'Mug': [0.14, 0.14, 0.16], 'Plate': [0.24, 0.24, 0.04], 'Potato': [0.12, 0.1, 0.1], 'Tomato': [0.12, 0.12, 0.1], 'Knife': [0.26, 0.04, 0.03], 'SprayBottle': [0.12, 0.1, 0.24], 'SoapBottle': [0.12, 0.1, 0.22], 'CounterTop': [0.9, 0.8, 0.18], 'DiningTable': [0.9, 0.9, 0.75], 'CoffeeTable': [0.8, 0.8, 0.45], 'Shelf': [0.8, 0.35, 1.2], 'Cabinet': [0.8, 0.6, 1.2], 'Fridge': [0.8, 0.7, 1.7], 'Microwave': [0.45, 0.45, 0.35], 'SinkBasin': [0.7, 0.75, 0.9], 'LightSwitch': [0.08, 0.04, 0.12], 'Faucet': [0.16, 0.16, 0.2], 'FloorLamp': [0.25, 0.25, 1.4], 'DeskLamp': [0.25, 0.25, 0.5]}
CONTAINER_TYPES = {'Cabinet', 'Drawer', 'Fridge', 'Microwave', 'GarbageCan', 'Bowl', 'Cup', 'Mug', 'Plate'}
CUTTABLE_FOOD_TYPES = {'Apple', 'Potato', 'Tomato'}
SOAKABLE_OBJECT_TYPES = CUTTABLE_FOOD_TYPES | {'Bowl', 'Cup', 'Mug', 'Plate'}
WIPEABLE_OBJECT_TYPES = SOAKABLE_OBJECT_TYPES | {'SprayBottle', 'SoapBottle'}
CLEANING_TOOL_TYPES = {'Rag', 'Sponge', 'Cloth', 'Towel'}
SPAWN_SUPPORT_ROLE_PREFIX = 'spawn_support_'
SPAWN_SUPPORT_CLEARANCE = 0.02
SPAWN_SUPPORT_MARGIN = 0.06
SPAWN_SUPPORT_MIN_CENTER_DISTANCE = 0.35
SPAWN_TABLE_OBJECT_TYPES = {'DiningTable', 'CoffeeTable', 'SideTable', 'CounterTop'}
SPAWN_SUPPORT_TYPE_PRIORITY = {'DiningTable': 0, 'CoffeeTable': 1, 'SideTable': 2, 'CounterTop': 3, 'Shelf': 20, 'Cabinet': 30, 'Microwave': 40, 'SinkBasin': 50}
SPAWN_SUPPORT_OBJECT_TYPES = set(SPAWN_SUPPORT_TYPE_PRIORITY)
NON_SPAWN_SUPPORT_CATEGORIES = {'floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'}

def generate_stage4_tasks(*, profiles: Sequence[Mapping[str, Any]], templates: Sequence[Mapping[str, Any]], match_records: Sequence[Mapping[str, Any]], template_ids: Sequence[str] | None=None, out_of_scope_failure_code: str='TEMPLATE_NOT_IN_STAGE4_SCOPE', max_tasks_per_template_scene: int=1, seed: int=0) -> dict[str, Any]:
    rng = random.Random(seed)
    templates_by_id = {str(template.get('template_id')): template for template in templates}
    profiles_by_scene = {str(profile.get('scene_model')): profile for profile in profiles}
    if template_ids is None:
        allowed = set(load_active_template_ids())
        template_filter_failure_code = TEMPLATE_DISABLED_FAILURE_CODE
    else:
        allowed = {str(template_id) for template_id in template_ids}
        template_filter_failure_code = out_of_scope_failure_code
    tasks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in sorted(match_records, key=lambda item: (str(item.get('scene_model')), str(item.get('template_id')))):
        template_id = str(record.get('template_id'))
        scene_model = str(record.get('scene_model'))
        if template_id not in allowed:
            rejected.append(reject_record(record, template_filter_failure_code))
            continue
        template = templates_by_id.get(template_id)
        profile = profiles_by_scene.get(scene_model)
        if template is None or profile is None:
            rejected.append(reject_record(record, 'MISSING_TEMPLATE_OR_PROFILE'))
            continue
        if not bool(record.get('profile_match')) and (not str(record.get('status', '')).startswith('matched')):
            rejected.append(reject_record(record, 'NO_STAGE3_MATCH_BINDING'))
            continue
        unsupported = unsupported_goal_types(template)
        if unsupported:
            rejected.append(reject_record(record, 'UNSUPPORTED_GOAL_PREDICATE', {'goal_types': unsupported}))
            continue
        generated_for_pair = 0
        for binding in record.get('bindings') or []:
            if generated_for_pair >= max_tasks_per_template_scene:
                break
            try:
                task = build_task_spec_from_binding(template=template, match_record=record, binding=binding, scene_profile=profile, seed=seed, variant_index=generated_for_pair, rng=rng)
                validate_behavior_task_spec(task)
            except Exception as exc:
                rejected.append(reject_record(record, 'TASK_SPEC_VALIDATION_FAILED', {'error': str(exc)}))
                continue
            tasks.append(task)
            generated_for_pair += 1
        if generated_for_pair == 0:
            rejected.append(reject_record(record, 'NO_VALID_BINDING_MATERIALIZED'))
    return {'tasks': tasks, 'rejected': rejected, 'summary': summarize_generated_tasks(tasks, rejected)}

def build_task_spec_from_binding(*, template: Mapping[str, Any], match_record: Mapping[str, Any], binding: Mapping[str, Any], scene_profile: Mapping[str, Any], seed: int, variant_index: int, rng: random.Random) -> dict[str, Any]:
    template_id = str(template.get('template_id'))
    effective_binding = effective_binding_for_template(template, binding, scene_profile=scene_profile)
    roles = build_roles(effective_binding)
    init = translate_initial_constraints(template.get('initial_state_constraints') or [], effective_binding)
    init_assertions = deferred_initial_assertions(template.get('initial_state_constraints') or [], effective_binding)
    goal_predicates = translate_goal_predicates(template.get('goal_predicates') or [], effective_binding, template)
    goal_predicates = effective_goal_predicates_for_template(template, effective_binding, goal_predicates)
    assign_spawn_centers(roles, goal_predicates, scene_profile=scene_profile)
    materialize_relocate_if(effective_relocation_template(template), roles, scene_profile=scene_profile)
    effective_roles_for_template(template, roles, scene_profile=scene_profile)
    family_spec = materialize_family_spec_template(template, effective_binding, scene_profile)
    family_spec = effective_family_spec_for_template(template, family_spec, roles=roles)
    instruction = render_instruction(template, effective_binding, rng, scope_room_label=(family_spec or {}).get('scope', {}).get('label'))
    instruction = effective_instruction_for_template(template, effective_binding, instruction, roles=roles)
    disturbance_spec, failure_spec, unsupported_scheduler_specs = materialize_scheduler_specs(template)
    disturbance_spec = effective_disturbance_spec_for_template(template, disturbance_spec, roles=roles)
    readiness_debt = stage4_readiness_debt(effective_binding, family_spec, disturbance_spec, failure_spec, unsupported_scheduler_specs, template)
    scene_model = str(match_record.get('scene_model'))
    layout_variant_id = scene_profile.get('layout_variant_id')
    scene_token = scene_model
    if layout_variant_id:
        scene_token = '{}_{}'.format(scene_model, _safe_task_id_token(str(layout_variant_id)))
    task_id = 'behavior_{}_{}_seed{:04d}_{:02d}'.format(scene_token, template_id, seed, variant_index)
    source = {'type': 'template', 'scene_model': scene_model, 'source_json': scene_profile.get('source_json'), 'layout_variant_id': layout_variant_id, 'template_id': template_id, 'template_version': template.get('template_version'), 'stage3_match_status': match_record.get('status'), 'occupancy_bounds': occupancy_bounds_from_profile(scene_profile), 'start_pose': start_pose_from_profile(scene_profile, roles=roles, goal_predicates=goal_predicates), 'room_catalog': room_catalog_from_profile(scene_profile)}
    task = {'id': task_id, 'version': 'stage4.0', 'backend': 'behavior', 'category': str(template.get('category')), 'source': source, 'template': {'template_id': template_id, 'template_version': template.get('template_version'), 'template_file': match_record.get('template_file')}, 'instruction': instruction, 'roles': roles, 'init': init, 'init_source_predicates': list(template.get('initial_state_constraints') or []), 'goal': {'type': 'predicates', 'require_model_stop': bool((template.get('goal_options') or {}).get('require_model_stop', False)), 'bddl_goal': None, 'predicates': goal_predicates}, 'generation': {'stage': 'stage4_profile_driven', 'seed': seed, 'stage4_readiness': stage4_readiness_from_debt(readiness_debt), 'stage4_readiness_debt': sorted(readiness_debt), 'required_init_mutations': effective_binding.get('required_init_mutations') or [], 'runtime_capabilities_required': effective_binding.get('runtime_capabilities_required') or [], 'binding_notes': effective_binding.get('notes') or [], 'deferred_init_assertions': init_assertions, 'family_spec_materialized': family_spec, 'unsupported_scheduler_specs': unsupported_scheduler_specs}, 'reset_plan': {'state_and_relation_init': init, 'assertions': init_assertions, 'required_init_mutations': effective_binding.get('required_init_mutations') or [], 'runtime_capabilities_required': effective_binding.get('runtime_capabilities_required') or []}, 'oracle_teacher_hint': {'family': (template.get('family_spec_template') or {}).get('family') or template.get('category'), 'expected_events': list(template.get('expected_events') or []), 'expected_brain_behaviors': list(template.get('expected_brain_behaviors') or []), 'expected_plan_skeleton': goal_plan_skeleton(goal_predicates)}, 'metrics_config': {'track_predicate_progress': True, 'track_tool_success_rate': True, 'require_model_stop': bool((template.get('goal_options') or {}).get('require_model_stop', False))}, 'online_requirements': {'expected_events': list(template.get('expected_events') or []), 'expected_brain_behaviors': list(template.get('expected_brain_behaviors') or [])}, 'validation': {'status': 'generated_pending_reset_validation', 'checks': []}, 'tags': stage4_tags(template_id)}
    if family_spec is not None:
        task['family_spec'] = family_spec
    if disturbance_spec:
        task['disturbance_spec'] = disturbance_spec
    if failure_spec:
        task['failure_spec'] = failure_spec
    task = dict(materialize_tidy_policy_initial_violations(task))
    from scoper.benchmark.behavior.scene.concrete_categories import apply_concrete_public_names
    apply_concrete_public_names(task)
    from scoper.benchmark.common.task_identity_policy import enabled, assert_distinct
    if enabled():
        assert_distinct(task)
    return task

def _safe_task_id_token(text: str) -> str:
    chars = []
    for ch in str(text):
        if ch.isalnum() or ch in {'-', '_'}:
            chars.append(ch)
        else:
            chars.append('_')
    token = ''.join(chars).strip('_')
    return token or 'layout_variant'

def distinct_secondary_asset(primary, scene_profile):
    from scoper.benchmark.common.task_identity_policy import enabled
    if not enabled():
        return copy.deepcopy(dict(primary))
    from scoper.benchmark.behavior.task_generation.scene_matcher import serialize_role_binding
    families = ({'Bowl', 'Plate', 'Mug', 'Cup'}, {'Apple', 'Potato', 'Tomato', 'Lettuce', 'Bread', 'Egg'}, {'Book', 'Pencil', 'Pen'}, {'SoapBar', 'SoapBottle', 'SprayBottle', 'DishSponge'})
    typ = primary.get('object_type')
    family = next((group for group in families if typ in group), set())
    choices = [c for c in (scene_profile or {}).get('spawnable_catalog', []) if c.get('object_type') in family - {typ} and c.get('pickupable') and ('license' not in str(c.get('synset', '')))]
    if not choices:
        raise ValueError('NO_DISTINCT_COMPATIBLE_SECONDARY_ASSET')
    choices.sort(key=lambda c: (str(c.get('object_type')), str(c.get('object_id'))))
    return serialize_role_binding('distinct_secondary', choices[0])

def effective_binding_for_template(template: Mapping[str, Any], binding: Mapping[str, Any], *, scene_profile=None) -> dict[str, Any]:
    template_id = str(template.get('template_id'))
    if template_id not in {'mid_execution_container_re_closed_001', 'mid_execution_state_re_dirty_001', 'mid_execution_object_relocated_001'}:
        return copy.deepcopy(dict(binding))
    effective = copy.deepcopy(dict(binding))
    roles = dict(effective.get('roles') or {})
    if template_id == 'mid_execution_state_re_dirty_001':
        primary_room_ids = {str((roles.get(role_name) or {}).get('room_id') or '') for role_name in ('sink_basin', 'faucet', 'target_receptacle')}
        if '' in primary_room_ids or len(primary_room_ids) != 1:
            raise ValueError('re_dirty_primary_roles_require_one_room')
    if template_id == 'mid_execution_container_re_closed_001' and 'target_object' in roles and ('target_object_2' not in roles):
        second = distinct_secondary_asset(roles['target_object'], scene_profile)
        second['source_kind'] = 'spawnable_asset'
        second['requires_spawn'] = True
        for key in ('object_id', 'position', 'center', 'spawn_placement'):
            second.pop(key, None)
        notes = list(second.get('notes') or [])
        notes.append('generated_second_container_reclosed_target')
        second['notes'] = notes
        roles['target_object_2'] = second
    if template_id == 'mid_execution_state_re_dirty_001' and 'target_object' in roles and ('secondary_object' not in roles):
        secondary = distinct_secondary_asset(roles['target_object'], scene_profile)
        secondary['source_kind'] = 'spawnable_asset'
        secondary['requires_spawn'] = True
        secondary['init_states'] = {'dirty': False}
        for key in ('object_id', 'position', 'center', 'spawn_placement'):
            secondary.pop(key, None)
        notes = list(secondary.get('notes') or [])
        notes.append('generated_secondary_re_dirty_work_item')
        secondary['notes'] = notes
        roles['secondary_object'] = secondary
    effective['roles'] = roles
    notes = list(effective.get('notes') or [])
    if template_id == 'mid_execution_container_re_closed_001' and 'target_object_2:generated_for_container_re_closed_sequence' not in notes:
        notes.append('target_object_2:generated_for_container_re_closed_sequence')
    if template_id == 'mid_execution_state_re_dirty_001' and 'secondary_object:generated_for_re_dirty_interruption_task' not in notes:
        notes.append('secondary_object:generated_for_re_dirty_interruption_task')
    effective['notes'] = notes
    return effective

def effective_relocation_template(template: Mapping[str, Any]) -> Mapping[str, Any]:
    if str(template.get('template_id')) != 'mid_execution_state_re_dirty_001':
        return template
    effective = copy.deepcopy(dict(template))
    requires = copy.deepcopy(dict(effective.get('requires') or {}))
    target = copy.deepcopy(dict(requires.get('target_object') or {}))
    target['same_room_as'] = 'sink_basin'
    target['relocate_if'] = {'modes': ['same_room_as']}
    requires['target_object'] = target
    effective['requires'] = requires
    return effective

def effective_roles_for_template(template: Mapping[str, Any], roles: dict[str, Any], *, scene_profile: Mapping[str, Any] | None=None) -> None:
    template_id = str(template.get('template_id'))
    if template_id == 'mid_execution_object_relocated_001':
        if 'faucet' not in roles:
            faucet = select_interrupt_faucet_role(roles, scene_profile)
            if faucet is not None:
                roles['faucet'] = materialize_profile_support_role(faucet, 'faucet')
                roles['faucet']['generated_role'] = 'object_relocated_interrupt_faucet'
                roles['faucet']['init_states'] = dict(roles['faucet'].get('init_states') or {})
                roles['faucet']['init_states']['toggled_on'] = False
        if 'staging_receptacle' not in roles:
            target_object = roles.get('target_object')
            candidate = select_object_relocated_staging_receptacle(roles, target_object, scene_profile)
            if candidate is not None:
                entry = materialize_profile_support_role(candidate, 'staging_receptacle')
                entry.pop('auxiliary_spawn_support', None)
                entry['generated_role'] = 'object_relocated_staging_receptacle'
                roles['staging_receptacle'] = entry
        return
    if template_id != 'mid_execution_state_re_dirty_001':
        return
    if 'secondary_receptacle' in roles:
        return
    secondary = roles.get('secondary_object') or roles.get('target_object')
    if not secondary:
        return
    candidate = select_re_dirty_secondary_receptacle(roles, secondary, scene_profile)
    if candidate is None:
        raise ValueError('re_dirty_secondary_table_receptacle_unavailable')
    entry = materialize_profile_support_role(candidate, 'secondary_receptacle')
    entry.pop('auxiliary_spawn_support', None)
    entry['generated_role'] = 're_dirty_secondary_task_receptacle'
    roles['secondary_receptacle'] = entry

def select_interrupt_faucet_role(roles: Mapping[str, Any], scene_profile: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    for role_name in ('faucet', 'interrupt_role'):
        role = roles.get(role_name)
        if isinstance(role, Mapping) and str(role.get('object_type') or '') == 'Faucet':
            return role
    if not scene_profile:
        return None
    sink_center = role_center_or_center(roles.get('sink_basin') or {})
    target_center = role_center_or_center(roles.get('target_object') or {}) or role_center_or_center(roles.get('target_receptacle') or {})
    candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    used_object_ids = {str(role.get('object_id')) for role in roles.values() if role.get('object_id')}
    for candidate in scene_profile.get('objects') or []:
        object_id = str(candidate.get('object_id') or '')
        if not object_id or object_id in used_object_ids:
            continue
        if str(candidate.get('object_type') or '') != 'Faucet':
            continue
        if not candidate.get('toggleable'):
            continue
        center = role_center_or_center(candidate)
        score = (xy_distance(center, sink_center), xy_distance(center, target_center), str(candidate.get('room_id') or ''), object_id)
        candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]

def select_object_relocated_staging_receptacle(roles: Mapping[str, Any], target_object: Mapping[str, Any] | None, scene_profile: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not target_object:
        return None
    candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    used_object_ids = {str(role.get('object_id')) for role in roles.values() if role.get('object_id')}
    excluded_roles = {'target_object', 'target_receptacle', 'faucet', 'sink_basin'}
    excluded_object_ids = {str((roles.get(role_name) or {}).get('object_id')) for role_name in excluded_roles if (roles.get(role_name) or {}).get('object_id')}
    anchor_center = role_center_or_center(roles.get('target_object') or {}) or role_center_or_center(roles.get('target_receptacle') or {})
    final_center = role_center_or_center(roles.get('target_receptacle') or {})
    for role_name, role in roles.items():
        if role_name in excluded_roles or role.get('requires_spawn'):
            continue
        if not spawn_support_can_hold(role, target_object):
            continue
        if support_overlaps_role(role, target_object):
            continue
        center = role_center_or_center(role)
        candidates.append(((0, 0 if is_table_like_spawn_support(role) else 1, xy_distance(center, anchor_center), -xy_distance(center, final_center), str(role.get('category') or ''), str(role.get('object_id') or role_name)), role))
    if scene_profile:
        for candidate in scene_profile.get('objects') or []:
            object_id = str(candidate.get('object_id') or '')
            if not object_id or object_id in used_object_ids or object_id in excluded_object_ids:
                continue
            if candidate.get('requires_spawn'):
                continue
            if not spawn_support_can_hold(candidate, target_object):
                continue
            if support_overlaps_role(candidate, target_object):
                continue
            center = role_center_or_center(candidate)
            candidates.append(((1, 0 if is_table_like_spawn_support(candidate) else 1, xy_distance(center, anchor_center), -xy_distance(center, final_center), str(candidate.get('category') or ''), object_id), candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]

def select_re_dirty_secondary_receptacle(roles: Mapping[str, Any], secondary_object: Mapping[str, Any], scene_profile: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    candidates: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
    used_object_ids = {str(role.get('object_id')) for role in roles.values() if role.get('object_id')}
    excluded_roles = {'target_object', 'secondary_object', 'target_receptacle', 'sink_basin', 'faucet'}
    excluded_object_ids = {str((roles.get(role_name) or {}).get('object_id')) for role_name in excluded_roles if (roles.get(role_name) or {}).get('object_id')}
    sink = roles.get('sink_basin') or {}
    target_receptacle = roles.get('target_receptacle') or {}
    sink_center = role_center_or_center(sink)
    target_center = role_center_or_center(target_receptacle) or sink_center
    sink_room = str(sink.get('room_id') or '')
    for candidate in (scene_profile or {}).get('objects') or []:
        object_id = str(candidate.get('object_id') or '')
        if not object_id or object_id in used_object_ids or object_id in excluded_object_ids:
            continue
        if candidate.get('requires_spawn'):
            continue
        if not is_table_like_spawn_support(candidate):
            continue
        if not spawn_support_can_hold(candidate, secondary_object):
            continue
        if support_overlaps_role(candidate, secondary_object):
            continue
        center = role_center_or_center(candidate)
        if center is None:
            continue
        sink_distance = xy_distance(center, sink_center)
        target_distance = xy_distance(center, target_center)
        candidate_room = str(candidate.get('room_id') or '')
        same_room_penalty = 1 if sink_room and candidate_room == sink_room else 0
        near_penalty = 1 if sink_distance < 2.0 else 0
        score = (near_penalty, same_room_penalty, -sink_distance, -target_distance, SPAWN_SUPPORT_TYPE_PRIORITY.get(str(candidate.get('object_type') or ''), 100), str(candidate.get('category') or ''), object_id)
        candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]

def effective_goal_predicates_for_template(template: Mapping[str, Any], binding: Mapping[str, Any], goal_predicates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = [copy.deepcopy(dict(predicate)) for predicate in goal_predicates]
    if str(template.get('template_id')) != 'mid_execution_container_re_closed_001':
        return out
    roles = binding.get('roles') or {}
    if 'target_object_2' not in roles:
        return out
    second_goal = {'type': goal_relation_for_receptacle(binding, 'target_receptacle'), 'args': {'object': 'target_object_2', 'receptacle': 'target_receptacle'}}
    if not any((predicate == second_goal for predicate in out)):
        insert_at = len(out)
        for idx, predicate in enumerate(out):
            if predicate.get('type') == 'object_closed':
                insert_at = idx
                break
        out.insert(insert_at, second_goal)
    return out

def effective_family_spec_for_template(template: Mapping[str, Any], family_spec: Mapping[str, Any] | None, *, roles: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    template_id = str(template.get('template_id'))
    if family_spec is None:
        return family_spec
    effective = copy.deepcopy(dict(family_spec))
    primary = (effective.get('mid_execution') or {}).get('primary_task') or {}
    if template_id == 'mid_execution_container_re_closed_001' and isinstance(primary, dict):
        primary['target_objects'] = ['target_object', 'target_object_2']
    elif template_id == 'mid_execution_object_relocated_001' and isinstance(primary, dict):
        primary['belief_navigation_enabled'] = True
        primary['scenario'] = str(primary.get('scenario') or 'staging_interrupt')
        primary['relocation_activation'] = 'after_staging_interrupt'
        primary['staging_receptacle_role'] = str(primary.get('staging_receptacle_role') or 'staging_receptacle')
        primary['interrupt_role'] = str(primary.get('interrupt_role') or 'faucet')
        if 'interrupt_tools' not in primary:
            primary['interrupt_tools'] = ['navigate_to', 'toggle_on', 'toggle_off']
    elif template_id == 'semantic_relocate_after_interruption_001' and isinstance(primary, dict):
        primary['belief_navigation_enabled'] = True
        primary['scenario'] = str(primary.get('scenario') or 'staging_interrupt')
        primary['relocation_activation'] = 'after_staging_interrupt'
    elif template_id == 'mid_execution_state_re_dirty_001' and isinstance(primary, dict):
        sink_role = str(primary.get('sink_role') or 'sink_basin')
        secondary_receptacle_role = 'secondary_receptacle' if 'secondary_receptacle' in (roles or {}) else str(primary.get('receptacle_role') or 'target_receptacle')
        primary['staging_receptacle_role'] = sink_role
        primary['secondary_task'] = {'type': 'place_object', 'object_role': 'secondary_object', 'receptacle_role': secondary_receptacle_role}
    return effective

def effective_disturbance_spec_for_template(template: Mapping[str, Any], disturbance_spec: Sequence[Mapping[str, Any]], *, roles: Mapping[str, Any] | None=None) -> list[dict[str, Any]]:
    specs = [copy.deepcopy(dict(item)) for item in disturbance_spec]
    template_id = str(template.get('template_id'))
    if template_id == 'mid_execution_state_re_dirty_001':
        secondary_receptacle_role = 'secondary_receptacle' if 'secondary_receptacle' in (roles or {}) else 'target_receptacle'
        for item in specs:
            if item.get('type') != 'dirty_object':
                continue
            item['requires_out_of_view'] = True
            item['after_step'] = 0
            item['activation'] = {'type': 'after_tool_result', 'tool': 'put_down', 'target_role': secondary_receptacle_role, 'success': True, 'state_delta_object_role': 'secondary_object', 'requires_target_not_held': True, 'required_parent_role': 'sink_basin'}
        return specs
    if template_id == 'mid_execution_object_relocated_001':
        for item in specs:
            if item.get('type') != 'relocate_object':
                continue
            item['requires_out_of_view'] = True
            item['after_step'] = 0
            item['activation'] = {'type': 'after_tool_result', 'tool': 'put_down', 'target_role': 'staging_receptacle', 'success': True, 'after_distinct_tool_count': 3, 'requires_target_not_held': True, 'required_parent_role': 'staging_receptacle'}
            extra = dict(item.get('extra') or {})
            exclude_roles = list(extra.get('exclude_roles') or [])
            for role in ('target_receptacle', 'staging_receptacle', 'target_object', 'faucet'):
                if role not in exclude_roles:
                    exclude_roles.append(role)
            extra['exclude_roles'] = exclude_roles
            extra['avoid_current_view'] = True
            extra['avoid_last_seen_view'] = True
            extra['prefer_different_room'] = True
            extra['min_distance'] = 1.0
            extra['max_distance'] = 4.0
            extra['max_candidates'] = 8
            item['extra'] = extra
        return specs
    if template_id == 'semantic_relocate_after_interruption_001':
        for item in specs:
            if item.get('type') != 'relocate_object':
                continue
            item['requires_out_of_view'] = True
            item['after_step'] = 0
            activation = dict(item.get('activation') or {})
            activation.setdefault('type', 'after_tool_result')
            activation.setdefault('tool', 'put_down')
            activation.setdefault('target_role', 'staging_receptacle')
            activation.setdefault('success', True)
            activation.setdefault('after_distinct_tool_count', 2)
            activation.setdefault('requires_target_not_held', True)
            activation.setdefault('required_parent_role', 'staging_receptacle')
            item['activation'] = activation
            extra = dict(item.get('extra') or {})
            exclude_roles = list(extra.get('exclude_roles') or [])
            for role in ('target_receptacle', 'staging_receptacle', 'target_object', 'faucet'):
                if role not in exclude_roles:
                    exclude_roles.append(role)
            extra['exclude_roles'] = exclude_roles
            extra['avoid_current_view'] = True
            extra['avoid_last_seen_view'] = True
            extra['prefer_different_room'] = True
            extra['min_distance'] = 1.0
            extra['max_distance'] = 4.0
            extra['max_candidates'] = 8
            item['extra'] = extra
        return specs
    if template_id != 'mid_execution_container_re_closed_001':
        return specs
    for item in specs:
        if item.get('type') != 'close_object':
            continue
        item['requires_out_of_view'] = True
        item['after_step'] = 0
        item['activation'] = {'type': 'after_tool_result', 'tool': 'put_down', 'target_role': 'target_receptacle', 'success': True, 'min_delay_steps': 1, 'after_distinct_tool_count': 1, 'exclude_target_roles': ['target_receptacle'], 'requires_target_not_held': False}
    return specs

def effective_instruction_for_template(template: Mapping[str, Any], binding: Mapping[str, Any], instruction: str, *, roles: Mapping[str, Any] | None=None) -> str:
    template_id = str(template.get('template_id'))
    if template_id == 'mid_execution_object_relocated_001':
        labels = {role_name: str(role.get('category') or role.get('object_type') or role_name).replace('_', ' ') for role_name, role in (binding.get('roles') or {}).items()}
        labels.update({role_name: str(role.get('category') or role.get('object_type') or role_name).replace('_', ' ') for role_name, role in (roles or {}).items()})
        target_object = labels.get('target_object', 'object')
        staging_receptacle = labels.get('staging_receptacle', 'staging receptacle')
        target_receptacle = labels.get('target_receptacle', 'target receptacle')
        return 'Find the {target_object}, stage it on the {staging_receptacle}, turn the faucet on and back off, then return and put the {target_object} in the {target_receptacle} even if it has moved.'.format(target_object=target_object, staging_receptacle=staging_receptacle, target_receptacle=target_receptacle)
    if template_id == 'mid_execution_state_re_dirty_001':
        labels = {role_name: str(role.get('category') or role.get('object_type') or role_name).replace('_', ' ') for role_name, role in (binding.get('roles') or {}).items()}
        labels.update({role_name: str(role.get('category') or role.get('object_type') or role_name).replace('_', ' ') for role_name, role in (roles or {}).items()})
        target_object = labels.get('target_object', 'object')
        secondary_object = labels.get('secondary_object', 'object')
        target_receptacle = labels.get('target_receptacle', 'target receptacle')
        secondary_receptacle = labels.get('secondary_receptacle', target_receptacle)
        return 'Clean the {target_object} at the sink, leave it staged there, put the {secondary_description} in the {secondary_receptacle}, then return to check the {target_object}; if it has become dirty again, clean it again before putting it in the {target_receptacle}.'.format(target_object=target_object, secondary_object=secondary_object, secondary_description=('second ' if secondary_object == target_object else '') + secondary_object, secondary_receptacle=secondary_receptacle, target_receptacle=target_receptacle)
    if template_id != 'mid_execution_container_re_closed_001':
        return instruction
    labels = {role_name: str(role.get('category') or role.get('object_type') or role_name).replace('_', ' ') for role_name, role in (binding.get('roles') or {}).items()}
    target_object = labels.get('target_object', 'object')
    target_object_2 = labels.get('target_object_2', target_object)
    receptacle = labels.get('target_receptacle', 'receptacle')
    if target_object == target_object_2:
        raise ValueError('RECLOSED_REQUIRES_DISTINGUISHABLE_OBJECTS')
    return 'Put the {} and the {} in the {}, then close it.'.format(target_object, target_object_2, receptacle)

def build_roles(binding: Mapping[str, Any]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    synset_counts: Counter[str] = Counter()
    for role_name, role in sorted((binding.get('roles') or {}).items()):
        synset = str(role.get('synset') or '{}.n.01'.format(role.get('category') or role_name))
        synset_counts[synset] += 1
        entry = {'synset': synset, 'bddl_inst': '{}_{}'.format(synset, synset_counts[synset]), 'category': str(role.get('category') or role.get('object_type') or role_name), 'object_id': role.get('object_id'), 'object_type': role.get('object_type'), 'source_kind': role.get('source_kind'), 'requires_spawn': bool(role.get('requires_spawn')), 'abilities': role_abilities(role), 'extent': role_extent(role), 'init_states': role_init_states(role), 'room_id': role.get('room_id'), 'position': role.get('position')}
        for field in ('bbox_size', 'scale'):
            vector = number_vector(role.get(field), 3)
            if vector is not None:
                entry[field] = vector
        for field in ('model', 'catalog_object_id'):
            value = role.get(field)
            if value not in (None, 'unknown'):
                entry[field] = value
        for field in ('root_position', 'root_orientation_xyzw'):
            vector = number_vector(role.get(field), 3 if field == 'root_position' else 4)
            if vector is not None:
                entry[field] = vector
            elif field == 'root_position' and isinstance(role.get(field), Mapping):
                position = role[field]
                if all((key in position for key in ('x', 'y', 'z'))):
                    try:
                        entry[field] = {key: float(position[key]) for key in ('x', 'y', 'z')}
                    except (TypeError, ValueError):
                        pass
        root_yaw = role.get('root_yaw')
        if isinstance(root_yaw, (int, float)):
            entry['root_yaw'] = float(root_yaw)
        interaction_faces = role.get('interaction_faces')
        if isinstance(interaction_faces, Sequence) and (not isinstance(interaction_faces, (str, bytes))):
            entry['interaction_faces'] = [copy.deepcopy(dict(face)) for face in interaction_faces if isinstance(face, Mapping)]
        if isinstance(role.get('spawn_constraints'), Mapping):
            entry['spawn_constraints'] = copy.deepcopy(dict(role['spawn_constraints']))
        center = role_center(role)
        if center is not None:
            entry['center'] = center
        roles[str(role_name)] = entry
    return roles

def assign_spawn_centers(roles: dict[str, Any], goal_predicates: Sequence[Mapping[str, Any]], *, scene_profile: Mapping[str, Any] | None=None) -> None:
    anchors = spawn_goal_anchors(goal_predicates)
    support_slots: dict[str, int] = {}
    placed_spawns: list[dict[str, Any]] = []
    fallback_anchor = next((name for name, role in roles.items() if 'center' in role and (not role.get('requires_spawn'))), None)
    for role_name in list(roles):
        role = roles[role_name]
        if not role.get('requires_spawn') or 'center' in role:
            continue
        anchor_name = anchors.get(role_name) or fallback_anchor
        anchor = roles.get(anchor_name or '')
        support_name, support = choose_spawn_support(role_name, role, roles, anchor_name=anchor_name, goal_anchor_name=anchors.get(role_name), scene_profile=scene_profile, support_slots=support_slots, placed_spawns=placed_spawns)
        if support_name and support:
            occupied_items = surface_occupants_for_support(scene_profile, support)
            slot, center, offset = available_spawn_support_slot(role, support_name, support, support_slots, placed_spawns, occupied_items=occupied_items)
            if slot is None or center is None or offset is None:
                continue
            support_slots[support_name] = max(support_slots.get(support_name, 0), slot + 1)
            role['center'] = center
            role['spawn_placement'] = {'mode': 'on_support_surface', 'anchor_role': anchor_name, 'support_role': support_name, 'offset': offset, 'slot': slot, 'clearance': SPAWN_SUPPORT_CLEARANCE, 'background_occupants_checked': len(occupied_items)}
            placed_spawns.append({'role': role_name, 'center': center, 'extent': role.get('extent') or [0.2, 0.2, 0.2]})
            continue
        if not anchor or 'center' not in anchor:
            continue
        ax, ay, _ = anchor['center']
        anchor_extent = anchor.get('extent') or [0.8, 0.8, 0.8]
        extent = role.get('extent') or [0.2, 0.2, 0.2]
        offset = float(anchor_extent[0]) * 0.5 + float(extent[0]) * 0.5 + 0.35
        role['center'] = [float(ax) + offset, float(ay), float(extent[2]) * 0.5]
        role['spawn_placement'] = {'mode': 'near_anchor_floor_fallback', 'anchor_role': anchor_name, 'offset': [offset, 0.0, 0.0], 'fallback_reason': 'no_available_spawn_support_surface'}
        placed_spawns.append({'role': role_name, 'center': role['center'], 'extent': role.get('extent') or [0.2, 0.2, 0.2]})

def materialize_relocate_if(template: Mapping[str, Any], roles: dict[str, Any], *, scene_profile: Mapping[str, Any] | None=None) -> None:
    support_slots: dict[str, int] = {}
    placed: list[dict[str, Any]] = []
    for role_name, requirement in (template.get('requires') or {}).items():
        relocate = requirement.get('relocate_if') if isinstance(requirement, Mapping) else None
        role = roles.get(str(role_name))
        if not isinstance(relocate, Mapping) or not isinstance(role, dict):
            continue
        if 'pickupable' not in set(role.get('abilities') or []):
            continue
        modes = [str(mode) for mode in relocate.get('modes') or []]
        fallback = relocate.get('fallback_to')
        if isinstance(fallback, str) and fallback not in modes:
            modes.append(fallback)
        anchor_name = relocate.get('to_floor_anchor_role') or relocate.get('anchor_role') or requirement.get('same_room_as')
        anchor = roles.get(str(anchor_name)) if isinstance(anchor_name, str) else None
        if not isinstance(anchor, Mapping):
            continue
        room_id = str(anchor.get('room_id') or '')
        same_room_required = 'same_room_as' in modes and str(role.get('room_id') or '') != room_id
        floor_required = 'near_primary_object_floor' in modes
        if not same_room_required and (not floor_required):
            continue
        if floor_required:
            center, offset = relocation_floor_center(role, anchor, roles=roles, scene_profile=scene_profile)
            placement = {'mode': 'near_anchor_floor_fallback', 'anchor_role': str(anchor_name), 'offset': offset, 'relocation_mode': 'near_primary_object_floor'}
        else:
            support_name, support = choose_spawn_support(str(role_name), role, roles, anchor_name=str(anchor_name), goal_anchor_name=None, scene_profile=scene_profile, support_slots=support_slots, placed_spawns=placed, room_id=room_id or None)
            if support_name and support:
                slot, center, offset = available_spawn_support_slot(role, str(support_name), support, support_slots, placed, occupied_items=surface_occupants_for_support(scene_profile, support))
            else:
                slot, center, offset = (None, None, None)
            if slot is not None and center is not None and (offset is not None):
                support_slots[str(support_name)] = slot + 1
                placement = {'mode': 'on_support_surface', 'anchor_role': str(anchor_name), 'support_role': str(support_name), 'offset': offset, 'slot': slot, 'clearance': SPAWN_SUPPORT_CLEARANCE, 'relocation_mode': 'same_room_as'}
            else:
                center, offset = relocation_floor_center(role, anchor, roles=roles, scene_profile=scene_profile)
                placement = {'mode': 'near_anchor_floor_fallback', 'anchor_role': str(anchor_name), 'offset': offset, 'relocation_mode': 'same_room_as', 'fallback_reason': 'no_available_room_support_surface'}
        role['center'] = [float(value) for value in center]
        role['room_id'] = room_id or role.get('room_id')
        role['spawn_placement'] = placement
        placed.append({'role': str(role_name), 'center': role['center'], 'extent': role.get('extent') or [0.2, 0.2, 0.2]})

def relocation_floor_center(role: Mapping[str, Any], anchor: Mapping[str, Any], *, roles: Mapping[str, Mapping[str, Any]], scene_profile: Mapping[str, Any] | None, candidate_validator: Any | None=None) -> tuple[list[float], list[float]]:
    anchor_center = role_center_or_center(anchor) or [0.0, 0.0, 0.0]
    anchor_extent = number_vector(anchor.get('extent'), 3) or [0.8, 0.8, 0.8]
    extent = number_vector(role.get('extent'), 3) or [0.2, 0.2, 0.2]
    radius = max(float(anchor_extent[0]), float(anchor_extent[1])) * 0.5
    radius += max(float(extent[0]), float(extent[1])) * 0.5 + 0.35
    blockers = relocation_floor_blockers(role, roles, scene_profile)
    diagonal = math.sqrt(0.5)
    directions = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (diagonal, diagonal), (-diagonal, diagonal), (-diagonal, -diagonal), (diagonal, -diagonal))
    for ring in range(4):
        candidate_radius = radius + 0.35 * ring
        for direction_x, direction_y in directions:
            center = [float(anchor_center[0]) + direction_x * candidate_radius, float(anchor_center[1]) + direction_y * candidate_radius, float(extent[2]) * 0.5]
            if not relocation_center_within_bounds(center, extent, scene_profile):
                continue
            if any((aabb_vectors_overlap(center, extent, blocker_center, blocker_extent, tolerance=0.01) for blocker_center, blocker_extent in blockers)):
                continue
            if candidate_validator is not None and (not bool(candidate_validator(center, extent))):
                continue
            offset = [float(center[0]) - float(anchor_center[0]), float(center[1]) - float(anchor_center[1]), float(center[2]) - float(anchor_center[2])]
            return (center, offset)
    raise ValueError('No collision-free floor relocation candidate near anchor.')

def relocation_floor_blockers(role: Mapping[str, Any], roles: Mapping[str, Mapping[str, Any]], scene_profile: Mapping[str, Any] | None) -> list[tuple[list[float], list[float]]]:
    target_id = str(role.get('object_id') or '')
    candidates = list(roles.values()) + list((scene_profile or {}).get('objects') or [])
    blockers = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if target_id and str(candidate.get('object_id') or '') == target_id:
            continue
        category = str(candidate.get('category') or '').lower()
        if category in {'floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'}:
            continue
        center = role_center_or_center(candidate)
        candidate_extent = number_vector(candidate.get('extent'), 3) or number_vector(candidate.get('bbox_size'), 3)
        if center is not None and candidate_extent is not None:
            blockers.append((center, candidate_extent))
    return blockers

def relocation_center_within_bounds(center: Sequence[float], extent: Sequence[float], scene_profile: Mapping[str, Any] | None) -> bool:
    bounds = (scene_profile or {}).get('bounds') or {}
    values = (bounds.get('min_x'), bounds.get('min_z'), bounds.get('max_x'), bounds.get('max_z'))
    if not all((isinstance(value, (int, float)) for value in values)):
        return True
    min_x, min_y, max_x, max_y = [float(value) for value in values]
    half_x, half_y = (float(extent[0]) * 0.5, float(extent[1]) * 0.5)
    return min_x + half_x <= float(center[0]) <= max_x - half_x and min_y + half_y <= float(center[1]) <= max_y - half_y

def spawn_goal_anchors(goal_predicates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    anchors: dict[str, str] = {}
    for predicate in goal_predicates:
        args = predicate.get('args') or {}
        obj = args.get('object')
        rec = args.get('receptacle')
        if isinstance(obj, str) and isinstance(rec, str):
            anchors[obj] = rec
    return anchors

def choose_spawn_support(role_name: str, role: Mapping[str, Any], roles: dict[str, Any], *, anchor_name: str | None, goal_anchor_name: str | None, scene_profile: Mapping[str, Any] | None, support_slots: Mapping[str, int], placed_spawns: Sequence[Mapping[str, Any]], room_id: str | None=None) -> tuple[str | None, Mapping[str, Any] | None]:
    existing = existing_spawn_support_candidates(role_name, role, roles, anchor_name, goal_anchor_name, room_id=room_id)
    profile_candidates = profile_spawn_support_candidates(role, roles, anchor_name, goal_anchor_name, scene_profile, room_id=room_id)
    for table_like_only in (True, False):
        for support_name, support in existing:
            if table_like_only and (not is_table_like_spawn_support(support)):
                continue
            if not table_like_only and is_table_like_spawn_support(support):
                continue
            occupied_items = surface_occupants_for_support(scene_profile, support)
            slot, _center, _offset = available_spawn_support_slot(role, support_name, support, support_slots, placed_spawns, occupied_items=occupied_items)
            if slot is not None:
                return (support_name, support)
        for profile_candidate in profile_candidates:
            if table_like_only and (not is_table_like_spawn_support(profile_candidate)):
                continue
            if not table_like_only and is_table_like_spawn_support(profile_candidate):
                continue
            support_role_name = next_auxiliary_spawn_support_role_name(roles)
            support = materialize_profile_support_role(profile_candidate, support_role_name)
            occupied_items = surface_occupants_for_support(scene_profile, support)
            slot, _center, _offset = available_spawn_support_slot(role, support_role_name, support, support_slots, placed_spawns, occupied_items=occupied_items)
            if slot is None:
                continue
            roles[support_role_name] = support
            return (support_role_name, roles[support_role_name])
    return (None, None)

def existing_spawn_support_candidates(role_name: str, role: Mapping[str, Any], roles: Mapping[str, Any], anchor_name: str | None, goal_anchor_name: str | None, *, room_id: str | None=None) -> list[tuple[str, Mapping[str, Any]]]:
    anchor = roles.get(anchor_name or '')
    goal_anchor = roles.get(goal_anchor_name or '')
    anchor_center = number_vector((anchor or {}).get('center'), 3) if anchor else None
    out: list[tuple[str, Mapping[str, Any], tuple[Any, ...]]] = []
    for support_name, support in roles.items():
        if support_name == role_name:
            continue
        if support.get('requires_spawn'):
            continue
        if room_id and str(support.get('room_id') or '') != room_id:
            continue
        if goal_anchor_name and support_name == goal_anchor_name:
            continue
        if goal_anchor and support_overlaps_role(support, goal_anchor):
            continue
        if not spawn_support_can_hold(support, role):
            continue
        out.append((support_name, support, spawn_support_sort_key(support, anchor, anchor_center)))
    out.sort(key=lambda item: item[2])
    return [(name, support) for name, support, _key in out]

def profile_spawn_support_candidates(role: Mapping[str, Any], roles: Mapping[str, Any], anchor_name: str | None, goal_anchor_name: str | None, scene_profile: Mapping[str, Any] | None, *, room_id: str | None=None) -> list[Mapping[str, Any]]:
    if not scene_profile:
        return []
    anchor = roles.get(anchor_name or '')
    goal_anchor = roles.get(goal_anchor_name or '')
    anchor_center = role_center_or_center(anchor or {})
    used_object_ids = {str(item.get('object_id')) for item in roles.values() if item.get('object_id')}
    candidates: list[tuple[Mapping[str, Any], tuple[Any, ...]]] = []
    for candidate in scene_profile.get('objects') or []:
        object_id = str(candidate.get('object_id') or '')
        if not object_id or object_id in used_object_ids:
            continue
        if candidate.get('requires_spawn'):
            continue
        if room_id and str(candidate.get('room_id') or '') != room_id:
            continue
        if not spawn_support_can_hold(candidate, role):
            continue
        if goal_anchor and support_overlaps_role(candidate, goal_anchor):
            continue
        candidates.append((candidate, spawn_support_sort_key(candidate, anchor, anchor_center)))
    candidates.sort(key=lambda item: item[1])
    return [candidate for candidate, _key in candidates]

def spawn_support_sort_key(support: Mapping[str, Any], anchor: Mapping[str, Any] | None, anchor_center: Sequence[float] | None) -> tuple[Any, ...]:
    support_center = role_center_or_center(support)
    room_penalty = 0
    if anchor and anchor.get('room_id') and (support.get('room_id') != anchor.get('room_id')):
        room_penalty = 1
    if anchor_center and support_center:
        distance = math.hypot(float(support_center[0]) - float(anchor_center[0]), float(support_center[1]) - float(anchor_center[1]))
    else:
        distance = 999.0
    object_type = str(support.get('object_type') or '')
    category = str(support.get('category') or '')
    return (room_penalty, SPAWN_SUPPORT_TYPE_PRIORITY.get(object_type, 100), distance, category, str(support.get('object_id') or ''))

def spawn_support_can_hold(support: Mapping[str, Any], role: Mapping[str, Any]) -> bool:
    if not supports_generic_surface_placement(support):
        return False
    category = str(support.get('category') or '').lower()
    if category in NON_SPAWN_SUPPORT_CATEGORIES:
        return False
    if support.get('pickupable'):
        return False
    support_extent = number_vector(support.get('extent'), 3) or number_vector(support.get('bbox_size'), 3)
    role_extent_value = number_vector(role.get('extent'), 3) or number_vector(role.get('bbox_size'), 3)
    support_center = role_center_or_center(support)
    if support_extent is None or role_extent_value is None or support_center is None:
        return False
    if support_extent[0] <= 0.0 or support_extent[1] <= 0.0 or support_extent[2] <= 0.0:
        return False
    usable_x = support_extent[0] * 0.5 - role_extent_value[0] * 0.5 - SPAWN_SUPPORT_MARGIN
    usable_y = support_extent[1] * 0.5 - role_extent_value[1] * 0.5 - SPAWN_SUPPORT_MARGIN
    if usable_x < 0.0 or usable_y < 0.0:
        return False
    object_type = str(support.get('object_type') or '')
    abilities = set(support.get('abilities') or [])
    has_surface_semantics = object_type in SPAWN_SUPPORT_OBJECT_TYPES or 'support_surface' in abilities or 'placeable_surface' in abilities or ('heatSource' in abilities)
    if not has_surface_semantics:
        return False
    return spawn_support_matches_surface_constraints(support, role)

def spawn_support_matches_surface_constraints(support: Mapping[str, Any], role: Mapping[str, Any]) -> bool:
    support_extent = number_vector(support.get('extent'), 3) or number_vector(support.get('bbox_size'), 3)
    support_center = role_center_or_center(support)
    object_type = str(support.get('object_type') or '')
    constraints = surface_constraints_from_policy(_spawn_surface_policy(), 'task_spawn', override=_spawn_constraints_override(role))
    if object_type in SPAWN_SUPPORT_OBJECT_TYPES and (not _spawn_constraints_override(role).get('required_abilities_any')):
        constraints = dict(constraints)
        constraints['required_abilities_any'] = []
    descriptor = support_surface_descriptor_from_values(category=str(support.get('category') or ''), object_type=object_type, abilities=[str(item) for item in support.get('abilities') or []], center=support_center, extent=support_extent)
    return support_matches_surface_constraints(descriptor, constraints)

def surface_occupants_for_support(scene_profile: Mapping[str, Any] | None, support: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not scene_profile:
        return []
    support_center = role_center_or_center(support)
    support_extent = number_vector(support.get('extent'), 3) or number_vector(support.get('bbox_size'), 3)
    if support_center is None or support_extent is None:
        return []
    support_room = str(support.get('room_id') or '')
    support_object_id = str(support.get('object_id') or '')
    surface_z = float(support_center[2]) + float(support_extent[2]) * 0.5
    out: list[dict[str, Any]] = []
    for obj in scene_profile.get('objects') or []:
        if not isinstance(obj, Mapping):
            continue
        object_id = str(obj.get('object_id') or '')
        if support_object_id and object_id == support_object_id:
            continue
        if not _is_spawn_surface_occupant(obj):
            continue
        if support_room and obj.get('room_id') and (str(obj.get('room_id')) != support_room):
            continue
        center = role_center_or_center(obj)
        extent = number_vector(obj.get('extent'), 3) or number_vector(obj.get('bbox_size'), 3) or role_extent(obj)
        if center is None or extent is None:
            continue
        bottom_z = float(center[2]) - float(extent[2]) * 0.5
        if abs(bottom_z - surface_z) > 0.08:
            continue
        if abs(float(center[0]) - float(support_center[0])) + float(extent[0]) * 0.5 > float(support_extent[0]) * 0.5 + 0.03:
            continue
        if abs(float(center[1]) - float(support_center[1])) + float(extent[1]) * 0.5 > float(support_extent[1]) * 0.5 + 0.03:
            continue
        out.append({'object_id': object_id, 'category': str(obj.get('category') or obj.get('object_type') or 'object'), 'center': center, 'extent': extent, 'source': 'scene_profile_surface_occupant'})
    return out

def _is_spawn_surface_occupant(obj: Mapping[str, Any]) -> bool:
    abilities = {str(item) for item in obj.get('abilities') or []}
    object_type = str(obj.get('object_type') or '')
    return bool(obj.get('pickupable') or 'pickupable' in abilities or (object_type in DEFAULT_EXTENTS and object_type not in SPAWN_SUPPORT_OBJECT_TYPES and (object_type not in {'CounterTop', 'Cabinet', 'Fridge', 'Microwave', 'SinkBasin'})))

@lru_cache(maxsize=1)
def _spawn_surface_policy() -> Mapping[str, Any]:
    try:
        return load_rules().receptacle_policy
    except Exception:
        return {}

def _spawn_constraints_override(role: Mapping[str, Any]) -> dict[str, Any]:
    raw = role.get('spawn_constraints')
    if not isinstance(raw, Mapping):
        return {}
    out = dict(raw)
    if 'support_categories' in raw and 'allowed_support_categories' not in out:
        out['allowed_support_categories'] = list(raw.get('support_categories') or [])
    if 'support_object_types' in raw and 'allowed_object_types' not in out:
        out['allowed_object_types'] = list(raw.get('support_object_types') or [])
    return out

def support_overlaps_role(support: Mapping[str, Any], role: Mapping[str, Any] | None) -> bool:
    if not role:
        return False
    support_center = role_center_or_center(support)
    role_center_value = role_center_or_center(role)
    support_extent = number_vector(support.get('extent'), 3) or number_vector(support.get('bbox_size'), 3)
    role_extent_value = number_vector(role.get('extent'), 3) or number_vector(role.get('bbox_size'), 3)
    if support_center is None or role_center_value is None or support_extent is None or (role_extent_value is None):
        return False
    return aabb_vectors_overlap(role_center_value, role_extent_value, support_center, support_extent, tolerance=0.01)

def is_table_like_spawn_support(support: Mapping[str, Any]) -> bool:
    return str(support.get('object_type') or '') in SPAWN_TABLE_OBJECT_TYPES

def add_auxiliary_spawn_support_role(roles: dict[str, Any], candidate: Mapping[str, Any]) -> str:
    role_name = next_auxiliary_spawn_support_role_name(roles)
    entry = materialize_profile_support_role(candidate, role_name)
    roles[role_name] = entry
    return role_name

def next_auxiliary_spawn_support_role_name(roles: Mapping[str, Any]) -> str:
    index = 1
    while '{}{}'.format(SPAWN_SUPPORT_ROLE_PREFIX, index) in roles:
        index += 1
    return '{}{}'.format(SPAWN_SUPPORT_ROLE_PREFIX, index)

def materialize_profile_support_role(candidate: Mapping[str, Any], role_name: str) -> dict[str, Any]:
    synset = str(candidate.get('synset') or '{}.n.01'.format(candidate.get('category') or role_name))
    entry = {'synset': synset, 'bddl_inst': str(candidate.get('object_id') or '{}_1'.format(synset)), 'category': str(candidate.get('category') or candidate.get('object_type') or role_name), 'object_id': candidate.get('object_id'), 'object_type': candidate.get('object_type'), 'source_kind': candidate.get('source_kind') or 'scene_instance', 'requires_spawn': False, 'abilities': role_abilities(candidate), 'extent': role_extent(candidate), 'init_states': role_init_states(candidate), 'room_id': candidate.get('room_id'), 'position': candidate.get('position'), 'auxiliary_spawn_support': True}
    for field in ('bbox_size', 'scale'):
        vector = number_vector(candidate.get(field), 3)
        if vector is not None:
            entry[field] = vector
    for field in ('model', 'catalog_object_id'):
        value = candidate.get(field)
        if value not in (None, 'unknown'):
            entry[field] = value
    center = role_center_or_center(candidate)
    if center is not None:
        entry['center'] = center
    return entry

def spawn_center_on_support(role: Mapping[str, Any], support: Mapping[str, Any], slot: int) -> tuple[list[float], list[float]]:
    support_center = role_center_or_center(support) or [0.0, 0.0, 0.0]
    support_extent = number_vector(support.get('extent'), 3) or [0.8, 0.8, 0.8]
    role_extent_value = number_vector(role.get('extent'), 3) or [0.2, 0.2, 0.2]
    sx, sy, sz = [float(v) for v in support_center]
    usable_x = max(support_extent[0] * 0.5 - role_extent_value[0] * 0.5 - SPAWN_SUPPORT_MARGIN, 0.0)
    usable_y = max(support_extent[1] * 0.5 - role_extent_value[1] * 0.5 - SPAWN_SUPPORT_MARGIN, 0.0)
    direction = int(slot) % 4
    lane = int(slot) // 4
    lane_shift_x = min(usable_x, lane * max(role_extent_value[0] + SPAWN_SUPPORT_MARGIN, 0.08))
    lane_shift_y = min(usable_y, lane * max(role_extent_value[1] + SPAWN_SUPPORT_MARGIN, 0.08))
    if direction == 0:
        offset_x, offset_y = (usable_x, -lane_shift_y)
    elif direction == 1:
        offset_x, offset_y = (-usable_x, lane_shift_y)
    elif direction == 2:
        offset_x, offset_y = (-lane_shift_x, usable_y)
    else:
        offset_x, offset_y = (lane_shift_x, -usable_y)
    top_z = sz + support_extent[2] * 0.5
    center = [float(sx + offset_x), float(sy + offset_y), float(top_z + role_extent_value[2] * 0.5 + SPAWN_SUPPORT_CLEARANCE)]
    return (center, [float(offset_x), float(offset_y), float(center[2] - sz)])

def available_spawn_support_slot(role: Mapping[str, Any], support_name: str, support: Mapping[str, Any], support_slots: Mapping[str, int], placed_spawns: Sequence[Mapping[str, Any]], *, occupied_items: Sequence[Mapping[str, Any]] | None=None) -> tuple[int | None, list[float] | None, list[float] | None]:
    start_slot = int(support_slots.get(support_name, 0))
    extent = number_vector(role.get('extent'), 3) or [0.2, 0.2, 0.2]
    blockers = list(placed_spawns) + list(occupied_items or [])
    for slot in range(start_slot, start_slot + 32):
        center, offset = spawn_center_on_support(role, support, slot)
        if any((aabb_vectors_overlap(center, extent, item.get('center'), item.get('extent'), tolerance=0.01) for item in blockers)):
            continue
        if any((spawn_centers_too_close(center, item.get('center')) for item in blockers)):
            continue
        return (slot, center, offset)
    return (None, None, None)

def aabb_vectors_overlap(center_a: Sequence[float], extent_a: Sequence[float], center_b: Any, extent_b: Any, *, tolerance: float) -> bool:
    center_b_vec = number_vector(center_b, 3)
    extent_b_vec = number_vector(extent_b, 3)
    if center_b_vec is None or extent_b_vec is None:
        return False
    for axis in range(3):
        delta = abs(float(center_a[axis]) - float(center_b_vec[axis]))
        limit = float(extent_a[axis]) * 0.5 + float(extent_b_vec[axis]) * 0.5 - float(tolerance)
        if delta >= limit:
            return False
    return True

def spawn_centers_too_close(center_a: Sequence[float], center_b: Any) -> bool:
    center_b_vec = number_vector(center_b, 3)
    if center_b_vec is None:
        return False
    return math.hypot(float(center_a[0]) - center_b_vec[0], float(center_a[1]) - center_b_vec[1]) < SPAWN_SUPPORT_MIN_CENTER_DISTANCE

def role_center_or_center(role: Mapping[str, Any]) -> list[float] | None:
    return number_vector(role.get('center'), 3) or role_center(role)

def xy_distance(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if a is None or b is None:
        return float('inf')
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except Exception:
        return float('inf')

def role_abilities(role: Mapping[str, Any]) -> list[str]:
    abilities = set()
    if role.get('pickupable'):
        abilities.add('pickupable')
    if role.get('receptacle'):
        object_type = str(role.get('object_type') or '')
        abilities.add('container' if object_type in CONTAINER_TYPES else 'support_surface')
        abilities.add('receptacle')
    if role.get('openable'):
        abilities.update({'openable', 'container'})
    if role.get('toggleable'):
        abilities.add('toggleable')
    object_type = str(role.get('object_type') or '')
    if object_type in {'Microwave', 'Oven', 'StoveBurner'}:
        abilities.update({'heatSource', 'toggleable', 'support_surface'})
    if object_type == 'Fridge':
        abilities.add('coldSource')
    if object_type == 'Faucet':
        abilities.update({'toggleable', 'water_source'})
    if object_type == 'Knife':
        abilities.add('cutting_tool')
    if object_type in CLEANING_TOOL_TYPES:
        abilities.add('cleaning_tool')
    if object_type in CUTTABLE_FOOD_TYPES:
        abilities.add('cuttable')
    if object_type in SOAKABLE_OBJECT_TYPES:
        abilities.add('soakable')
    if object_type in WIPEABLE_OBJECT_TYPES:
        abilities.add('wipeable')
    return sorted(abilities or {'support_surface'})

def role_extent(role: Mapping[str, Any]) -> list[float]:
    position = role.get('position')
    object_type = str(role.get('object_type') or '')
    default = DEFAULT_EXTENTS.get(object_type, [0.2, 0.2, 0.2])
    raw_extent = number_vector(role.get('extent'), 3) or number_vector(role.get('bbox_size'), 3)
    if raw_extent is not None:
        return [max(0.04, float(raw_extent[0])), max(0.04, float(raw_extent[1])), max(0.04, float(raw_extent[2]))]
    if isinstance(position, Mapping) and object_type in DEFAULT_EXTENTS:
        return list(default)
    return list(default)

def number_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        vector = [float(v) for v in value]
    except Exception:
        return None
    if not all((math.isfinite(v) for v in vector)):
        return None
    return vector

def role_center(role: Mapping[str, Any]) -> list[float] | None:
    if role.get('requires_spawn'):
        return None
    pos = role.get('position')
    if not isinstance(pos, Mapping):
        return None
    return [float(pos['x']), float(pos['z']), float(pos['y'])]

def role_init_states(role: Mapping[str, Any]) -> dict[str, Any]:
    object_type = str(role.get('object_type') or '')
    states: dict[str, Any] = {}
    if object_type in CUTTABLE_FOOD_TYPES:
        states['sliced'] = False
    if object_type in SOAKABLE_OBJECT_TYPES:
        states['saturated'] = {}
    if object_type in CUTTABLE_FOOD_TYPES:
        states['cooked'] = False
        states['cooled'] = False
    if object_type in CONTAINER_TYPES:
        states.setdefault('filled', {})
        states.setdefault('contains', {})
        abilities = set(role.get('abilities') or [])
        if role.get('openable') or 'openable' in abilities:
            is_open = role.get('is_open')
            states.setdefault('open', is_open if isinstance(is_open, bool) else False)
    return states

def translate_initial_constraints(constraints: Sequence[Mapping[str, Any]], binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    init: list[dict[str, Any]] = []
    bound_roles = set((binding.get('roles') or {}).keys())
    for constraint in constraints:
        ctype = str(constraint.get('type') or '')
        role = constraint.get('object_role')
        if ctype in STATE_INIT_MAP and role in bound_roles:
            init.append({'predicate': STATE_INIT_MAP[ctype], 'args': [role]})
        elif ctype == 'object_in_receptacle':
            obj = constraint.get('object_role')
            rec = constraint.get('receptacle_role')
            if obj in bound_roles and rec in bound_roles:
                init.append({'predicate': relation_for_receptacle(binding, str(rec)), 'args': [obj, rec]})
    return init

def deferred_initial_assertions(constraints: Sequence[Mapping[str, Any]], binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []
    bound_roles = set((binding.get('roles') or {}).keys())
    for constraint in constraints:
        ctype = str(constraint.get('type') or '')
        if ctype != 'object_not_in_receptacle':
            continue
        obj = constraint.get('object_role')
        rec = constraint.get('receptacle_role')
        if obj in bound_roles and rec in bound_roles:
            assertions.append({'type': ctype, 'args': {'object': obj, 'receptacle': rec}, 'relation': relation_for_receptacle(binding, str(rec))})
    return assertions

def translate_goal_predicates(predicates: Sequence[Mapping[str, Any]], binding: Mapping[str, Any], template: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for predicate in predicates:
        if not predicate_include_if_satisfied(predicate, binding):
            continue
        ptype = str(predicate.get('type') or '')
        if ptype == 'object_in_receptacle':
            obj = str(predicate.get('object_role'))
            rec = str(predicate.get('receptacle_role'))
            out.append({'type': goal_relation_for_receptacle(binding, rec), 'args': {'object': obj, 'receptacle': rec}})
        elif ptype == 'object_not_in_receptacle':
            obj = str(predicate.get('object_role'))
            rec = str(predicate.get('receptacle_role'))
            out.append({'type': 'object_not_in_receptacle', 'args': {'object': obj, 'receptacle': rec}})
        elif ptype == 'object_in_any_receptacle':
            obj = str(predicate.get('object_role'))
            rec = choose_any_receptacle_role(predicate, binding)
            out.append({'type': goal_relation_for_receptacle(binding, rec), 'args': {'object': obj, 'receptacle': rec}})
        elif ptype == 'object_not_in_any_receptacle':
            obj = str(predicate.get('object_role'))
            rec = choose_any_receptacle_role(predicate, binding)
            out.append({'type': 'object_not_in_receptacle', 'args': {'object': obj, 'receptacle': rec}})
        elif ptype in SUPPORTED_STAGE4_GOAL_TYPES:
            role = str(predicate.get('object_role') or (predicate.get('args') or {}).get('object'))
            out.append({'type': normalize_goal_type(ptype), 'args': unary_goal_args(ptype, role, binding, template)})
        else:
            raise ValueError("Unsupported Stage 4 goal predicate '{}'.".format(ptype))
    return out

def predicate_include_if_satisfied(predicate: Mapping[str, Any], binding: Mapping[str, Any]) -> bool:
    include_if = predicate.get('include_if')
    roles = binding.get('roles') or {}
    if include_if is None:
        return True
    if isinstance(include_if, str):
        return include_if in roles
    if not isinstance(include_if, Mapping):
        raise ValueError('Predicate include_if must be a mapping or role string.')
    role_name = include_if.get('role')
    field_name = include_if.get('field')
    if not isinstance(role_name, str) or role_name not in roles:
        return False
    if field_name is None:
        return True
    if not isinstance(field_name, str) or not field_name:
        raise ValueError('Predicate include_if.field must be a non-empty string.')
    return roles[role_name].get(field_name) == include_if.get('equals', True)

def choose_any_receptacle_role(predicate: Mapping[str, Any], binding: Mapping[str, Any]) -> str:
    roles = binding.get('roles') or {}
    for role in predicate.get('receptacle_roles') or []:
        if isinstance(role, str) and role in roles:
            return role
    raise ValueError('No bound receptacle role for {}.'.format(predicate.get('type')))

def normalize_goal_type(ptype: str) -> str:
    return 'object_cooked' if ptype == 'object_cooked' else ptype

def unary_goal_args(ptype: str, object_role: str, binding: Mapping[str, Any], template: Mapping[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {'object': object_role}
    if ptype in {'object_hot', 'object_cooked'}:
        heatsource = infer_role_for_goal(binding, preferred_names=('microwave', 'oven', 'stove', 'stoveburner', 'heatsource', 'heat_source'), ability='heatSource')
        if heatsource:
            args['heatsource'] = heatsource
    elif ptype == 'object_cool':
        coldsource = infer_role_for_goal(binding, preferred_names=('fridge', 'freezer', 'coldsource', 'cold_source', 'cool_source'), ability='coldSource')
        if coldsource:
            args['coldsource'] = coldsource
    elif ptype == 'object_sliced':
        tool = infer_role_for_goal(binding, preferred_names=('knife', 'tool', 'cutting_tool'), ability='sliceable')
        if tool:
            args['tool'] = tool
    elif ptype in {'object_filled', 'object_soaked'}:
        source = infer_role_for_goal(binding, preferred_names=('faucet', 'sink', 'sink_basin', 'water_source'), ability='water_source')
        if source:
            args['source'] = source
    return args

def infer_role_for_goal(binding: Mapping[str, Any], *, preferred_names: Sequence[str], ability: str) -> str | None:
    roles = binding.get('roles') or {}
    for role_name in preferred_names:
        if role_name in roles:
            return role_name
    for role_name, role in sorted(roles.items()):
        abilities = set(role.get('abilities') or [])
        object_type = str(role.get('object_type') or '')
        if ability in abilities:
            return str(role_name)
        if ability == 'heatSource' and object_type in {'Microwave', 'Oven', 'StoveBurner'}:
            return str(role_name)
        if ability == 'coldSource' and object_type in {'Fridge'}:
            return str(role_name)
        if ability == 'water_source' and object_type in {'Faucet', 'SinkBasin'}:
            return str(role_name)
    return None

def relation_for_receptacle(binding: Mapping[str, Any], role_name: str) -> str:
    role = (binding.get('roles') or {}).get(role_name) or {}
    return 'inside' if str(role.get('object_type')) in CONTAINER_TYPES else 'ontop'

def goal_relation_for_receptacle(binding: Mapping[str, Any], role_name: str) -> str:
    return 'object_in_receptacle' if relation_for_receptacle(binding, role_name) == 'inside' else 'object_on_receptacle'

def render_instruction(template: Mapping[str, Any], binding: Mapping[str, Any], rng: random.Random, *, scope_room_label: str | None=None) -> str:
    options = list(template.get('instruction_templates') or [])
    template_text = rng.choice(options) if options else str(template.get('template_id'))
    labels = {role_name: str(role.get('category') or role.get('object_type') or role_name).replace('_', ' ') for role_name, role in (binding.get('roles') or {}).items()}
    if isinstance(scope_room_label, str) and scope_room_label:
        labels['scope_room_label'] = scope_room_label
    try:
        return template_text.format(**labels)
    except Exception:
        return template_text

def occupancy_bounds_from_profile(profile: Mapping[str, Any]) -> list[float]:
    bounds = profile.get('bounds') or {}
    values = [bounds.get('min_x'), bounds.get('min_z'), bounds.get('max_x'), bounds.get('max_z')]
    if any((not isinstance(value, (int, float)) for value in values)):
        return [-8.0, -8.0, 8.0, 8.0]
    min_x, min_y, max_x, max_y = [float(value) for value in values]
    return [min_x - 2.0, min_y - 2.0, max_x + 2.0, max_y + 2.0]

def start_pose_from_profile(profile: Mapping[str, Any], *, roles: Mapping[str, Any] | None=None, goal_predicates: Sequence[Mapping[str, Any]] | None=None) -> dict[str, Any]:
    room_records = [room for room in profile.get('rooms') or [] if isinstance(room, Mapping) and room.get('room_id') and room.get('representative_reachable_pose')]
    if not room_records:
        return {'x': 0.0, 'y': 0.0, 'z': 1.55, 'yaw': 0.0, 'pitch': 0.0, 'selection_source': 'profile_default_origin'}
    room_by_id = {str(room.get('room_id')): room for room in room_records}
    preferred_room_id = preferred_start_room_id(roles or {}, goal_predicates or [], room_by_id)
    if preferred_room_id is None:
        preferred_room_id = str(room_records[0].get('room_id'))
    room = room_by_id.get(preferred_room_id) or room_records[0]
    pose = (room.get('representative_reachable_pose') or {}).get('position')
    if isinstance(pose, Mapping):
        return {'x': float(pose['x']), 'y': float(pose['z']), 'z': 1.55, 'yaw': 0.0, 'pitch': 0.0, 'room_id': str(room.get('room_id')), 'selection_source': 'task_relevant_room_representative_pose' if preferred_room_id else 'first_profile_room_representative_pose', 'pose_source': str((room.get('representative_reachable_pose') or {}).get('source') or 'unknown')}
    return {'x': 0.0, 'y': 0.0, 'z': 1.55, 'yaw': 0.0, 'pitch': 0.0, 'room_id': str(room.get('room_id')), 'selection_source': 'profile_room_pose_missing_origin_fallback'}

def preferred_start_room_id(roles: Mapping[str, Any], goal_predicates: Sequence[Mapping[str, Any]], room_by_id: Mapping[str, Mapping[str, Any]]) -> str | None:
    if not roles or not room_by_id:
        return None
    role_scores: Counter[str] = Counter()
    for predicate in goal_predicates:
        args = predicate.get('args') if isinstance(predicate, Mapping) else None
        if not isinstance(args, Mapping):
            continue
        for key in ('object', 'receptacle'):
            role_name = args.get(key)
            role = roles.get(str(role_name)) if role_name is not None else None
            room_id = str((role or {}).get('room_id') or '')
            if room_id in room_by_id:
                role_scores[room_id] += 2 if key == 'receptacle' else 1
    for role_name, role in roles.items():
        room_id = str((role or {}).get('room_id') or '')
        if room_id in room_by_id and (not role.get('requires_spawn')):
            role_scores[room_id] += 1
    if not role_scores:
        return None
    return min(role_scores, key=lambda room_id: (-int(role_scores[room_id]), str(room_by_id[room_id].get('room_type') or ''), room_id))

def room_catalog_from_profile(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    rooms: list[dict[str, Any]] = []
    for room in profile.get('rooms') or []:
        if not isinstance(room, Mapping) or not room.get('room_id'):
            continue
        rooms.append({'room_id': str(room.get('room_id')), 'room_type': str(room.get('room_type') or str(room.get('room_id')).rsplit('_', 1)[0]), 'object_count': int(room.get('object_count') or 0), 'source': str(room.get('source') or 'profile_in_rooms'), 'geometry_status': str(room.get('geometry_status') or 'estimated_from_object_centers'), 'reachable_status': str(room.get('reachable_status') or 'profile_fallback'), 'representative_reachable_pose': copy.deepcopy(room.get('representative_reachable_pose'))})
    return rooms

def unsupported_goal_types(template: Mapping[str, Any]) -> list[str]:
    goal_types = {str(predicate.get('type')) for predicate in template.get('goal_predicates') or []}
    return sorted(goal_types - SUPPORTED_STAGE4_GOAL_TYPES)

def stage4_readiness(binding: Mapping[str, Any]) -> str:
    return stage4_readiness_from_debt(stage4_readiness_debt(binding, None, [], [], [], {}))

def stage4_readiness_from_debt(debt: set[str]) -> str:
    requirements = set(debt)
    if not requirements:
        return 'profile_only_candidate'
    if requirements <= {'spawn_object', 'spawn_placement'}:
        return 'requires_spawn_reset'
    return 'requires_runtime_reset_validation'

def stage4_readiness_debt(binding: Mapping[str, Any], family_spec: Mapping[str, Any] | None, disturbance_spec: Sequence[Mapping[str, Any]], failure_spec: Sequence[Mapping[str, Any]], unsupported_scheduler_specs: Sequence[Mapping[str, Any]], template: Mapping[str, Any]) -> set[str]:
    debt = set(binding.get('required_init_mutations') or [])
    debt.update(binding.get('runtime_capabilities_required') or [])
    if family_spec is not None:
        family = str(family_spec.get('family') or template.get('category') or '')
        if family:
            debt.add('family_spec:{}'.format(family))
        if family in {'prepare_then_place', 'prepare_multi_stage'}:
            debt.add('family_teacher:prepare')
        elif family == 'tidy_by_policy':
            debt.add('family_policy:tidy')
        elif family in {'put_away_by_category', 'put_away_visual_judgment', 'put_away_cross_room'}:
            debt.add('family_policy:put_away')
        elif family in {'recover_and_store_incident_object', 'recover_multi_incident'}:
            debt.add('family_teacher:recover')
            debt.add('visual_discoverability')
        elif family == 'human_interactive_execution':
            debt.add('human_interaction')
        elif family == 'mid_execution_state_disturbance':
            debt.add('family_scheduler:mid_execution')
        elif family == 'disturbance_recovery':
            debt.add('family_scheduler:disturbance_recovery')
        elif family == 'explore_then_act':
            debt.add('exploration_policy')
        elif family == 'composite':
            debt.add('composite_executor')
    if disturbance_spec:
        debt.add('scheduler:disturbance')
    if failure_spec:
        debt.add('scheduler:failure_injection')
    for spec in unsupported_scheduler_specs:
        mode = spec.get('mode') or spec.get('type') or 'unknown'
        debt.add('scheduler:unsupported:{}'.format(mode))
    return debt
SUPPORTED_STAGE4_FAILURE_MODES = {'grasp_slip', 'nav_blocked_partway', 'drop_beside', 'partial_open', 'partial_close', 'nav_wrong_target', 'block', 'report_fail', 'delayed_success', 'partial_execution', 'wrong_target'}

def materialize_scheduler_specs(template: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    spec = template.get('disturbance_spec')
    if not isinstance(spec, Mapping):
        return ([], [], [])
    disturbances = [copy.deepcopy(dict(item)) for item in spec.get('disturbances') or [] if isinstance(item, Mapping)]
    failures = []
    unsupported = []
    for item in spec.get('failures') or []:
        if not isinstance(item, Mapping):
            continue
        copied = copy.deepcopy(dict(item))
        if copied.get('mode') in SUPPORTED_STAGE4_FAILURE_MODES:
            failures.append(copied)
        else:
            unsupported.append(copied)
    return (disturbances, failures, unsupported)

def materialize_family_spec_template(template: Mapping[str, Any], binding: Mapping[str, Any], scene_profile: Mapping[str, Any]) -> dict[str, Any] | None:
    family_spec = template.get('family_spec_template')
    if not isinstance(family_spec, Mapping):
        return None
    out = copy.deepcopy(dict(family_spec))
    scope = out.get('scope')
    roles = binding.get('roles') or {}
    if isinstance(scope, Mapping):
        scope = materialize_room_scope(scope, roles=roles, room_catalog=build_room_type_catalog(scene_profile.get('rooms') or []))
        out['scope'] = scope
    put_away = out.get('put_away')
    if isinstance(put_away, dict):
        target_prefix = put_away.pop('target_roles_prefix', None)
        distractor_prefix = put_away.pop('protected_distractor_roles_prefix', None)
        relocate_only = relocate_target_only_roles(template)
        if isinstance(target_prefix, str):
            put_away['target_roles'] = sorted((role for role in roles if role.startswith(target_prefix) and role not in relocate_only))
        if isinstance(distractor_prefix, str):
            put_away['protected_distractor_roles'] = sorted((role for role in roles if role.startswith(distractor_prefix) and role not in relocate_only))
    prepare = out.get('prepare')
    if isinstance(prepare, dict) and (not prepare.get('stages')):
        stages = prepare_stages_from_template(prepare, roles)
        if stages:
            prepare['stages'] = stages
    observation_conditioned = out.get('observation_conditioned')
    if isinstance(observation_conditioned, Mapping):
        out['observation_conditioned'] = materialize_observation_conditioned_spec(observation_conditioned, roles=roles)
    return out

def prepare_stages_from_template(prepare: Mapping[str, Any], roles: Mapping[str, Any]) -> list[dict[str, Any]]:
    subtemplate = str(prepare.get('subtemplate') or '').lower()
    stages: list[dict[str, Any]] = []
    tool_roles = [str(role) for role in prepare.get('tool_roles') or [] if role]
    device_roles = [str(role) for role in prepare.get('device_roles') or [] if role]
    final_receptacle = prepare.get('final_receptacle_role')
    if 'clean' in subtemplate or 'wash' in subtemplate:
        stages.append({'type': 'wash', 'method': 'wipe' if tool_roles else 'soak', 'tool': first_matching_role(tool_roles, ('rag', 'sponge', 'cloth', 'cleaning')), 'source': first_matching_role(device_roles, ('faucet', 'sink', 'basin'))})
    if 'heat' in subtemplate:
        stages.append({'type': 'heat', 'heatsource': first_matching_role(device_roles, ('microwave', 'oven', 'stove'))})
    if 'cool' in subtemplate:
        stages.append({'type': 'cool', 'coldsource': first_matching_role(device_roles, ('fridge', 'freezer'))})
    if 'slice' in subtemplate:
        stages.append({'type': 'slice', 'tool': first_matching_role(tool_roles or list(roles), ('knife', 'cutting', 'tool'))})
    if final_receptacle:
        stages.append({'type': 'store', 'receptacle': str(final_receptacle)})
    return [stage for stage in stages if stage.get('type')]

def first_matching_role(role_names: Sequence[str], preferred_tokens: Sequence[str]) -> str | None:
    names = [str(name) for name in role_names if name]
    for token in preferred_tokens:
        for name in names:
            if token in name:
                return name
    return names[0] if names else None

def relocate_target_only_roles(template: Mapping[str, Any]) -> set[str]:
    return {str(role) for role, requirement in (template.get('requires') or {}).items() if isinstance(requirement, Mapping) and requirement.get('relocate_target_only')}

def goal_plan_skeleton(goal_predicates: Sequence[Mapping[str, Any]]) -> list[str]:
    skeleton = []
    for predicate in goal_predicates:
        ptype = str(predicate.get('type') or '')
        if ptype in {'object_in_receptacle', 'object_on_receptacle'}:
            skeleton.extend(['navigate_to_object', 'pick_up', 'navigate_to_receptacle', 'place'])
        elif ptype in {'object_open', 'object_closed', 'object_toggled_on', 'object_toggled_off'}:
            skeleton.extend(['navigate_to_stateful_object', ptype.replace('object_', '')])
        elif ptype in {'object_hot', 'object_cooked'}:
            skeleton.extend(['navigate_to_object', 'pick_up', 'navigate_to_heat_source', 'heat'])
        elif ptype == 'object_cool':
            skeleton.extend(['navigate_to_object', 'pick_up', 'navigate_to_cold_source', 'cool'])
        elif ptype == 'object_clean':
            skeleton.extend(['navigate_to_object', 'clean'])
        elif ptype == 'object_sliced':
            skeleton.extend(['navigate_to_tool', 'pick_up_tool', 'navigate_to_object', 'cut'])
        elif ptype in {'object_filled', 'object_soaked'}:
            skeleton.extend(['navigate_to_object', 'pick_up', 'navigate_to_water_source', ptype.replace('object_', '')])
        elif ptype == 'agent_near':
            skeleton.append('navigate_to_object')
        else:
            skeleton.append(ptype)
    return skeleton

def stage4_tags(template_id: str) -> list[str]:
    tags = ['stage4_candidate', 'profile_driven', template_id]
    if template_id in set(STAGE4_5_GOLD_TEMPLATE_IDS):
        tags.append('stage4_5_gold')
    return tags

def reject_record(record: Mapping[str, Any], code: str, extra: Mapping[str, Any] | None=None) -> dict[str, Any]:
    out = {'scene_model': record.get('scene_model'), 'template_id': record.get('template_id'), 'category': record.get('category'), 'failure_code': code, 'stage3_status': record.get('status')}
    if extra:
        out.update(dict(extra))
    return out

def summarize_generated_tasks(tasks: list[Mapping[str, Any]], rejected: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {'generated_total': len(tasks), 'rejected_total': len(rejected), 'generated_by_template': dict(sorted(Counter((task['template']['template_id'] for task in tasks)).items())), 'generated_by_scene': dict(sorted(Counter((task['source']['scene_model'] for task in tasks)).items())), 'generated_by_stage3_status': dict(sorted(Counter((task['source'].get('stage3_match_status') for task in tasks)).items())), 'generated_by_readiness': dict(sorted(Counter(((task.get('generation') or {}).get('stage4_readiness') for task in tasks)).items())), 'rejection_codes': dict(sorted(Counter((item['failure_code'] for item in rejected)).items()))}

def validate_reset_and_oracle(task_spec: Mapping[str, Any], *, rng_seed: int=0, check_oracle: bool=True) -> dict[str, Any]:
    result: dict[str, Any] = {'task_id': task_spec.get('id'), 'structure_ok': False, 'reset_ok': False, 'init_ok': False, 'initial_goal_false': False, 'oracle_ok': None, 'failure_code': None, 'failure_reason': None, 'placement_report': None}
    try:
        validate_behavior_task_spec(task_spec)
        result['structure_ok'] = True
        env = BehaviorClosedLoopToolEnv()
        env.reset_to_task(task_spec, rng_seed=rng_seed)
        result['reset_ok'] = True
        result['placement_report'] = env.scene_result.placement_report
        result['init_ok'] = init_predicates_satisfied(env, task_spec)
        initial_goal = env.check_goal(model_stopped=False)
        result['initial_goal_false'] = initial_goal.get('task_success') == 0.0
        result['initial_goal'] = initial_goal
        if check_oracle:
            ep = EpisodeRunner(env, max_high_level_steps=50).run(task_spec, brain=default_dispatcher().build(task_spec), rng_seed=rng_seed)
            result['oracle_ok'] = ep.final_goal.get('task_success') == 1.0
            result['final_goal'] = ep.final_goal
            result['metrics'] = ep.metrics
            if not result['oracle_ok']:
                result['failure_code'] = 'ORACLE_PRECHECK_FAILED'
                result['failure_reason'] = str(ep.final_goal)
    except Exception as exc:
        if not result['structure_ok']:
            result['failure_code'] = 'TASK_SPEC_VALIDATION_FAILED'
        elif not result['reset_ok']:
            result['failure_code'] = 'RESET_FAILED'
        else:
            result['failure_code'] = 'VALIDATION_EXCEPTION'
        result['failure_reason'] = str(exc)
    if result['failure_code'] is None and (not result['init_ok']):
        result['failure_code'] = 'INIT_PREDICATES_UNSATISFIED'
    if result['failure_code'] is None and (not result['initial_goal_false']):
        result['failure_code'] = 'INITIAL_GOAL_ALREADY_TRUE'
    return result

def init_predicates_satisfied(env: BehaviorClosedLoopToolEnv, task_spec: Mapping[str, Any]) -> bool:
    for entry in task_spec.get('init') or []:
        predicate = entry.get('predicate')
        args = entry.get('args') or []
        if not args:
            return False
        obj_name = env.role_map.get(str(args[0]), str(args[0]))
        obj = env.core.objects.get(obj_name)
        if obj is None:
            return False
        states = obj.states or {}
        if predicate == 'open' and states.get('open') is not True:
            return False
        if predicate == 'closed' and states.get('open') is not False:
            return False
        if predicate == 'toggled_on' and states.get('toggled_on') is not True:
            return False
        if predicate == 'toggled_off' and states.get('toggled_on') is not False:
            return False
        if predicate == 'dirty' and states.get('dirty') is not True:
            return False
        if predicate == 'clean' and states.get('dirty') is not False:
            return False
        if predicate == 'sliced' and states.get('sliced') is not True:
            return False
        if predicate == 'cooled' and states.get('cooled') is not True:
            return False
    return True
__all__ = ['STAGE4_5_GOLD_TEMPLATE_IDS', 'generate_stage4_tasks', 'validate_reset_and_oracle']
