from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, List, Mapping, Optional, Sequence
from scoper.benchmark.behavior.bddl_adapter import BDDLAdapter
from scoper.benchmark.behavior.scene.doors import is_door_like
from scoper.benchmark.behavior.scene import placement_symbolic
from scoper.benchmark.behavior.scene.placement_symbolic import SceneInitResult
_SCENE_PROVIDED_CATEGORIES = frozenset({'floors', 'floor', 'walls', 'wall', 'ceilings', 'ceiling'})
_IDENTITY_QUAT_XYZW = [0.0, 0.0, 0.0, 1.0]
_HEAT_SOURCE_OBJECT_TYPES = frozenset({'Stove', 'StoveBurner', 'Oven', 'Microwave', 'Range'})
_HEAT_SOURCE_CATEGORY_TOKENS = ('stove', 'burner', 'oven', 'microwave', 'range')
_WATER_SOURCE_ABILITIES = frozenset({'water_source', 'faucet_like', 'fluid_source', 'fluidSource', 'particleSource', 'particle_source'})

@dataclass
class RenderSceneInitResult:
    symbolic: SceneInitResult
    og_objects: List[dict]
    render_report: dict = field(default_factory=dict)
    render_aliases: dict[str, str] = field(default_factory=dict)
    render_alias_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

def build(task_spec: Mapping[str, Any], adapter: Optional[BDDLAdapter]=None, rng_seed: int=0, scene_layout_mode: str='task_roles', load_doors: bool=False, particle_systems_enabled: bool=True) -> RenderSceneInitResult:
    from scoper.benchmark.behavior.render.asset_resolver import require_task_render_assets
    task_spec = require_task_render_assets(task_spec, scene_layout_mode=scene_layout_mode, load_doors=load_doors).task
    adapter = adapter or BDDLAdapter()
    symbolic = placement_symbolic.build(task_spec, rng_seed=rng_seed)
    if scene_layout_mode == 'source_json_full':
        from scoper.benchmark.behavior.scene.spawn_collision_repair import repair_task_spawn_collisions_after_merge
        from scoper.benchmark.behavior.scene.variant_layout import merge_source_json_layout
        merged = merge_source_json_layout(task_spec, symbolic.aabb_objects, symbolic.role_map, load_doors=load_doors)
        repaired_objects, repair_report = repair_task_spawn_collisions_after_merge(merged.objects, role_map=symbolic.role_map, placements=(symbolic.placement_report or {}).get('placements') or {})
        _sync_symbolic_role_objects(symbolic, repaired_objects)
        symbolic.placement_report['scene_layout_mode'] = 'source_json_full'
        symbolic.placement_report['source_json_full_layout'] = merged.report
        symbolic.placement_report['post_merge_collision_repairs'] = repair_report
    full_scene_object_ids = _full_scene_object_ids(task_spec) if scene_layout_mode == 'source_json_full' else set()
    synset_by_instance: dict[str, str] = {}
    role_by_instance: dict[str, Mapping[str, Any]] = {}
    roles = task_spec.get('roles', {})
    for role_name, role in roles.items():
        inst = symbolic.role_map.get(role_name, role.get('bddl_inst') or role_name)
        role_by_instance[str(inst)] = role
        if role.get('synset'):
            synset_by_instance[inst] = str(role['synset'])
    alias_parent_by_instance = _derived_faucet_alias_parents(roles, symbolic.role_map, full_scene_object_ids=full_scene_object_ids)
    alias_abilities_by_parent = _derived_faucet_render_abilities_by_parent(roles, symbolic.role_map, alias_parent_by_instance, particle_systems_enabled=particle_systems_enabled)
    og_objects: List[dict] = []
    skipped: List[str] = []
    render_aliases: dict[str, str] = {}
    render_alias_metadata: dict[str, dict[str, Any]] = {}
    aliased_objects: list[dict[str, str]] = []
    unresolved_aliases: list[dict[str, str]] = []
    door_loading_disabled_objects: list[str] = []
    for obj in symbolic.aabb_objects:
        if not load_doors and is_door_like(obj.category, obj.name):
            skipped.append(obj.name)
            door_loading_disabled_objects.append(str(obj.name))
            continue
        if obj.category in _SCENE_PROVIDED_CATEGORIES:
            skipped.append(obj.name)
            continue
        role = role_by_instance.get(str(obj.name), {})
        object_id = str(role.get('object_id') or '')
        if scene_layout_mode == 'source_json_full' and object_id and (object_id in full_scene_object_ids) and (not bool(role.get('requires_spawn'))):
            alias_name = _og_safe_name(str(obj.name))
            target_name = _og_safe_name(object_id)
            render_aliases[alias_name] = target_name
            render_alias_metadata[alias_name] = {'alias_kind': 'source_scene_object', 'core_name': str(obj.name), 'source_scene_object_id': object_id, 'render_name': target_name, 'pose_sync_allowed': True, 'held_sync_allowed': True, 'visual_state_sync_allowed': True}
            skipped.append(obj.name)
            continue
        if scene_layout_mode != 'source_json_full' and str(role.get('source_kind') or '') == 'scene_instance' and (not bool(role.get('requires_spawn'))):
            skipped.append(obj.name)
            continue
        if _is_derived_faucet(role, obj.category):
            parent_inst = alias_parent_by_instance.get(str(obj.name))
            if parent_inst:
                alias_name = _og_safe_name(str(obj.name))
                target_name = _og_safe_name(str(parent_inst))
                render_aliases[alias_name] = target_name
                render_alias_metadata[alias_name] = {'alias_kind': 'derived_parent', 'core_name': str(obj.name), 'parent_object': str(parent_inst), 'render_name': target_name, 'pose_sync_allowed': False, 'held_sync_allowed': False, 'visual_state_sync_allowed': True}
                aliased_objects.append({'name': alias_name, 'target': target_name, 'category': str(obj.category), 'reason': 'derived_faucet_uses_parent_sink_render_asset'})
            else:
                unresolved_aliases.append({'name': _og_safe_name(str(obj.name)), 'category': str(obj.category), 'reason': 'derived_faucet_parent_sink_not_found'})
            skipped.append(obj.name)
            continue
        synset = synset_by_instance.get(obj.name)
        category = _resolve_render_category(synset, obj.category, role, adapter)
        if category is None:
            skipped.append(obj.name)
            continue
        load_entry = {'type': 'DatasetObject', 'prim_path': '/World/{}'.format(_og_safe_name(obj.name)), 'name': _og_safe_name(obj.name), 'category': category, 'position': [float(obj.center[0]), float(obj.center[1]), float(obj.center[2])], 'orientation': list(_IDENTITY_QUAT_XYZW), 'bounding_box': [float(obj.extent[0]), float(obj.extent[1]), float(obj.extent[2])], 'fixed_base': 'pickupable' not in tuple(obj.abilities), 'visual_only': True, '_desired_aabb_center': [float(obj.center[0]), float(obj.center[1]), float(obj.center[2])], '_symbolic_name': str(obj.name)}
        render_abilities = _render_abilities_for_role(role, obj.category, particle_systems_enabled=particle_systems_enabled)
        _merge_render_abilities(render_abilities, alias_abilities_by_parent.get(str(obj.name)))
        if render_abilities:
            load_entry['abilities'] = render_abilities
        model = role.get('model')
        if not isinstance(model, str) or not model or model == 'unknown':
            raise ValueError('renderer asset preflight did not resolve an explicit model')
        load_entry['model'] = model
        og_objects.append(load_entry)
    task_frame_context = []
    if scene_layout_mode != 'source_json_full':
        floor_entries = _task_floor_render_entries(symbolic.aabb_objects)
        og_objects.extend(floor_entries)
        task_frame_context.extend(({'name': entry['name'], 'category': entry['category'], 'reason': 'task_roles_floor_context'} for entry in floor_entries))
    report = {'rng_seed': rng_seed, 'loaded_objects': [o['name'] for o in og_objects], 'skipped_objects': skipped, 'render_aliases': dict(sorted(render_aliases.items())), 'render_alias_metadata': dict(sorted(render_alias_metadata.items())), 'aliased_objects': aliased_objects, 'unresolved_aliases': unresolved_aliases, 'scene_provided_categories': sorted(_SCENE_PROVIDED_CATEGORIES), 'render_layout_mode': str(scene_layout_mode or 'task_roles'), 'source_json_scene_object_count': len(full_scene_object_ids), 'task_frame_context_objects': task_frame_context, 'load_doors': bool(load_doors), 'door_loading_disabled_objects': door_loading_disabled_objects, 'particle_systems_enabled': bool(particle_systems_enabled), 'excluded_particle_systems': [] if particle_systems_enabled else ['water']}
    return RenderSceneInitResult(symbolic=symbolic, og_objects=og_objects, render_report=report, render_aliases=render_aliases, render_alias_metadata=render_alias_metadata)

def _sync_symbolic_role_objects(symbolic: SceneInitResult, repaired_objects: Sequence[Any]) -> None:
    role_names = {str(name) for name in symbolic.role_map.values()}
    repaired_by_name = {str(obj.name): obj for obj in repaired_objects if str(obj.name) in role_names}
    for obj in symbolic.aabb_objects:
        repaired = repaired_by_name.get(str(obj.name))
        if repaired is None:
            continue
        obj.center = tuple((float(v) for v in repaired.center))
        obj.extent = tuple((float(v) for v in repaired.extent))
        obj.states = dict(repaired.states or {})
        obj.root_position = repaired.root_position
        obj.root_orientation_xyzw = repaired.root_orientation_xyzw
        obj.root_yaw = repaired.root_yaw
        obj.pose_source = repaired.pose_source

def _full_scene_object_ids(task_spec: Mapping[str, Any]) -> set[str]:
    source_json = (task_spec.get('source') or {}).get('source_json')
    if not source_json:
        return set()
    try:
        from scoper.benchmark.behavior.scene.variant_layout import _resolve_source_json
        path = _resolve_source_json(source_json)
        import json
        scene = json.loads(path.read_text(encoding='utf-8'))
        init_info = (scene.get('objects_info') or {}).get('init_info') or {}
        return {str(name) for name in init_info}
    except Exception:
        return set()

def _resolve_category(synset: Optional[str], fallback: str, adapter: BDDLAdapter) -> Optional[str]:
    if synset and adapter.is_valid_synset(synset):
        cats = adapter.categories_for_synset(synset)
        if cats:
            return str(cats[0])
    return str(fallback) if fallback else None

def _resolve_render_category(synset: Optional[str], fallback: str, role: Mapping[str, Any], adapter: BDDLAdapter) -> Optional[str]:
    model = role.get('model')
    role_category = role.get('category')
    if isinstance(model, str) and model and (model != 'unknown') and role_category:
        return str(role_category)
    return _resolve_category(synset, fallback, adapter)

def _is_derived_faucet(role: Mapping[str, Any], category: str) -> bool:
    return str(category) == 'derived_faucet' or (str(role.get('source_kind') or '') == 'derived_scene_part' and str(role.get('object_type') or '') == 'Faucet')

def _derived_faucet_alias_parents(roles: Mapping[str, Mapping[str, Any]], role_map: Mapping[str, str], *, full_scene_object_ids: set[str] | None=None) -> dict[str, str]:
    parent_by_instance: dict[str, str] = {}
    full_scene_object_ids = set(full_scene_object_ids or set())
    object_id_to_role_name = {str(role.get('object_id')): str(role_name) for role_name, role in roles.items() if role.get('object_id')}
    for role_name, role in roles.items():
        inst = str(role_map.get(str(role_name), role.get('bddl_inst') or role_name))
        if not _is_derived_faucet(role, str(role.get('category') or '')):
            continue
        parent_object_id = _derived_parent_object_id(role)
        parent_role_name = object_id_to_role_name.get(parent_object_id)
        if not parent_role_name:
            if parent_object_id and parent_object_id in full_scene_object_ids:
                parent_by_instance[inst] = parent_object_id
            continue
        parent_role = roles[parent_role_name]
        parent_inst = str(role_map.get(parent_role_name, parent_role.get('bddl_inst') or parent_role_name))
        parent_by_instance[inst] = parent_inst
    return parent_by_instance

def _derived_faucet_render_abilities_by_parent(roles: Mapping[str, Mapping[str, Any]], role_map: Mapping[str, str], alias_parent_by_instance: Mapping[str, str], *, particle_systems_enabled: bool=True) -> dict[str, dict[str, Any]]:
    abilities_by_parent: dict[str, dict[str, Any]] = {}
    role_by_instance: dict[str, Mapping[str, Any]] = {}
    for role_name, role in roles.items():
        inst = str(role_map.get(str(role_name), role.get('bddl_inst') or role_name))
        role_by_instance[inst] = role
    for faucet_inst, parent_inst in alias_parent_by_instance.items():
        role = role_by_instance.get(str(faucet_inst), {})
        abilities = _render_abilities_for_role(role, str(role.get('category') or 'derived_faucet'), particle_systems_enabled=particle_systems_enabled)
        if not abilities:
            continue
        target = abilities_by_parent.setdefault(str(parent_inst), {})
        _merge_render_abilities(target, abilities)
    return abilities_by_parent

def _render_abilities_for_role(role: Mapping[str, Any], category: str, *, particle_systems_enabled: bool=True) -> dict[str, Any]:
    role_abilities = {str(ability) for ability in role.get('abilities') or []}
    render_abilities: dict[str, Any] = {}
    if 'toggleable' in role_abilities:
        render_abilities['toggleable'] = {}
    if _role_is_heat_source(role, category, role_abilities):
        render_abilities['toggleable'] = {}
        render_abilities['heatSource'] = {'requires_toggled_on': True}
    if _role_is_water_source(role, category, role_abilities):
        render_abilities['toggleable'] = {}
        if particle_systems_enabled:
            render_abilities['particleSource'] = {'conditions': {'water': [(_particle_toggled_on_condition(), True)]}, 'initial_speed': 0.0}
            render_abilities['particleSink'] = {'conditions': {'water': []}}
    return render_abilities

def _role_is_heat_source(role: Mapping[str, Any], category: str, abilities: set[str]) -> bool:
    object_type = str(role.get('object_type') or '')
    text = ' '.join((str(value).lower() for value in (category, role.get('category', ''), role.get('synset', ''), object_type)))
    return 'heatSource' in abilities or 'heat_source' in abilities or object_type in _HEAT_SOURCE_OBJECT_TYPES or any((token in text for token in _HEAT_SOURCE_CATEGORY_TOKENS))

def _role_is_water_source(role: Mapping[str, Any], category: str, abilities: set[str]) -> bool:
    object_type = str(role.get('object_type') or '')
    text = ' '.join((str(value).lower() for value in (category, role.get('category', ''), role.get('synset', ''), object_type)))
    return bool(_WATER_SOURCE_ABILITIES & abilities) or _is_derived_faucet(role, category) or object_type == 'Faucet' or ('faucet' in text)

def _merge_render_abilities(target: dict[str, Any], source: Mapping[str, Any] | None) -> None:
    if not source:
        return
    for ability, params in source.items():
        if ability not in target:
            target[str(ability)] = deepcopy(params)
            continue
        if isinstance(target[ability], dict) and isinstance(params, Mapping):
            _deep_merge_dict(target[ability], params)

def _deep_merge_dict(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, Mapping):
            _deep_merge_dict(target[key], value)
        else:
            target[key] = deepcopy(value)

def _particle_toggled_on_condition() -> Any:
    try:
        from omnigibson.utils.constants import ParticleModifyCondition
        return ParticleModifyCondition.TOGGLEDON
    except Exception:
        return 'toggled_on'

def _derived_parent_object_id(role: Mapping[str, Any]) -> str:
    object_id = str(role.get('object_id') or '')
    if '::' in object_id:
        return object_id.split('::', 1)[0]
    parents = role.get('parent_receptacles') or []
    if parents:
        return str(parents[0])
    return ''

def _og_safe_name(instance_name: str) -> str:
    return instance_name.replace('.', '_').replace('-', '_')

def _task_floor_render_entries(aabb_objects: List[Any]) -> List[dict]:
    entries: List[dict] = []
    for obj in aabb_objects:
        if str(getattr(obj, 'category', '')).strip().lower() not in {'floor', 'floors'}:
            continue
        center = [float(value) for value in getattr(obj, 'center', (0.0, 0.0, -0.02))]
        extent = [float(value) for value in getattr(obj, 'extent', (10.0, 6.0, 0.04))]
        extent[0] = max(extent[0], 1.0)
        extent[1] = max(extent[1], 1.0)
        extent[2] = max(extent[2], 0.02)
        safe_name = '{}_render_floor'.format(_og_safe_name(str(getattr(obj, 'name', 'task_floor'))))
        entries.append({'type': 'PrimitiveObject', 'primitive_type': 'Cube', 'relative_prim_path': '/{}'.format(safe_name), 'name': safe_name, 'category': 'floors', 'position': center, 'orientation': list(_IDENTITY_QUAT_XYZW), 'scale': extent, 'fixed_base': True, 'visual_only': True, 'rgba': [0.58, 0.55, 0.5, 1.0], '_desired_aabb_center': center, '_symbolic_name': str(getattr(obj, 'name', safe_name)), '_renderer_context_object': True})
    return entries
__all__ = ['RenderSceneInitResult', 'build']
