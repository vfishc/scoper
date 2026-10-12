from __future__ import annotations
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping
POLICY_SCHEMA_VERSION = 'behavior-particle-segmentation-policy.v2'
REPLAY_STATE_SCHEMA_VERSION = 'behavior-render-replay-state.v1'
REPLAY_VISIBILITY_SOURCE = 'behavior_particle_suppressed_replay_segmentation'
_WATER_SOURCE_ABILITIES = {'fluidsource', 'fluid_source', 'particlesource', 'particle_source', 'water_source', 'faucet_like'}
_WATER_SOURCE_CATEGORIES = {'derived_faucet', 'faucet'}
_PARTICLE_ABILITIES = ('particleSource', 'particleSink')

def particle_segmentation_policy(task_spec: Mapping[str, Any]) -> Mapping[str, Any]:
    affected_roles = []
    reasons = set()
    roles = task_spec.get('roles') if isinstance(task_spec, Mapping) else None
    for role_name, role in sorted(dict(roles or {}).items()):
        if not isinstance(role, Mapping):
            continue
        abilities = {_normalized_token(value) for value in role.get('abilities') or []}
        category = _normalized_token(role.get('category'))
        object_type = _normalized_token(role.get('object_type'))
        synset = _normalized_token(role.get('synset'))
        role_text = _normalized_token(role_name)
        water_source = bool(abilities & _WATER_SOURCE_ABILITIES) or category in _WATER_SOURCE_CATEGORIES
        water_source = water_source or object_type == 'faucet' or 'faucet' in synset or ('faucet' in role_text)
        if not water_source:
            continue
        affected_roles.append(str(role_name))
        if abilities & _WATER_SOURCE_ABILITIES:
            reasons.add('water_source_ability')
        if category in _WATER_SOURCE_CATEGORIES or object_type == 'faucet' or 'faucet' in synset:
            reasons.add('water_source_identity')
    required = bool(affected_roles)
    source_particle_systems_enabled = required
    return {'schema_version': POLICY_SCHEMA_VERSION, 'mode': 'paired_render_state_replay' if required else 'inline', 'required': required, 'excluded_particle_systems': ['water'], 'source_particle_systems_enabled': source_particle_systems_enabled, 'source_native_particle_source_visuals_enabled': False, 'source_particle_source_stream_visuals_enabled': required, 'source_water_visual_mode': 'continuous_mesh' if required else 'excluded', 'replay_particle_systems_enabled': False, 'replay_particle_source_stream_visuals_enabled': False, 'affected_roles': affected_roles, 'reasons': sorted(reasons)}

def particle_suppressed_scene_file(scene_file: str | Path, *, ability_resolver: Callable[[str], Mapping[str, Any] | None], cache_root: str | Path | None=None) -> tuple[Path, Mapping[str, Any]]:
    source = Path(scene_file).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f'particle replay scene file does not exist: {source}')
    raw = source.read_bytes()
    try:
        scene = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'particle replay scene file is invalid JSON: {source}') from exc
    if not isinstance(scene, dict):
        raise ValueError('particle replay scene JSON must be an object')
    init_info = (scene.get('objects_info') or {}).get('init_info') or {}
    if not isinstance(init_info, dict):
        raise ValueError('particle replay scene object init_info is invalid')
    removed_rows = []
    explicit_ability_objects = 0
    taxonomy_ability_objects = 0
    for object_name, object_info in sorted(init_info.items()):
        if not isinstance(object_info, dict):
            raise ValueError(f'particle replay init_info is invalid for {object_name}')
        args = object_info.get('args')
        if not isinstance(args, dict):
            raise ValueError(f'particle replay init args are invalid for {object_name}')
        explicit = args.get('abilities')
        if explicit is None:
            category = str(args.get('category') or 'object')
            resolved = ability_resolver(category)
            abilities = deepcopy(dict(resolved or {}))
            taxonomy_ability_objects += 1
            ability_source = 'taxonomy'
        elif isinstance(explicit, Mapping):
            abilities = deepcopy(dict(explicit))
            explicit_ability_objects += 1
            ability_source = 'scene_json'
        else:
            raise ValueError(f'particle replay abilities are invalid for {object_name}')
        removed = [ability for ability in _PARTICLE_ABILITIES if abilities.pop(ability, None) is not None]
        args['abilities'] = abilities
        if removed:
            removed_rows.append({'object': str(object_name), 'category': str(args.get('category') or 'object'), 'ability_source': ability_source, 'removed_abilities': removed})
    removed_systems = _remove_initial_water_systems(scene)
    payload = json.dumps(scene, indent=2, sort_keys=False, ensure_ascii=False, allow_nan=False).encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()
    cache_dir = Path(cache_root) if cache_root is not None else Path(tempfile.gettempdir()) / 'eb_behavior_particle_suppressed_scenes'
    cache_dir.mkdir(parents=True, exist_ok=True)
    sanitized = cache_dir / f'{source.stem}_{digest[:20]}.json'
    cache_hit = sanitized.is_file() and hashlib.sha256(sanitized.read_bytes()).hexdigest() == digest
    if not cache_hit:
        temporary = sanitized.with_name(f'.{sanitized.name}.{os.getpid()}.tmp')
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, sanitized)
        finally:
            temporary.unlink(missing_ok=True)
    report = {'enabled': True, 'mode': 'preconstruction_scene_json_sanitization', 'source_scene_file': str(source), 'sanitized_scene_file': str(sanitized), 'source_scene_sha256': hashlib.sha256(raw).hexdigest(), 'sanitized_scene_sha256': digest, 'cache_hit': cache_hit, 'object_count': len(init_info), 'explicit_ability_object_count': explicit_ability_objects, 'taxonomy_ability_object_count': taxonomy_ability_objects, 'removed_ability_count': sum((len(row['removed_abilities']) for row in removed_rows)), 'removed_initial_systems': removed_systems, 'objects': removed_rows}
    return (sanitized, report)

def _remove_initial_water_systems(scene: dict[str, Any]) -> list[str]:
    state = scene.get('state')
    if not isinstance(state, dict):
        return []
    registry = state.get('registry') if isinstance(state.get('registry'), dict) else state
    systems = registry.get('system_registry') if isinstance(registry, dict) else None
    if not isinstance(systems, dict):
        return []
    removed = []
    for system_name in tuple(systems):
        if str(system_name).strip().casefold() == 'water':
            systems.pop(system_name)
            removed.append(str(system_name))
    return sorted(removed)

def finalize_render_replay_state(state: Mapping[str, Any], *, episode_id: str, frame_id: str) -> Mapping[str, Any]:
    value = deepcopy(dict(state))
    value['schema_version'] = REPLAY_STATE_SCHEMA_VERSION
    value['episode_id'] = str(episode_id)
    value['frame_id'] = str(frame_id)
    value.pop('state_sha256', None)
    value['state_sha256'] = _sha256_json(value)
    validate_render_replay_state(value, episode_id=episode_id, frame_id=frame_id)
    return value

def validate_render_replay_state(state: Mapping[str, Any], *, episode_id: str | None=None, frame_id: str | None=None) -> None:
    required = {'schema_version', 'episode_id', 'frame_id', 'camera_pose', 'object_names', 'object_poses', 'visual_states', 'held_pose', 'render_identity_sha256', 'state_sha256'}
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError('render replay state fields are invalid')
    if state['schema_version'] != REPLAY_STATE_SCHEMA_VERSION:
        raise ValueError('render replay state schema is invalid')
    if episode_id is not None and state['episode_id'] != episode_id:
        raise ValueError('render replay state episode_id is invalid')
    if frame_id is not None and state['frame_id'] != frame_id:
        raise ValueError('render replay state frame_id is invalid')
    if not isinstance(state['camera_pose'], Mapping):
        raise ValueError('render replay camera pose is invalid')
    object_names = state['object_names']
    if not isinstance(object_names, list) or object_names != sorted(object_names) or len(object_names) != len(set(object_names)) or any((not isinstance(item, str) or not item for item in object_names)):
        raise ValueError('render replay object_names are invalid')
    if not isinstance(state['object_poses'], Mapping) or not isinstance(state['visual_states'], Mapping):
        raise ValueError('render replay object state maps are invalid')
    for digest_name in ('render_identity_sha256', 'state_sha256'):
        digest = state[digest_name]
        if not isinstance(digest, str) or len(digest) != 64 or any((c not in '0123456789abcdef' for c in digest)):
            raise ValueError(f'render replay {digest_name} is invalid')
    unhashed = deepcopy(dict(state))
    supplied = unhashed.pop('state_sha256')
    if supplied != _sha256_json(unhashed):
        raise ValueError('render replay state hash mismatch')

def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def _normalized_token(value: Any) -> str:
    return str(value or '').strip().replace('-', '_').casefold()
__all__ = ['POLICY_SCHEMA_VERSION', 'REPLAY_STATE_SCHEMA_VERSION', 'REPLAY_VISIBILITY_SOURCE', 'finalize_render_replay_state', 'particle_suppressed_scene_file', 'particle_segmentation_policy', 'validate_render_replay_state']
