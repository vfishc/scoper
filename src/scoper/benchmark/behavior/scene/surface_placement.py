from __future__ import annotations
import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.behavior.paths import repo_root
PLACEMENT_GEOMETRY_ENV = 'EB_BEHAVIOR_PLACEMENT_GEOMETRY'
ASSETS_ROOT_ENV = 'EB_BEHAVIOR_ASSETS_ROOT'
VALID_PLACEMENT_GEOMETRY_MODES = frozenset({'aabb', 'asset_bbox', 'link_bbox'})
DEFAULT_PLACEMENT_GEOMETRY_MODE = 'aabb'
_SUPPORT_LINK_TOKENS = ('top', 'surface', 'counter', 'table', 'shelf', 'desktop', 'worktop')
_EXCLUDED_LINK_TOKENS = ('door', 'drawer', 'handle', 'knob', 'leg', 'wheel', 'caster', 'switch', 'faucet', 'tap')

def placement_geometry_mode(task_spec: Mapping[str, Any]) -> str:
    options = task_spec.get('reset_options') if isinstance(task_spec, Mapping) else None
    raw = None
    if isinstance(options, Mapping):
        raw = options.get('placement_geometry')
    if raw in (None, ''):
        raw = os.environ.get(PLACEMENT_GEOMETRY_ENV, DEFAULT_PLACEMENT_GEOMETRY_MODE)
    mode = str(raw or DEFAULT_PLACEMENT_GEOMETRY_MODE).strip().lower()
    if mode not in VALID_PLACEMENT_GEOMETRY_MODES:
        return DEFAULT_PLACEMENT_GEOMETRY_MODE
    return mode

def assets_root_for_task(task_spec: Mapping[str, Any]) -> Path:
    options = task_spec.get('reset_options') if isinstance(task_spec, Mapping) else None
    raw = options.get('assets_root') if isinstance(options, Mapping) else None
    if raw in (None, ''):
        raw = os.environ.get(ASSETS_ROOT_ENV)
    if raw in (None, ''):
        return repo_root() / 'datasets' / 'behavior-1k-assets'
    return Path(str(raw))

def normalize_roles_for_surface_placement(roles: Mapping[str, Mapping[str, Any]], *, mode: str, assets_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    copied = {str(name): copy.deepcopy(dict(role)) for name, role in roles.items()}
    reports: list[dict[str, Any]] = []
    if mode == 'aabb':
        return (copied, reports)
    for role_name, role in copied.items():
        report = _normalize_role_geometry(role_name, role, assets_root=assets_root)
        if report is not None:
            reports.append(report)
    return (copied, reports)

def resolve_on_support_surface(*, role_name: str, role: Mapping[str, Any], support_role_name: str, support_role: Mapping[str, Any], object_center: Sequence[float], object_extent: Sequence[float], support_center: Sequence[float], support_extent: Sequence[float], mode: str, assets_root: Path, clearance: float) -> tuple[tuple[float, float, float], dict[str, Any]]:
    old_center = _vector3(object_center) or (0.0, 0.0, 0.0)
    extent = _vector3(object_extent) or (0.2, 0.2, 0.2)
    support_center_vec = _vector3(support_center) or (0.0, 0.0, 0.0)
    support_extent_vec = _vector3(support_extent) or (0.2, 0.2, 0.2)
    aabb_surface_z = float(support_center_vec[2]) + float(support_extent_vec[2]) * 0.5
    surface = _surface_z_from_asset(support_role, mode=mode, assets_root=assets_root, fallback_surface_z=aabb_surface_z)
    surface_z = float(surface['surface_z'])
    new_center = (float(old_center[0]), float(old_center[1]), float(surface_z + float(extent[2]) * 0.5 + float(clearance)))
    report = {'kind': 'spawn_surface', 'role': str(role_name), 'support_role': str(support_role_name), 'mode': str(mode), 'surface_source': surface.get('surface_source'), 'surface_z': surface_z, 'aabb_surface_z': aabb_surface_z, 'old_center': [float(v) for v in old_center], 'new_center': [float(v) for v in new_center], 'object_extent': [float(v) for v in extent], 'clearance': float(clearance)}
    if surface.get('fallback_reason'):
        report['fallback_reason'] = surface['fallback_reason']
    if surface.get('metadata_path'):
        report['metadata_path'] = surface['metadata_path']
    if surface.get('link_name'):
        report['link_name'] = surface['link_name']
    if role.get('spawn_placement'):
        report['spawn_placement'] = dict(role.get('spawn_placement') or {})
    return (new_center, report)

def _normalize_role_geometry(role_name: str, role: dict[str, Any], *, assets_root: Path) -> dict[str, Any] | None:
    root_center = _root_center_from_position(role)
    if root_center is None:
        return None
    metadata, metadata_path, reason = _metadata_for_role(role, assets_root)
    if metadata is None:
        return {'kind': 'role_geometry', 'role': str(role_name), 'mode': 'asset_bbox', 'status': 'fallback', 'fallback_reason': reason}
    scale = _scale3(role)
    bbox = _scaled_vector(metadata.get('bbox_size'), scale)
    offset = _scaled_vector(metadata.get('base_link_offset'), scale, default=(0.0, 0.0, 0.0))
    if bbox is None:
        return {'kind': 'role_geometry', 'role': str(role_name), 'mode': 'asset_bbox', 'status': 'fallback', 'fallback_reason': 'missing_bbox_size', 'metadata_path': str(metadata_path)}
    old_center = _vector3(role.get('center'))
    old_extent = _vector3(role.get('extent'))
    new_center = (float(root_center[0] + offset[0]), float(root_center[1] + offset[1]), float(root_center[2] + offset[2]))
    role['center'] = [float(v) for v in new_center]
    role['extent'] = [float(v) for v in bbox]
    return {'kind': 'role_geometry', 'role': str(role_name), 'mode': 'asset_bbox', 'status': 'normalized', 'metadata_path': str(metadata_path), 'old_center': None if old_center is None else [float(v) for v in old_center], 'new_center': [float(v) for v in new_center], 'old_extent': None if old_extent is None else [float(v) for v in old_extent], 'new_extent': [float(v) for v in bbox]}

def _surface_z_from_asset(role: Mapping[str, Any], *, mode: str, assets_root: Path, fallback_surface_z: float) -> dict[str, Any]:
    if mode == 'aabb':
        return {'surface_z': float(fallback_surface_z), 'surface_source': 'aabb'}
    metadata, metadata_path, reason = _metadata_for_role(role, assets_root)
    if metadata is None:
        return {'surface_z': float(fallback_surface_z), 'surface_source': 'aabb', 'fallback_reason': reason}
    root_center = _root_center_from_position(role)
    scale = _scale3(role)
    offset = _scaled_vector(metadata.get('base_link_offset'), scale, default=(0.0, 0.0, 0.0))
    bbox = _scaled_vector(metadata.get('bbox_size'), scale)
    if root_center is None:
        center = _vector3(role.get('center'))
        if center is not None and offset is not None:
            root_center = (float(center[0] - offset[0]), float(center[1] - offset[1]), float(center[2] - offset[2]))
    if root_center is None or offset is None or bbox is None:
        return {'surface_z': float(fallback_surface_z), 'surface_source': 'aabb', 'fallback_reason': 'missing_root_or_bbox', 'metadata_path': str(metadata_path)}
    if mode == 'link_bbox':
        link = _best_support_link_surface(metadata, scale)
        if link is not None:
            return {'surface_z': float(root_center[2] + link['top_z']), 'surface_source': 'link_bbox', 'link_name': link['link_name'], 'metadata_path': str(metadata_path)}
    return {'surface_z': float(root_center[2] + offset[2] + bbox[2] * 0.5), 'surface_source': 'asset_bbox', 'fallback_reason': None if mode == 'asset_bbox' else 'no_support_link_bbox', 'metadata_path': str(metadata_path)}

def _best_support_link_surface(metadata: Mapping[str, Any], scale: tuple[float, float, float]) -> dict[str, Any] | None:
    raw = metadata.get('link_bounding_boxes')
    if not isinstance(raw, Mapping):
        return None
    candidates: list[dict[str, Any]] = []
    for link_name, payload in raw.items():
        box = _axis_aligned_box(payload)
        if box is None:
            continue
        extent, center = box
        extent = (float(extent[0]) * float(scale[0]), float(extent[1]) * float(scale[1]), float(extent[2]) * float(scale[2]))
        center = (float(center[0]) * float(scale[0]), float(center[1]) * float(scale[1]), float(center[2]) * float(scale[2]))
        area = max(0.0, extent[0]) * max(0.0, extent[1])
        if area <= 0.0 or extent[2] <= 0.0:
            continue
        name = str(link_name).lower()
        if any((token in name for token in _EXCLUDED_LINK_TOKENS)):
            continue
        token_bonus = 1 if any((token in name for token in _SUPPORT_LINK_TOKENS)) else 0
        candidates.append({'link_name': str(link_name), 'top_z': float(center[2] + extent[2] * 0.5), 'area': float(area), 'token_bonus': token_bonus})
    if not candidates:
        return None
    max_area = max((item['area'] for item in candidates))
    large_enough = [item for item in candidates if item['area'] >= max_area * 0.1]
    pool = large_enough or candidates
    return max(pool, key=lambda item: (item['token_bonus'], item['top_z'], item['area']))

def _axis_aligned_box(payload: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not isinstance(payload, Mapping):
        return None
    for source in ('collision', 'visual'):
        source_payload = payload.get(source)
        if not isinstance(source_payload, Mapping):
            continue
        box = source_payload.get('axis_aligned')
        if not isinstance(box, Mapping):
            continue
        extent = _vector3(box.get('extent'))
        transform = box.get('transform')
        if extent is None or not isinstance(transform, (list, tuple)) or len(transform) < 3:
            continue
        try:
            center = (float(transform[0][3]), float(transform[1][3]), float(transform[2][3]))
        except Exception:
            continue
        return (extent, center)
    return None

def _metadata_for_role(role: Mapping[str, Any], assets_root: Path) -> tuple[Mapping[str, Any] | None, Path | None, str | None]:
    category = str(role.get('category') or '').strip()
    model = str(role.get('model') or '').strip()
    if not category:
        return (None, None, 'missing_category')
    if not model or model == 'unknown':
        return (None, None, 'missing_model')
    path = _metadata_path(assets_root, category, model)
    if path is None:
        return (None, None, 'metadata_not_found')
    data = _load_metadata(str(path))
    if data is None:
        return (None, path, 'metadata_unreadable')
    return (data, path, None)

def _metadata_path(assets_root: Path, category: str, model: str) -> Path | None:
    for group in ('objects', 'systems'):
        path = Path(assets_root) / group / str(category) / str(model) / 'misc' / 'metadata.json'
        if path.exists():
            return path
    return None

@lru_cache(maxsize=4096)
def _load_metadata(path: str) -> Mapping[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None

def _root_center_from_position(role: Mapping[str, Any]) -> tuple[float, float, float] | None:
    pos = role.get('position')
    if not isinstance(pos, Mapping):
        return None
    try:
        return (float(pos['x']), float(pos['z']), float(pos['y']))
    except Exception:
        return None

def _scale3(role: Mapping[str, Any]) -> tuple[float, float, float]:
    return _vector3(role.get('scale')) or (1.0, 1.0, 1.0)

def _scaled_vector(value: Any, scale: tuple[float, float, float], *, default: tuple[float, float, float] | None=None) -> tuple[float, float, float] | None:
    vector = _vector3(value)
    if vector is None:
        vector = default
    if vector is None:
        return None
    return (float(vector[0]) * float(scale[0]), float(vector[1]) * float(scale[1]), float(vector[2]) * float(scale[2]))

def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return None
__all__ = ['ASSETS_ROOT_ENV', 'DEFAULT_PLACEMENT_GEOMETRY_MODE', 'PLACEMENT_GEOMETRY_ENV', 'VALID_PLACEMENT_GEOMETRY_MODES', 'assets_root_for_task', 'normalize_roles_for_surface_placement', 'placement_geometry_mode', 'resolve_on_support_surface']
