from __future__ import annotations
from typing import Any, Mapping, Sequence
DEFAULT_SUPPORT_SURFACE_CONSTRAINTS: dict[str, Any] = {'min_surface_z': 0.25, 'max_surface_z': 1.35, 'min_surface_extent_xy': [0.25, 0.25], 'max_surface_extent_xy': [3.0, 3.0], 'allowed_support_categories': [], 'denied_support_categories': ['floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'], 'required_abilities_any': ['support_surface', 'placeable_surface', 'heatSource'], 'allowed_object_types': []}

def surface_constraints_from_policy(policy: Mapping[str, Any] | None, purpose: str, *, override: Mapping[str, Any] | None=None) -> dict[str, Any]:
    constraints = dict(DEFAULT_SUPPORT_SURFACE_CONSTRAINTS)
    raw = (policy or {}).get('support_surface_constraints') if isinstance(policy, Mapping) else None
    if isinstance(raw, Mapping):
        default = raw.get('default')
        if isinstance(default, Mapping):
            constraints.update(default)
        specific = raw.get(str(purpose))
        if isinstance(specific, Mapping):
            constraints.update(specific)
    if isinstance(override, Mapping):
        constraints.update(override)
    return constraints

def support_surface_descriptor_from_values(*, category: str | None=None, object_type: str | None=None, abilities: Sequence[str] | None=None, center: Sequence[float] | None=None, extent: Sequence[float] | None=None, surface_z: float | None=None) -> dict[str, Any]:
    extent_xyz = _vector3(extent)
    center_xyz = _vector3(center)
    if surface_z is None and center_xyz is not None and (extent_xyz is not None):
        surface_z = float(center_xyz[2]) + float(extent_xyz[2]) * 0.5
    return {'category': str(category or ''), 'object_type': str(object_type or ''), 'abilities': sorted({str(item) for item in abilities or [] if str(item)}), 'center': list(center_xyz) if center_xyz is not None else None, 'extent': list(extent_xyz) if extent_xyz is not None else None, 'surface_z': float(surface_z) if surface_z is not None else None}

def support_matches_surface_constraints(descriptor: Mapping[str, Any], constraints: Mapping[str, Any]) -> bool:
    category = str(descriptor.get('category') or '')
    object_type = str(descriptor.get('object_type') or '')
    abilities = {str(item) for item in descriptor.get('abilities') or []}
    denied_categories = {str(item) for item in constraints.get('denied_support_categories') or []}
    if category and category in denied_categories:
        return False
    allowed_categories = {str(item) for item in constraints.get('allowed_support_categories') or []}
    if allowed_categories and category not in allowed_categories:
        return False
    allowed_types = {str(item) for item in constraints.get('allowed_object_types') or []}
    if allowed_types and object_type not in allowed_types:
        return False
    required_any = {str(item) for item in constraints.get('required_abilities_any') or []}
    if required_any and (not abilities.intersection(required_any)):
        return False
    surface_z = _number(descriptor.get('surface_z'))
    if surface_z is None:
        return False
    min_z = _number(constraints.get('min_surface_z'))
    max_z = _number(constraints.get('max_surface_z'))
    if min_z is not None and surface_z < min_z:
        return False
    if max_z is not None and surface_z > max_z:
        return False
    extent = _vector3(descriptor.get('extent'))
    if extent is None:
        return False
    min_xy = _vector2(constraints.get('min_surface_extent_xy'))
    max_xy = _vector2(constraints.get('max_surface_extent_xy'))
    if min_xy is not None and (extent[0] < min_xy[0] or extent[1] < min_xy[1]):
        return False
    if max_xy is not None and (extent[0] > max_xy[0] or extent[1] > max_xy[1]):
        return False
    return True

def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value)

def _vector2(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except Exception:
        return None

def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return None
__all__ = ['DEFAULT_SUPPORT_SURFACE_CONSTRAINTS', 'support_matches_surface_constraints', 'support_surface_descriptor_from_values', 'surface_constraints_from_policy']
