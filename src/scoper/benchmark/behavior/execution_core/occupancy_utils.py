from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from PIL import Image, ImageDraw
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose, yaw_towards
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject, OccupancyConfig, OccupancyMap, _map_mask_for_grid

@dataclass(frozen=True)
class StartSelectionConfig:
    min_start_distance: float = 3.0
    max_start_distance: float = 0.0
    preferred_start_distance: float = 3.5
    start_edge_margin: float = 0.6
    start_candidates_per_goal: int = 6

    def as_dict(self) -> dict[str, float | int]:
        return {'min_start_distance': float(self.min_start_distance), 'max_start_distance': float(self.max_start_distance), 'preferred_start_distance': float(self.preferred_start_distance), 'start_edge_margin': float(self.start_edge_margin), 'start_candidates_per_goal': int(self.start_candidates_per_goal)}

@dataclass
class SceneTraversability:
    floor_map: np.ndarray | None = None
    map_resolution: float | None = None
    map_size: int | None = None
    floor: int = 0
    source: str | None = None
    indoor_mask: np.ndarray | None = None
    doorway_opening_objects: tuple[AABBObject, ...] = ()
    doorway_opening_allowed_map: Any | None = None
    doorway_opening_allowed_map_resolution: float | None = None
    doorway_opening_allowed_map_size: int | None = None
    metadata: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.floor_map is not None

    def occupancy_kwargs(self) -> dict[str, Any]:
        if self.floor_map is None:
            return {}
        kwargs = {'traversability_map': self.floor_map, 'traversability_map_resolution': self.map_resolution, 'traversability_map_size': self.map_size, 'traversability_floor': self.floor, 'traversability_source': self.source or 'scene.trav_map'}
        if self.doorway_opening_objects:
            kwargs['doorway_opening_objects'] = tuple(self.doorway_opening_objects)
            allowed_map = self.doorway_opening_allowed_map
            allowed_resolution = self.doorway_opening_allowed_map_resolution
            allowed_size = self.doorway_opening_allowed_map_size
            if allowed_map is None and self.indoor_mask is not None:
                allowed_map = self.indoor_mask
                allowed_resolution = self.map_resolution
                allowed_size = self.map_size
            if allowed_map is not None:
                kwargs['doorway_opening_allowed_map'] = allowed_map
                kwargs['doorway_opening_allowed_map_resolution'] = allowed_resolution or self.map_resolution
                kwargs['doorway_opening_allowed_map_size'] = allowed_size or self.map_size
        return kwargs

def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.array(value)

def _merged_doorway_opening_objects(*traversabilities: SceneTraversability) -> tuple[AABBObject, ...]:
    merged: list[AABBObject] = []
    seen: set[tuple[str, str]] = set()
    for traversability in traversabilities:
        for obj in tuple(getattr(traversability, 'doorway_opening_objects', ()) or ()):
            key = (str(getattr(obj, 'name', '')), str(getattr(obj, 'category', '')))
            if key in seen:
                continue
            seen.add(key)
            merged.append(obj)
    return tuple(merged)

def _doorway_opening_allowed_kwargs(*traversabilities: SceneTraversability) -> dict[str, Any]:
    for traversability in traversabilities:
        allowed_map = getattr(traversability, 'doorway_opening_allowed_map', None)
        if allowed_map is not None:
            return {'doorway_opening_allowed_map': allowed_map, 'doorway_opening_allowed_map_resolution': getattr(traversability, 'doorway_opening_allowed_map_resolution', None) or getattr(traversability, 'map_resolution', None), 'doorway_opening_allowed_map_size': getattr(traversability, 'doorway_opening_allowed_map_size', None) or getattr(traversability, 'map_size', None)}
        indoor_mask = getattr(traversability, 'indoor_mask', None)
        if indoor_mask is not None:
            return {'doorway_opening_allowed_map': indoor_mask, 'doorway_opening_allowed_map_resolution': getattr(traversability, 'map_resolution', None), 'doorway_opening_allowed_map_size': getattr(traversability, 'map_size', None)}
    return {}

def bounds_from_aabbs(aabb_objects: Sequence[AABBObject], padding: float=0.75) -> tuple[float, float, float, float]:
    if not aabb_objects:
        raise ValueError('Cannot infer occupancy bounds without objects')
    min_x = min((obj.center[0] - obj.extent[0] * 0.5 for obj in aabb_objects)) - padding
    min_y = min((obj.center[1] - obj.extent[1] * 0.5 for obj in aabb_objects)) - padding
    max_x = max((obj.center[0] + obj.extent[0] * 0.5 for obj in aabb_objects)) + padding
    max_y = max((obj.center[1] + obj.extent[1] * 0.5 for obj in aabb_objects)) + padding
    return (min_x, min_y, max_x, max_y)

def load_indoor_segmentation_mask(assets_root: str | Path, scene_model: str, floor: int, shape: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    path = Path(assets_root) / 'scenes' / scene_model / 'layout' / f'floor_insseg_{floor}.png'
    info: dict[str, Any] = {'enabled': False, 'source': str(path)}
    if not path.exists():
        info['reason'] = 'missing_floor_insseg'
        return (None, info)
    try:
        image = Image.open(path).convert('L')
        original_shape = [image.height, image.width]
        if (image.height, image.width) != tuple(shape):
            image = image.resize((int(shape[1]), int(shape[0])), Image.Resampling.NEAREST)
        mask = np.array(image) > 0
    except Exception as exc:
        info['reason'] = f'failed_to_load_floor_insseg: {exc}'
        return (None, info)
    info.update({'enabled': True, 'original_shape': original_shape, 'shape': list(mask.shape), 'free_pixels': int(np.count_nonzero(mask))})
    return (mask, info)

def scene_traversability_from_scene(scene: Any, *, scene_model: str, assets_root: str | Path, floor: int=0, occupancy_resolution: float=0.1, use_scene_traversability: bool=True, use_indoor_segmentation_mask: bool=True) -> SceneTraversability:
    if not use_scene_traversability:
        return SceneTraversability(floor=floor, metadata={'enabled': False, 'reason': 'disabled'})
    trav_map = getattr(scene, 'trav_map', None)
    floor_maps = getattr(trav_map, 'floor_map', None)
    if trav_map is None or floor_maps is None or floor >= len(floor_maps):
        return SceneTraversability(floor=floor, metadata={'enabled': False, 'reason': 'scene_trav_map_unavailable'})
    floor_map = to_numpy(floor_maps[floor])
    if floor_map.ndim != 2 or floor_map.size == 0:
        return SceneTraversability(floor=floor, metadata={'enabled': False, 'reason': f'invalid_floor_map_shape_{tuple(floor_map.shape)}'})
    floor_map = floor_map.copy()
    free_pixels_before = int(np.count_nonzero(floor_map > 0))
    indoor_mask = None
    indoor_mask_info: dict[str, Any] = {'enabled': False}
    if use_indoor_segmentation_mask:
        indoor_mask, indoor_mask_info = load_indoor_segmentation_mask(assets_root, scene_model, floor, floor_map.shape)
        if indoor_mask is not None:
            floor_map = np.where(indoor_mask, floor_map, 0).astype(floor_map.dtype, copy=False)
    source = 'scene.trav_map.floor_map'
    if indoor_mask_info.get('enabled'):
        source = f'{source}&floor_insseg'
    map_resolution = float(getattr(trav_map, 'map_resolution', occupancy_resolution))
    map_size = int(getattr(trav_map, 'map_size', floor_map.shape[0]))
    metadata = {'enabled': True, 'source': source, 'floor': floor, 'shape': list(floor_map.shape), 'map_resolution': map_resolution, 'map_size': map_size, 'free_pixels_before_indoor_mask': free_pixels_before, 'free_pixels': int(np.count_nonzero(floor_map > 0)), 'indoor_mask': indoor_mask_info}
    return SceneTraversability(floor_map=floor_map, map_resolution=map_resolution, map_size=map_size, floor=floor, source=source, indoor_mask=indoor_mask, metadata=metadata)

def write_scene_traversability_debug(traversability: SceneTraversability, debug_dir: str | Path) -> dict[str, str]:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if traversability.floor_map is not None:
        path = debug_dir / 'scene_traversability_floor_masked.png'
        Image.fromarray(np.where(np.asarray(traversability.floor_map) > 0, 255, 0).astype(np.uint8)).save(path)
        paths['masked_traversability_png'] = str(path)
    if traversability.indoor_mask is not None:
        path = debug_dir / 'scene_indoor_insseg_mask.png'
        Image.fromarray(np.where(np.asarray(traversability.indoor_mask), 255, 0).astype(np.uint8)).save(path)
        paths['indoor_mask_png'] = str(path)
    return paths

def scene_mesh_occupancy_from_scene(scene: Any, *, floor: int=0, floor_height: float=0.0, occupancy_resolution: float=0.1, map_size: int | None=None, navigation_height_band: tuple[float, float]=(0.05, 2.0), structural_categories: Sequence[str]=('wall', 'walls')) -> SceneTraversability:
    if scene is None:
        return SceneTraversability(floor=floor, metadata={'enabled': False, 'reason': 'scene_unavailable'})
    resolution = float(occupancy_resolution)
    if resolution <= 0:
        return SceneTraversability(floor=floor, metadata={'enabled': False, 'reason': 'invalid_resolution'})
    resolved_map_size = _resolve_mesh_occupancy_map_size(scene, resolution, map_size)
    if resolved_map_size <= 0:
        return SceneTraversability(floor=floor, metadata={'enabled': False, 'reason': 'invalid_map_size'})
    band_min = float(floor_height) + float(navigation_height_band[0])
    band_max = float(floor_height) + float(navigation_height_band[1])
    if band_min > band_max:
        band_min, band_max = (band_max, band_min)
    occupied_image = Image.new('L', (resolved_map_size, resolved_map_size), 0)
    draw = ImageDraw.Draw(occupied_image)
    category_filter = {str(category).lower() for category in structural_categories}
    mesh_records: list[dict[str, Any]] = []
    object_count = 0
    mesh_count = 0
    face_count = 0
    rasterized_face_count = 0
    line_width = max(1, int(math.ceil(0.05 / resolution)))
    for og_obj in getattr(scene, 'objects', []) or []:
        category = str(getattr(og_obj, 'category', '') or '').lower()
        if category not in category_filter:
            continue
        object_count += 1
        object_mesh_count = 0
        object_rasterized_faces = 0
        for mesh_info in _iter_og_object_trimeshes(og_obj):
            mesh_count += 1
            object_mesh_count += 1
            vertices = np.asarray(getattr(mesh_info['mesh'], 'vertices', []), dtype=np.float64)
            faces = np.asarray(getattr(mesh_info['mesh'], 'faces', []), dtype=np.int64)
            if vertices.ndim != 2 or vertices.shape[1] < 3 or faces.ndim != 2 or (faces.shape[1] < 3):
                continue
            face_count += int(len(faces))
            rasterized = _rasterize_height_band_faces(draw, vertices=vertices, faces=faces[:, :3], map_size=resolved_map_size, resolution=resolution, band_min=band_min, band_max=band_max, line_width=line_width)
            rasterized_face_count += rasterized
            object_rasterized_faces += rasterized
        mesh_records.append({'name': str(getattr(og_obj, 'name', '')), 'category': category, 'mesh_count': object_mesh_count, 'rasterized_faces': object_rasterized_faces})
    occupied = np.asarray(occupied_image) > 0
    free = ~occupied
    metadata = {'enabled': True, 'source': 'scene.mesh_projection', 'floor': floor, 'floor_height': float(floor_height), 'height_band': [float(band_min), float(band_max)], 'map_resolution': resolution, 'map_size': int(resolved_map_size), 'shape': [int(resolved_map_size), int(resolved_map_size)], 'structural_categories': sorted(category_filter), 'structural_object_count': int(object_count), 'mesh_count': int(mesh_count), 'face_count': int(face_count), 'rasterized_face_count': int(rasterized_face_count), 'occupied_pixels': int(np.count_nonzero(occupied)), 'free_pixels': int(np.count_nonzero(free)), 'objects': mesh_records}
    return SceneTraversability(floor_map=np.where(free, 255, 0).astype(np.uint8), map_resolution=resolution, map_size=int(resolved_map_size), floor=floor, source='scene.mesh_projection', metadata=metadata)

def scene_wall_slice_occupancy_from_scene(scene: Any, *, floor: int=0, floor_height: float=0.0, occupancy_resolution: float=0.1, map_size: int | None=None, slice_heights: Sequence[float]=(0.2, 0.8, 1.2), structural_categories: Sequence[str]=('wall', 'walls'), visual_fallback: bool=False, exterior_barriers: Sequence[AABBObject]=(), interior_mask: Any | None=None, interior_mask_resolution: float | None=None, interior_mask_size: int | None=None) -> dict[str, Any]:
    if scene is None:
        return {'enabled': False, 'reason': 'scene_unavailable', 'floor': floor}
    resolution = float(occupancy_resolution)
    if resolution <= 0:
        return {'enabled': False, 'reason': 'invalid_resolution', 'floor': floor}
    resolved_map_size = _resolve_mesh_occupancy_map_size(scene, resolution, map_size)
    if resolved_map_size <= 0:
        return {'enabled': False, 'reason': 'invalid_map_size', 'floor': floor}
    resolved_slice_heights = [float(height) for height in slice_heights]
    category_filter = {str(category).lower() for category in structural_categories}
    occupied_by_height = {float(height): np.zeros((resolved_map_size, resolved_map_size), dtype=bool) for height in resolved_slice_heights}
    segment_counts = {float(height): 0 for height in resolved_slice_heights}
    mesh_records: list[dict[str, Any]] = []
    object_count = 0
    mesh_count = 0
    face_count = 0
    line_width = max(1, int(math.ceil(0.02 / resolution)))
    for og_obj in getattr(scene, 'objects', []) or []:
        category = str(getattr(og_obj, 'category', '') or '').lower()
        if category not in category_filter:
            continue
        object_count += 1
        object_mesh_count = 0
        object_segment_count = 0
        for mesh_info in _iter_og_object_trimeshes(og_obj, visual_fallback=visual_fallback):
            if not visual_fallback and str(mesh_info.get('purpose') or '') != 'collision':
                continue
            mesh_count += 1
            object_mesh_count += 1
            vertices = np.asarray(getattr(mesh_info['mesh'], 'vertices', []), dtype=np.float64)
            faces = np.asarray(getattr(mesh_info['mesh'], 'faces', []), dtype=np.int64)
            if vertices.ndim != 2 or vertices.shape[1] < 3 or faces.ndim != 2 or (faces.shape[1] < 3):
                continue
            face_count += int(len(faces))
            for height in resolved_slice_heights:
                z = float(floor_height) + float(height)
                rasterized = _rasterize_height_slice_segments(occupied_by_height[float(height)], vertices=vertices, faces=faces[:, :3], map_size=resolved_map_size, resolution=resolution, z=z, line_width=line_width)
                segment_counts[float(height)] += rasterized
                object_segment_count += rasterized
        mesh_records.append({'name': str(getattr(og_obj, 'name', '')), 'category': category, 'mesh_count': object_mesh_count, 'rasterized_segments': object_segment_count})
    structural_occupied = np.zeros((resolved_map_size, resolved_map_size), dtype=bool)
    for slice_occupied in occupied_by_height.values():
        structural_occupied |= slice_occupied
    exterior_barrier_map, exterior_barrier_records = _rasterize_exterior_barriers(exterior_barriers, map_size=resolved_map_size, resolution=resolution)
    interior = None
    if interior_mask is not None:
        interior = _map_mask_for_grid(interior_mask, height=resolved_map_size, width=resolved_map_size, origin_xy=(-resolved_map_size * resolution / 2.0, -resolved_map_size * resolution / 2.0), resolution=resolution, map_resolution=interior_mask_resolution, map_size=interior_mask_size)
        exterior = ~interior
        exterior_method = 'indoor_mask'
    else:
        exterior_seed_barriers = structural_occupied | exterior_barrier_map
        exterior = _exterior_mask_from_occupied(exterior_seed_barriers)
        exterior_method = 'border_floodfill'
    occupied = structural_occupied | exterior
    free = ~occupied
    source = f'scene.wall_collision_height_slices.{exterior_method}'
    metadata = {'enabled': True, 'source': source, 'floor': floor, 'floor_height': float(floor_height), 'slice_heights': [float(height) for height in resolved_slice_heights], 'map_resolution': resolution, 'map_size': int(resolved_map_size), 'shape': [int(resolved_map_size), int(resolved_map_size)], 'structural_categories': sorted(category_filter), 'visual_fallback': bool(visual_fallback), 'structural_object_count': int(object_count), 'mesh_count': int(mesh_count), 'face_count': int(face_count), 'line_width_px': int(line_width), 'segment_counts': {str(height): int(count) for height, count in sorted(segment_counts.items())}, 'structural_occupied_pixels': int(np.count_nonzero(structural_occupied)), 'exterior_barrier_pixels': int(np.count_nonzero(exterior_barrier_map)), 'exterior_barriers': exterior_barrier_records, 'exterior_method': exterior_method, 'interior_mask_used': interior is not None, 'exterior_blocked_pixels': int(np.count_nonzero(exterior)), 'occupied_pixels': int(np.count_nonzero(occupied)), 'free_pixels': int(np.count_nonzero(free)), 'objects': mesh_records}
    return {'enabled': True, 'floor_map': np.where(free, 255, 0).astype(np.uint8), 'occupied_map': occupied, 'structural_occupied_map': structural_occupied, 'exterior_barrier_map': exterior_barrier_map, 'exterior_map': exterior, 'occupied_by_height': occupied_by_height, 'map_resolution': resolution, 'map_size': int(resolved_map_size), 'floor': floor, 'source': source, 'indoor_mask': interior, 'metadata': metadata}

def scene_wall_slice_occupancy_as_traversability(wall_slice_occupancy: Mapping[str, Any]) -> SceneTraversability:
    if not wall_slice_occupancy.get('enabled') or wall_slice_occupancy.get('floor_map') is None:
        return SceneTraversability(floor=int(wall_slice_occupancy.get('floor') or 0), metadata={'enabled': False, 'reason': wall_slice_occupancy.get('reason') or 'wall_slice_occupancy_unavailable', 'source': wall_slice_occupancy.get('source') or 'scene.wall_collision_height_slices'})
    metadata = dict(wall_slice_occupancy.get('metadata') or {})
    return SceneTraversability(floor_map=np.asarray(wall_slice_occupancy['floor_map']), map_resolution=float(wall_slice_occupancy.get('map_resolution') or metadata.get('map_resolution') or 0.1), map_size=int(wall_slice_occupancy.get('map_size') or metadata.get('map_size') or np.asarray(wall_slice_occupancy['floor_map']).shape[0]), floor=int(wall_slice_occupancy.get('floor') or metadata.get('floor') or 0), source=str(wall_slice_occupancy.get('source') or metadata.get('source') or 'scene.wall_collision_height_slices'), indoor_mask=wall_slice_occupancy.get('indoor_mask'), metadata=metadata)

def indoor_mask_as_traversability(reference: SceneTraversability) -> SceneTraversability:
    if reference.indoor_mask is None:
        return SceneTraversability(floor=reference.floor, metadata={'enabled': False, 'reason': 'indoor_mask_unavailable', 'source': reference.source})
    indoor = np.asarray(reference.indoor_mask) > 0
    metadata = {'enabled': True, 'source': 'floor_insseg', 'floor': reference.floor, 'shape': [int(indoor.shape[0]), int(indoor.shape[1])], 'map_resolution': reference.map_resolution, 'map_size': reference.map_size, 'free_pixels': int(np.count_nonzero(indoor)), 'reference_source': reference.source}
    return SceneTraversability(floor_map=np.where(indoor, 255, 0).astype(np.uint8), map_resolution=reference.map_resolution, map_size=reference.map_size, floor=reference.floor, source='floor_insseg', indoor_mask=indoor, doorway_opening_objects=tuple(reference.doorway_opening_objects), **_doorway_opening_allowed_kwargs(reference), metadata=metadata)

def mask_traversability_with_reference(traversability: SceneTraversability, reference: SceneTraversability, *, label: str='reference') -> SceneTraversability:
    if traversability.floor_map is None or reference.floor_map is None:
        return traversability
    target_map = np.asarray(traversability.floor_map)
    reference_map = np.asarray(reference.floor_map)
    if target_map.ndim != 2 or reference_map.ndim != 2 or target_map.size == 0 or (reference_map.size == 0):
        return traversability
    target_resolution = float(traversability.map_resolution or 0.0)
    reference_resolution = float(reference.map_resolution or 0.0)
    target_size = int(traversability.map_size or target_map.shape[0])
    reference_size = int(reference.map_size or reference_map.shape[0])
    if target_resolution <= 0.0 or reference_resolution <= 0.0 or target_size <= 0 or (reference_size <= 0):
        return traversability
    height, width = target_map.shape
    world_x = (np.arange(width, dtype=np.float64) + 0.5 - target_size / 2.0) * target_resolution
    world_y = (np.arange(height, dtype=np.float64) + 0.5 - target_size / 2.0) * target_resolution
    reference_cols = np.floor(world_x / reference_resolution + reference_size / 2.0).astype(np.int32)
    reference_rows = np.floor(world_y / reference_resolution + reference_size / 2.0).astype(np.int32)
    valid_cols = (reference_cols >= 0) & (reference_cols < reference_map.shape[1])
    valid_rows = (reference_rows >= 0) & (reference_rows < reference_map.shape[0])
    reference_free = np.zeros((height, width), dtype=bool)
    if np.any(valid_cols) and np.any(valid_rows):
        row_indices = np.nonzero(valid_rows)[0]
        col_indices = np.nonzero(valid_cols)[0]
        sampled = reference_map[reference_rows[row_indices][:, None], reference_cols[col_indices][None, :]]
        reference_free[np.ix_(row_indices, col_indices)] = sampled > 0
    masked = np.where(reference_free, target_map, 0).astype(target_map.dtype, copy=False)
    metadata = dict(traversability.metadata or {})
    metadata['reference_mask'] = {'label': str(label), 'source': reference.source, 'map_resolution': reference.map_resolution, 'map_size': reference.map_size, 'free_pixels_before': int(np.count_nonzero(target_map > 0)), 'free_pixels_after': int(np.count_nonzero(masked > 0))}
    source = '{}&{}'.format(traversability.source or 'traversability', str(label))
    return SceneTraversability(floor_map=masked, map_resolution=target_resolution, map_size=target_size, floor=traversability.floor, source=source, indoor_mask=traversability.indoor_mask, doorway_opening_objects=_merged_doorway_opening_objects(traversability, reference), **_doorway_opening_allowed_kwargs(traversability, reference), metadata=metadata)

def write_scene_mesh_occupancy_debug(mesh_occupancy: SceneTraversability, debug_dir: str | Path, *, navigation_objects: Sequence[AABBObject]=(), reference_traversability: SceneTraversability | None=None, agent_radius: float=0.2, safety_margin: float=0.05) -> dict[str, str]:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if mesh_occupancy.floor_map is None:
        return paths
    free = np.asarray(mesh_occupancy.floor_map) > 0
    occupied = ~free
    raw_free_path = debug_dir / 'mesh_occupancy_free_raw.png'
    raw_occupied_path = debug_dir / 'mesh_occupancy_occupied_raw.png'
    Image.fromarray(np.where(free, 255, 0).astype(np.uint8)).save(raw_free_path)
    Image.fromarray(np.where(occupied, 255, 0).astype(np.uint8)).save(raw_occupied_path)
    paths['mesh_free_raw_png'] = str(raw_free_path)
    paths['mesh_occupied_raw_png'] = str(raw_occupied_path)
    radius_cells = int(math.ceil(max(0.0, float(agent_radius) + float(safety_margin)) / float(mesh_occupancy.map_resolution or 0.1)))
    if radius_cells > 0:
        free_eroded = _erode_bool_free_square(free, radius_cells)
    else:
        free_eroded = free.copy()
    eroded_path = debug_dir / 'mesh_occupancy_free_eroded_preview.png'
    Image.fromarray(np.where(free_eroded, 255, 0).astype(np.uint8)).save(eroded_path)
    paths['mesh_free_eroded_preview_png'] = str(eroded_path)
    door_overlay_path = debug_dir / 'mesh_occupancy_with_door_aabbs.png'
    _write_door_overlay_image(free, door_overlay_path, map_size=int(mesh_occupancy.map_size or free.shape[0]), resolution=float(mesh_occupancy.map_resolution or 0.1), navigation_objects=navigation_objects)
    paths['mesh_with_door_aabbs_png'] = str(door_overlay_path)
    if reference_traversability is not None and reference_traversability.floor_map is not None:
        reference = np.asarray(reference_traversability.floor_map) > 0
        if reference.shape == free.shape:
            diff_path = debug_dir / 'mesh_vs_scene_traversability_diff.png'
            diff = np.zeros((*free.shape, 3), dtype=np.uint8)
            both_free = free & reference
            both_blocked = ~free & ~reference
            mesh_free_scene_blocked = free & ~reference
            mesh_blocked_scene_free = ~free & reference
            diff[both_free] = (245, 245, 245)
            diff[both_blocked] = (0, 0, 0)
            diff[mesh_free_scene_blocked] = (0, 180, 255)
            diff[mesh_blocked_scene_free] = (255, 120, 0)
            Image.fromarray(diff).save(diff_path)
            paths['mesh_vs_scene_traversability_diff_png'] = str(diff_path)
    return paths

def write_scene_wall_slice_occupancy_debug(wall_slice_occupancy: Mapping[str, Any], debug_dir: str | Path) -> dict[str, str]:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if not wall_slice_occupancy.get('enabled'):
        return paths
    occupied = np.asarray(wall_slice_occupancy.get('occupied_map'), dtype=bool)
    if occupied.ndim != 2 or occupied.size == 0:
        return paths
    structural_occupied = np.asarray(wall_slice_occupancy.get('structural_occupied_map', occupied), dtype=bool)
    if structural_occupied.shape != occupied.shape:
        structural_occupied = occupied
    exterior = np.asarray(wall_slice_occupancy.get('exterior_map', np.zeros_like(occupied)), dtype=bool)
    if exterior.shape != occupied.shape:
        exterior = np.zeros_like(occupied, dtype=bool)
    free = ~occupied
    structural_occupied_path = debug_dir / 'wall_slice_occupancy_occupied_union.png'
    structural_free_path = debug_dir / 'wall_slice_occupancy_free_union.png'
    occupied_path = debug_dir / 'wall_slice_occupancy_occupied_with_exterior.png'
    free_path = debug_dir / 'wall_slice_occupancy_free_interior.png'
    exterior_path = debug_dir / 'wall_slice_occupancy_exterior_floodfill.png'
    Image.fromarray(np.where(structural_occupied, 255, 0).astype(np.uint8)).save(structural_occupied_path)
    Image.fromarray(np.where(~structural_occupied, 255, 0).astype(np.uint8)).save(structural_free_path)
    Image.fromarray(np.where(occupied, 255, 0).astype(np.uint8)).save(occupied_path)
    Image.fromarray(np.where(free, 255, 0).astype(np.uint8)).save(free_path)
    Image.fromarray(np.where(exterior, 255, 0).astype(np.uint8)).save(exterior_path)
    paths['wall_slice_occupied_union_png'] = str(structural_occupied_path)
    paths['wall_slice_free_union_png'] = str(structural_free_path)
    paths['wall_slice_occupied_with_exterior_png'] = str(occupied_path)
    paths['wall_slice_free_interior_png'] = str(free_path)
    paths['wall_slice_exterior_floodfill_png'] = str(exterior_path)
    slice_panels = []
    for height, slice_occupied in sorted((wall_slice_occupancy.get('occupied_by_height') or {}).items()):
        safe_height = str(height).replace('.', 'p').replace('-', 'm')
        path = debug_dir / f'wall_slice_occupancy_occupied_h_{safe_height}.png'
        Image.fromarray(np.where(np.asarray(slice_occupied, dtype=bool), 255, 0).astype(np.uint8)).save(path)
        paths[f'wall_slice_occupied_h_{safe_height}_png'] = str(path)
        slice_panels.append((f'h={float(height):.2f}m', np.asarray(slice_occupied, dtype=bool)))
    montage_path = debug_dir / 'wall_slice_occupancy_no_inflation_montage.png'
    _write_wall_slice_montage(slice_panels, structural_occupied, montage_path)
    paths['wall_slice_no_inflation_montage_png'] = str(montage_path)
    final_montage_path = debug_dir / 'wall_slice_occupancy_exterior_montage.png'
    _write_wall_slice_montage([*slice_panels, ('structural union', structural_occupied), ('exterior', exterior)], occupied, final_montage_path)
    paths['wall_slice_exterior_montage_png'] = str(final_montage_path)
    return paths

def _write_wall_slice_montage(slice_panels: Sequence[tuple[str, np.ndarray]], occupied_union: np.ndarray, path: Path) -> None:
    panels: list[tuple[str, np.ndarray]] = [(str(title), np.asarray(mask, dtype=bool)) for title, mask in slice_panels]
    panels.append(('union', np.asarray(occupied_union, dtype=bool)))
    if not panels:
        return
    scale = max(2, int(math.ceil(512 / max((max(mask.shape) for _, mask in panels)))))
    header_h = 24
    cell_w = max((mask.shape[1] * scale for _, mask in panels))
    cell_h = max((mask.shape[0] * scale + header_h for _, mask in panels))
    cols = min(3, len(panels))
    rows = int(math.ceil(len(panels) / cols))
    pad = 10
    montage = Image.new('RGB', (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), (235, 238, 242))
    for idx, (title, mask) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        panel = Image.new('RGB', (cell_w, cell_h), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 6), title, fill=(0, 0, 0))
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rgb[mask] = (0, 0, 0)
        rgb[~mask] = (245, 245, 245)
        image = Image.fromarray(np.flipud(rgb), mode='RGB').resize((mask.shape[1] * scale, mask.shape[0] * scale), Image.Resampling.NEAREST)
        panel.paste(image, (0, header_h))
        montage.paste(panel, (x, y))
    montage.save(path)

def _resolve_mesh_occupancy_map_size(scene: Any, resolution: float, map_size: int | None) -> int:
    if map_size is not None:
        return int(map_size)
    trav_map = getattr(scene, 'trav_map', None)
    scene_map_size = getattr(trav_map, 'map_size', None)
    if scene_map_size:
        return int(scene_map_size)
    bounds = []
    for og_obj in getattr(scene, 'objects', []) or []:
        try:
            center = to_numpy(getattr(og_obj, 'aabb_center')).astype(float)
            extent = to_numpy(getattr(og_obj, 'aabb_extent')).astype(float)
        except Exception:
            continue
        if center.shape[0] < 2 or extent.shape[0] < 2:
            continue
        bounds.extend([abs(float(center[0] - extent[0] * 0.5)), abs(float(center[0] + extent[0] * 0.5)), abs(float(center[1] - extent[1] * 0.5)), abs(float(center[1] + extent[1] * 0.5))])
    if not bounds:
        return 0
    return int(math.ceil((2.0 * max(bounds) + 1.0) / resolution))

def _exterior_mask_from_occupied(occupied: np.ndarray) -> np.ndarray:
    barriers = np.asarray(occupied, dtype=bool)
    if barriers.ndim != 2 or barriers.size == 0:
        return np.zeros_like(barriers, dtype=bool)
    height, width = barriers.shape
    exterior = np.zeros_like(barriers, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def seed(row: int, col: int) -> None:
        if barriers[row, col] or exterior[row, col]:
            return
        exterior[row, col] = True
        queue.append((row, col))
    for col in range(width):
        seed(0, col)
        seed(height - 1, col)
    for row in range(height):
        seed(row, 0)
        seed(row, width - 1)
    while queue:
        row, col = queue.popleft()
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= next_row < height and 0 <= next_col < width and (not barriers[next_row, next_col]) and (not exterior[next_row, next_col]):
                exterior[next_row, next_col] = True
                queue.append((next_row, next_col))
    return exterior

def _rasterize_exterior_barriers(barriers: Sequence[AABBObject], *, map_size: int, resolution: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    barrier_map = np.zeros((int(map_size), int(map_size)), dtype=bool)
    records: list[dict[str, Any]] = []
    if map_size <= 0 or resolution <= 0:
        return (barrier_map, records)
    for obj in barriers:
        states = getattr(obj, 'states', {}) or {}
        if isinstance(states, Mapping):
            margin = states.get('exterior_floodfill_barrier_margin', states.get('navigation_opening_margin', 0.0))
        else:
            margin = 0.0
        try:
            margin = max(0.0, float(margin))
        except Exception:
            margin = 0.0
        x0 = float(obj.center[0]) - float(obj.extent[0]) * 0.5 - margin
        x1 = float(obj.center[0]) + float(obj.extent[0]) * 0.5 + margin
        y0 = float(obj.center[1]) - float(obj.extent[1]) * 0.5 - margin
        y1 = float(obj.center[1]) + float(obj.extent[1]) * 0.5 + margin
        c0, r0 = _world_xy_to_map_pixel(x0, y0, int(map_size), float(resolution))
        c1, r1 = _world_xy_to_map_pixel(x1, y1, int(map_size), float(resolution))
        c0, c1 = sorted((c0, c1))
        r0, r1 = sorted((r0, r1))
        c0 = max(0, c0)
        r0 = max(0, r0)
        c1 = min(int(map_size) - 1, c1)
        r1 = min(int(map_size) - 1, r1)
        if c0 > c1 or r0 > r1:
            continue
        barrier_map[r0:r1 + 1, c0:c1 + 1] = True
        records.append({'name': str(getattr(obj, 'name', '')), 'category': str(getattr(obj, 'category', '')), 'cell_bounds': [int(r0), int(c0), int(r1), int(c1)], 'margin': float(margin)})
    return (barrier_map, records)

def _iter_og_object_trimeshes(og_obj: Any, *, visual_fallback: bool=True) -> list[dict[str, Any]]:
    try:
        from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh
    except Exception:
        return []
    meshes: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    links = getattr(og_obj, 'links', {}) or {}
    for link in links.values():
        geom_sources = []
        collision_meshes = list((getattr(link, 'collision_meshes', {}) or {}).values())
        visual_meshes = list((getattr(link, 'visual_meshes', {}) or {}).values())
        geom_sources.extend((('collision', mesh) for mesh in collision_meshes))
        if visual_fallback and (not collision_meshes):
            geom_sources.extend((('visual', mesh) for mesh in visual_meshes))
        for purpose, geom in geom_sources:
            prim = getattr(geom, 'prim', None)
            if prim is None:
                continue
            prim_path = str(getattr(getattr(prim, 'GetPath', lambda: '')(), 'pathString', ''))
            if prim_path in seen_paths:
                continue
            seen_paths.add(prim_path)
            try:
                mesh = mesh_prim_to_trimesh_mesh(prim, include_normals=False, include_texcoord=False, world_frame=True)
            except Exception:
                continue
            meshes.append({'mesh': mesh, 'purpose': purpose, 'prim_path': prim_path})
    return meshes

def _rasterize_height_band_faces(draw: ImageDraw.ImageDraw, *, vertices: np.ndarray, faces: np.ndarray, map_size: int, resolution: float, band_min: float, band_max: float, line_width: int) -> int:
    rasterized = 0
    for face in faces:
        tri = vertices[face]
        z_min = float(np.min(tri[:, 2]))
        z_max = float(np.max(tri[:, 2]))
        if z_max < band_min or z_min > band_max:
            continue
        points = [_world_xy_to_map_pixel(float(vertex[0]), float(vertex[1]), map_size, resolution) for vertex in tri]
        if not _points_touch_image(points, map_size):
            continue
        clamped = [(min(max(col, 0), map_size - 1), min(max(row, 0), map_size - 1)) for col, row in points]
        try:
            draw.polygon(clamped, fill=255)
            draw.line(clamped + [clamped[0]], fill=255, width=line_width)
        except Exception:
            continue
        rasterized += 1
    return rasterized

def _rasterize_height_slice_segments(occupied: np.ndarray, *, vertices: np.ndarray, faces: np.ndarray, map_size: int, resolution: float, z: float, line_width: int) -> int:
    image = Image.fromarray(np.where(occupied, 255, 0).astype(np.uint8), mode='L')
    draw = ImageDraw.Draw(image)
    rasterized = 0
    for face in faces:
        tri = vertices[face]
        segment = _triangle_height_slice_segment_xy(tri, z=z)
        if segment is None:
            continue
        p0 = _world_xy_to_map_pixel(float(segment[0][0]), float(segment[0][1]), map_size, resolution)
        p1 = _world_xy_to_map_pixel(float(segment[1][0]), float(segment[1][1]), map_size, resolution)
        if not _points_touch_image([p0, p1], map_size):
            continue
        p0 = (min(max(p0[0], 0), map_size - 1), min(max(p0[1], 0), map_size - 1))
        p1 = (min(max(p1[0], 0), map_size - 1), min(max(p1[1], 0), map_size - 1))
        draw.line([p0, p1], fill=255, width=line_width)
        rasterized += 1
    occupied[:] = np.asarray(image) > 0
    return rasterized

def _triangle_height_slice_segment_xy(tri: np.ndarray, *, z: float, eps: float=1e-06) -> tuple[np.ndarray, np.ndarray] | None:
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = float(np.linalg.norm(normal))
    if norm <= eps:
        return None
    normal = normal / norm
    if abs(float(normal[2])) > 0.85:
        return None
    z_values = tri[:, 2]
    if np.max(z_values) < z - eps or np.min(z_values) > z + eps:
        return None
    if np.all(np.abs(z_values - z) <= eps):
        return None
    points: list[np.ndarray] = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        p0 = tri[i]
        p1 = tri[j]
        d0 = float(p0[2] - z)
        d1 = float(p1[2] - z)
        if abs(d0) <= eps and abs(d1) <= eps:
            continue
        if abs(d0) <= eps:
            points.append(p0[:2].copy())
            continue
        if abs(d1) <= eps:
            points.append(p1[:2].copy())
            continue
        if d0 * d1 < 0.0:
            t = (z - float(p0[2])) / float(p1[2] - p0[2])
            points.append((p0[:2] + t * (p1[:2] - p0[:2])).copy())
    unique = _unique_xy_points(points, eps=eps)
    if len(unique) < 2:
        return None
    if len(unique) == 2:
        return (unique[0], unique[1])
    best = (unique[0], unique[1])
    best_dist = -1.0
    for i, p0 in enumerate(unique):
        for p1 in unique[i + 1:]:
            dist = float(np.linalg.norm(p1 - p0))
            if dist > best_dist:
                best = (p0, p1)
                best_dist = dist
    return best

def _unique_xy_points(points: Sequence[np.ndarray], *, eps: float) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if any((float(np.linalg.norm(point - existing)) <= eps for existing in unique)):
            continue
        unique.append(point)
    return unique

def _world_xy_to_map_pixel(x: float, y: float, map_size: int, resolution: float) -> tuple[int, int]:
    col = int(math.floor(x / resolution + map_size / 2.0))
    row = int(math.floor(y / resolution + map_size / 2.0))
    return (col, row)

def _points_touch_image(points: Sequence[tuple[int, int]], map_size: int) -> bool:
    cols = [point[0] for point in points]
    rows = [point[1] for point in points]
    return not (max(cols) < 0 or min(cols) >= map_size or max(rows) < 0 or (min(rows) >= map_size))

def _erode_bool_free_square(free: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return free.copy()
    padded = np.pad(free, radius_cells, mode='constant', constant_values=False)
    out = np.zeros_like(free, dtype=bool)
    size = 2 * radius_cells + 1
    for row in range(free.shape[0]):
        for col in range(free.shape[1]):
            window = padded[row:row + size, col:col + size]
            out[row, col] = bool(np.all(window))
    return out

def _write_door_overlay_image(free: np.ndarray, path: Path, *, map_size: int, resolution: float, navigation_objects: Sequence[AABBObject]) -> None:
    scale = max(1, int(math.ceil(512 / max(free.shape))))
    image = Image.fromarray(np.where(free, 255, 0).astype(np.uint8)).resize((free.shape[1] * scale, free.shape[0] * scale), Image.Resampling.NEAREST)
    rgb = image.convert('RGB')
    draw = ImageDraw.Draw(rgb)
    for obj in navigation_objects:
        x0 = float(obj.center[0]) - float(obj.extent[0]) * 0.5
        x1 = float(obj.center[0]) + float(obj.extent[0]) * 0.5
        y0 = float(obj.center[1]) - float(obj.extent[1]) * 0.5
        y1 = float(obj.center[1]) + float(obj.extent[1]) * 0.5
        c0, r0 = _world_xy_to_map_pixel(x0, y0, map_size, resolution)
        c1, r1 = _world_xy_to_map_pixel(x1, y1, map_size, resolution)
        c0, c1 = sorted((c0, c1))
        r0, r1 = sorted((r0, r1))
        color = (0, 200, 255) if (getattr(obj, 'states', {}) or {}).get('open') is True else (255, 0, 0)
        draw.rectangle((c0 * scale, r0 * scale, (c1 + 1) * scale, (r1 + 1) * scale), outline=color, width=max(2, scale))
        cc, rr = _world_xy_to_map_pixel(float(obj.center[0]), float(obj.center[1]), map_size, resolution)
        draw.ellipse((cc * scale - 3, rr * scale - 3, cc * scale + 3, rr * scale + 3), fill=(0, 0, 255))
    rgb.save(path)

def make_occupancy_config(aabb_objects: Sequence[AABBObject], *, resolution: float=0.1, agent_radius: float=0.25, safety_margin: float=0.05, ignored_categories: set[str] | Sequence[str]=('floors', 'ceilings'), ignored_object_names: set[str] | Sequence[str]=(), traversability: SceneTraversability | None=None, bounds: tuple[float, float, float, float] | None=None, padding: float=0.75) -> OccupancyConfig:
    return OccupancyConfig(resolution=resolution, agent_radius=agent_radius, safety_margin=safety_margin, bounds=bounds if bounds is not None else bounds_from_aabbs(aabb_objects, padding=padding), ignored_categories=set(ignored_categories), ignored_object_names=set(ignored_object_names), **{} if traversability is None else traversability.occupancy_kwargs())

def start_edge_clearance_m(occupancy: OccupancyMap, row: int, col: int) -> float:
    clearance_cells = min(row, col, occupancy.height - 1 - row, occupancy.width - 1 - col)
    return max(0.0, float(clearance_cells) * occupancy.resolution)

def sample_starts_from_component(occupancy: OccupancyMap, component: int, target_xy: tuple[float, float], height: float, config: StartSelectionConfig) -> list[EgoPose]:
    rows, cols = np.where(occupancy.component_labels == component)
    if len(rows) == 0:
        return []
    candidates: list[tuple[tuple[float, float, float, float, float], tuple[float, float]]] = []
    fallback_candidates: list[tuple[tuple[float, float, float, float, float], tuple[float, float]]] = []
    stride = max(1, len(rows) // 1200)
    preferred_distance = float(config.preferred_start_distance or config.min_start_distance)
    max_distance = float(config.max_start_distance)
    for row, col in zip(rows[::stride], cols[::stride]):
        row = int(row)
        col = int(col)
        xy = occupancy.cell_to_world((row, col))
        dist = math.dist(xy, target_xy)
        if dist < config.min_start_distance:
            continue
        if max_distance and dist > max_distance:
            continue
        edge_clearance = start_edge_clearance_m(occupancy, row, col)
        score = (abs(dist - preferred_distance), -edge_clearance, dist, xy[0], xy[1])
        fallback_candidates.append((score, xy))
        if edge_clearance >= config.start_edge_margin:
            candidates.append((score, xy))
    if not candidates:
        candidates = fallback_candidates
    candidates.sort(key=lambda item: item[0])
    starts = []
    for _, xy in candidates[:config.start_candidates_per_goal]:
        starts.append(EgoPose(xy[0], xy[1], z=height, yaw=yaw_towards(xy, target_xy), pitch=0.0))
    return starts

def start_pose_debug(occupancy: OccupancyMap, pose: EgoPose, goal_xy: tuple[float, float]) -> dict[str, Any]:
    row, col = occupancy.world_to_cell(pose.xy)
    return {'start_pose': pose.as_dict(), 'start_cell': [int(row), int(col)], 'start_component': int(occupancy.component_at(pose.xy)), 'start_edge_clearance_m': start_edge_clearance_m(occupancy, int(row), int(col)), 'start_distance_to_goal': float(math.dist(pose.xy, goal_xy)), 'goal_xy': [float(goal_xy[0]), float(goal_xy[1])], 'goal_cell': list(occupancy.world_to_cell(goal_xy)), 'goal_component': int(occupancy.component_at(goal_xy))}

def draw_occupancy_path_overlay(occupancy: OccupancyMap, pose_records: Sequence[dict[str, Any]], out_path: str | Path, *, role_markers: dict[str, AABBObject] | None=None, scale: int=6, show_legend: bool=True) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = np.where(occupancy.occupied, 30, 235).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)
    rgb[occupancy.free & ~occupancy.eroded_free] = np.array([170, 170, 170], dtype=np.uint8)
    image = Image.fromarray(np.flipud(rgb), mode='RGB').resize((occupancy.width * scale, occupancy.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    def world_to_px(xy):
        row, col = occupancy.world_to_cell(xy)
        return (col * scale + scale // 2, (occupancy.height - 1 - row) * scale + scale // 2)
    path_points = [world_to_px((float(record['x']), float(record['y']))) for record in pose_records]
    if len(path_points) > 1:
        draw.line(path_points, fill=(220, 40, 40), width=4)
    if path_points:
        stride = max(1, len(path_points) // 24)
        for point in path_points[::stride]:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=(255, 120, 120))
        start = path_points[0]
        end = path_points[-1]
        draw.ellipse((start[0] - 9, start[1] - 9, start[0] + 9, start[1] + 9), fill=(0, 190, 0), outline=(0, 80, 0))
        draw.text((start[0] + 11, start[1] - 11), 'START', fill=(0, 110, 0))
        draw.ellipse((end[0] - 9, end[1] - 9, end[0] + 9, end[1] + 9), fill=(30, 110, 255), outline=(0, 40, 120))
        draw.text((end[0] + 11, end[1] - 11), 'END', fill=(20, 70, 200))
    seen: set[str] = set()
    for label, obj in (role_markers or {}).items():
        if obj.name in seen:
            continue
        seen.add(obj.name)
        px, py = world_to_px(obj.xy_center)
        draw.rectangle((px - 7, py - 7, px + 7, py + 7), fill=(255, 220, 0), outline=(120, 100, 0))
        draw.text((px + 10, py - 10), label, fill=(100, 80, 0))
    if not show_legend:
        image.save(out_path)
        return out_path
    legend_x, legend_y = (8, 8)
    draw.rectangle((legend_x, legend_y, legend_x + 230, legend_y + 92), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.rectangle((legend_x + 8, legend_y + 10, legend_x + 22, legend_y + 24), fill=(30, 30, 30))
    draw.text((legend_x + 30, legend_y + 8), 'occupied', fill=(0, 0, 0))
    draw.rectangle((legend_x + 8, legend_y + 32, legend_x + 22, legend_y + 46), fill=(170, 170, 170))
    draw.text((legend_x + 30, legend_y + 30), 'eroded margin', fill=(0, 0, 0))
    draw.line((legend_x + 8, legend_y + 62, legend_x + 34, legend_y + 62), fill=(220, 40, 40), width=4)
    draw.text((legend_x + 42, legend_y + 54), 'rendered ego path', fill=(0, 0, 0))
    draw.rectangle((legend_x + 8, legend_y + 74, legend_x + 22, legend_y + 88), fill=(255, 220, 0))
    draw.text((legend_x + 42, legend_y + 72), 'role targets', fill=(0, 0, 0))
    image.save(out_path)
    return out_path
