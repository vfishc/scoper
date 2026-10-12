from __future__ import annotations
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None
from scoper.benchmark.behavior.scene_geometry.asset_catalog import AssetCatalog, scaled_bbox
from scoper.benchmark.behavior.scene_geometry.category_classifier import CategoryClassifier
from scoper.benchmark.behavior.scene_geometry.scene_io import SceneObjectRecord, parse_scene_objects, quat_to_yaw
_ROOM_GEOMETRY_CACHE: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[slice, slice]]] = {}
_TRAVERSABLE_MASK_CACHE: dict[int, np.ndarray] = {}
_TRAVERSABLE_CROP_CACHE: dict[tuple[int, int, int, int, int, int, int], np.ndarray] = {}

class SpatialIndex:

    def __init__(self, cell_size: float=2.0):
        self.cell_size = float(cell_size)
        self._grid: dict[tuple[int, int], list[Footprint]] = {}
        self._footprints: list[Footprint] = []

    def add(self, footprint: Footprint) -> None:
        self._footprints.append(footprint)
        x1, y1, x2, y2 = footprint.aabb()
        c1_x = int(x1 / self.cell_size)
        c1_y = int(y1 / self.cell_size)
        c2_x = int(x2 / self.cell_size)
        c2_y = int(y2 / self.cell_size)
        for cx in range(c1_x, c2_x + 1):
            for cy in range(c1_y, c2_y + 1):
                key = (cx, cy)
                if key not in self._grid:
                    self._grid[key] = []
                self._grid[key].append(footprint)

    def query_aabb(self, x1: float, y1: float, x2: float, y2: float, *, room_id: str | None=None) -> list[Footprint]:
        c1_x = int(x1 / self.cell_size)
        c1_y = int(y1 / self.cell_size)
        c2_x = int(x2 / self.cell_size)
        c2_y = int(y2 / self.cell_size)
        candidates = set()
        for cx in range(c1_x, c2_x + 1):
            for cy in range(c1_y, c2_y + 1):
                key = (cx, cy)
                if key in self._grid:
                    for fp in self._grid[key]:
                        if room_id is not None and fp.room is not None and (fp.room != room_id):
                            continue
                        candidates.add(fp)
        return list(candidates)

    def query_footprint(self, footprint: Footprint, *, clearance: float=0.0) -> list[Footprint]:
        x1, y1, x2, y2 = footprint.aabb(clearance=clearance)
        candidates = self.query_aabb(x1, y1, x2, y2, room_id=footprint.room)
        return [fp for fp in candidates if fp is not footprint]

    def all_footprints(self) -> list[Footprint]:
        return list(self._footprints)

    def __len__(self) -> int:
        return len(self._footprints)

def build_spatial_index(footprints: list[Footprint], *, cell_size: float=2.0) -> SpatialIndex:
    index = SpatialIndex(cell_size=cell_size)
    for fp in footprints:
        index.add(fp)
    return index

@dataclass(frozen=True)
class Footprint:
    name: str
    category: str
    room: str | None
    center_xy: tuple[float, float]
    extent_xy: tuple[float, float]
    yaw: float
    source: str = 'existing'
    owner: str | None = None

    @property
    def area(self) -> float:
        return float(self.extent_xy[0]) * float(self.extent_xy[1])

    def aabb(self, *, clearance: float=0.0) -> tuple[float, float, float, float]:
        ex = float(self.extent_xy[0]) / 2.0 + float(clearance)
        ey = float(self.extent_xy[1]) / 2.0 + float(clearance)
        return (float(self.center_xy[0]) - ex, float(self.center_xy[1]) - ey, float(self.center_xy[0]) + ex, float(self.center_xy[1]) + ey)

@dataclass(frozen=True)
class FreeSpaceMetrics:
    free_ratio: float
    connected_ratio: float
    occupied_area_ratio: float
    free_cells: int
    largest_component_cells: int
    traversable_cells: int

@dataclass(frozen=True)
class RoomMaskCrop:
    free: np.ndarray
    room: np.ndarray
    row_offset: int
    col_offset: int
    room_slice: tuple[slice, slice]
PROTECTED_ZONE_SOURCES = {'door_clearance', 'openable_clearance'}
FLOOR_LAYER_CATEGORIES = {'carpet', 'rug', 'mat'}
DOOR_CLEARANCE_CATEGORIES = {'door', 'elevator_door', 'garage_door', 'openable_window', 'sliding_door'}
OPENABLE_CLEARANCE_CATEGORIES = {'bottom_cabinet', 'bottom_cabinet_no_top', 'dishwasher', 'fridge', 'nightstand', 'oven', 'washer', 'wardrobe'}
_MASK_ENUMERATION_CELL_LIMIT = 1000000

def existing_footprints(scene: dict, catalog: AssetCatalog, classifier: CategoryClassifier | None=None, *, include_roles: set[str] | None=None) -> list[Footprint]:
    footprints: list[Footprint] = []
    for record in parse_scene_objects(scene).values():
        asset = catalog.get_model(record.category, record.model)
        if asset is None:
            continue
        role = getattr(classifier.get(record.category), 'role', None) if classifier else None
        if include_roles is not None and role not in include_roles:
            continue
        bbox = scaled_bbox(asset, record.scale)
        yaw = quat_to_yaw(record.ori)
        extent = footprint_extent_for_yaw((bbox[0], bbox[1]), yaw)
        footprints.append(Footprint(name=record.name, category=record.category, room=record.primary_room, center_xy=(record.pos[0], record.pos[1]), extent_xy=extent, yaw=yaw, owner=record.name))
    return footprints

def protected_zones_from_scene(scene: dict, catalog: AssetCatalog, classifier: CategoryClassifier | None=None, *, policy: dict | None=None) -> list[Footprint]:
    zones: list[Footprint] = []
    for record in parse_scene_objects(scene).values():
        if not record.primary_room:
            continue
        asset = catalog.get_model(record.category, record.model)
        if asset is None:
            continue
        zones.extend(protected_zones_for_object(name=record.name, category=record.category, room=record.primary_room, asset=asset, center_xy=(record.pos[0], record.pos[1]), yaw=quat_to_yaw(record.ori), scale=record.scale, classifier=classifier, policy=policy))
    return zones

def protected_zones_for_object(*, name: str, category: str, room: str | None, asset: object, center_xy: tuple[float, float], yaw: float, scale: tuple[float, float, float] | list[float] | None=None, classifier: CategoryClassifier | None=None, policy: dict | None=None) -> list[Footprint]:
    if not room:
        return []
    policy = policy or {}
    category = str(category)
    bbox = scaled_bbox(asset, scale)
    out: list[Footprint] = []
    if is_door_clearance_category(category):
        out.extend(_bidirectional_clearance_zones(name=name, category=category, room=room, center_xy=center_xy, yaw=yaw, bbox_xy=(bbox[0], bbox[1]), source='door_clearance', depth=float(policy.get('door_clearance_depth_m', 0.85)), side_margin=float(policy.get('door_clearance_side_margin_m', 0.18)), normal_axis='thin'))
    elif is_openable_clearance_object(category, asset, classifier):
        out.extend(_bidirectional_clearance_zones(name=name, category=category, room=room, center_xy=center_xy, yaw=yaw, bbox_xy=(bbox[0], bbox[1]), source='openable_clearance', depth=float(policy.get('openable_clearance_depth_m', 0.55)), side_margin=float(policy.get('openable_clearance_side_margin_m', 0.12)), normal_axis='y'))
    return out

def is_door_clearance_category(category: str) -> bool:
    text = str(category)
    return text in DOOR_CLEARANCE_CATEGORIES or 'door' in text

def is_openable_clearance_object(category: str, asset: object, classifier: CategoryClassifier | None=None) -> bool:
    text = str(category)
    if is_door_clearance_category(text):
        return False
    if text in OPENABLE_CLEARANCE_CATEGORIES:
        return True
    metadata = getattr(asset, 'metadata', {}) or {}
    openable_joint_ids = metadata.get('openable_joint_ids')
    if isinstance(openable_joint_ids, list) and openable_joint_ids:
        if classifier is None:
            return True
        role = getattr(classifier.get(text), 'role', '')
        return role in {'fixture', 'large_floor', 'support_surface'}
    return False

def footprint_extent_for_yaw(extent_xy: tuple[float, float], yaw: float) -> tuple[float, float]:
    c = abs(math.cos(float(yaw)))
    s = abs(math.sin(float(yaw)))
    x, y = (float(extent_xy[0]), float(extent_xy[1]))
    return (x * c + y * s, x * s + y * c)

def aabb_overlap_ratio(a: Footprint, b: Footprint, *, clearance: float=0.0) -> float:
    ax1, ay1, ax2, ay2 = a.aabb(clearance=clearance)
    bx1, by1, bx2, by2 = b.aabb(clearance=clearance)
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    denom = max(min(a.area, b.area), 1e-08)
    return inter / denom

def room_free_mask(layout: object, room_id: str, occupied: Iterable[Footprint], *, clearance_m: float=0.0) -> np.ndarray | None:
    crop = room_free_mask_crop(layout, room_id, occupied, clearance_m=clearance_m)
    if crop is None:
        return None
    free = np.zeros_like(np.asarray(layout.room_ins_map), dtype=bool)
    free[crop.room_slice] = crop.free
    return free

def room_free_mask_crop(layout: object, room_id: str, occupied: Iterable[Footprint], *, clearance_m: float=0.0) -> RoomMaskCrop | None:
    room, _, _, room_slice = _cached_room_geometry(layout, room_id)
    if room.size == 0:
        return None
    free = np.asarray(room, dtype=bool).copy()
    for fp in occupied:
        if fp.room is not None and fp.room != room_id:
            continue
        _burn_footprint_crop(free, layout, room_slice, fp, clearance_m=clearance_m)
    return RoomMaskCrop(free=free, room=room, row_offset=int(room_slice[0].start or 0), col_offset=int(room_slice[1].start or 0), room_slice=room_slice)

def room_free_space_metrics(layout: object, room_id: str, occupied: Iterable[Footprint], *, clearance_m: float=0.0, density_excluded_sources: set[str] | None=None, density_excluded_categories: set[str] | None=None, compute_connected: bool=True) -> FreeSpaceMetrics:
    crop = room_free_mask_crop(layout, room_id, occupied, clearance_m=clearance_m)
    if crop is None:
        return FreeSpaceMetrics(0.0, 0.0, 0.0, 0, 0, 0)
    room_mask = crop.room
    traversable = room_mask & traversable_mask_crop(layout, crop.room_slice, room_mask.shape)
    traversable_cells = int(traversable.sum())
    free_traversable = crop.free & traversable
    free_cells = int(free_traversable.sum())
    if compute_connected:
        largest = _largest_component_size(free_traversable)
    else:
        largest = free_cells
    room_area = max(float(room_mask.sum()) * float(layout.map_resolution) ** 2, 1e-08)
    occupied_area = 0.0
    excluded_sources = density_excluded_sources if density_excluded_sources is not None else PROTECTED_ZONE_SOURCES
    excluded_categories = density_excluded_categories if density_excluded_categories is not None else FLOOR_LAYER_CATEGORIES
    for fp in occupied:
        if fp.room != room_id:
            continue
        if fp.source in excluded_sources or fp.category in excluded_categories:
            continue
        occupied_area += fp.area
    return FreeSpaceMetrics(free_ratio=free_cells / traversable_cells if traversable_cells else 0.0, connected_ratio=largest / traversable_cells if traversable_cells else 0.0, occupied_area_ratio=occupied_area / room_area, free_cells=free_cells, largest_component_cells=int(largest), traversable_cells=traversable_cells)

def candidate_fits(layout: object, room_id: str, free_mask: np.ndarray | RoomMaskCrop, center_xy: tuple[float, float], extent_xy: tuple[float, float], *, clearance_m: float=0.0) -> bool:
    rows, cols = footprint_rows_cols(layout, center_xy, extent_xy, clearance_m=clearance_m)
    if rows.size == 0 or cols.size == 0:
        return False
    if isinstance(free_mask, RoomMaskCrop):
        local_rows = rows - int(free_mask.row_offset)
        local_cols = cols - int(free_mask.col_offset)
        if local_rows.min() < 0 or local_cols.min() < 0 or local_rows.max() >= free_mask.free.shape[0] or (local_cols.max() >= free_mask.free.shape[1]):
            return False
        patch_free = free_mask.free[np.ix_(local_rows, local_cols)]
        patch_room = free_mask.room[np.ix_(local_rows, local_cols)]
        return bool(patch_free.all() and patch_room.all())
    room = layout.room_mask(room_id)
    if room is None:
        return False
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= free_mask.shape[0] or (cols.max() >= free_mask.shape[1]):
        return False
    patch_free = free_mask[np.ix_(rows, cols)]
    patch_room = np.asarray(room, dtype=bool)[np.ix_(rows, cols)]
    return bool(patch_free.all() and patch_room.all())

def footprint_rows_cols(layout: object, center_xy: tuple[float, float], extent_xy: tuple[float, float], *, clearance_m: float=0.0) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = (float(center_xy[0]), float(center_xy[1]))
    ex = float(extent_xy[0]) / 2.0 + float(clearance_m)
    ey = float(extent_xy[1]) / 2.0 + float(clearance_m)
    r1, c1 = layout.world_to_map((cx - ex, cy - ey))
    r2, c2 = layout.world_to_map((cx + ex, cy + ey))
    row_min, row_max = sorted((int(r1), int(r2)))
    col_min, col_max = sorted((int(c1), int(c2)))
    return (np.arange(row_min, row_max + 1), np.arange(col_min, col_max + 1))

def sample_anchor_pose(*, layout: object, room_id: str, extent_xy: tuple[float, float], anchor: str, occupied: list[Footprint], rng: random.Random, clearance_m: float, edge_band_m: float, corner_band_m: float, max_attempts: int, avoid_center_xy: tuple[float, float] | None=None, min_center_distance_m: float=0.0, candidate_filter: Callable[[Footprint], bool] | None=None) -> tuple[tuple[float, float], float, Footprint] | None:
    free = room_free_mask_crop(layout, room_id, occupied, clearance_m=clearance_m)
    if free is None or not np.any(free.free):
        return None
    _, boundary_distance, (center_row, center_col), _ = _cached_room_geometry(layout, room_id)
    row_offset = free.row_offset
    col_offset = free.col_offset
    center_row_local = float(center_row) - float(row_offset)
    center_col_local = float(center_col) - float(col_offset)
    center_mask = free.free & traversable_mask_crop(layout, free.room_slice, free.free.shape)
    if not np.any(center_mask):
        center_mask = free.free
    resolution = float(layout.map_resolution)
    edge_band_px = max(1, int(round(float(edge_band_m) / resolution)))
    corner_band_px = max(1, int(round(float(corner_band_m) / resolution)))
    candidates = _sample_anchor_cells(center_mask, anchor, rng, max_attempts=max_attempts, boundary_distance=boundary_distance, edge_band_px=edge_band_px, corner_band_px=corner_band_px, center_row=center_row_local, center_col=center_col_local)
    if not candidates:
        return None
    yaws = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
    for row, col in candidates[:max_attempts]:
        local_rc = (int(row), int(col))
        if not _candidate_matches_anchor(anchor, local_rc, boundary_distance, edge_band_px, corner_band_px, center_row_local, center_col_local):
            continue
        center_xy = layout.map_to_world((int(row) + row_offset, int(col) + col_offset))
        if avoid_center_xy is not None and min_center_distance_m > 0:
            if math.hypot(float(center_xy[0]) - float(avoid_center_xy[0]), float(center_xy[1]) - float(avoid_center_xy[1])) < float(min_center_distance_m):
                continue
        rng.shuffle(yaws)
        for yaw in yaws:
            candidate_extent = footprint_extent_for_yaw(extent_xy, yaw)
            if not candidate_fits(layout, room_id, free, center_xy, candidate_extent, clearance_m=clearance_m):
                continue
            fp = Footprint(name='candidate', category='candidate', room=room_id, center_xy=(float(center_xy[0]), float(center_xy[1])), extent_xy=candidate_extent, yaw=yaw, source='candidate')
            if candidate_filter is not None and (not candidate_filter(fp)):
                continue
            return (fp.center_xy, yaw, fp)
    return None

def _sample_anchor_cells(mask: np.ndarray, anchor: str, rng: random.Random, *, max_attempts: int, boundary_distance: np.ndarray, edge_band_px: int, corner_band_px: int, center_row: float, center_col: float) -> list[tuple[int, int]]:
    target = max(int(max_attempts) * 6, 256)
    if mask.size <= _MASK_ENUMERATION_CELL_LIMIT:
        rows, cols = np.where(mask)
        if rows.size == 0:
            return []
        sample_count = min(int(rows.size), target)
        if sample_count < int(rows.size):
            indices = rng.sample(range(int(rows.size)), sample_count)
            candidates = [(int(rows[i]), int(cols[i])) for i in indices]
        else:
            candidates = list(zip(rows.tolist(), cols.tolist()))
            rng.shuffle(candidates)
        return _rank_anchor_cells(candidates, anchor, boundary_distance, edge_band_px, corner_band_px, center_row, center_col)
    height, width = (int(mask.shape[0]), int(mask.shape[1]))
    candidates: list[tuple[int, int]] = []
    seen: set[int] = set()

    def add(row: int, col: int) -> None:
        if row < 0 or col < 0 or row >= height or (col >= width):
            return
        if not bool(mask[row, col]):
            return
        if not _candidate_matches_anchor(anchor, (row, col), boundary_distance, edge_band_px, corner_band_px, center_row, center_col):
            return
        key = row * width + col
        if key in seen:
            return
        seen.add(key)
        candidates.append((row, col))
    biased_attempts = max(target * 80, int(max_attempts) * 300, 4096)
    for _ in range(biased_attempts):
        row, col = _draw_anchor_biased_cell(anchor, rng, height, width, edge_band_px, corner_band_px)
        add(row, col)
        if len(candidates) >= target:
            break
    if len(candidates) < int(max_attempts):
        random_attempts = max(target * 160, int(max_attempts) * 600, 8192)
        for _ in range(random_attempts):
            add(rng.randrange(height), rng.randrange(width))
            if len(candidates) >= target:
                break
    if len(candidates) < int(max_attempts):
        row_attempts = max(int(max_attempts) * 20, 512)
        for _ in range(row_attempts):
            row = rng.randrange(height)
            valid_cols = np.flatnonzero(mask[row])
            if valid_cols.size == 0:
                continue
            col = int(valid_cols[rng.randrange(int(valid_cols.size))])
            add(row, col)
            if len(candidates) >= target:
                break
    return _rank_anchor_cells(candidates, anchor, boundary_distance, edge_band_px, corner_band_px, center_row, center_col)

def _draw_anchor_biased_cell(anchor: str, rng: random.Random, height: int, width: int, edge_band_px: int, corner_band_px: int) -> tuple[int, int]:
    if anchor == 'in_corner':
        row_band = max(1, min(height, corner_band_px * 3))
        col_band = max(1, min(width, corner_band_px * 3))
        row = rng.randrange(row_band) if rng.random() < 0.5 else height - 1 - rng.randrange(row_band)
        col = rng.randrange(col_band) if rng.random() < 0.5 else width - 1 - rng.randrange(col_band)
        return (int(row), int(col))
    if anchor != 'in_middle':
        band = max(1, min(max(height, width), edge_band_px * 3))
        side = rng.randrange(4)
        if side == 0:
            return (int(rng.randrange(min(height, band))), int(rng.randrange(width)))
        if side == 1:
            return (int(height - 1 - rng.randrange(min(height, band))), int(rng.randrange(width)))
        if side == 2:
            return (int(rng.randrange(height)), int(rng.randrange(min(width, band))))
        return (int(rng.randrange(height)), int(width - 1 - rng.randrange(min(width, band))))
    return (int(rng.randrange(height)), int(rng.randrange(width)))

def _rank_anchor_cells(candidates: list[tuple[int, int]], anchor: str, boundary_distance: np.ndarray, edge_band_px: int, corner_band_px: int, center_row: float, center_col: float) -> list[tuple[int, int]]:
    return sorted(candidates, key=lambda rc: _anchor_score(anchor, (int(rc[0]), int(rc[1])), boundary_distance, center_row, center_col, edge_band_px, corner_band_px), reverse=True)

def _crop_mask_like(mask: np.ndarray, crop: tuple[slice, slice], target_shape: tuple[int, int]) -> np.ndarray:
    target_rows, target_cols = (int(target_shape[0]), int(target_shape[1]))
    out = np.zeros((target_rows, target_cols), dtype=bool)
    row_start = max(0, int(crop[0].start or 0))
    col_start = max(0, int(crop[1].start or 0))
    row_stop = min(mask.shape[0], int(crop[0].stop if crop[0].stop is not None else mask.shape[0]))
    col_stop = min(mask.shape[1], int(crop[1].stop if crop[1].stop is not None else mask.shape[1]))
    if row_stop <= row_start or col_stop <= col_start:
        return out
    cropped = mask[row_start:row_stop, col_start:col_stop]
    rows = min(target_rows, cropped.shape[0])
    cols = min(target_cols, cropped.shape[1])
    out[:rows, :cols] = cropped[:rows, :cols]
    return out

def traversable_mask_crop(layout: object, crop: tuple[slice, slice], target_shape: tuple[int, int]) -> np.ndarray:
    layout_id = id(layout)
    row_start = int(crop[0].start or 0)
    row_stop = int(crop[0].stop if crop[0].stop is not None else row_start)
    col_start = int(crop[1].start or 0)
    col_stop = int(crop[1].stop if crop[1].stop is not None else col_start)
    target_rows = int(target_shape[0])
    target_cols = int(target_shape[1])
    key = (layout_id, row_start, row_stop, col_start, col_stop, target_rows, target_cols)
    cached = _TRAVERSABLE_CROP_CACHE.get(key)
    if cached is not None:
        return cached
    traversable = _TRAVERSABLE_MASK_CACHE.get(layout_id)
    if traversable is None:
        traversable = np.asarray(layout.traversable_mask(), dtype=bool)
        _TRAVERSABLE_MASK_CACHE[layout_id] = traversable
    out = _crop_mask_like(traversable, crop, (target_rows, target_cols))
    _TRAVERSABLE_CROP_CACHE[key] = out
    return out

def clear_geometry_caches() -> None:
    _ROOM_GEOMETRY_CACHE.clear()
    _TRAVERSABLE_MASK_CACHE.clear()
    _TRAVERSABLE_CROP_CACHE.clear()

def _bidirectional_clearance_zones(*, name: str, category: str, room: str, center_xy: tuple[float, float], yaw: float, bbox_xy: tuple[float, float], source: str, depth: float, side_margin: float, normal_axis: str) -> list[Footprint]:
    x, y = (float(bbox_xy[0]), float(bbox_xy[1]))
    if normal_axis == 'thin':
        axis = 'x' if x <= y else 'y'
    else:
        axis = str(normal_axis)
    if axis == 'x':
        local_extent = (max(float(depth), 0.05), max(y + 2.0 * float(side_margin), 0.05))
        offset_distance = x / 2.0 + float(depth) / 2.0
        local_offsets = [(-offset_distance, 0.0), (offset_distance, 0.0)]
    else:
        local_extent = (max(x + 2.0 * float(side_margin), 0.05), max(float(depth), 0.05))
        offset_distance = y / 2.0 + float(depth) / 2.0
        local_offsets = [(0.0, -offset_distance), (0.0, offset_distance)]
    zones = []
    for index, offset in enumerate(local_offsets):
        zone_center = _transform_offset(center_xy, yaw, offset)
        zones.append(Footprint(name=f'{name}__{source}_{index}', category=category, room=room, center_xy=zone_center, extent_xy=footprint_extent_for_yaw(local_extent, yaw), yaw=yaw, source=source, owner=str(name)))
    return zones

def _transform_offset(center_xy: tuple[float, float], yaw: float, offset_xy: tuple[float, float]) -> tuple[float, float]:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    ox, oy = (float(offset_xy[0]), float(offset_xy[1]))
    return (float(center_xy[0]) + ox * c - oy * s, float(center_xy[1]) + ox * s + oy * c)

def _largest_component_size(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    if cv2 is not None:
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8), connectivity=4, ltype=cv2.CV_32S)
        if component_count <= 1:
            return 0
        return int(stats[1:, cv2.CC_STAT_AREA].max())
    visited = np.zeros(mask.shape, dtype=bool)
    rows, cols = np.where(mask)
    largest = 0
    for start_row, start_col in zip(rows.tolist(), cols.tolist()):
        if visited[start_row, start_col]:
            continue
        count = 0
        queue: deque[tuple[int, int]] = deque([(int(start_row), int(start_col))])
        visited[start_row, start_col] = True
        while queue:
            row, col = queue.popleft()
            count += 1
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if nr < 0 or nc < 0 or nr >= mask.shape[0] or (nc >= mask.shape[1]):
                    continue
                if visited[nr, nc] or not mask[nr, nc]:
                    continue
                visited[nr, nc] = True
                queue.append((nr, nc))
        largest = max(largest, count)
    return int(largest)

def _cached_room_geometry(layout: object, room_id: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[slice, slice]]:
    key = (id(layout), str(room_id))
    cached = _ROOM_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    room = np.asarray(layout.room_mask(room_id), dtype=bool)
    bounds = _mask_bounds(room)
    if bounds is not None:
        row_min, row_max, col_min, col_max = bounds
        room_slice = (slice(row_min, row_max + 1), slice(col_min, col_max + 1))
    else:
        room_slice = (slice(0, 0), slice(0, 0))
    room_crop = room[room_slice]
    boundary_distance = _distance_to_boundary(room_crop)
    center_local = _mask_centroid(room_crop)
    center = (center_local[0] + float(room_slice[0].start or 0), center_local[1] + float(room_slice[1].start or 0))
    cached = (room_crop, boundary_distance, center, room_slice)
    _ROOM_GEOMETRY_CACHE[key] = cached
    return cached

def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.size == 0:
        return None
    row_has = np.any(mask, axis=1)
    col_has = np.any(mask, axis=0)
    rows = np.flatnonzero(row_has)
    cols = np.flatnonzero(col_has)
    if rows.size == 0 or cols.size == 0:
        return None
    return (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))

def _burn_footprint(mask: np.ndarray, layout: object, footprint: Footprint, *, clearance_m: float) -> None:
    rows, cols = footprint_rows_cols(layout, footprint.center_xy, footprint.extent_xy, clearance_m=clearance_m)
    rows = rows[(rows >= 0) & (rows < mask.shape[0])]
    cols = cols[(cols >= 0) & (cols < mask.shape[1])]
    if rows.size and cols.size:
        mask[np.ix_(rows, cols)] = False

def _burn_footprint_crop(mask: np.ndarray, layout: object, room_slice: tuple[slice, slice], footprint: Footprint, *, clearance_m: float) -> None:
    rows, cols = footprint_rows_cols(layout, footprint.center_xy, footprint.extent_xy, clearance_m=clearance_m)
    row_offset = int(room_slice[0].start or 0)
    col_offset = int(room_slice[1].start or 0)
    rows = rows - row_offset
    cols = cols - col_offset
    rows = rows[(rows >= 0) & (rows < mask.shape[0])]
    cols = cols[(cols >= 0) & (cols < mask.shape[1])]
    if rows.size and cols.size:
        mask[np.ix_(rows, cols)] = False

def _distance_to_boundary(mask: np.ndarray) -> np.ndarray:
    bounds = _mask_bounds(mask)
    dist = np.zeros(mask.shape, dtype=np.float32)
    if bounds is None:
        return dist
    min_r, max_r, min_c, max_c = bounds
    row_ids = np.arange(mask.shape[0], dtype=np.float32)
    col_ids = np.arange(mask.shape[1], dtype=np.float32)
    row_dist = np.minimum(row_ids - float(min_r), float(max_r) - row_ids)
    col_dist = np.minimum(col_ids - float(min_c), float(max_c) - col_ids)
    dist = np.minimum(row_dist[:, None], col_dist[None, :]).astype(np.float32, copy=False)
    dist[~mask] = 0
    return dist

def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    total = int(mask.sum())
    if total == 0:
        return (mask.shape[0] / 2.0, mask.shape[1] / 2.0)
    row_counts = mask.sum(axis=1, dtype=np.float64)
    col_counts = mask.sum(axis=0, dtype=np.float64)
    row_mean = float(np.dot(np.arange(mask.shape[0], dtype=np.float64), row_counts) / float(total))
    col_mean = float(np.dot(np.arange(mask.shape[1], dtype=np.float64), col_counts) / float(total))
    return (row_mean, col_mean)

def _anchor_score(anchor: str, row_col: tuple[int, int], distance: np.ndarray, center_row: float, center_col: float, edge_band_px: int, corner_band_px: int) -> float:
    row, col = row_col
    d = float(distance[row, col])
    center_dist = math.hypot(row - center_row, col - center_col)
    if anchor == 'in_middle':
        return d - 0.01 * center_dist
    if anchor == 'in_corner':
        return max(0.0, corner_band_px - d) + 0.002 * center_dist
    return max(0.0, edge_band_px - d) + 0.001 * center_dist

def _candidate_matches_anchor(anchor: str, row_col: tuple[int, int], distance: np.ndarray, edge_band_px: int, corner_band_px: int, center_row: float, center_col: float) -> bool:
    row, col = row_col
    d = float(distance[row, col])
    if anchor == 'in_middle':
        return d >= max(2.0, edge_band_px * 0.65)
    if anchor == 'in_corner':
        return d <= corner_band_px
    return d <= edge_band_px
__all__ = ['Footprint', 'FreeSpaceMetrics', 'RoomMaskCrop', 'SpatialIndex', 'PROTECTED_ZONE_SOURCES', 'FLOOR_LAYER_CATEGORIES', 'aabb_overlap_ratio', 'build_spatial_index', 'candidate_fits', 'existing_footprints', 'footprint_extent_for_yaw', 'footprint_rows_cols', 'is_door_clearance_category', 'is_openable_clearance_object', 'protected_zones_for_object', 'protected_zones_from_scene', 'room_free_space_metrics', 'room_free_mask', 'room_free_mask_crop', 'sample_anchor_pose', 'clear_geometry_caches', 'traversable_mask_crop']
