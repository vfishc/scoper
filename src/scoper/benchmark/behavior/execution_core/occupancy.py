from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from scoper.benchmark.behavior.execution_core.trace import CheckResult, write_json

@dataclass
class AABBObject:
    name: str
    category: str
    center: tuple[float, float, float]
    extent: tuple[float, float, float]
    visible: bool = True
    abilities: tuple[str, ...] = ()
    states: dict[str, Any] = field(default_factory=dict)
    root_position: tuple[float, float, float] | None = None
    root_orientation_xyzw: tuple[float, float, float, float] | None = None
    root_yaw: float | None = None
    pose_source: str | None = None

    @property
    def xy_center(self) -> tuple[float, float]:
        return (self.center[0], self.center[1])

@dataclass
class OccupancyConfig:
    resolution: float = 0.1
    agent_radius: float = 0.25
    safety_margin: float = 0.05
    floor: int = 0
    floor_height: float = 0.0
    navigation_height_band: tuple[float, float] = (0.05, 2.0)
    bounds: tuple[float, float, float, float] | None = None
    padding: float = 0.5
    ignored_categories: set[str] = field(default_factory=lambda: {'floors', 'ceilings'})
    ignored_object_names: set[str] = field(default_factory=set)
    traversability_map: Any | None = None
    traversability_map_resolution: float | None = None
    traversability_map_size: int | None = None
    traversability_floor: int = 0
    traversability_source: str | None = None
    line_of_sight_traversability_map: Any | None = None
    line_of_sight_traversability_map_resolution: float | None = None
    line_of_sight_traversability_map_size: int | None = None
    line_of_sight_traversability_source: str | None = None
    navigation_opening_margin: float = 0.12
    doorway_opening_objects: tuple[AABBObject, ...] = ()
    doorway_opening_allowed_map: Any | None = None
    doorway_opening_allowed_map_resolution: float | None = None
    doorway_opening_allowed_map_size: int | None = None

class OccupancyMap:

    def __init__(self, source: str, resolution: float, origin_xy: tuple[float, float], occupied: np.ndarray, floor: int=0, metadata: dict | None=None) -> None:
        self.source = source
        self.resolution = float(resolution)
        self.origin_xy = origin_xy
        self.floor = floor
        self.occupied = occupied.astype(bool)
        self.free = ~self.occupied
        self.eroded_free = self.free.copy()
        self.component_labels = np.full(self.occupied.shape, -1, dtype=np.int32)
        self.metadata = metadata or {}
        self.traversability_free: np.ndarray | None = None
        self.line_of_sight_traversability_free: np.ndarray | None = None

    @classmethod
    def from_aabb_objects(cls, objects: Sequence[AABBObject], config: OccupancyConfig) -> 'OccupancyMap':
        bounds = config.bounds or _bounds_from_objects(objects, config.padding)
        min_x, min_y, max_x, max_y = bounds
        width = int(math.ceil((max_x - min_x) / config.resolution))
        height = int(math.ceil((max_y - min_y) / config.resolution))
        if width <= 0 or height <= 0:
            raise ValueError(f'Invalid occupancy bounds: {bounds}')
        occupied = np.zeros((height, width), dtype=bool)
        rasterized: list[dict] = []
        for obj in objects:
            if not _should_rasterize(obj, config):
                continue
            x0 = obj.center[0] - obj.extent[0] * 0.5
            x1 = obj.center[0] + obj.extent[0] * 0.5
            y0 = obj.center[1] - obj.extent[1] * 0.5
            y1 = obj.center[1] + obj.extent[1] * 0.5
            c0 = math.floor((x0 - min_x) / config.resolution)
            c1 = math.floor((x1 - min_x) / config.resolution)
            r0 = math.floor((y0 - min_y) / config.resolution)
            r1 = math.floor((y1 - min_y) / config.resolution)
            c0, c1 = (max(0, c0), min(width - 1, c1))
            r0, r1 = (max(0, r0), min(height - 1, r1))
            if c0 <= c1 and r0 <= r1:
                occupied[r0:r1 + 1, c0:c1 + 1] = True
                rasterized.append({'name': obj.name, 'category': obj.category, 'cell_bounds': [r0, c0, r1, c1]})
        traversability_mask = _traversability_mask_for_grid(height=height, width=width, origin_xy=(min_x, min_y), resolution=config.resolution, config=config)
        line_of_sight_traversability_mask = _line_of_sight_traversability_mask_for_grid(height=height, width=width, origin_xy=(min_x, min_y), resolution=config.resolution, config=config)
        traversability_metadata = None
        if traversability_mask is not None:
            opened_navigation_barriers = _carve_open_navigation_barriers(traversability_mask, objects, origin_xy=(min_x, min_y), resolution=config.resolution, config=config)
            doorway_openings_carved = _carve_doorway_openings(traversability_mask, config.doorway_opening_objects, origin_xy=(min_x, min_y), resolution=config.resolution, config=config)
            outside_traversable = ~traversability_mask
            occupied |= outside_traversable
            traversability_metadata = {'source': config.traversability_source or 'traversability_map', 'floor': config.traversability_floor, 'map_resolution': config.traversability_map_resolution, 'map_size': config.traversability_map_size, 'free_cells': int(np.count_nonzero(traversability_mask)), 'blocked_cells': int(traversability_mask.size - np.count_nonzero(traversability_mask)), 'opened_navigation_barriers': opened_navigation_barriers, 'doorway_openings_carved': doorway_openings_carved}
        line_of_sight_traversability_metadata = None
        if line_of_sight_traversability_mask is not None:
            line_of_sight_traversability_metadata = {'source': config.line_of_sight_traversability_source or 'line_of_sight_traversability_map', 'map_resolution': config.line_of_sight_traversability_map_resolution, 'map_size': config.line_of_sight_traversability_map_size, 'free_cells': int(np.count_nonzero(line_of_sight_traversability_mask)), 'blocked_cells': int(line_of_sight_traversability_mask.size - np.count_nonzero(line_of_sight_traversability_mask))}
        metadata = {'bounds': bounds, 'resolution': config.resolution, 'grid_shape': [height, width], 'origin_xy': [min_x, min_y], 'agent_radius': config.agent_radius, 'safety_margin': config.safety_margin, 'ignored_categories': sorted(config.ignored_categories), 'ignored_object_names': sorted(config.ignored_object_names), 'rasterized_objects': rasterized}
        if traversability_metadata is not None:
            metadata['traversability'] = traversability_metadata
        if line_of_sight_traversability_metadata is not None:
            metadata['line_of_sight_traversability'] = line_of_sight_traversability_metadata
        occ = cls(source='generated_aabb', resolution=config.resolution, origin_xy=(min_x, min_y), occupied=occupied, floor=config.floor, metadata=metadata)
        if traversability_mask is not None:
            occ.traversability_free = traversability_mask.astype(bool, copy=True)
        if line_of_sight_traversability_mask is not None:
            occ.line_of_sight_traversability_free = line_of_sight_traversability_mask.astype(bool, copy=True)
        occ.erode(config.agent_radius + config.safety_margin)
        occ.compute_connected_components()
        return occ

    @property
    def height(self) -> int:
        return self.occupied.shape[0]

    @property
    def width(self) -> int:
        return self.occupied.shape[1]

    def world_to_cell(self, xy: Sequence[float]) -> tuple[int, int]:
        col = math.floor((float(xy[0]) - self.origin_xy[0]) / self.resolution)
        row = math.floor((float(xy[1]) - self.origin_xy[1]) / self.resolution)
        return (row, col)

    def cell_to_world(self, cell: Sequence[int]) -> tuple[float, float]:
        row, col = (int(cell[0]), int(cell[1]))
        return (self.origin_xy[0] + (col + 0.5) * self.resolution, self.origin_xy[1] + (row + 0.5) * self.resolution)

    def in_bounds_cell(self, cell: Sequence[int]) -> bool:
        row, col = (int(cell[0]), int(cell[1]))
        return 0 <= row < self.height and 0 <= col < self.width

    def is_cell_free(self, cell: Sequence[int], eroded: bool=True) -> bool:
        if not self.in_bounds_cell(cell):
            return False
        row, col = (int(cell[0]), int(cell[1]))
        grid = self.eroded_free if eroded else self.free
        return bool(grid[row, col])

    def is_free(self, xy: Sequence[float], eroded: bool=True) -> bool:
        return self.is_cell_free(self.world_to_cell(xy), eroded=eroded)

    def erode(self, radius_m: float) -> None:
        radius_cells = int(math.ceil(radius_m / self.resolution))
        if radius_cells <= 0:
            self.eroded_free = self.free.copy()
            return
        self.eroded_free = _erode_free_square(self.free, radius_cells)
        self.metadata['erosion_radius_m'] = radius_m
        self.metadata['erosion_radius_cells'] = radius_cells

    def compute_connected_components(self) -> None:
        labels = np.full(self.eroded_free.shape, -1, dtype=np.int32)
        label = 0
        for row in range(self.height):
            for col in range(self.width):
                if not self.eroded_free[row, col] or labels[row, col] != -1:
                    continue
                _flood_fill(self.eroded_free, labels, row, col, label)
                label += 1
        self.component_labels = labels
        self.metadata['component_count'] = label

    def component_at(self, xy: Sequence[float]) -> int:
        cell = self.world_to_cell(xy)
        if not self.in_bounds_cell(cell):
            return -1
        return int(self.component_labels[cell[0], cell[1]])

    def validate_path(self, path_xy: Sequence[Sequence[float]]) -> CheckResult:
        if not path_xy:
            return CheckResult.fail('empty_path')
        start_cell = self.world_to_cell(path_xy[0])
        goal_cell = self.world_to_cell(path_xy[-1])
        if not self.is_cell_free(start_cell):
            return CheckResult.fail('start_occupied', start_cell=start_cell)
        if not self.is_cell_free(goal_cell):
            return CheckResult.fail('goal_occupied', goal_cell=goal_cell)
        checked = 0
        for segment_index, (start, end) in enumerate(zip(path_xy[:-1], path_xy[1:])):
            start_cell = self.world_to_cell(start)
            end_cell = self.world_to_cell(end)
            for cell in cells_on_segment(start_cell, end_cell):
                checked += 1
                if not self.is_cell_free(cell):
                    return CheckResult.fail('path_collides', segment_index=segment_index, collision_cell=cell, checked_cells=checked)
        return CheckResult.pass_(checked_cells=checked, waypoint_count=len(path_xy))

    def write_debug_artifacts(self, out_dir: str | Path, prefix: str='occupancy') -> dict[str, str]:
        from PIL import Image
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {'occupied_png': out_dir / f'{prefix}_occupied.png', 'eroded_free_png': out_dir / f'{prefix}_eroded_free.png', 'components_png': out_dir / f'{prefix}_components.png', 'metadata_json': out_dir / f'{prefix}_metadata.json'}
        Image.fromarray(np.where(self.occupied, 0, 255).astype(np.uint8)).save(paths['occupied_png'])
        Image.fromarray(np.where(self.eroded_free, 255, 0).astype(np.uint8)).save(paths['eroded_free_png'])
        Image.fromarray(_components_to_rgb(self.component_labels)).save(paths['components_png'])
        write_json(paths['metadata_json'], self.metadata)
        return {key: str(path) for key, path in paths.items()}

def cells_on_segment(start: Sequence[int], end: Sequence[int]) -> list[tuple[int, int]]:
    r0, c0 = (int(start[0]), int(start[1]))
    r1, c1 = (int(end[0]), int(end[1]))
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    step_r = 1 if r0 < r1 else -1
    step_c = 1 if c0 < c1 else -1
    cells: list[tuple[int, int]] = []
    if dc > dr:
        err = dc / 2
        row = r0
        for col in range(c0, c1 + step_c, step_c):
            cells.append((row, col))
            err -= dr
            if err < 0:
                row += step_r
                err += dc
    else:
        err = dr / 2
        col = c0
        for row in range(r0, r1 + step_r, step_r):
            cells.append((row, col))
            err -= dc
            if err < 0:
                col += step_c
                err += dr
    return cells

def _bounds_from_objects(objects: Sequence[AABBObject], padding: float) -> tuple[float, float, float, float]:
    if not objects:
        raise ValueError('Cannot infer occupancy bounds without objects')
    min_x = min((obj.center[0] - obj.extent[0] * 0.5 for obj in objects)) - padding
    min_y = min((obj.center[1] - obj.extent[1] * 0.5 for obj in objects)) - padding
    max_x = max((obj.center[0] + obj.extent[0] * 0.5 for obj in objects)) + padding
    max_y = max((obj.center[1] + obj.extent[1] * 0.5 for obj in objects)) + padding
    return (min_x, min_y, max_x, max_y)

def _traversability_mask_for_grid(*, height: int, width: int, origin_xy: tuple[float, float], resolution: float, config: OccupancyConfig) -> np.ndarray | None:
    if config.traversability_map is None:
        return None
    return _map_mask_for_grid(config.traversability_map, height=height, width=width, origin_xy=origin_xy, resolution=resolution, map_resolution=config.traversability_map_resolution, map_size=config.traversability_map_size)

def _line_of_sight_traversability_mask_for_grid(*, height: int, width: int, origin_xy: tuple[float, float], resolution: float, config: OccupancyConfig) -> np.ndarray | None:
    if config.line_of_sight_traversability_map is None:
        return None
    return _map_mask_for_grid(config.line_of_sight_traversability_map, height=height, width=width, origin_xy=origin_xy, resolution=resolution, map_resolution=config.line_of_sight_traversability_map_resolution, map_size=config.line_of_sight_traversability_map_size)

def _map_mask_for_grid(source_map: Any, *, height: int, width: int, origin_xy: tuple[float, float], resolution: float, map_resolution: float | None, map_size: int | None) -> np.ndarray:
    trav_map = np.asarray(source_map)
    if trav_map.ndim == 3:
        trav_map = trav_map[..., 0]
    if trav_map.ndim != 2 or trav_map.size == 0:
        raise ValueError(f'Invalid traversability map shape: {trav_map.shape}')
    trav_resolution = float(map_resolution or resolution)
    trav_size = int(map_size or trav_map.shape[0])
    if trav_resolution <= 0 or trav_size <= 0:
        raise ValueError(f'Invalid traversability map metadata: resolution={trav_resolution}, size={trav_size}')
    world_x = origin_xy[0] + (np.arange(width, dtype=np.float64) + 0.5) * resolution
    world_y = origin_xy[1] + (np.arange(height, dtype=np.float64) + 0.5) * resolution
    map_cols = np.floor(world_x / trav_resolution + trav_size / 2.0).astype(np.int32)
    map_rows = np.floor(world_y / trav_resolution + trav_size / 2.0).astype(np.int32)
    valid_cols = (map_cols >= 0) & (map_cols < trav_map.shape[1])
    valid_rows = (map_rows >= 0) & (map_rows < trav_map.shape[0])
    mask = np.zeros((height, width), dtype=bool)
    if not np.any(valid_cols) or not np.any(valid_rows):
        return mask
    row_indices = np.nonzero(valid_rows)[0]
    col_indices = np.nonzero(valid_cols)[0]
    sampled = trav_map[map_rows[row_indices][:, None], map_cols[col_indices][None, :]]
    mask[np.ix_(row_indices, col_indices)] = sampled > 0
    return mask

def _carve_open_navigation_barriers(traversability_mask: np.ndarray, objects: Sequence[AABBObject], *, origin_xy: tuple[float, float], resolution: float, config: OccupancyConfig) -> list[dict[str, Any]]:
    opened: list[dict[str, Any]] = []
    height, width = traversability_mask.shape
    for obj in objects:
        if not _is_open_navigation_barrier(obj):
            continue
        margin = _navigation_opening_margin(obj, config)
        r0, c0, r1, c1 = _aabb_cell_bounds(obj, origin_xy=origin_xy, resolution=resolution, width=width, height=height, margin=margin)
        if r0 > r1 or c0 > c1:
            continue
        traversability_mask[r0:r1 + 1, c0:c1 + 1] = True
        opened.append({'name': obj.name, 'category': obj.category, 'cell_bounds': [int(r0), int(c0), int(r1), int(c1)], 'margin': float(margin)})
    return opened

def _carve_doorway_openings(traversability_mask: np.ndarray, objects: Sequence[AABBObject], *, origin_xy: tuple[float, float], resolution: float, config: OccupancyConfig) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_changed = 0
    if not objects:
        return {'object_count': 0, 'changed_cells': 0, 'objects': records}
    height, width = traversability_mask.shape
    allowed_mask = _doorway_opening_allowed_mask_for_grid(height=height, width=width, origin_xy=origin_xy, resolution=resolution, config=config)
    for obj in objects:
        margin = _navigation_opening_margin(obj, config)
        r0, c0, r1, c1 = _aabb_cell_bounds(obj, origin_xy=origin_xy, resolution=resolution, width=width, height=height, margin=margin)
        if r0 > r1 or c0 > c1:
            continue
        region = traversability_mask[r0:r1 + 1, c0:c1 + 1]
        if allowed_mask is None:
            changed = int(np.count_nonzero(~region))
            traversability_mask[r0:r1 + 1, c0:c1 + 1] = True
        else:
            allowed = allowed_mask[r0:r1 + 1, c0:c1 + 1]
            changed = int(np.count_nonzero(~region & allowed))
            if np.any(allowed):
                updated = region.copy()
                updated[allowed] = True
                traversability_mask[r0:r1 + 1, c0:c1 + 1] = updated
        total_changed += changed
        records.append({'name': obj.name, 'category': obj.category, 'cell_bounds': [int(r0), int(c0), int(r1), int(c1)], 'margin': float(margin), 'changed_cells': int(changed)})
    return {'object_count': int(len(records)), 'changed_cells': int(total_changed), 'objects': records}

def _doorway_opening_allowed_mask_for_grid(*, height: int, width: int, origin_xy: tuple[float, float], resolution: float, config: OccupancyConfig) -> np.ndarray | None:
    if config.doorway_opening_allowed_map is None:
        return None
    return _map_mask_for_grid(config.doorway_opening_allowed_map, height=height, width=width, origin_xy=origin_xy, resolution=resolution, map_resolution=config.doorway_opening_allowed_map_resolution, map_size=config.doorway_opening_allowed_map_size)

def _aabb_cell_bounds(obj: AABBObject, *, origin_xy: tuple[float, float], resolution: float, width: int, height: int, margin: float=0.0) -> tuple[int, int, int, int]:
    x0 = float(obj.center[0]) - float(obj.extent[0]) * 0.5 - margin
    x1 = float(obj.center[0]) + float(obj.extent[0]) * 0.5 + margin
    y0 = float(obj.center[1]) - float(obj.extent[1]) * 0.5 - margin
    y1 = float(obj.center[1]) + float(obj.extent[1]) * 0.5 + margin
    c0 = math.floor((x0 - origin_xy[0]) / resolution)
    c1 = math.floor((x1 - origin_xy[0]) / resolution)
    r0 = math.floor((y0 - origin_xy[1]) / resolution)
    r1 = math.floor((y1 - origin_xy[1]) / resolution)
    return (max(0, r0), max(0, c0), min(height - 1, r1), min(width - 1, c1))

def _is_open_navigation_barrier(obj: AABBObject) -> bool:
    return _is_navigation_barrier(obj) and _open_state_value(obj) is True

def _is_navigation_barrier(obj: AABBObject) -> bool:
    states = getattr(obj, 'states', {}) or {}
    if isinstance(states, dict):
        barrier = str(states.get('navigation_barrier') or '').lower()
        if barrier in {'door', 'openable_door', 'navigation_barrier'}:
            return True
        if bool(states.get('blocks_navigation_when_closed')):
            return True
    category = str(getattr(obj, 'category', '') or '').lower()
    name = str(getattr(obj, 'name', '') or '').lower()
    return 'door' in category or 'door' in name

def _open_state_value(obj: AABBObject) -> bool | None:
    states = getattr(obj, 'states', {}) or {}
    if not isinstance(states, dict):
        return None
    if 'open' in states:
        return bool(states['open'])
    for key, value in states.items():
        if str(getattr(key, '__name__', '')) != 'Open':
            continue
        try:
            return bool(value.get_value())
        except Exception:
            return None
    return None

def _navigation_opening_margin(obj: AABBObject, config: OccupancyConfig) -> float:
    states = getattr(obj, 'states', {}) or {}
    if isinstance(states, dict) and 'navigation_opening_margin' in states:
        try:
            return max(0.0, float(states['navigation_opening_margin']))
        except Exception:
            pass
    return max(0.0, float(config.navigation_opening_margin))

def _should_rasterize(obj: AABBObject, config: OccupancyConfig) -> bool:
    if not obj.visible:
        return False
    if bool((getattr(obj, 'states', {}) or {}).get('occupancy_ignored')):
        return False
    if _is_open_navigation_barrier(obj):
        return False
    if obj.name in config.ignored_object_names:
        return False
    if obj.category in config.ignored_categories:
        return False
    min_z = obj.center[2] - obj.extent[2] * 0.5
    max_z = obj.center[2] + obj.extent[2] * 0.5
    band_min = config.floor_height + config.navigation_height_band[0]
    band_max = config.floor_height + config.navigation_height_band[1]
    return max_z >= band_min and min_z <= band_max

def _erode_free_square(free: np.ndarray, radius_cells: int) -> np.ndarray:
    occupied = ~free
    padded = np.pad(occupied.astype(np.uint8), radius_cells, mode='constant', constant_values=1)
    integral = np.pad(padded, ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius_cells + 1
    h, w = free.shape
    window_sum = integral[size:size + h, size:size + w] - integral[:h, size:size + w] - integral[size:size + h, :w] + integral[:h, :w]
    return free & (window_sum == 0)

def _flood_fill(free: np.ndarray, labels: np.ndarray, start_row: int, start_col: int, label: int) -> None:
    q: deque[tuple[int, int]] = deque([(start_row, start_col)])
    labels[start_row, start_col] = label
    while q:
        row, col = q.popleft()
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= next_row < free.shape[0] and 0 <= next_col < free.shape[1] and free[next_row, next_col] and (labels[next_row, next_col] == -1):
                labels[next_row, next_col] = label
                q.append((next_row, next_col))

def _components_to_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    mask = labels >= 0
    values = labels[mask].astype(np.uint32)
    rgb[mask, 0] = (values * 53 + 80) % 255
    rgb[mask, 1] = (values * 97 + 120) % 255
    rgb[mask, 2] = (values * 193 + 160) % 255
    return rgb
