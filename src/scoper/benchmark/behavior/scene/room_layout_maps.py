from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image
from scoper.benchmark.behavior.paths import resolve_repo_path
DEFAULT_MAP_RESOLUTION = 0.01
MAX_ROOM_LAYOUT_PIXELS = 512000000

@dataclass(frozen=True)
class RoomLayoutMaps:
    scene_model: str
    floor: int
    map_resolution: float
    map_size: int
    room_ins_map: np.ndarray
    room_sem_map: np.ndarray
    trav_map: np.ndarray
    room_ins_name_to_ins_id: dict[str, int]
    room_sem_name_to_ins_name: dict[str, list[str]]
    source_paths: dict[str, str]

    def room_mask(self, room_id: str) -> np.ndarray | None:
        ins_id = self.room_ins_name_to_ins_id.get(str(room_id))
        if ins_id is None:
            return None
        return self.room_ins_map == int(ins_id)

    def traversable_mask(self) -> np.ndarray:
        return self.trav_map > 0

    def map_to_world(self, row_col: Sequence[float]) -> tuple[float, float]:
        row = float(row_col[0])
        col = float(row_col[1])
        return ((col - self.map_size / 2.0) * self.map_resolution, (row - self.map_size / 2.0) * self.map_resolution)

    def world_to_map(self, xy: Sequence[float]) -> tuple[int, int]:
        x = float(xy[0])
        y = float(xy[1])
        return (int(y / self.map_resolution + self.map_size / 2.0), int(x / self.map_resolution + self.map_size / 2.0))

def load_behavior_room_layout_maps(scene_model: str, *, assets_root: str | Path | None=None, floor: int=0) -> RoomLayoutMaps:
    root = Path(assets_root) if assets_root else resolve_repo_path('datasets/behavior-1k-assets')
    layout_dir = root / 'scenes' / str(scene_model) / 'layout'
    ins_path = layout_dir / f'floor_insseg_{floor}.png'
    sem_path = layout_dir / f'floor_semseg_{floor}.png'
    trav_path = layout_dir / f'floor_trav_{floor}.png'
    categories_path = root / 'metadata' / 'room_categories.txt'
    missing = [str(path) for path in (ins_path, sem_path, trav_path, categories_path) if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing room layout files: {}'.format(', '.join(missing)))
    room_ins = _load_grayscale(ins_path)
    room_sem = _load_grayscale(sem_path, shape=room_ins.shape)
    trav = _load_grayscale(trav_path, shape=room_ins.shape)
    if room_ins.shape[0] != room_ins.shape[1]:
        raise ValueError('Room instance segmentation map must be square: {}'.format(room_ins.shape))
    room_categories = [line.strip() for line in categories_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    instance_names, sem_to_instances = _build_room_instance_names(room_ins, room_sem, room_categories)
    return RoomLayoutMaps(scene_model=str(scene_model), floor=int(floor), map_resolution=DEFAULT_MAP_RESOLUTION, map_size=int(room_ins.shape[0]), room_ins_map=room_ins, room_sem_map=room_sem, trav_map=trav, room_ins_name_to_ins_id=instance_names, room_sem_name_to_ins_name=sem_to_instances, source_paths={'floor_insseg': str(ins_path), 'floor_semseg': str(sem_path), 'floor_trav': str(trav_path), 'room_categories': str(categories_path)})

def _load_grayscale(path: Path, *, shape: tuple[int, int] | None=None) -> np.ndarray:
    with _allow_large_room_layout_maps():
        image = Image.open(path).convert('L')
    if shape is not None and (image.height, image.width) != tuple(shape):
        image = image.resize((int(shape[1]), int(shape[0])), Image.Resampling.NEAREST)
    return np.array(image)

@contextmanager
def _allow_large_room_layout_maps():
    previous = Image.MAX_IMAGE_PIXELS
    try:
        if previous is None:
            yield
        else:
            Image.MAX_IMAGE_PIXELS = max(int(previous), MAX_ROOM_LAYOUT_PIXELS)
            yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous

def _build_room_instance_names(room_ins: np.ndarray, room_sem: np.ndarray, room_categories: Sequence[str]) -> tuple[dict[str, int], dict[str, list[str]]]:
    sem_to_ins: dict[int, list[int]] = {}
    for ins_id in sorted((int(value) for value in np.unique(room_ins) if int(value) != 0)):
        rows, cols = np.where(room_ins == ins_id)
        if len(rows) == 0:
            continue
        sem_id = int(room_sem[int(rows[0]), int(cols[0])])
        if sem_id <= 0 or sem_id > len(room_categories):
            continue
        sem_to_ins.setdefault(sem_id, []).append(ins_id)
    instance_names: dict[str, int] = {}
    sem_to_instances: dict[str, list[str]] = {}
    for sem_id in sorted(sem_to_ins):
        sem_name = str(room_categories[sem_id - 1])
        sem_to_instances[sem_name] = []
        for index, ins_id in enumerate(sorted(sem_to_ins[sem_id])):
            ins_name = f'{sem_name}_{index}'
            instance_names[ins_name] = int(ins_id)
            sem_to_instances[sem_name].append(ins_name)
    return (instance_names, sem_to_instances)
__all__ = ['RoomLayoutMaps', 'load_behavior_room_layout_maps']
