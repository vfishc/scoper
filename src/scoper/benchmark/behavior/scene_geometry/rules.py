from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from scoper.benchmark.behavior.scene_geometry.paths import data_dir

@dataclass(frozen=True)
class AugmentationRules:
    manual_overrides: dict[str, set[str]]
    placement_policy: dict[str, Any]
    receptacle_policy: dict[str, Any]
    room_density_profiles: dict[str, dict[str, int]]
    semantic_groups: dict[str, Any]

    def density_for_room_type(self, room_type: str | None) -> dict[str, int]:
        default = dict(self.room_density_profiles.get('default') or {'min_large': 1, 'target_large': 3, 'target_small': 4})
        if room_type and room_type in self.room_density_profiles:
            default.update(self.room_density_profiles[room_type])
        return default

def load_rules(root: str | Path | None=None) -> AugmentationRules:
    base = Path(root) if root is not None else data_dir()
    overrides_raw = _load_json(base / 'manual_category_overrides.json')
    overrides = {str(key): {str(item) for item in value} for key, value in overrides_raw.items()}
    return AugmentationRules(manual_overrides=overrides, placement_policy=_load_json(base / 'placement_policy.json'), receptacle_policy=_load_json(base / 'receptacle_policy.json'), semantic_groups=_load_json(base / 'semantic_groups.json'), room_density_profiles={str(key): {str(k): int(v) for k, v in dict(value).items()} for key, value in _load_json(base / 'room_density_profiles.json').items()})

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))
__all__ = ['AugmentationRules', 'load_rules']
