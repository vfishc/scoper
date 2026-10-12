"""Scene usage statistics used as priors for asset-driven generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scoper.benchmark.behavior.scene_geometry.paths import assets_root as default_assets_root
from scoper.benchmark.behavior.scene_geometry.scene_io import parse_scene_objects, room_type_from_instance


@dataclass(frozen=True)
class SceneStatistics:
    scene_count: int
    used_categories: dict[str, int]
    room_category_counts: dict[str, dict[str, int]]

    def category_prior(self, room_type: str | None, category: str) -> float:
        if not room_type:
            return 0.0
        rows = self.room_category_counts.get(str(room_type), {})
        total = sum(rows.values())
        if total <= 0:
            return 0.0
        return float(rows.get(str(category), 0)) / float(total)

    def top_categories_for_room(self, room_type: str, *, limit: int = 30) -> list[tuple[str, int]]:
        rows = self.room_category_counts.get(str(room_type), {})
        return sorted(rows.items(), key=lambda item: (-item[1], item[0]))[: int(limit)]

    def summary(self) -> dict[str, Any]:
        return {
            "scene_count": self.scene_count,
            "used_category_count": len(self.used_categories),
            "used_categories": dict(sorted(self.used_categories.items(), key=lambda item: (-item[1], item[0]))),
            "room_category_counts": {
                room: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
                for room, counts in sorted(self.room_category_counts.items())
            },
        }


def build_scene_statistics(assets_root: str | Path | None = None) -> SceneStatistics:
    root = Path(assets_root) if assets_root is not None else default_assets_root()
    scene_root = root / "scenes"
    used: dict[str, int] = {}
    room_counts: dict[str, dict[str, int]] = {}
    scene_count = 0
    for path in sorted(scene_root.glob("*/json/*_best.json")):
        try:
            scene = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scene_count += 1
        for record in parse_scene_objects(scene).values():
            if not record.category:
                continue
            used[record.category] = used.get(record.category, 0) + 1
            for room in record.rooms:
                room_type = room_type_from_instance(room)
                room_counts.setdefault(room_type, {})
                room_counts[room_type][record.category] = room_counts[room_type].get(record.category, 0) + 1
    return SceneStatistics(scene_count=scene_count, used_categories=used, room_category_counts=room_counts)


def analyze_scene_density(scene: dict[str, Any], classifier: Any) -> dict[str, Any]:
    rooms: dict[str, dict[str, Any]] = {}
    for record in parse_scene_objects(scene).values():
        record_rooms = record.rooms or ("?",)
        role = getattr(classifier.get(record.category), "role", "unknown") if classifier is not None else "unknown"
        for room in record_rooms:
            room_type = room_type_from_instance(room) if room != "?" else "?"
            row = rooms.setdefault(
                room,
                {
                    "room_type": room_type,
                    "object_count": 0,
                    "roles": {},
                    "categories": {},
                },
            )
            row["object_count"] += 1
            row["roles"][role] = row["roles"].get(role, 0) + 1
            row["categories"][record.category] = row["categories"].get(record.category, 0) + 1
    return {"rooms": rooms}


__all__ = ["SceneStatistics", "analyze_scene_density", "build_scene_statistics"]
