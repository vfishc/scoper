"""Variant generator for asset-driven BEHAVIOR-1K scene augmentation."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scoper.benchmark.behavior.scene.room_layout_maps import load_behavior_room_layout_maps

from scoper.benchmark.behavior.scene_geometry.asset_catalog import (
    AssetCatalog,
    ModelAsset,
    bbox_bottom_z,
    bbox_top_z,
    root_z_for_bbox_bottom,
    root_z_for_floor,
    scaled_bbox,
)
from scoper.benchmark.behavior.scene_geometry.category_classifier import CategoryClassifier
from scoper.benchmark.behavior.scene_geometry.geometry import (
    Footprint,
    FreeSpaceMetrics,
    FLOOR_LAYER_CATEGORIES,
    PROTECTED_ZONE_SOURCES,
    SpatialIndex,
    aabb_overlap_ratio,
    build_spatial_index,
    candidate_fits,
    clear_geometry_caches,
    existing_footprints,
    _largest_component_size,
    footprint_rows_cols,
    footprint_extent_for_yaw,
    is_openable_clearance_object,
    protected_zones_for_object,
    protected_zones_from_scene,
    room_free_mask,
    room_free_mask_crop,
    sample_anchor_pose,
    traversable_mask_crop,
)
from scoper.benchmark.behavior.scene_geometry.paths import assets_root as default_assets_root
from scoper.benchmark.behavior.scene_geometry.rules import AugmentationRules, load_rules
from scoper.benchmark.behavior.scene_geometry.scene_io import (
    add_dataset_object,
    clone_scene,
    ensure_scene_augmentation_metadata,
    load_scene_json,
    parse_scene_objects,
    remove_object,
    room_type_from_instance,
    set_object_model,
    set_object_pose,
    set_object_room,
    quat_to_yaw,
    yaw_to_quat,
)
from scoper.benchmark.behavior.scene_geometry.scene_statistics import SceneStatistics
from scoper.benchmark.behavior.scene_geometry.surface_occupancy import (
    ObjectAABB,
    aabb3d_overlap_ratio,
    aabb3d_overlaps,
    object_aabb_from_pose,
    object_aabb_from_record,
    small_object_aabbs_from_scene,
    xy_center_distance,
)
from scoper.benchmark.behavior.scene_geometry.surface_constraints import (
    support_matches_surface_constraints,
    support_surface_descriptor_from_values,
    surface_constraints_from_policy,
)
from scoper.benchmark.behavior.scene_geometry.taxonomy_index import TaxonomyIndex


_ROOM_USABILITY_STATE_CACHE_MAX = 16
_FOOTPRINT_SIGNATURE_CACHE_MAX = 4096


@dataclass(frozen=True)
class VariantConfig:
    add_large: bool = True
    add_small: bool = True
    model_swap: bool = True
    pose_jitter: bool = True
    max_variants_per_scene: int = 3
    large_layout_mode: str = "balanced"
    large_rewrite_fraction: float = 0.6
    preserve_task_objects: bool = True
    min_large_per_room: int | None = None
    space_budget: float | None = None
    diversity_level: float = 1.0
    room_relayout_allow_removal: bool | None = None
    room_relayout_max_removals_per_room: int | None = None
    room_subset_mode: str = "all"
    room_subset_size: int = 0
    active_rooms: tuple[str, ...] = ()
    exclude_room_types: tuple[str, ...] = ()
    stationary_category_swap: bool = True
    stationary_category_swap_probability: float = 0.65


@dataclass(frozen=True)
class VariantResult:
    scene: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True)
class GroupMemberPlacement:
    record: Any | None
    role_id: str
    category: str
    asset: ModelAsset
    scale: tuple[float, float, float]
    offset_xy: tuple[float, float]
    relative_yaw: float
    fixed_base: bool
    visual_only: bool


@dataclass(frozen=True)
class GroupPlacement:
    group_id: str
    anchor_xy: tuple[float, float]
    yaw: float
    footprint: Footprint
    members: tuple[GroupMemberPlacement, ...]


@dataclass(frozen=True)
class GroupFootprintSpec:
    extent_xy: tuple[float, float]
    center_offset_xy: tuple[float, float]


@dataclass(frozen=True)
class LayoutUsabilityConfig:
    space_budget: float
    min_traversable_free_ratio: float
    min_connected_free_ratio: float
    clearance_m: float
    check_connected_during_sampling: bool


@dataclass(frozen=True)
class RoomUsabilityState:
    crop: Any
    traversable: np.ndarray
    room_area: float
    occupied_area: float
    current_metrics: FreeSpaceMetrics


@dataclass(frozen=True)
class RoomRelayoutFallback:
    occupied: list[Footprint]
    status: str


@dataclass(frozen=True)
class RoomRelayoutAttemptResult:
    scene: dict[str, Any]
    summary: dict[str, Any]
    occupied: list[Footprint]


@dataclass(frozen=True)
class ActiveRoomSelection:
    rooms: frozenset[str]
    mode: str
    requested_size: int
    eligible_rooms: tuple[str, ...]
    adjacency: dict[str, tuple[str, ...]]


class SceneVariantGenerator:
    def __init__(
        self,
        *,
        catalog: AssetCatalog | None = None,
        taxonomy: TaxonomyIndex | None = None,
        classifier: CategoryClassifier | None = None,
        statistics: SceneStatistics | None = None,
        rules: AugmentationRules | None = None,
        assets_root: str | Path | None = None,
    ):
        self.assets_root = Path(assets_root) if assets_root is not None else default_assets_root()
        self.catalog = catalog or AssetCatalog.build(self.assets_root)
        self.taxonomy = taxonomy or TaxonomyIndex.build()
        self.rules = rules or load_rules()
        self.classifier = classifier or CategoryClassifier(self.catalog, self.taxonomy, self.rules)
        self.statistics = statistics
        self._layout_cache: dict[tuple[str, int, str], Any] = {}
        self._room_usability_state_cache: dict[tuple[Any, ...], RoomUsabilityState] = {}
        self._footprint_signature_cache: dict[
            tuple[int, str],
            tuple[list[Footprint], tuple[int, ...], tuple[tuple[Any, ...], ...]],
        ] = {}

    def generate_from_file(
        self,
        *,
        scene_model: str,
        input_json: str | Path,
        seed: int,
        config: VariantConfig | None = None,
    ) -> VariantResult:
        return self.generate(scene_model=scene_model, scene=load_scene_json(input_json), source_json=str(input_json), seed=seed, config=config)

    def generate(
        self,
        *,
        scene_model: str,
        scene: dict[str, Any],
        source_json: str,
        seed: int,
        config: VariantConfig | None = None,
    ) -> VariantResult:
        cfg = config or VariantConfig()
        rng = random.Random(int(seed))
        clear_geometry_caches()
        self._room_usability_state_cache.clear()
        self._footprint_signature_cache.clear()
        variant = clone_scene(scene)
        layout = self._load_layout(str(scene_model), floor=0)
        self.classifier.classify_all()
        usability = self._layout_usability_config(cfg)
        diversity_level = _clamp(float(cfg.diversity_level), 0.0, 2.5)
        active_selection = self._select_active_rooms(layout, rng, cfg)
        active_rooms = active_selection.rooms
        all_rooms = set(str(room_id) for room_id in layout.room_ins_name_to_ins_id)

        summary: dict[str, Any] = {
            "scene_model": str(scene_model),
            "seed": int(seed),
            "source_json": str(source_json),
            "large_layout_mode": str(cfg.large_layout_mode),
            "large_rewrite_fraction": float(cfg.large_rewrite_fraction),
            "space_budget": float(usability.space_budget),
            "diversity_level": float(diversity_level),
            "room_subset_mode": active_selection.mode,
            "room_subset_size": int(active_selection.requested_size),
            "active_rooms": sorted(active_rooms),
            "active_room_count": len(active_rooms),
            "inactive_room_count": max(0, len(all_rooms) - len(active_rooms)),
            "eligible_room_count": len(active_selection.eligible_rooms),
            "added_large": 0,
            "removed_large": 0,
            "discarded_large": 0,
            "relocated_large": 0,
            "rewritten_large": 0,
            "room_relayout_rooms": 0,
            "placed_groups": 0,
            "failed_groups": 0,
            "group_objects_relocated": 0,
            "category_swaps": 0,
            "stationary_category_swaps": 0,
            "added_small": 0,
            "model_swaps": 0,
            "pose_jitters": 0,
            "space_rejections": 0,
            "door_clearance_rejections": 0,
            "openable_clearance_rejections": 0,
            "model_swap_safety_rejections": 0,
            "small_collision_rejections": 0,
            "small_placement_failures": 0,
            "small_collision_repairs": 0,
            "relocated_small": 0,
            "removed_small": 0,
            "small_collision_unresolved": 0,
            "skipped": {},
            "operations": [],
        }

        occupied = existing_footprints(
            variant,
            self.catalog,
            self.classifier,
            include_roles={"large_floor", "support_surface", "fixture"},
        )
        protected_zones = protected_zones_from_scene(
            variant,
            self.catalog,
            self.classifier,
            policy=self.rules.placement_policy,
        )
        occupied.extend(protected_zones)
        task_relevant = _task_relevant_names(variant) if cfg.preserve_task_objects else set()
        if cfg.add_large:
            occupied = self._apply_large_layout_mode(variant, layout, occupied, rng, summary, cfg, task_relevant, usability, active_rooms)
            occupied = self._add_large_objects(variant, layout, occupied, rng, summary, usability, active_rooms)
        if cfg.stationary_category_swap:
            self._apply_stationary_category_swaps(variant, layout, rng, summary, usability, active_rooms, cfg)
        if cfg.model_swap:
            self._apply_model_swaps(variant, layout, rng, summary, usability, active_rooms)
        if cfg.pose_jitter:
            self._apply_pose_jitter(variant, layout, rng, summary, diversity_level, active_rooms)
        records = parse_scene_objects(variant)
        if cfg.add_small:
            self._add_small_objects(
                variant,
                records,
                layout,
                rng,
                summary,
                conservative_supports=cfg.large_layout_mode == "room_relayout",
                diversity_level=diversity_level,
                active_rooms=active_rooms,
            )
            self._repair_small_object_collisions(
                variant,
                layout,
                rng,
                summary,
                task_relevant=task_relevant,
                active_rooms=active_rooms,
            )

        metadata = ensure_scene_augmentation_metadata(variant)
        metadata.update(
            {
                "generator": "scoper.benchmark.behavior.scene_geometry",
                "version": "0.1",
                "seed": int(seed),
                "source_json": str(source_json),
                "operations": {
                    "large_layout_mode": str(cfg.large_layout_mode),
                    "large_rewrite_fraction": float(cfg.large_rewrite_fraction),
                    "space_budget": float(usability.space_budget),
                    "diversity_level": float(diversity_level),
                    "room_subset_mode": active_selection.mode,
                    "room_subset_size": int(active_selection.requested_size),
                    "active_rooms": sorted(active_rooms),
                    "active_room_count": len(active_rooms),
                    "inactive_room_count": max(0, len(all_rooms) - len(active_rooms)),
                    "added_large": summary["added_large"],
                    "removed_large": summary["removed_large"],
                    "discarded_large": summary["discarded_large"],
                    "relocated_large": summary["relocated_large"],
                    "rewritten_large": summary["rewritten_large"],
                    "room_relayout_rooms": summary["room_relayout_rooms"],
                    "placed_groups": summary["placed_groups"],
                    "failed_groups": summary["failed_groups"],
                    "group_objects_relocated": summary["group_objects_relocated"],
                    "category_swaps": summary["category_swaps"],
                    "stationary_category_swaps": summary["stationary_category_swaps"],
                    "added_small": summary["added_small"],
                    "model_swaps": summary["model_swaps"],
                    "pose_jitters": summary["pose_jitters"],
                    "space_rejections": summary["space_rejections"],
                    "door_clearance_rejections": summary["door_clearance_rejections"],
                    "openable_clearance_rejections": summary["openable_clearance_rejections"],
                    "model_swap_safety_rejections": summary["model_swap_safety_rejections"],
                    "small_collision_rejections": summary["small_collision_rejections"],
                    "small_placement_failures": summary["small_placement_failures"],
                    "small_collision_repairs": summary["small_collision_repairs"],
                    "relocated_small": summary["relocated_small"],
                    "removed_small": summary["removed_small"],
                    "small_collision_unresolved": summary["small_collision_unresolved"],
                },
                "operation_log": summary["operations"],
                "skipped": summary["skipped"],
            }
        )
        self._room_usability_state_cache.clear()
        self._footprint_signature_cache.clear()
        clear_geometry_caches()
        return VariantResult(scene=variant, summary=summary)

    def _load_layout(self, scene_model: str, *, floor: int = 0) -> Any:
        key = (str(scene_model), int(floor), str(self.assets_root))
        cached = self._layout_cache.get(key)
        if cached is None:
            cached = load_behavior_room_layout_maps(str(scene_model), assets_root=self.assets_root, floor=int(floor))
            self._layout_cache[key] = cached
        return cached

    def _apply_large_layout_mode(
        self,
        scene: dict[str, Any],
        layout: object,
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any],
        config: VariantConfig,
        task_relevant: set[str],
        usability: LayoutUsabilityConfig,
        active_rooms: frozenset[str],
    ) -> list[Footprint]:
        mode = str(config.large_layout_mode or "off")
        if mode == "off":
            return occupied
        if mode == "jitter":
            return self._jitter_existing_large(scene, layout, occupied, rng, summary, task_relevant, usability, active_rooms)
        if mode == "room_relayout":
            return self._room_relayout_existing_large(scene, layout, occupied, rng, summary, config, task_relevant, usability, active_rooms)
        if mode not in {"balanced", "rewrite"}:
            _skip(summary, f"unknown_large_layout_mode_{mode}")
            return occupied
        return self._rewrite_existing_large(scene, layout, occupied, rng, summary, config, task_relevant, usability, active_rooms)

    def _rewrite_existing_large(
        self,
        scene: dict[str, Any],
        layout: object,
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any],
        config: VariantConfig,
        task_relevant: set[str],
        usability: LayoutUsabilityConfig,
        active_rooms: frozenset[str],
    ) -> list[Footprint]:
        mode = str(config.large_layout_mode or "balanced")
        fraction = _clamp(float(config.large_rewrite_fraction), 0.0, 1.0)
        if mode == "rewrite":
            fraction = max(fraction, 0.8)
        records = parse_scene_objects(scene)
        occupied_by_name = {fp.name: fp for fp in occupied}
        max_overlap = float(self.rules.placement_policy.get("max_large_overlap_ratio", 0.08))
        by_room: dict[str, list[Any]] = {}
        for record in records.values():
            if record.name in task_relevant:
                continue
            if not record.primary_room:
                continue
            if record.primary_room not in active_rooms:
                continue
            if not self._is_rewrite_candidate(record, mode):
                continue
            by_room.setdefault(record.primary_room, []).append(record)

        for room_id, candidates in sorted(by_room.items()):
            room_type = room_type_from_instance(room_id)
            allowed = self._allowed_large_categories_for_room(room_type)
            eligible = [record for record in candidates if record.category in allowed]
            if not eligible:
                continue
            rng.shuffle(eligible)
            eligible.sort(key=lambda record: _rewrite_priority(record.category))
            target_count = self._rewrite_target_count(len(eligible), fraction)
            selected = self._select_rewrite_candidates(eligible, target_count, rng)
            for record in selected:
                local_occupied = _without_object_footprints(occupied, record.name)
                replacement_category = self._sample_rewrite_category(record.category, room_type, mode, rng, summary)
                if replacement_category is None:
                    _skip(summary, f"rewrite_no_category_{record.category}")
                    continue
                replacement_asset = self._sample_rewrite_model(record, replacement_category, rng, summary)
                if replacement_asset is None:
                    _skip(summary, f"rewrite_no_model_{replacement_category}")
                    continue
                placed = self._sample_large_pose(
                    layout=layout,
                    room_id=room_id,
                    category=replacement_category,
                    asset=replacement_asset,
                    scale=record.scale,
                    occupied=local_occupied,
                    rng=rng,
                    summary=summary,
                    usability=usability,
                    clearance_m=float(self.rules.placement_policy.get("large_rewrite_clearance_m", self.rules.placement_policy.get("large_clearance_m", 0.18))),
                    avoid_center_xy=(record.pos[0], record.pos[1]),
                    min_center_distance_m=float(self.rules.placement_policy.get("large_jitter_translate_m", 0.18)) * (2.0 if mode == "rewrite" else 1.0),
                )
                if placed is None:
                    placed = self._sample_local_relocation_pose(
                        layout=layout,
                        room_id=room_id,
                        record=record,
                        asset=replacement_asset,
                        occupied=local_occupied,
                        rng=rng,
                        summary=summary,
                        usability=usability,
                    )
                    if placed is None and replacement_category == record.category:
                        current_asset = self.catalog.get_model(record.category, record.model)
                        if current_asset is not None and current_asset.model != replacement_asset.model:
                            replacement_asset = current_asset
                            placed = self._sample_local_relocation_pose(
                                layout=layout,
                                room_id=room_id,
                                record=record,
                                asset=replacement_asset,
                                occupied=local_occupied,
                                rng=rng,
                                summary=summary,
                                usability=usability,
                            )
                    if placed is None:
                        _skip(summary, f"rewrite_placement_failed_{replacement_category}")
                        continue
                xy, yaw, fp = placed
                if any(aabb_overlap_ratio(fp, other) > max_overlap for other in local_occupied if other.room == room_id):
                    _skip(summary, f"rewrite_overlap_{replacement_category}")
                    continue
                old_model = record.model
                old_category = record.category
                old_pos = record.pos
                old_yaw = record.ori
                new_pos = (xy[0], xy[1], root_z_for_floor(replacement_asset, record.scale))
                new_ori = yaw_to_quat(yaw)
                if replacement_category == record.category:
                    set_object_model(scene, record.name, replacement_asset.model)
                    rewritten_name = record.name
                    removed_name = None
                else:
                    removed_name = record.name
                    remove_object(scene, record.name)
                    rewritten_name = self._unique_name(scene, replacement_category, replacement_asset.model, int(summary["seed"]), summary["rewritten_large"])
                    add_dataset_object(
                        scene,
                        name=rewritten_name,
                        category=replacement_category,
                        model=replacement_asset.model,
                        room=room_id,
                        pos=new_pos,
                        ori=new_ori,
                        fixed_base=record.fixed_base,
                        visual_only=record.visual_only,
                        scale=record.scale,
                    )
                    summary["removed_large"] += 1
                if replacement_category == record.category:
                    set_object_pose(scene, record.name, pos=new_pos, ori=new_ori)
                    set_object_room(scene, record.name, room_id)
                new_fp = Footprint(
                    name=rewritten_name,
                    category=replacement_category,
                    room=room_id,
                    center_xy=xy,
                    extent_xy=fp.extent_xy,
                    yaw=yaw,
                    source="rewritten_large",
                )
                occupied_by_name.pop(record.name, None)
                occupied_by_name[rewritten_name] = new_fp
                occupied = local_occupied + [new_fp] + self._protected_zones_for_placed_object(
                    rewritten_name,
                    replacement_category,
                    room_id,
                    replacement_asset,
                    xy,
                    yaw,
                    record.scale,
                )
                summary["rewritten_large"] += 1
                summary["relocated_large"] += 1
                if replacement_asset.model != old_model:
                    summary["model_swaps"] += 1
                summary["operations"].append(
                    {
                        "op": "rewrite_large",
                        "name": rewritten_name,
                        "old_name": record.name,
                        "removed_name": removed_name,
                        "category": replacement_category,
                        "old_category": old_category,
                        "model": replacement_asset.model,
                        "old_model": old_model,
                        "room": room_id,
                        "old_pos": list(old_pos),
                        "new_pos": [float(new_pos[0]), float(new_pos[1]), float(new_pos[2])],
                        "old_ori": list(old_yaw),
                        "new_ori": list(new_ori),
                    }
                )
        return occupied

    def _room_relayout_existing_large(
        self,
        scene: dict[str, Any],
        layout: object,
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any],
        config: VariantConfig,
        task_relevant: set[str],
        usability: LayoutUsabilityConfig,
        active_rooms: frozenset[str],
    ) -> list[Footprint]:
        records = parse_scene_objects(scene)
        occupied_by_name = {fp.name: fp for fp in occupied}
        movable_by_room: dict[str, list[Any]] = {}
        for record in records.values():
            if record.name in task_relevant or not record.primary_room:
                continue
            if record.primary_room not in active_rooms:
                continue
            if not self._is_rewrite_candidate(record, "room_relayout"):
                continue
            movable_by_room.setdefault(record.primary_room, []).append(record)

        for room_id, room_records in sorted(movable_by_room.items()):
            room_type = room_type_from_instance(room_id)
            group_defs = self._semantic_groups_for_room(room_type)
            if not group_defs:
                continue
            relayout_records = self._select_room_relayout_records(room_records, room_type, config, rng)
            if len(relayout_records) < 2:
                continue
            attempt = self._attempt_room_relayout(
                scene=scene,
                layout=layout,
                occupied=occupied,
                rng=rng,
                summary=summary,
                config=config,
                usability=usability,
                room_id=room_id,
                room_type=room_type,
                group_defs=group_defs,
                relayout_records=relayout_records,
            )
            if attempt is None:
                _skip(summary, f"room_relayout_no_valid_plan_{room_type}")
                continue
            scene.clear()
            scene.update(attempt.scene)
            summary.clear()
            summary.update(attempt.summary)
            occupied = attempt.occupied
            occupied_by_name = {fp.name: fp for fp in occupied}
        return list(occupied_by_name.values())

    def _attempt_room_relayout(
        self,
        *,
        scene: dict[str, Any],
        layout: object,
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any],
        config: VariantConfig,
        usability: LayoutUsabilityConfig,
        room_id: str,
        room_type: str,
        group_defs: list[dict[str, Any]],
        relayout_records: list[Any],
    ) -> RoomRelayoutAttemptResult | None:
        max_room_removals = self._room_relayout_max_removals(config, len(relayout_records))
        allow_room_removal = self._room_relayout_allow_removal(config)
        for group_def in self._ordered_group_defs(group_defs, room_type, rng, summary):
            max_member_attempts = max(1, int(self.rules.placement_policy.get("room_relayout_group_member_attempts", 4)))
            attempted_member_signatures: set[tuple[tuple[str, str, str], ...]] = set()
            for member_attempt in range(max_member_attempts):
                attempt = self._attempt_room_relayout_group(
                    scene=scene,
                    layout=layout,
                    occupied=occupied,
                    rng=rng,
                    summary=summary,
                    config=config,
                    usability=usability,
                    room_id=room_id,
                    room_type=room_type,
                    group_def=group_def,
                    relayout_records=relayout_records,
                    attempted_member_signatures=attempted_member_signatures,
                    force_compact_group_models=member_attempt > 0,
                )
                if attempt is not None:
                    return attempt
        return None

    def _attempt_room_relayout_group(
        self,
        *,
        scene: dict[str, Any],
        layout: object,
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any],
        config: VariantConfig,
        usability: LayoutUsabilityConfig,
        room_id: str,
        room_type: str,
        group_def: dict[str, Any],
        relayout_records: list[Any],
        attempted_member_signatures: set[tuple[tuple[str, str, str], ...]],
        force_compact_group_models: bool = False,
    ) -> RoomRelayoutAttemptResult | None:
        max_room_removals = self._room_relayout_max_removals(config, len(relayout_records))
        allow_room_removal = self._room_relayout_allow_removal(config)
        trial_scene = clone_scene(scene)
        trial_summary = copy.deepcopy(summary)
        if force_compact_group_models:
            trial_summary["_force_compact_group_models"] = True
        discarded_before = int(trial_summary.get("discarded_large", 0))
        room_discards = 0
        available = {record.name: record for record in relayout_records}
        members = self._select_group_members(group_def, room_type, available, rng, trial_summary)
        if members is None or len(members) < 2:
            trial_summary["failed_groups"] += 1
            _skip(summary, f"group_member_selection_failed_{room_type}_{group_def.get('id', 'unknown')}")
            return None
        signature = tuple(sorted((member.role_id, member.category, member.asset.model) for member in members))
        if signature in attempted_member_signatures:
            _skip(summary, f"group_member_duplicate_{room_type}_{group_def.get('id', 'unknown')}")
            return None
        attempted_member_signatures.add(signature)
        group_member_names = {member.record.name for member in members if member.record is not None}
        candidate_relocatable = {
            record.name
            for record in available.values()
            if record.name not in group_member_names and self._is_room_relayout_soft_obstacle(record)
        }
        removed_for_plan = group_member_names
        fixed_occupied = [fp for fp in occupied if fp.name not in removed_for_plan and fp.owner not in removed_for_plan]
        placement = self._try_place_semantic_group_members(
            group_def=group_def,
            room_id=room_id,
            members=members,
            occupied=fixed_occupied,
            layout=layout,
            rng=rng,
            summary=trial_summary,
            usability=usability,
        )
        if placement is None:
            trial_summary["failed_groups"] += 1
            _skip(summary, f"group_placement_failed_{room_type}_{group_def.get('id', 'unknown')}")
            return None
        fixed_occupied = self._commit_semantic_group(trial_scene, placement, room_id, rng, trial_summary, fixed_occupied)
        for member in placement.members:
            if member.record is not None:
                available.pop(member.record.name, None)
        trial_summary["placed_groups"] += 1

        optional_probability = self._effective_probability(
            float(self.rules.placement_policy.get("room_relayout_optional_single_probability", 0.35)),
            trial_summary,
            max_value=0.75,
        )
        for record in list(available.values()):
            if record.name not in candidate_relocatable:
                continue
            if rng.random() > optional_probability:
                _skip(trial_summary, f"room_relayout_optional_kept_{record.category}")
                continue
            placed = self._place_room_relayout_single(
                scene=trial_scene,
                record=record,
                room_id=room_id,
                room_type=room_type,
                occupied=fixed_occupied,
                layout=layout,
                rng=rng,
                summary=trial_summary,
                usability=usability,
            )
            if placed is None:
                _skip(trial_summary, f"room_relayout_single_failed_{record.category}")
                continue
            fixed_occupied = placed
        if room_discards > max_room_removals:
            _skip(summary, f"room_relayout_plan_rejected_removals_{room_type}_{group_def.get('id', 'unknown')}")
            return None
        trial_summary["room_relayout_rooms"] += 1
        trial_summary.pop("_force_compact_group_models", None)
        trial_summary["operations"].append(
            {
                "op": "room_relayout",
                "room": room_id,
                "room_type": room_type,
                "selected_objects": len(relayout_records),
                "discarded_large": room_discards,
                "max_removals": max_room_removals,
                "placed_groups": [placement.group_id],
            }
        )
        trial_occupied = [fp for fp in fixed_occupied if fp.room != room_id] + [fp for fp in fixed_occupied if fp.room == room_id]
        return RoomRelayoutAttemptResult(scene=trial_scene, summary=trial_summary, occupied=trial_occupied)

    def _semantic_groups_for_room(self, room_type: str) -> list[dict[str, Any]]:
        groups_by_room = self.rules.semantic_groups.get("room_groups") or {}
        return [dict(item) for item in groups_by_room.get(str(room_type), []) if isinstance(item, dict)]

    def _ordered_group_defs(
        self,
        group_defs: list[dict[str, Any]],
        room_type: str,
        rng: random.Random,
        summary: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        preferred = {
            "living_room": ("sofa_tv_coffee_table", "sofa_coffee_table"),
            "television_room": ("sofa_tv_coffee_table",),
            "bedroom": ("bed_nightstand", "bed_desk_chair"),
            "childs_room": ("bed_desk_chair",),
            "dining_room": ("table_chairs",),
            "kitchen": ("table_chairs",),
            "private_office": ("desk_chair_bookshelf",),
            "shared_office": ("desk_chair_bookshelf",),
        }.get(str(room_type), ())
        rows = list(group_defs)
        rng.shuffle(rows)
        diversity_level = float((summary or {}).get("diversity_level", 1.0))
        if rng.random() > min(0.65, max(0.0, diversity_level - 1.0) * 0.35):
            rows.sort(key=lambda row: preferred.index(str(row.get("id"))) if str(row.get("id")) in preferred else len(preferred))
        return rows

    def _select_room_relayout_records(self, records: list[Any], room_type: str, config: VariantConfig, rng: random.Random) -> list[Any]:
        fraction = max(0.75, _clamp(float(config.large_rewrite_fraction), 0.0, 1.0))
        rows = [record for record in records if record.category in self._allowed_large_categories_for_room(room_type)]
        rng.shuffle(rows)
        rows.sort(key=lambda record: _rewrite_priority(record.category))
        min_count = int(config.min_large_per_room) if config.min_large_per_room is not None else 2
        target = max(min_count, int(round(len(rows) * fraction)))
        return rows[: min(len(rows), target)]

    def _try_place_semantic_group(
        self,
        *,
        group_def: dict[str, Any],
        room_id: str,
        room_type: str,
        available: dict[str, Any],
        occupied: list[Footprint],
        layout: object,
        rng: random.Random,
        summary: dict[str, Any],
        usability: LayoutUsabilityConfig,
    ) -> GroupPlacement | None:
        members = self._select_group_members(group_def, room_type, available, rng, summary)
        if members is None or len(members) < 2:
            return None
        return self._try_place_semantic_group_members(
            group_def=group_def,
            room_id=room_id,
            members=members,
            occupied=occupied,
            layout=layout,
            rng=rng,
            summary=summary,
            usability=usability,
        )

    def _try_place_semantic_group_members(
        self,
        *,
        group_def: dict[str, Any],
        room_id: str,
        members: list[GroupMemberPlacement],
        occupied: list[Footprint],
        layout: object,
        rng: random.Random,
        summary: dict[str, Any],
        usability: LayoutUsabilityConfig,
    ) -> GroupPlacement | None:
        footprint_spec = self._group_footprint_spec(members)
        policy = self.rules.placement_policy
        anchors = [str(anchor) for anchor in group_def.get("anchors") or ["in_middle"]]
        rng.shuffle(anchors)
        for anchor in anchors:
            placed = sample_anchor_pose(
                layout=layout,
                room_id=room_id,
                extent_xy=footprint_spec.extent_xy,
                anchor=anchor,
                occupied=occupied,
                rng=rng,
                clearance_m=float(policy.get("large_rewrite_clearance_m", policy.get("large_clearance_m", 0.18))),
                edge_band_m=float(policy.get("edge_band_m", 0.75)),
                corner_band_m=float(policy.get("corner_band_m", 0.9)),
                max_attempts=int(policy.get("max_candidate_attempts", 180)),
            )
            if placed is None:
                continue
            footprint_center_xy, yaw, fp = placed
            anchor_xy = _group_anchor_from_footprint_center(footprint_center_xy, yaw, footprint_spec.center_offset_xy)
            if not self._group_members_fit(layout, room_id, anchor_xy, yaw, members, occupied, summary, usability):
                continue
            return GroupPlacement(
                group_id=str(group_def.get("id") or "semantic_group"),
                anchor_xy=anchor_xy,
                yaw=yaw,
                footprint=Footprint(
                    name="semantic_group",
                    category="semantic_group",
                    room=room_id,
                    center_xy=footprint_center_xy,
                    extent_xy=fp.extent_xy,
                    yaw=yaw,
                    source="semantic_group",
                ),
                members=tuple(members),
            )
        fallback = self._sample_group_member_fit_pose(
            layout=layout,
            room_id=room_id,
            footprint_spec=footprint_spec,
            members=members,
            occupied=occupied,
            rng=rng,
            summary=summary,
            usability=usability,
        )
        if fallback is not None:
            anchor_xy, yaw, fp = fallback
            return GroupPlacement(
                group_id=str(group_def.get("id") or "semantic_group"),
                anchor_xy=anchor_xy,
                yaw=yaw,
                footprint=fp,
                members=tuple(members),
            )
        return None

    def _sample_group_member_fit_pose(
        self,
        *,
        layout: object,
        room_id: str,
        footprint_spec: GroupFootprintSpec,
        members: list[GroupMemberPlacement],
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any],
        usability: LayoutUsabilityConfig,
    ) -> tuple[tuple[float, float], float, Footprint] | None:
        policy = self.rules.placement_policy
        clearance = float(policy.get("large_rewrite_clearance_m", policy.get("large_clearance_m", 0.18)))
        free = room_free_mask_crop(layout, room_id, occupied, clearance_m=clearance)
        if free is None:
            return None
        center_mask = free.free & traversable_mask_crop(layout, free.room_slice, free.free.shape)
        if not np.any(center_mask):
            center_mask = free.free
        rows, cols = np.where(center_mask)
        if rows.size == 0:
            return None
        sample_count = min(int(rows.size), int(policy.get("max_candidate_attempts", 64)) * 8)
        indices = rng.sample(range(int(rows.size)), sample_count) if sample_count < int(rows.size) else list(range(int(rows.size)))
        rng.shuffle(indices)
        for idx in indices:
            xy = layout.map_to_world((int(rows[idx]) + free.row_offset, int(cols[idx]) + free.col_offset))
            yaws = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
            rng.shuffle(yaws)
            for yaw in yaws:
                candidate_extent = footprint_extent_for_yaw(footprint_spec.extent_xy, yaw)
                if not candidate_fits(layout, room_id, free, xy, candidate_extent, clearance_m=clearance):
                    continue
                anchor_xy = _group_anchor_from_footprint_center(xy, yaw, footprint_spec.center_offset_xy)
                if not self._group_members_fit(layout, room_id, anchor_xy, yaw, members, occupied, summary, usability):
                    continue
                fp = Footprint(
                    name="semantic_group",
                    category="semantic_group",
                    room=room_id,
                    center_xy=(float(xy[0]), float(xy[1])),
                    extent_xy=candidate_extent,
                    yaw=yaw,
                    source="semantic_group",
                )
                return anchor_xy, yaw, fp
        return None

    def _select_group_members(
        self,
        group_def: dict[str, Any],
        room_type: str,
        available: dict[str, Any],
        rng: random.Random,
        summary: dict[str, Any] | None = None,
    ) -> list[GroupMemberPlacement] | None:
        used: set[str] = set()
        members: list[GroupMemberPlacement] = []
        allowed_in_room = self._allowed_large_categories_for_room(room_type)
        for role in group_def.get("roles") or []:
            if not isinstance(role, dict):
                continue
            categories = [str(category) for category in role.get("categories") or [] if str(category) in allowed_in_room]
            if bool((summary or {}).get("_force_compact_group_models", False)):
                expanded: list[str] = []
                for category in categories:
                    expanded.append(category)
                    expanded.extend(
                        candidate
                        for candidate in self._compatible_categories_for(category)
                        if candidate in allowed_in_room and self._is_spawnable_large(candidate)
                    )
                categories = _dedupe_preserve_order(expanded)
                rng.shuffle(categories)
            if not categories:
                if role.get("required", False):
                    return None
                continue
            record = self._pick_available_record_for_role(categories, available, used, rng)
            if record is None:
                if role.get("required", False):
                    return None
                continue
            replacement_category = self._pick_semantic_replacement_category(record.category, categories, room_type, rng, summary)
            asset = self._sample_group_model(record, replacement_category, rng, summary)
            if asset is None:
                if role.get("required", False):
                    return None
                continue
            used.add(record.name)
            offset = role.get("offset") or [0.0, 0.0]
            members.append(
                GroupMemberPlacement(
                    record=record,
                    role_id=str(role.get("id") or replacement_category),
                    category=replacement_category,
                    asset=asset,
                    scale=record.scale,
                    offset_xy=(float(offset[0]), float(offset[1])),
                    relative_yaw=float(role.get("yaw", 0.0)),
                    fixed_base=record.fixed_base,
                    visual_only=record.visual_only,
                )
            )
        return members

    def _pick_available_record_for_role(self, categories: list[str], available: dict[str, Any], used: set[str], rng: random.Random) -> Any | None:
        direct = [record for record in available.values() if record.name not in used and record.category in categories]
        if direct:
            direct.sort(key=lambda record: categories.index(record.category))
            return direct[0]
        compatible = []
        target_categories = set(categories)
        for record in available.values():
            if record.name in used:
                continue
            options = set(self._compatible_categories_for(record.category))
            if options & target_categories:
                compatible.append(record)
        if compatible:
            rng.shuffle(compatible)
            compatible.sort(key=lambda record: _rewrite_priority(record.category))
            return compatible[0]
        return None

    def _pick_semantic_replacement_category(
        self,
        current_category: str,
        role_categories: list[str],
        room_type: str,
        rng: random.Random,
        summary: dict[str, Any] | None = None,
    ) -> str:
        allowed = self._allowed_large_categories_for_room(room_type)
        compatible = [category for category in self._compatible_categories_for(current_category) if category in role_categories and category in allowed]
        force_compact = bool((summary or {}).get("_force_compact_group_models", False))
        if current_category in role_categories and current_category in allowed and not force_compact:
            return current_category
        if compatible:
            return rng.choice(compatible)
        choices = [category for category in role_categories if category in allowed and self._is_spawnable_large(category)]
        return rng.choice(choices) if choices else str(current_category)

    def _compatible_categories_for(self, category: str) -> list[str]:
        compat = self.rules.semantic_groups.get("semantic_compatibility") or {}
        values = compat.get(str(category))
        if isinstance(values, list) and values:
            return [str(item) for item in values]
        return [str(category)]

    def _group_footprint_spec(self, members: list[GroupMemberPlacement]) -> GroupFootprintSpec:
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for member in members:
            bbox = scaled_bbox(member.asset, member.scale)
            extent = footprint_extent_for_yaw((bbox[0], bbox[1]), member.relative_yaw)
            ox, oy = member.offset_xy
            min_x = min(min_x, ox - extent[0] / 2.0)
            max_x = max(max_x, ox + extent[0] / 2.0)
            min_y = min(min_y, oy - extent[1] / 2.0)
            max_y = max(max_y, oy + extent[1] / 2.0)
        if not members:
            return GroupFootprintSpec(extent_xy=(0.0, 0.0), center_offset_xy=(0.0, 0.0))
        return GroupFootprintSpec(
            extent_xy=(max(0.05, max_x - min_x), max(0.05, max_y - min_y)),
            center_offset_xy=((min_x + max_x) / 2.0, (min_y + max_y) / 2.0),
        )

    def _group_members_fit(
        self,
        layout: object,
        room_id: str,
        anchor_xy: tuple[float, float],
        group_yaw: float,
        members: list[GroupMemberPlacement],
        occupied: list[Footprint],
        summary: dict[str, Any] | None = None,
        usability: LayoutUsabilityConfig | None = None,
    ) -> bool:
        policy = self.rules.placement_policy
        free = room_free_mask_crop(
            layout,
            room_id,
            occupied,
            clearance_m=float(policy.get("large_rewrite_clearance_m", policy.get("large_clearance_m", 0.18))),
        )
        if free is None:
            return False
        member_fps: list[Footprint] = []
        max_overlap = float(policy.get("max_large_overlap_ratio", 0.08))
        for member in members:
            xy = _transform_group_offset(anchor_xy, group_yaw, member.offset_xy)
            yaw = group_yaw + member.relative_yaw
            bbox = scaled_bbox(member.asset, member.scale)
            extent = footprint_extent_for_yaw((bbox[0], bbox[1]), yaw)
            if not candidate_fits(layout, room_id, free, xy, extent, clearance_m=float(policy.get("large_rewrite_clearance_m", 0.05))):
                return False
            name = member.record.name if member.record is not None else member.role_id
            fp = Footprint(
                name=name,
                category=member.category,
                room=room_id,
                center_xy=xy,
                extent_xy=extent,
                yaw=yaw,
                source="semantic_group_member",
                owner=name,
            )
            if any(aabb_overlap_ratio(fp, other) > max_overlap for other in member_fps):
                return False
            if any(aabb_overlap_ratio(fp, other) > max_overlap for other in occupied if other.room == room_id):
                return False
            member_fps.append(fp)
            member_fps.extend(self._protected_zones_for_placed_object(name, member.category, room_id, member.asset, xy, yaw, member.scale))
        if usability is not None and not self._candidate_preserves_usability(layout, room_id, occupied, member_fps, usability, summary):
            return False
        return True

    def _commit_semantic_group(
        self,
        scene: dict[str, Any],
        placement: GroupPlacement,
        room_id: str,
        rng: random.Random,
        summary: dict[str, Any],
        occupied: list[Footprint],
    ) -> list[Footprint]:
        out = list(occupied)
        for member in placement.members:
            xy = _transform_group_offset(placement.anchor_xy, placement.yaw, member.offset_xy)
            yaw = placement.yaw + member.relative_yaw
            pos = (xy[0], xy[1], root_z_for_floor(member.asset, member.scale))
            ori = yaw_to_quat(yaw)
            old_name = member.record.name if member.record is not None else ""
            old_category = member.record.category if member.record is not None else ""
            old_model = member.record.model if member.record is not None else ""
            old_pos = member.record.pos if member.record is not None else (0.0, 0.0, 0.0)
            old_ori = member.record.ori if member.record is not None else (0.0, 0.0, 0.0, 1.0)
            if member.record is not None and member.category == member.record.category:
                name = member.record.name
                set_object_model(scene, name, member.asset.model)
                set_object_pose(scene, name, pos=pos, ori=ori)
                set_object_room(scene, name, room_id)
                removed_name = None
            else:
                removed_name = old_name or None
                if old_name:
                    remove_object(scene, old_name)
                name = self._unique_name(scene, member.category, member.asset.model, int(summary["seed"]), summary["rewritten_large"])
                add_dataset_object(
                    scene,
                    name=name,
                    category=member.category,
                    model=member.asset.model,
                    room=room_id,
                    pos=pos,
                    ori=ori,
                    fixed_base=member.fixed_base,
                    visual_only=member.visual_only,
                    scale=member.scale,
                )
                if old_name:
                    summary["removed_large"] += 1
                summary["category_swaps"] += 1
            bbox = scaled_bbox(member.asset, member.scale)
            out.append(
                Footprint(
                    name=name,
                    category=member.category,
                    room=room_id,
                    center_xy=xy,
                    extent_xy=footprint_extent_for_yaw((bbox[0], bbox[1]), yaw),
                    yaw=yaw,
                    source="semantic_group_member",
                    owner=name,
                )
            )
            out.extend(self._protected_zones_for_placed_object(name, member.category, room_id, member.asset, xy, yaw, member.scale))
            summary["rewritten_large"] += 1
            summary["relocated_large"] += 1
            summary["group_objects_relocated"] += 1
            if member.asset.model != old_model:
                summary["model_swaps"] += 1
            summary["operations"].append(
                {
                    "op": "rewrite_large",
                    "mode": "room_relayout",
                    "semantic_group": placement.group_id,
                    "role": member.role_id,
                    "name": name,
                    "old_name": old_name,
                    "removed_name": removed_name,
                    "category": member.category,
                    "old_category": old_category,
                    "model": member.asset.model,
                    "old_model": old_model,
                    "room": room_id,
                    "old_pos": list(old_pos),
                    "new_pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "old_ori": list(old_ori),
                    "new_ori": list(ori),
                }
            )
        return out

    def _place_room_relayout_single(
        self,
        *,
        scene: dict[str, Any],
        record: Any,
        room_id: str,
        room_type: str,
        occupied: list[Footprint],
        layout: object,
        rng: random.Random,
        summary: dict[str, Any],
        usability: LayoutUsabilityConfig,
    ) -> list[Footprint] | None:
        local_occupied = _without_object_footprints(occupied, record.name)
        replacement_category = self._sample_rewrite_category(record.category, room_type, "room_relayout", rng, summary)
        if replacement_category is None:
            return None
        replacement_asset = self._sample_rewrite_model(record, replacement_category, rng, summary)
        if replacement_asset is None:
            return None
        placed = self._sample_large_pose(
            layout=layout,
            room_id=room_id,
            category=replacement_category,
            asset=replacement_asset,
            scale=record.scale,
            occupied=local_occupied,
            rng=rng,
            summary=summary,
            usability=usability,
            clearance_m=float(self.rules.placement_policy.get("large_rewrite_clearance_m", self.rules.placement_policy.get("large_clearance_m", 0.18))),
            avoid_center_xy=(record.pos[0], record.pos[1]),
            min_center_distance_m=float(self.rules.placement_policy.get("large_jitter_translate_m", 0.18)),
        )
        if placed is None:
            placed = self._sample_local_relocation_pose(
                layout=layout,
                room_id=room_id,
                record=record,
                asset=replacement_asset,
                occupied=local_occupied,
                rng=rng,
                summary=summary,
                usability=usability,
            )
        if placed is None:
            return None
        xy, yaw, fp = placed
        pos = (xy[0], xy[1], root_z_for_floor(replacement_asset, record.scale))
        ori = yaw_to_quat(yaw)
        if replacement_category == record.category:
            name = record.name
            removed_name = None
            set_object_model(scene, name, replacement_asset.model)
            set_object_pose(scene, name, pos=pos, ori=ori)
            set_object_room(scene, name, room_id)
        else:
            remove_object(scene, record.name)
            removed_name = record.name
            name = self._unique_name(scene, replacement_category, replacement_asset.model, int(summary["seed"]), summary["rewritten_large"])
            add_dataset_object(
                scene,
                name=name,
                category=replacement_category,
                model=replacement_asset.model,
                room=room_id,
                pos=pos,
                ori=ori,
                fixed_base=record.fixed_base,
                visual_only=record.visual_only,
                scale=record.scale,
            )
            summary["removed_large"] += 1
            summary["category_swaps"] += 1
        new_fp = Footprint(
            name=name,
            category=replacement_category,
            room=room_id,
            center_xy=xy,
            extent_xy=fp.extent_xy,
            yaw=yaw,
            source="room_relayout_single",
            owner=name,
        )
        summary["rewritten_large"] += 1
        summary["relocated_large"] += 1
        if replacement_asset.model != record.model:
            summary["model_swaps"] += 1
        summary["operations"].append(
            {
                "op": "rewrite_large",
                "mode": "room_relayout",
                "name": name,
                "old_name": record.name,
                "removed_name": removed_name,
                "category": replacement_category,
                "old_category": record.category,
                "model": replacement_asset.model,
                "old_model": record.model,
                "room": room_id,
                "old_pos": list(record.pos),
                "new_pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "old_ori": list(record.ori),
                "new_ori": list(ori),
            }
        )
        return local_occupied + [new_fp] + self._protected_zones_for_placed_object(name, replacement_category, room_id, replacement_asset, xy, yaw, record.scale)

    def _keep_or_remove_room_relayout_original(
        self,
        *,
        scene: dict[str, Any],
        record: Any,
        occupied: list[Footprint],
        layout: object,
        summary: dict[str, Any],
        usability: LayoutUsabilityConfig,
        allow_remove: bool,
    ) -> RoomRelayoutFallback:
        asset = self.catalog.get_model(record.category, record.model)
        if asset is None:
            return RoomRelayoutFallback(occupied=occupied, status="missing_asset")
        local_occupied = _without_object_footprints(occupied, record.name)
        yaw = quat_to_yaw(record.ori)
        bbox = scaled_bbox(asset, record.scale)
        fp = Footprint(
            name=record.name,
            category=record.category,
            room=record.primary_room,
            center_xy=(record.pos[0], record.pos[1]),
            extent_xy=footprint_extent_for_yaw((bbox[0], bbox[1]), yaw),
            yaw=yaw,
            source="existing",
            owner=record.name,
        )
        candidate_fps = [fp] + self._protected_zones_for_placed_object(
            record.name,
            record.category,
            str(record.primary_room),
            asset,
            fp.center_xy,
            yaw,
            record.scale,
        )
        max_overlap = float(self.rules.placement_policy.get("max_large_overlap_ratio", 0.08))
        overlaps_existing = any(
            aabb_overlap_ratio(fp, other) > max_overlap
            for other in local_occupied
            if other.room == record.primary_room and other.source not in {"door_clearance", "openable_clearance"}
        )
        if (
            record.primary_room
            and not overlaps_existing
            and self._candidate_preserves_usability(layout, record.primary_room, local_occupied, candidate_fps, usability, None)
        ):
            summary["operations"].append(
                {
                    "op": "room_relayout_keep_original",
                    "name": record.name,
                    "category": record.category,
                    "model": record.model,
                    "room": record.primary_room,
                    "reason": "placement_failed_original_still_usable",
                }
            )
            _skip(summary, f"room_relayout_kept_original_{record.category}")
            return RoomRelayoutFallback(occupied=local_occupied + candidate_fps, status="kept")
        if allow_remove:
            remove_object(scene, record.name)
            summary["removed_large"] += 1
            summary["discarded_large"] += 1
            summary["operations"].append(
                {
                    "op": "remove_large",
                    "mode": "room_relayout",
                    "name": record.name,
                    "category": record.category,
                    "model": record.model,
                    "room": record.primary_room,
                    "reason": "space_budget_or_clearance",
                }
            )
            _skip(summary, f"room_relayout_removed_original_{record.category}")
            return RoomRelayoutFallback(occupied=local_occupied, status="removed")
        _skip(summary, f"room_relayout_kept_original_{record.category}")
        summary["operations"].append(
            {
                "op": "room_relayout_keep_original",
                "name": record.name,
                "category": record.category,
                "model": record.model,
                "room": record.primary_room,
                "reason": "removal_disabled_original_may_exceed_budget",
            }
        )
        return RoomRelayoutFallback(occupied=local_occupied + candidate_fps, status="kept_forced")

    def _jitter_existing_large(
        self,
        scene: dict[str, Any],
        layout: object,
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any],
        task_relevant: set[str],
        usability: LayoutUsabilityConfig,
        active_rooms: frozenset[str],
    ) -> list[Footprint]:
        probability = float(self.rules.placement_policy.get("large_jitter_probability", 0.45))
        max_translate = float(self.rules.placement_policy.get("large_jitter_translate_m", 0.18))
        max_yaw = math.radians(float(self.rules.placement_policy.get("large_jitter_yaw_deg", 10.0)))
        max_overlap = float(self.rules.placement_policy.get("max_large_overlap_ratio", 0.08))
        records = parse_scene_objects(scene)
        occupied_by_name = {fp.name: fp for fp in occupied}
        for record in records.values():
            if record.name in task_relevant or not self._is_jitter_candidate(record):
                continue
            if record.primary_room not in active_rooms:
                continue
            role = getattr(self.classifier.get(record.category), "role", "")
            if role in {"large_floor", "support_surface"} and not self._is_category_allowed_for_room(
                record.category,
                room_type_from_instance(str(record.primary_room)),
                role,
            ):
                continue
            if rng.random() > probability:
                continue
            old_fp = occupied_by_name.get(record.name)
            if old_fp is None:
                continue
            local_occupied = _without_object_footprints(occupied, record.name)
            old_yaw = old_fp.yaw
            current_asset = self.catalog.get_model(record.category, record.model)
            base_extent_xy = old_fp.extent_xy
            if current_asset is not None:
                current_bbox = scaled_bbox(current_asset, record.scale)
                base_extent_xy = (current_bbox[0], current_bbox[1])
            for _ in range(16):
                xy = (
                    record.pos[0] + rng.uniform(-max_translate, max_translate),
                    record.pos[1] + rng.uniform(-max_translate, max_translate),
                )
                yaw = old_yaw + rng.uniform(-max_yaw, max_yaw)
                extent = footprint_extent_for_yaw(base_extent_xy, yaw)
                fp = Footprint(record.name, record.category, record.primary_room, xy, extent, yaw, source="relocated_large", owner=record.name)
                if not self._large_candidate_stays_in_room(layout, str(record.primary_room), xy, extent):
                    continue
                candidate_fps = [fp]
                if current_asset is not None:
                    candidate_fps.extend(self._protected_zones_for_placed_object(record.name, record.category, record.primary_room, current_asset, xy, yaw, record.scale))
                if not self._candidate_preserves_usability(layout, str(record.primary_room), local_occupied, candidate_fps, usability, summary):
                    continue
                if any(
                    aabb_overlap_ratio(fp, other) > max_overlap
                    for other in local_occupied
                    if other.room == fp.room and other.source not in {"door_clearance", "openable_clearance"}
                ):
                    continue
                new_z = root_z_for_floor(current_asset, record.scale) if current_asset is not None else record.pos[2]
                set_object_pose(scene, record.name, pos=(xy[0], xy[1], new_z), ori=yaw_to_quat(yaw))
                occupied = local_occupied + candidate_fps
                occupied_by_name[record.name] = fp
                summary["relocated_large"] += 1
                summary["operations"].append(
                    {
                        "op": "relocate_large",
                        "name": record.name,
                        "category": record.category,
                        "model": record.model,
                        "room": record.primary_room,
                        "old_pos": list(record.pos),
                        "new_pos": [float(xy[0]), float(xy[1]), float(new_z)],
                        "old_ori": list(record.ori),
                        "new_ori": list(yaw_to_quat(yaw)),
                    }
                )
                break
        return occupied

    def _add_large_objects(
        self,
        scene: dict[str, Any],
        layout: object,
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any],
        usability: LayoutUsabilityConfig,
        active_rooms: frozenset[str],
    ) -> list[Footprint]:
        policy = self.rules.placement_policy
        max_added_per_room = int(policy.get("max_added_large_per_room", 4))
        max_overlap = float(policy.get("max_large_overlap_ratio", 0.08))
        max_room_attempts = max(6, max_added_per_room * 4)

        for room_id in sorted(layout.room_ins_name_to_ins_id):
            if room_id not in active_rooms:
                continue
            room_type = room_type_from_instance(room_id)
            density = self.rules.density_for_room_type(room_type)
            target = int(density.get("target_large", 3))
            if target <= 0:
                continue
            room_existing = [fp for fp in occupied if fp.room == room_id]
            large_count = sum(1 for fp in room_existing if self._is_density_large(fp.category))
            if large_count >= target:
                continue
            if not self._room_can_accept_more_large(layout, room_id, occupied, usability, summary):
                _skip(summary, f"space_budget_full_{room_type}")
                continue
            additions = 0
            room_attempts = 0
            while large_count < target and additions < max_added_per_room and room_attempts < max_room_attempts:
                room_attempts += 1
                category = self._sample_large_category(room_type, rng)
                if category is None:
                    _skip(summary, "no_large_category")
                    break
                asset = self._sample_model_for_room(category, room_id, layout, rng)
                if asset is None:
                    _skip(summary, f"no_model_{category}")
                    continue
                placed = self._sample_large_pose(
                    layout=layout,
                    room_id=room_id,
                    category=category,
                    asset=asset,
                    occupied=occupied,
                    rng=rng,
                    summary=summary,
                    usability=usability,
                )
                if placed is None:
                    _skip(summary, f"placement_failed_{category}")
                    continue
                xy, yaw, fp = placed
                fp = Footprint(
                    name="candidate",
                    category=category,
                    room=room_id,
                    center_xy=xy,
                    extent_xy=fp.extent_xy,
                    yaw=yaw,
                    source="generated_large",
                    owner="candidate",
                )
                if any(aabb_overlap_ratio(fp, other) > max_overlap for other in occupied if other.room == room_id):
                    _skip(summary, f"overlap_{category}")
                    continue
                name = self._unique_name(scene, category, asset.model, int(summary["seed"]), summary["added_large"])
                add_dataset_object(
                    scene,
                    name=name,
                    category=category,
                    model=asset.model,
                    room=room_id,
                    pos=(xy[0], xy[1], root_z_for_floor(asset)),
                    ori=yaw_to_quat(yaw),
                    fixed_base=False,
                    visual_only=False,
                )
                occupied.append(
                    Footprint(
                        name=name,
                        category=category,
                        room=room_id,
                        center_xy=xy,
                        extent_xy=fp.extent_xy,
                        yaw=yaw,
                        source="generated_large",
                        owner=name,
                    )
                )
                occupied.extend(self._protected_zones_for_placed_object(name, category, room_id, asset, xy, yaw))
                summary["added_large"] += 1
                additions += 1
                large_count += 1
                summary["operations"].append({"op": "add_large", "name": name, "category": category, "model": asset.model, "room": room_id})
        return occupied

    def _add_small_objects(
        self,
        scene: dict[str, Any],
        records: dict[str, Any],
        layout: object,
        rng: random.Random,
        summary: dict[str, Any],
        conservative_supports: bool = False,
        diversity_level: float = 1.0,
        active_rooms: frozenset[str] | None = None,
    ) -> None:
        policy = self.rules.receptacle_policy
        support_categories = set(str(item) for item in policy.get("support_categories") or [])
        small_surface_constraints = surface_constraints_from_policy(policy, "small_prop")
        if "support_surface_constraints" not in policy:
            small_surface_constraints = dict(small_surface_constraints)
            small_surface_constraints["required_abilities_any"] = []
        if conservative_supports:
            support_categories &= {
                "bar",
                "breakfast_table",
                "coffee_table",
                "commercial_kitchen_table",
                "console_table",
                "countertop",
                "desk",
                "nightstand",
                "pedestal_table",
                "stand",
            }
        diversity_level = _clamp(float(diversity_level), 0.0, 2.5)
        max_per_parent = max(
            1,
            int(round(float(self.rules.placement_policy.get("max_added_small_per_receptacle", 5)) * max(0.6, min(1.5, diversity_level)))),
        )
        max_per_room = max(
            1,
            int(round(float(self.rules.placement_policy.get("max_added_small_per_room", 16)) * max(0.6, min(1.35, diversity_level)))),
        )
        small_clearance = max(0.0, float(self.rules.placement_policy.get("small_clearance_m", 0.015)))
        small_surface_margin = max(0.0, float(self.rules.placement_policy.get("small_surface_margin_m", 0.04)))
        small_max_overlap = max(0.0, float(self.rules.placement_policy.get("small_max_overlap_ratio", 0.0)))
        small_min_center_distance = max(0.0, float(self.rules.placement_policy.get("small_min_center_distance_m", 0.03)))
        attempts_per_child = max(1, int(self.rules.placement_policy.get("small_placement_attempts_per_child", 32)))
        summary.setdefault("small_collision_rejections", 0)
        summary.setdefault("small_placement_failures", 0)
        added_by_room: dict[str, int] = {}
        supports = [record for record in records.values() if record.category in support_categories]
        rng.shuffle(supports)
        small_occupied = small_object_aabbs_from_scene(scene, self.catalog, self.classifier)
        protected_zones = protected_zones_from_scene(
            scene,
            self.catalog,
            self.classifier,
            policy=self.rules.placement_policy,
        )
        for support in supports:
            if not support.rooms:
                continue
            room = support.rooms[0]
            if active_rooms is not None and room not in active_rooms:
                continue
            room_type = room_type_from_instance(room)
            target_room_small = int(self.rules.density_for_room_type(room_type).get("target_small", 4))
            target_room_small = max(1, int(round(target_room_small * max(0.75, min(1.35, diversity_level)))))
            if added_by_room.get(room, 0) >= min(max_per_room, target_room_small):
                continue
            parent_asset = self.catalog.get_model(support.category, support.model)
            if parent_asset is None:
                continue
            parent_bbox = scaled_bbox(parent_asset, support.scale)
            room_free = room_free_mask_crop(layout, room, [], clearance_m=0.0)
            surface_z = bbox_top_z(support.pos[2], parent_asset, support.scale)
            parent_yaw = quat_to_yaw(support.ori)
            parent_extent_xy = footprint_extent_for_yaw((parent_bbox[0], parent_bbox[1]), parent_yaw)
            descriptor = support_surface_descriptor_from_values(
                category=support.category,
                center=(support.pos[0], support.pos[1], support.pos[2]),
                extent=(parent_extent_xy[0], parent_extent_xy[1], parent_bbox[2]),
                surface_z=surface_z,
            )
            if not support_matches_surface_constraints(descriptor, small_surface_constraints):
                _skip(summary, f"small_support_surface_constraint_{support.category}")
                continue
            support_occupants = [
                item
                for item in small_occupied
                if self._small_aabb_on_support_top(
                    item,
                    support=support,
                    parent_asset=parent_asset,
                    surface_z=surface_z,
                    margin=small_surface_margin,
                    z_tolerance=0.08,
                )
            ]
            children = list(policy.get("children_by_support", {}).get(support.category) or policy.get("fallback_children") or [])
            children = self._filter_small_categories_for_room(room_type, children)
            rng.shuffle(children)
            if diversity_level > 1.15 and policy.get("fallback_children"):
                extra = self._filter_small_categories_for_room(room_type, [str(item) for item in policy.get("fallback_children") or []])
                rng.shuffle(extra)
                children = _dedupe_preserve_order(children + extra[: max(1, int(round(diversity_level)))])
            parent_added = 0
            for child_category in children:
                if parent_added >= max_per_parent or added_by_room.get(room, 0) >= min(max_per_room, target_room_small):
                    break
                if getattr(self.classifier.get(child_category), "role", "") != "small_prop":
                    _skip(summary, f"small_category_not_small_prop_{child_category}")
                    continue
                child_asset = self._sample_small_model(child_category, rng)
                if child_asset is None:
                    continue
                child_bbox = scaled_bbox(child_asset)
                if child_bbox[0] > parent_bbox[0] * 0.65 or child_bbox[1] > parent_bbox[1] * 0.65:
                    continue
                name = self._unique_name(scene, child_category, child_asset.model, int(summary["seed"]), summary["added_small"])
                placed: dict[str, Any] | None = None
                failure_counts: dict[str, int] = {}
                for attempt_index in range(attempts_per_child):
                    offset_x = rng.uniform(-0.28, 0.28) * parent_bbox[0]
                    offset_y = rng.uniform(-0.28, 0.28) * parent_bbox[1]
                    x = support.pos[0] + offset_x
                    y = support.pos[1] + offset_y
                    if not self._point_in_room_xy(layout, room, (x, y)):
                        failure_counts["outside_room_point"] = failure_counts.get("outside_room_point", 0) + 1
                        continue
                    z = root_z_for_bbox_bottom(child_asset, surface_z + small_clearance)
                    yaw = rng.choice([0.0, 1.57079632679, 3.14159265359, -1.57079632679])
                    small_extent = footprint_extent_for_yaw((child_bbox[0], child_bbox[1]), yaw)
                    small_fp = Footprint(name, child_category, room, (x, y), small_extent, yaw, source="added_small", owner=name)
                    if room_free is not None and not candidate_fits(layout, room, room_free, (x, y), small_extent, clearance_m=0.0):
                        failure_counts["footprint_outside_room"] = failure_counts.get("footprint_outside_room", 0) + 1
                        continue
                    if any(aabb_overlap_ratio(small_fp, zone) > 0.0 for zone in protected_zones if zone.room == room):
                        failure_counts["protected_zone_overlap"] = failure_counts.get("protected_zone_overlap", 0) + 1
                        continue
                    candidate_aabb = object_aabb_from_pose(
                        name=name,
                        category=child_category,
                        room=room,
                        pos=(x, y, z),
                        yaw=yaw,
                        asset=child_asset,
                        source="added_small",
                        owner=name,
                        parent=support.name,
                    )
                    if not self._small_aabb_on_support_top(
                        candidate_aabb,
                        support=support,
                        parent_asset=parent_asset,
                        surface_z=surface_z,
                        margin=small_surface_margin,
                        z_tolerance=0.08,
                    ):
                        failure_counts["outside_support_top"] = failure_counts.get("outside_support_top", 0) + 1
                        continue
                    blocker = self._small_collision_blocker(
                        candidate_aabb,
                        small_occupied,
                        support_occupants,
                        clearance=small_clearance,
                        max_overlap_ratio=small_max_overlap,
                        min_center_distance=small_min_center_distance,
                    )
                    if blocker is not None:
                        reason, _item = blocker
                        failure_counts[reason] = failure_counts.get(reason, 0) + 1
                        summary["small_collision_rejections"] += 1
                        continue
                    placed = {
                        "x": x,
                        "y": y,
                        "z": z,
                        "yaw": yaw,
                        "offset": [float(offset_x), float(offset_y), float(z - support.pos[2])],
                        "attempt_index": attempt_index,
                        "aabb": candidate_aabb,
                    }
                    break
                if placed is None:
                    summary["small_placement_failures"] += 1
                    if failure_counts:
                        reason = max(sorted(failure_counts), key=lambda key: failure_counts[key])
                        _skip(summary, f"small_placement_failed_{child_category}_{reason}")
                    else:
                        _skip(summary, f"small_placement_failed_{child_category}")
                    continue
                add_dataset_object(
                    scene,
                    name=name,
                    category=child_category,
                    model=child_asset.model,
                    room=room,
                    pos=(placed["x"], placed["y"], placed["z"]),
                    ori=yaw_to_quat(placed["yaw"]),
                    fixed_base=False,
                    visual_only=False,
                )
                small_occupied.append(placed["aabb"])
                support_occupants.append(placed["aabb"])
                summary["added_small"] += 1
                parent_added += 1
                added_by_room[room] = added_by_room.get(room, 0) + 1
                summary["operations"].append(
                    {
                        "op": "add_small",
                        "name": name,
                        "category": child_category,
                        "model": child_asset.model,
                        "room": room,
                        "parent": support.name,
                        "surface_z": float(surface_z),
                        "offset": placed["offset"],
                        "yaw": float(placed["yaw"]),
                        "attempt_index": int(placed["attempt_index"]),
                        "collision_checked": True,
                    }
                )

    def _small_aabb_on_support_top(
        self,
        item: ObjectAABB,
        *,
        support: Any,
        parent_asset: ModelAsset,
        surface_z: float,
        margin: float,
        z_tolerance: float,
    ) -> bool:
        parent_bbox = scaled_bbox(parent_asset, support.scale)
        parent_yaw = quat_to_yaw(support.ori)
        parent_extent = footprint_extent_for_yaw((parent_bbox[0], parent_bbox[1]), parent_yaw)
        allowed_x = max(float(parent_extent[0]) * 0.5 - float(margin), 0.0)
        allowed_y = max(float(parent_extent[1]) * 0.5 - float(margin), 0.0)
        half_x = float(item.extent_xyz[0]) * 0.5
        half_y = float(item.extent_xyz[1]) * 0.5
        if half_x > allowed_x or half_y > allowed_y:
            return False
        if abs(float(item.center_xyz[0]) - float(support.pos[0])) + half_x > allowed_x:
            return False
        if abs(float(item.center_xyz[1]) - float(support.pos[1])) + half_y > allowed_y:
            return False
        bottom_z = float(item.center_xyz[2]) - float(item.extent_xyz[2]) * 0.5
        return abs(bottom_z - float(surface_z)) <= float(z_tolerance)

    def _small_collision_blocker(
        self,
        candidate: ObjectAABB,
        small_occupied: list[ObjectAABB],
        support_occupants: list[ObjectAABB],
        *,
        clearance: float,
        max_overlap_ratio: float,
        min_center_distance: float,
    ) -> tuple[str, ObjectAABB] | None:
        for item in small_occupied:
            if candidate.room and item.room and candidate.room != item.room:
                continue
            if candidate.owner and item.owner and candidate.owner == item.owner:
                continue
            if aabb3d_overlaps(candidate, item, clearance=clearance, max_overlap_ratio=max_overlap_ratio):
                return "aabb_overlap", item
        if min_center_distance > 0.0:
            for item in support_occupants:
                if candidate.owner and item.owner and candidate.owner == item.owner:
                    continue
                if xy_center_distance(candidate, item) < float(min_center_distance):
                    return "center_too_close", item
        return None

    def _repair_small_object_collisions(
        self,
        scene: dict[str, Any],
        layout: object,
        rng: random.Random,
        summary: dict[str, Any],
        *,
        task_relevant: set[str],
        active_rooms: frozenset[str],
    ) -> None:
        max_passes = max(1, int(self.rules.placement_policy.get("small_repair_max_passes", 12)))
        max_overlap = max(0.0, float(self.rules.placement_policy.get("small_max_overlap_ratio", 0.0)))
        unresolved_seen: set[tuple[str, str]] = set()
        summary.setdefault("small_collision_repairs", 0)
        summary.setdefault("relocated_small", 0)
        summary.setdefault("removed_small", 0)
        summary.setdefault("small_collision_unresolved", 0)

        for _pass_index in range(max_passes):
            records = parse_scene_objects(scene)
            operation_by_name, parent_by_name, op_index = _small_operation_maps(summary)
            aabbs = self._small_aabbs_for_repair(records, operation_by_name, parent_by_name, task_relevant)
            pairs = self._small_collision_pairs(aabbs, max_overlap=max_overlap)
            if not pairs:
                return
            repaired = False
            for ratio, left, right in pairs:
                pair_key = tuple(sorted((left.name, right.name)))
                target_name = self._small_repair_target(left, right, operation_by_name, task_relevant, op_index)
                if target_name is None:
                    unresolved_seen.add(pair_key)
                    continue
                target = records.get(target_name)
                if target is None:
                    continue
                other = right if left.name == target_name else left
                relocated = self._relocate_added_small_object(
                    scene,
                    records,
                    layout,
                    target,
                    parent_by_name.get(target_name),
                    rng,
                    active_rooms=active_rooms,
                    task_relevant=task_relevant,
                )
                if relocated is not None:
                    summary["small_collision_repairs"] += 1
                    summary["relocated_small"] += 1
                    summary["operations"].append(
                        {
                            "op": "relocate_small",
                            "name": target.name,
                            "category": target.category,
                            "model": target.model,
                            "room": relocated["room"],
                            "parent": relocated.get("parent"),
                            "old_pos": list(target.pos),
                            "new_pos": list(relocated["pos"]),
                            "surface_z": relocated.get("surface_z"),
                            "offset": relocated.get("offset"),
                            "yaw": relocated.get("yaw"),
                            "attempt_index": relocated.get("attempt_index"),
                            "collision_checked": True,
                        }
                    )
                    summary["operations"].append(
                        {
                            "op": "small_collision_repair",
                            "action": "relocate",
                            "name": target.name,
                            "overlapped_with": other.name,
                            "ratio": round(float(ratio), 4),
                        }
                    )
                else:
                    remove_object(scene, target.name)
                    summary["small_collision_repairs"] += 1
                    summary["removed_small"] += 1
                    summary["operations"].append(
                        {
                            "op": "remove_small",
                            "name": target.name,
                            "category": target.category,
                            "model": target.model,
                            "room": target.primary_room,
                            "parent": parent_by_name.get(target.name),
                            "reason": "small_collision_unrepairable",
                        }
                    )
                    summary["operations"].append(
                        {
                            "op": "small_collision_repair",
                            "action": "remove",
                            "name": target.name,
                            "overlapped_with": other.name,
                            "ratio": round(float(ratio), 4),
                        }
                    )
                repaired = True
                break
            if not repaired:
                break

        if unresolved_seen:
            summary["small_collision_unresolved"] += len(unresolved_seen)
            for left, right in sorted(unresolved_seen):
                _skip(summary, f"small_collision_unresolved_{left}_{right}")

    def _small_aabbs_for_repair(
        self,
        records: dict[str, Any],
        operation_by_name: dict[str, str],
        parent_by_name: dict[str, str],
        task_relevant: set[str],
    ) -> list[ObjectAABB]:
        out: list[ObjectAABB] = []
        for record in records.values():
            role = getattr(self.classifier.get(record.category), "role", "")
            if role != "small_prop":
                continue
            asset = self.catalog.get_model(record.category, record.model)
            if asset is None:
                continue
            out.append(
                object_aabb_from_record(
                    record,
                    asset,
                    source=operation_by_name.get(record.name, "existing"),
                    owner=record.name,
                    parent=parent_by_name.get(record.name),
                    task_relevant=record.name in task_relevant,
                )
            )
        return out

    def _small_collision_pairs(
        self,
        aabbs: list[ObjectAABB],
        *,
        max_overlap: float,
    ) -> list[tuple[float, ObjectAABB, ObjectAABB]]:
        pairs: list[tuple[float, ObjectAABB, ObjectAABB]] = []
        for i, left in enumerate(aabbs):
            for right in aabbs[i + 1 :]:
                if left.room and right.room and left.room != right.room:
                    continue
                ratio = aabb3d_overlap_ratio(left, right)
                if ratio > float(max_overlap):
                    pairs.append((ratio, left, right))
        pairs.sort(key=lambda item: (-float(item[0]), item[1].name, item[2].name))
        return pairs

    def _small_repair_target(
        self,
        left: ObjectAABB,
        right: ObjectAABB,
        operation_by_name: dict[str, str],
        task_relevant: set[str],
        op_index: dict[str, int],
    ) -> str | None:
        candidates = []
        for item in (left, right):
            if item.name in task_relevant:
                continue
            if operation_by_name.get(item.name) == "added_small":
                candidates.append(item.name)
        if not candidates:
            return None
        return max(candidates, key=lambda name: int(op_index.get(name, -1)))

    def _relocate_added_small_object(
        self,
        scene: dict[str, Any],
        records: dict[str, Any],
        layout: object,
        record: Any,
        parent_name: str | None,
        rng: random.Random,
        *,
        active_rooms: frozenset[str],
        task_relevant: set[str],
    ) -> dict[str, Any] | None:
        parent = records.get(str(parent_name or ""))
        if parent is None or not parent.primary_room or parent.primary_room not in active_rooms:
            return None
        parent_asset = self.catalog.get_model(parent.category, parent.model)
        child_asset = self.catalog.get_model(record.category, record.model)
        if parent_asset is None or child_asset is None:
            return None
        room = str(parent.primary_room)
        parent_bbox = scaled_bbox(parent_asset, parent.scale)
        child_bbox = scaled_bbox(child_asset, record.scale)
        if child_bbox[0] > parent_bbox[0] * 0.65 or child_bbox[1] > parent_bbox[1] * 0.65:
            return None

        small_clearance = max(0.0, float(self.rules.placement_policy.get("small_clearance_m", 0.015)))
        small_surface_margin = max(0.0, float(self.rules.placement_policy.get("small_surface_margin_m", 0.04)))
        small_max_overlap = max(0.0, float(self.rules.placement_policy.get("small_max_overlap_ratio", 0.0)))
        small_min_center_distance = max(0.0, float(self.rules.placement_policy.get("small_min_center_distance_m", 0.03)))
        attempts = max(1, int(self.rules.placement_policy.get("small_repair_max_attempts", 48)))
        room_free = room_free_mask_crop(layout, room, [], clearance_m=0.0)
        protected_zones = protected_zones_from_scene(scene, self.catalog, self.classifier, policy=self.rules.placement_policy)
        occupied = [
            item
            for item in self._small_aabbs_for_repair(records, {}, {}, task_relevant)
            if item.name != record.name
        ]
        surface_z = bbox_top_z(parent.pos[2], parent_asset, parent.scale)
        support_occupants = [
            item
            for item in occupied
            if self._small_aabb_on_support_top(
                item,
                support=parent,
                parent_asset=parent_asset,
                surface_z=surface_z,
                margin=small_surface_margin,
                z_tolerance=0.08,
            )
        ]
        for attempt_index in range(attempts):
            offset_x = rng.uniform(-0.28, 0.28) * parent_bbox[0]
            offset_y = rng.uniform(-0.28, 0.28) * parent_bbox[1]
            x = float(parent.pos[0]) + offset_x
            y = float(parent.pos[1]) + offset_y
            if not self._point_in_room_xy(layout, room, (x, y)):
                continue
            z = root_z_for_bbox_bottom(child_asset, surface_z + small_clearance, record.scale)
            yaw = rng.choice([0.0, 1.57079632679, 3.14159265359, -1.57079632679])
            small_extent = footprint_extent_for_yaw((child_bbox[0], child_bbox[1]), yaw)
            small_fp = Footprint(record.name, record.category, room, (x, y), small_extent, yaw, source="repair_small", owner=record.name)
            if room_free is not None and not candidate_fits(layout, room, room_free, (x, y), small_extent, clearance_m=0.0):
                continue
            if any(aabb_overlap_ratio(small_fp, zone) > 0.0 for zone in protected_zones if zone.room == room):
                continue
            candidate_aabb = object_aabb_from_pose(
                name=record.name,
                category=record.category,
                room=room,
                pos=(x, y, z),
                yaw=yaw,
                asset=child_asset,
                scale=record.scale,
                source="relocated_small",
                owner=record.name,
                parent=parent.name,
                task_relevant=record.name in task_relevant,
            )
            if not self._small_aabb_on_support_top(
                candidate_aabb,
                support=parent,
                parent_asset=parent_asset,
                surface_z=surface_z,
                margin=small_surface_margin,
                z_tolerance=0.08,
            ):
                continue
            blocker = self._small_collision_blocker(
                candidate_aabb,
                occupied,
                support_occupants,
                clearance=small_clearance,
                max_overlap_ratio=small_max_overlap,
                min_center_distance=small_min_center_distance,
            )
            if blocker is not None:
                continue
            set_object_pose(scene, record.name, pos=(x, y, z), ori=yaw_to_quat(yaw))
            set_object_room(scene, record.name, room)
            return {
                "room": room,
                "parent": parent.name,
                "pos": (x, y, z),
                "surface_z": float(surface_z),
                "offset": [float(offset_x), float(offset_y), float(z - parent.pos[2])],
                "yaw": float(yaw),
                "attempt_index": int(attempt_index),
            }
        return None

    def _point_in_room_xy(self, layout: object, room_id: str, xy: tuple[float, float]) -> bool:
        room = layout.room_mask(room_id)
        if room is None:
            return True
        row, col = layout.world_to_map((float(xy[0]), float(xy[1])))
        if row < 0 or col < 0 or row >= room.shape[0] or col >= room.shape[1]:
            return False
        return bool(room[int(row), int(col)])

    def _apply_stationary_category_swaps(
        self,
        scene: dict[str, Any],
        layout: object,
        rng: random.Random,
        summary: dict[str, Any],
        usability: LayoutUsabilityConfig,
        active_rooms: frozenset[str],
        config: VariantConfig,
    ) -> None:
        probability = _clamp(
            float(config.stationary_category_swap_probability) * max(0.0, float(summary.get("diversity_level", 1.0))),
            0.0,
            0.95,
        )
        if probability <= 0:
            return
        touched_names = {
            str(value)
            for op in summary.get("operations", [])
            if isinstance(op, dict)
            for value in (op.get("name"), op.get("old_name"), op.get("removed_name"))
            if value
        }
        protected_categories = set(str(item) for item in self.rules.placement_policy.get("rewrite_protected_categories") or [])
        occupied = existing_footprints(
            scene,
            self.catalog,
            self.classifier,
            include_roles={"large_floor", "support_surface", "fixture"},
        )
        occupied.extend(
            protected_zones_from_scene(
                scene,
                self.catalog,
                self.classifier,
                policy=self.rules.placement_policy,
            )
        )
        records = list(parse_scene_objects(scene).values())
        rng.shuffle(records)
        for record in records:
            if record.name in touched_names or record.primary_room not in active_rooms:
                continue
            if record.fixed_base or record.category in protected_categories:
                continue
            role = getattr(self.classifier.get(record.category), "role", "")
            if role not in {"large_floor", "support_surface"}:
                continue
            if rng.random() > probability:
                continue
            room_id = str(record.primary_room)
            room_type = room_type_from_instance(room_id)
            category = self._sample_stationary_category(record.category, room_type, rng)
            if category is None or category == record.category:
                continue
            asset = self._sample_stationary_category_model(record, category, rng, summary)
            if asset is None:
                _skip(summary, f"stationary_category_no_model_{category}")
                continue
            yaw = quat_to_yaw(record.ori)
            bbox = scaled_bbox(asset, record.scale)
            candidate_fp = Footprint(
                name=record.name,
                category=category,
                room=room_id,
                center_xy=(record.pos[0], record.pos[1]),
                extent_xy=footprint_extent_for_yaw((bbox[0], bbox[1]), yaw),
                yaw=yaw,
                source="stationary_category_swap",
                owner=record.name,
            )
            local_occupied = _without_object_footprints(occupied, record.name)
            if not self._large_candidate_stays_in_room(layout, room_id, candidate_fp.center_xy, candidate_fp.extent_xy):
                _skip(summary, f"stationary_category_outside_room_{category}")
                continue
            candidate_fps = [candidate_fp] + self._protected_zones_for_placed_object(
                record.name,
                category,
                room_id,
                asset,
                candidate_fp.center_xy,
                yaw,
                record.scale,
            )
            if not self._candidate_preserves_usability(layout, room_id, local_occupied, candidate_fps, usability, summary):
                _skip(summary, f"stationary_category_safety_rejected_{category}")
                continue
            max_overlap = float(self.rules.placement_policy.get("max_large_overlap_ratio", 0.08))
            if any(aabb_overlap_ratio(candidate_fp, other) > max_overlap for other in local_occupied if other.room == room_id):
                _skip(summary, f"stationary_category_overlap_{category}")
                continue
            old_name = record.name
            old_category = record.category
            old_model = record.model
            old_asset = self.catalog.get_model(record.category, record.model)
            if old_asset is None:
                _skip(summary, f"stationary_category_missing_old_model_{record.category}")
                continue
            old_bottom_z = bbox_bottom_z(record.pos[2], old_asset, record.scale)
            remove_object(scene, old_name)
            new_name = self._unique_name(scene, category, asset.model, int(summary["seed"]), summary["stationary_category_swaps"])
            add_dataset_object(
                scene,
                name=new_name,
                category=category,
                model=asset.model,
                room=room_id,
                pos=(record.pos[0], record.pos[1], root_z_for_bbox_bottom(asset, old_bottom_z, record.scale)),
                ori=record.ori,
                fixed_base=record.fixed_base,
                visual_only=record.visual_only,
                scale=record.scale,
            )
            occupied = local_occupied + [
                Footprint(
                    name=new_name,
                    category=category,
                    room=room_id,
                    center_xy=candidate_fp.center_xy,
                    extent_xy=candidate_fp.extent_xy,
                    yaw=yaw,
                    source="stationary_category_swap",
                    owner=new_name,
                )
            ] + self._protected_zones_for_placed_object(new_name, category, room_id, asset, candidate_fp.center_xy, yaw, record.scale)
            summary["removed_large"] += 1
            summary["category_swaps"] += 1
            summary["stationary_category_swaps"] += 1
            summary["model_swaps"] += 1
            summary["operations"].append(
                {
                    "op": "stationary_category_swap",
                    "name": new_name,
                    "old_name": old_name,
                    "removed_name": old_name,
                    "category": category,
                    "old_category": old_category,
                    "model": asset.model,
                    "old_model": old_model,
                    "room": room_id,
                    "pos": list(record.pos),
                }
            )

    def _sample_stationary_category(self, current_category: str, room_type: str, rng: random.Random) -> str | None:
        allowed = self._allowed_large_categories_for_room(room_type)
        compatible = [
            category
            for category in self._compatible_categories_for(current_category)
            if category != current_category and category in allowed and self._is_spawnable_large(category)
        ]
        if compatible:
            return rng.choice(compatible)
        fallback = [
            category
            for category in sorted(allowed)
            if category != current_category and self._is_spawnable_large(category)
        ]
        return rng.choice(fallback) if fallback else None

    def _sample_stationary_category_model(self, record: Any, category: str, rng: random.Random, summary: dict[str, Any]) -> ModelAsset | None:
        current = self.catalog.get_model(record.category, record.model)
        current_bbox = scaled_bbox(current, record.scale) if current is not None else None
        candidates = []
        for asset in self.catalog.get_models(category):
            if asset.joint_count != 0:
                continue
            if current_bbox is not None and _relative_bbox_delta(current_bbox, scaled_bbox(asset, record.scale)) > 0.35:
                continue
            candidates.append(asset)
        return self._choose_model_candidate(candidates, current_bbox, record.scale, rng, summary) if candidates else None

    def _apply_model_swaps(
        self,
        scene: dict[str, Any],
        layout: object,
        rng: random.Random,
        summary: dict[str, Any],
        usability: LayoutUsabilityConfig,
        active_rooms: frozenset[str],
    ) -> None:
        probability = self._effective_probability(float(self.rules.placement_policy.get("model_swap_probability", 0.0)), summary, max_value=0.35)
        if probability <= 0:
            return
        protected_names = {
            str(op.get("name"))
            for op in summary.get("operations", [])
            if isinstance(op, dict) and op.get("name") and op.get("op") in {"rewrite_large", "relocate_large", "add_large"}
        }
        protected_categories = set(str(item) for item in self.rules.placement_policy.get("rewrite_protected_categories") or [])
        occupied = existing_footprints(
            scene,
            self.catalog,
            self.classifier,
            include_roles={"large_floor", "support_surface", "fixture"},
        )
        occupied.extend(
            protected_zones_from_scene(
                scene,
                self.catalog,
                self.classifier,
                policy=self.rules.placement_policy,
            )
        )
        for record in parse_scene_objects(scene).values():
            if record.name in protected_names:
                continue
            if record.primary_room not in active_rooms:
                continue
            role = getattr(self.classifier.get(record.category), "role", "")
            if role not in {"large_floor", "support_surface", "small_prop"}:
                continue
            if record.fixed_base or record.category in protected_categories:
                continue
            if not self._is_model_swap_allowed_in_rooms(record, role):
                continue
            if rng.random() > probability:
                continue
            current = self.catalog.get_model(record.category, record.model)
            if current is None:
                continue
            candidates = []
            current_bbox = scaled_bbox(current, record.scale)
            target_joint_count = _record_joint_count(record, current)
            for asset in self.catalog.get_models(record.category):
                if asset.model == record.model:
                    continue
                if asset.joint_count != target_joint_count:
                    continue
                bbox = scaled_bbox(asset, record.scale)
                if _relative_bbox_delta(current_bbox, bbox) <= 0.35:
                    candidates.append(asset)
            if not candidates:
                continue
            chosen = self._choose_model_candidate(candidates, current_bbox, record.scale, rng, summary)
            old_model = record.model
            old_bottom_z = bbox_bottom_z(record.pos[2], current, record.scale)
            yaw = quat_to_yaw(record.ori)
            chosen_bbox = scaled_bbox(chosen, record.scale)
            candidate_fp = Footprint(
                name=record.name,
                category=record.category,
                room=record.primary_room,
                center_xy=(record.pos[0], record.pos[1]),
                extent_xy=footprint_extent_for_yaw((chosen_bbox[0], chosen_bbox[1]), yaw),
                yaw=yaw,
                source="model_swap",
                owner=record.name,
            )
            local_occupied = _without_object_footprints(occupied, record.name)
            if role in {"large_floor", "support_surface"} and record.primary_room and not self._large_candidate_stays_in_room(
                layout,
                str(record.primary_room),
                candidate_fp.center_xy,
                candidate_fp.extent_xy,
            ):
                summary["model_swap_safety_rejections"] += 1
                _skip(summary, f"model_swap_outside_room_{record.category}")
                continue
            candidate_fps = [candidate_fp] + self._protected_zones_for_placed_object(
                record.name,
                record.category,
                str(record.primary_room),
                chosen,
                candidate_fp.center_xy,
                yaw,
                record.scale,
            )
            if record.primary_room and not self._candidate_preserves_usability(
                layout,
                str(record.primary_room),
                local_occupied,
                candidate_fps,
                usability,
                summary,
            ):
                summary["model_swap_safety_rejections"] += 1
                _skip(summary, f"model_swap_safety_rejected_{record.category}")
                continue
            max_overlap = float(self.rules.placement_policy.get("max_large_overlap_ratio", 0.08))
            if any(aabb_overlap_ratio(candidate_fp, other) > max_overlap for other in local_occupied if other.room == candidate_fp.room):
                summary["model_swap_safety_rejections"] += 1
                _skip(summary, f"model_swap_overlap_{record.category}")
                continue
            set_object_model(scene, record.name, chosen.model)
            set_object_pose(
                scene,
                record.name,
                pos=(record.pos[0], record.pos[1], root_z_for_bbox_bottom(chosen, old_bottom_z, record.scale)),
            )
            occupied = local_occupied + candidate_fps
            summary["model_swaps"] += 1
            summary["operations"].append(
                {
                    "op": "model_swap",
                    "name": record.name,
                    "category": record.category,
                    "room": record.primary_room,
                    "old_model": old_model,
                    "model": chosen.model,
                }
            )

    def _apply_pose_jitter(
        self,
        scene: dict[str, Any],
        layout: object,
        rng: random.Random,
        summary: dict[str, Any],
        diversity_level: float = 1.0,
        active_rooms: frozenset[str] | None = None,
    ) -> None:
        probability = _clamp(
            float(self.rules.placement_policy.get("pose_jitter_probability", 0.0)) * max(0.0, float(diversity_level)),
            0.0,
            0.4,
        )
        if probability <= 0:
            return
        jitter_roles = {"small_prop"}
        for record in parse_scene_objects(scene).values():
            role = getattr(self.classifier.get(record.category), "role", "")
            if active_rooms is not None and record.primary_room not in active_rooms:
                continue
            if role not in jitter_roles or rng.random() > probability:
                continue
            dx = rng.uniform(-0.03, 0.03)
            dy = rng.uniform(-0.03, 0.03)
            yaw = rng.uniform(-0.35, 0.35)
            set_object_pose(
                scene,
                record.name,
                pos=(record.pos[0] + dx, record.pos[1] + dy, record.pos[2]),
                ori=yaw_to_quat(yaw),
            )
            summary["pose_jitters"] += 1
            summary["operations"].append({"op": "pose_jitter", "name": record.name, "category": record.category, "room": record.primary_room})

    def _sample_large_category(self, room_type: str, rng: random.Random) -> str | None:
        policy = self.rules.placement_policy
        allowed = self._allowed_large_categories_for_room(room_type)
        fallback = [
            cat
            for cat in policy.get("fallback_large_by_room", {}).get(room_type, [])
            if self._is_spawnable_large(cat) and cat in allowed
        ]
        stats_candidates: list[tuple[str, float]] = []
        if self.statistics is not None:
            for category, count in self.statistics.top_categories_for_room(room_type, limit=80):
                if self._is_spawnable_large(category) and category in allowed:
                    stats_candidates.append((category, float(count)))
        candidates = [(cat, 8.0) for cat in fallback]
        candidates.extend(stats_candidates)
        if not candidates:
            candidates = [(cat, 1.0) for cat in sorted(allowed) if self._is_spawnable_large(cat)]
        return _weighted_choice(candidates, rng)

    def _sample_rewrite_category(
        self,
        current_category: str,
        room_type: str,
        mode: str,
        rng: random.Random,
        summary: dict[str, Any] | None = None,
    ) -> str | None:
        allowed = self._allowed_large_categories_for_room(room_type)
        swap_key = "rewrite_category_swap_probability" if mode == "rewrite" else "balanced_category_swap_probability"
        swap_probability = self._effective_probability(float(self.rules.placement_policy.get(swap_key, 0.0)), summary, max_value=0.75)
        if mode == "room_relayout":
            compatible = [
                category
                for category in self._compatible_categories_for(current_category)
                if category in allowed and self._is_spawnable_large(category)
            ]
            swap_probability = self._effective_probability(
                float(self.rules.placement_policy.get("room_relayout_category_swap_probability", 0.18)),
                summary,
                max_value=0.65,
            )
            if current_category in compatible and rng.random() > swap_probability:
                return current_category
            choices = [category for category in compatible if category != current_category]
            return _weighted_choice([(category, 1.0) for category in choices], rng) or (current_category if current_category in compatible else None)
        if current_category in allowed and self._is_spawnable_large(current_category) and rng.random() > swap_probability:
            # Prefer same-category rewrites so the object identity remains valid
            # and the result still represents "same type asset + layout" variety.
            return current_category
        compatible = [
            category
            for category in sorted(set(self._compatible_categories_for(current_category)))
            if category in allowed and self._is_spawnable_large(category)
        ]
        choices = [category for category in compatible if category != current_category]
        category = _weighted_choice([(category, 1.0) for category in choices], rng)
        if category is not None:
            return category
        if current_category in compatible:
            return current_category
        return None

    def _effective_probability(self, base: float, summary: dict[str, Any] | None = None, *, min_value: float = 0.0, max_value: float = 1.0) -> float:
        level = float((summary or {}).get("diversity_level", 1.0))
        return _clamp(float(base) * max(0.0, level), min_value, max_value)

    def _sample_rewrite_model(
        self,
        record: Any,
        category: str,
        rng: random.Random,
        summary: dict[str, Any] | None = None,
    ) -> ModelAsset | None:
        current = self.catalog.get_model(record.category, record.model)
        candidates: list[ModelAsset] = []
        current_bbox = scaled_bbox(current, record.scale) if current is not None else None
        target_joint_count = _record_joint_count(record, current)
        for asset in self.catalog.get_models(category):
            if category == record.category and asset.model == record.model:
                continue
            if category == record.category and asset.joint_count != target_joint_count:
                continue
            if category != record.category and asset.joint_count != 0:
                continue
            if current_bbox is not None:
                bbox = scaled_bbox(asset, record.scale)
                if _relative_bbox_delta(current_bbox, bbox) > 0.45:
                    continue
            candidates.append(asset)
        if candidates:
            return self._choose_model_candidate(candidates, current_bbox, record.scale, rng, summary)
        if category == record.category and current is not None:
            return current
        models = [asset for asset in self.catalog.get_models(category) if asset.joint_count == 0]
        return self._choose_model_candidate(models, current_bbox, record.scale, rng, summary) if models else None

    def _sample_group_model(
        self,
        record: Any,
        category: str,
        rng: random.Random,
        summary: dict[str, Any] | None = None,
    ) -> ModelAsset | None:
        current = self.catalog.get_model(record.category, record.model)
        swap_probability = self._effective_probability(
            float(self.rules.placement_policy.get("semantic_group_model_swap_probability", self.rules.placement_policy.get("model_swap_probability", 0.08))),
            summary,
            max_value=0.45,
        )
        force_compact = bool((summary or {}).get("_force_compact_group_models", False))
        if category == record.category and current is not None and not force_compact and rng.random() > swap_probability:
            return current
        current_bbox = scaled_bbox(current, record.scale) if current is not None else None
        target_joint_count = _record_joint_count(record, current)
        candidates: list[ModelAsset] = []
        for asset in self.catalog.get_models(category):
            if category == record.category and asset.model == record.model:
                continue
            if category == record.category and asset.joint_count != target_joint_count:
                continue
            if category != record.category and asset.joint_count != 0:
                continue
            if current_bbox is not None:
                bbox = scaled_bbox(asset, record.scale)
                if _relative_bbox_delta(current_bbox, bbox) > 0.45:
                    continue
            candidates.append(asset)
        if not candidates:
            return current if category == record.category and current is not None else None
        candidates.sort(key=lambda asset: scaled_bbox(asset, record.scale)[0] * scaled_bbox(asset, record.scale)[1])
        if force_compact:
            return rng.choice(candidates[: min(6, len(candidates))])
        if summary is not None and float(summary.get("diversity_level", 1.0)) > 1.05:
            return self._choose_model_candidate(candidates[: min(8, len(candidates))], current_bbox, record.scale, rng, summary)
        return rng.choice(candidates[: min(4, len(candidates))])

    def _choose_model_candidate(
        self,
        candidates: list[ModelAsset],
        current_bbox: tuple[float, float, float] | None,
        scale: tuple[float, float, float],
        rng: random.Random,
        summary: dict[str, Any] | None = None,
    ) -> ModelAsset:
        if not candidates:
            raise ValueError("candidates must not be empty")
        diversity_level = float((summary or {}).get("diversity_level", 1.0))
        if current_bbox is None or diversity_level <= 1.05 or len(candidates) <= 2:
            return rng.choice(candidates)
        ranked = sorted(
            candidates,
            key=lambda asset: _relative_bbox_delta(current_bbox, scaled_bbox(asset, scale)),
            reverse=True,
        )
        pool_size = max(2, min(len(ranked), int(round(2 + diversity_level * 3))))
        return rng.choice(ranked[:pool_size])

    def _sample_large_pose(
        self,
        *,
        layout: object,
        room_id: str,
        category: str,
        asset: ModelAsset,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any] | None = None,
        usability: LayoutUsabilityConfig | None = None,
        clearance_m: float | None = None,
        avoid_center_xy: tuple[float, float] | None = None,
        min_center_distance_m: float = 0.0,
    ) -> tuple[tuple[float, float], float, Footprint] | None:
        policy = self.rules.placement_policy
        clearance = float(clearance_m) if clearance_m is not None else float(policy.get("large_clearance_m", policy.get("default_clearance_m", 0.12)))
        bbox = scaled_bbox(asset, scale)
        extent_xy = (bbox[0], bbox[1])
        anchors = list(policy.get("anchors_by_category", {}).get(category) or ["on_edge"])
        rng.shuffle(anchors)
        for anchor in anchors:
            placed = sample_anchor_pose(
                layout=layout,
                room_id=room_id,
                extent_xy=extent_xy,
                anchor=str(anchor),
                occupied=occupied,
                rng=rng,
                clearance_m=clearance,
                edge_band_m=float(policy.get("edge_band_m", 0.75)),
                corner_band_m=float(policy.get("corner_band_m", 0.9)),
                max_attempts=int(policy.get("max_candidate_attempts", 180)),
                avoid_center_xy=avoid_center_xy,
                min_center_distance_m=min_center_distance_m,
                candidate_filter=(
                    lambda fp: self._candidate_preserves_usability(
                        layout,
                        room_id,
                        occupied,
                        self._candidate_with_clearance_zones(fp, category, asset, scale),
                        usability,
                        summary,
                    )
                    if usability is not None
                    else True
                ),
            )
            if placed is not None:
                return placed
        return None

    def _sample_local_relocation_pose(
        self,
        *,
        layout: object,
        room_id: str,
        record: Any,
        asset: ModelAsset,
        occupied: list[Footprint],
        rng: random.Random,
        summary: dict[str, Any] | None = None,
        usability: LayoutUsabilityConfig | None = None,
    ) -> tuple[tuple[float, float], float, Footprint] | None:
        policy = self.rules.placement_policy
        clearance = float(policy.get("large_rewrite_clearance_m", policy.get("default_clearance_m", 0.05)))
        free = room_free_mask_crop(layout, room_id, occupied, clearance_m=clearance)
        if free is None:
            return None
        bbox = scaled_bbox(asset, record.scale)
        base_extent = (bbox[0], bbox[1])
        old_yaw = quat_to_yaw(record.ori)
        cardinal_yaws = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
        yaws = [old_yaw]
        yaws.extend(cardinal_yaws)
        max_overlap = float(policy.get("max_large_overlap_ratio", 0.08))
        min_move = max(0.12, float(policy.get("large_jitter_translate_m", 0.18)))
        max_move = float(policy.get("large_rewrite_local_translate_m", 0.9))
        for _ in range(int(policy.get("max_candidate_attempts", 64)) * 3):
            distance = rng.uniform(min_move, max_move)
            angle = rng.uniform(-math.pi, math.pi)
            xy = (
                record.pos[0] + math.cos(angle) * distance,
                record.pos[1] + math.sin(angle) * distance,
            )
            yaw = rng.choice(yaws)
            if rng.random() < 0.35:
                yaw += rng.uniform(-math.radians(18.0), math.radians(18.0))
            extent = footprint_extent_for_yaw(base_extent, yaw)
            if not candidate_fits(layout, room_id, free, xy, extent, clearance_m=clearance):
                continue
            fp = Footprint(
                name=record.name,
                category=record.category,
                room=room_id,
                center_xy=(float(xy[0]), float(xy[1])),
                extent_xy=extent,
                yaw=yaw,
                source="rewritten_large_local",
                owner=record.name,
            )
            if any(aabb_overlap_ratio(fp, other) > max_overlap for other in occupied if other.room == room_id):
                continue
            candidate_fps = [fp] + self._protected_zones_for_placed_object(record.name, record.category, room_id, asset, fp.center_xy, yaw, record.scale)
            if usability is not None and not self._candidate_preserves_usability(layout, room_id, occupied, candidate_fps, usability, summary):
                continue
            return fp.center_xy, yaw, fp
        return None

    def _layout_usability_config(self, config: VariantConfig) -> LayoutUsabilityConfig:
        policy = self.rules.placement_policy
        default_budget = float(policy.get("default_space_budget", policy.get("max_occupied_area_ratio", 0.45)))
        min_budget = float(policy.get("min_space_budget", 0.25))
        max_budget = float(policy.get("max_space_budget", 0.65))
        budget = default_budget if config.space_budget is None else float(config.space_budget)
        budget = _clamp(budget, min_budget, max_budget)
        return LayoutUsabilityConfig(
            space_budget=budget,
            min_traversable_free_ratio=float(policy.get("min_traversable_free_ratio", 0.35)),
            min_connected_free_ratio=float(policy.get("min_connected_free_ratio", 0.28)),
            clearance_m=float(policy.get("usability_clearance_m", policy.get("large_rewrite_clearance_m", 0.05))),
            check_connected_during_sampling=bool(policy.get("check_connected_during_sampling", True)),
        )

    def _select_active_rooms(self, layout: object, rng: random.Random, config: VariantConfig) -> ActiveRoomSelection:
        mode = str(config.room_subset_mode or "all")
        all_rooms = tuple(sorted(str(room_id) for room_id in layout.room_ins_name_to_ins_id))
        excluded_types = {str(item) for item in config.exclude_room_types or () if str(item)}
        eligible = tuple(room for room in all_rooms if room_type_from_instance(room) not in excluded_types)
        adjacency = _room_adjacency(layout)
        if mode == "all":
            return ActiveRoomSelection(
                rooms=frozenset(all_rooms),
                mode=mode,
                requested_size=len(all_rooms),
                eligible_rooms=eligible,
                adjacency=adjacency,
            )
        explicit_rooms = tuple(str(room) for room in config.active_rooms or () if str(room))
        if mode == "explicit" or explicit_rooms:
            unknown = sorted(room for room in explicit_rooms if room not in layout.room_ins_name_to_ins_id)
            if unknown:
                raise ValueError("Unknown active room(s): {}".format(", ".join(unknown)))
            rooms = frozenset(explicit_rooms)
            if not rooms:
                raise ValueError("room_subset_mode='explicit' requires at least one active room")
            return ActiveRoomSelection(
                rooms=rooms,
                mode="explicit",
                requested_size=len(rooms),
                eligible_rooms=eligible,
                adjacency=adjacency,
            )
        if mode != "connected":
            raise ValueError(f"Unsupported room_subset_mode: {mode}")
        if not eligible:
            raise ValueError("No eligible rooms remain after applying exclude_room_types")
        requested_raw = int(config.room_subset_size or 0)
        requested = requested_raw if requested_raw > 0 else min(3, len(eligible))
        requested = max(1, min(requested, len(eligible)))
        rooms = _sample_connected_room_subset(adjacency, eligible, requested, rng)
        return ActiveRoomSelection(
            rooms=frozenset(rooms),
            mode=mode,
            requested_size=requested,
            eligible_rooms=eligible,
            adjacency=adjacency,
        )

    def _room_relayout_allow_removal(self, config: VariantConfig) -> bool:
        if config.room_relayout_allow_removal is not None:
            return bool(config.room_relayout_allow_removal)
        return bool(self.rules.placement_policy.get("room_relayout_allow_removal", False))

    def _room_relayout_max_removals(self, config: VariantConfig, selected_count: int) -> int:
        if config.room_relayout_max_removals_per_room is not None:
            return max(0, int(config.room_relayout_max_removals_per_room))
        raw = self.rules.placement_policy.get("room_relayout_max_removals_per_room")
        if raw is not None:
            return max(0, int(raw))
        fraction = float(self.rules.placement_policy.get("room_relayout_max_removal_fraction", 0.0))
        return max(0, int(math.floor(max(0, selected_count) * max(0.0, fraction))))

    def _rewrite_target_count(self, eligible_count: int, fraction: float) -> int:
        if eligible_count <= 0:
            return 0
        proportional = int(round(float(eligible_count) * float(fraction)))
        proportional = max(1, proportional)
        policy = self.rules.placement_policy
        budget_base = int(policy.get("rewrite_large_budget_base", 8))
        budget_scale = float(policy.get("rewrite_large_budget_sqrt_scale", 1.15))
        budget_cap = int(policy.get("max_rewrite_large_per_room", 36))
        budget = int(round(float(budget_base) + math.sqrt(float(eligible_count)) * float(budget_scale)))
        budget = max(1, min(budget_cap, budget))
        return min(proportional, budget)

    def _select_rewrite_candidates(self, eligible: list[Any], target_count: int, rng: random.Random) -> list[Any]:
        if target_count >= len(eligible):
            return list(eligible)
        room_counts: dict[str, int] = {}
        buckets: dict[str, list[Any]] = {}
        for record in eligible:
            room = str(record.primary_room or "")
            buckets.setdefault(str(record.category), []).append(record)
        categories = list(buckets.keys())
        rng.shuffle(categories)
        selected: list[Any] = []
        while len(selected) < target_count and categories:
            categories.sort(key=lambda category: (room_counts.get(category, 0), -len(buckets[category]), _rewrite_priority(category)))
            category = categories[0]
            bucket = buckets[category]
            if not bucket:
                categories.pop(0)
                continue
            record = bucket.pop(rng.randrange(len(bucket)))
            selected.append(record)
            room_counts[category] = room_counts.get(category, 0) + 1
            if not bucket:
                categories.pop(0)
        if len(selected) < target_count:
            remaining = [record for bucket in buckets.values() for record in bucket]
            rng.shuffle(remaining)
            selected.extend(remaining[: target_count - len(selected)])
        return selected

    def _candidate_footprint_for_category(self, fp: Footprint, category: str) -> Footprint:
        return Footprint(
            name=fp.name,
            category=str(category),
            room=fp.room,
            center_xy=fp.center_xy,
            extent_xy=fp.extent_xy,
            yaw=fp.yaw,
            source=fp.source,
            owner=fp.owner or fp.name,
        )

    def _candidate_with_clearance_zones(
        self,
        fp: Footprint,
        category: str,
        asset: ModelAsset,
        scale: tuple[float, float, float] | list[float] | None,
    ) -> list[Footprint]:
        candidate = self._candidate_footprint_for_category(fp, category)
        return [candidate] + self._protected_zones_for_placed_object(
            candidate.owner or candidate.name,
            category,
            str(candidate.room),
            asset,
            candidate.center_xy,
            candidate.yaw,
            scale,
        )

    def _candidate_preserves_usability(
        self,
        layout: object,
        room_id: str,
        occupied: list[Footprint],
        candidates: list[Footprint],
        usability: LayoutUsabilityConfig,
        summary: dict[str, Any] | None = None,
    ) -> bool:
        protected_overlap = self._protected_zone_overlap_reason(candidates, occupied)
        if protected_overlap == "door_clearance":
            if summary is not None:
                summary["door_clearance_rejections"] = summary.get("door_clearance_rejections", 0) + 1
                _skip(summary, "door_clearance_blocked")
            return False
        if protected_overlap == "openable_clearance":
            if summary is not None:
                summary["openable_clearance_rejections"] = summary.get("openable_clearance_rejections", 0) + 1
                _skip(summary, "openable_clearance_blocked")
            return False
        state = self._room_usability_state_cached(
            layout,
            room_id,
            occupied,
            clearance_m=usability.clearance_m,
            compute_connected=False,
        )
        if state is None:
            return False
        current_metrics = state.current_metrics
        traversable_cells = current_metrics.traversable_cells
        lost_free_cells = self._candidate_lost_free_cells(layout, room_id, state, candidates, usability.clearance_m)
        free_cells = max(0, int(current_metrics.free_cells) - int(lost_free_cells))
        metrics = FreeSpaceMetrics(
            free_ratio=(free_cells / traversable_cells) if traversable_cells else 0.0,
            connected_ratio=(free_cells / traversable_cells) if traversable_cells else 0.0,
            occupied_area_ratio=(state.occupied_area + self._footprint_occupied_area(room_id, candidates)) / state.room_area,
            free_cells=free_cells,
            largest_component_cells=free_cells,
            traversable_cells=traversable_cells,
        )
        if metrics.occupied_area_ratio > usability.space_budget and metrics.occupied_area_ratio > current_metrics.occupied_area_ratio + 0.01:
            if summary is not None:
                summary["space_rejections"] = summary.get("space_rejections", 0) + 1
                _skip(summary, "space_budget_exceeded")
            return False
        if metrics.free_ratio < usability.min_traversable_free_ratio and metrics.free_ratio < current_metrics.free_ratio - 0.01:
            if summary is not None:
                summary["space_rejections"] = summary.get("space_rejections", 0) + 1
                _skip(summary, "traversable_free_ratio_low")
            return False
        if usability.check_connected_during_sampling:
            current_connected = self._room_usability_state_cached(
                layout,
                room_id,
                occupied,
                clearance_m=usability.clearance_m,
                compute_connected=True,
            )
            if current_connected is None:
                return False
            if lost_free_cells > 0:
                candidate_connected = self._candidate_connected_metrics(
                    layout,
                    room_id,
                    current_connected,
                    candidates,
                    usability,
                )
                if (
                    candidate_connected.connected_ratio < usability.min_connected_free_ratio
                    and candidate_connected.connected_ratio < current_connected.current_metrics.connected_ratio - 0.01
                ):
                    if summary is not None:
                        summary["space_rejections"] = summary.get("space_rejections", 0) + 1
                        _skip(summary, "connected_free_ratio_low")
                    return False
        return True

    def _candidate_lost_free_cells(
        self,
        layout: object,
        room_id: str,
        state: RoomUsabilityState,
        candidates: list[Footprint],
        clearance_m: float,
    ) -> int:
        lost = 0
        touched: list[tuple[np.ndarray, np.ndarray]] = []
        for fp in candidates:
            if fp.room is not None and str(fp.room) != str(room_id):
                continue
            rows, cols = self._footprint_crop_rows_cols(layout, state.crop.room_slice, state.crop.free.shape, fp, clearance_m)
            if not rows.size or not cols.size:
                continue
            candidate_free = state.crop.free[np.ix_(rows, cols)]
            candidate_traversable = state.traversable[np.ix_(rows, cols)]
            newly_lost = candidate_free & candidate_traversable
            for prev_rows, prev_cols in touched:
                row_overlap = np.intersect1d(rows, prev_rows, assume_unique=True)
                col_overlap = np.intersect1d(cols, prev_cols, assume_unique=True)
                if row_overlap.size and col_overlap.size:
                    row_lookup = {int(value): idx for idx, value in enumerate(rows.tolist())}
                    col_lookup = {int(value): idx for idx, value in enumerate(cols.tolist())}
                    local_rows = [row_lookup[int(value)] for value in row_overlap.tolist()]
                    local_cols = [col_lookup[int(value)] for value in col_overlap.tolist()]
                    newly_lost[np.ix_(local_rows, local_cols)] = False
            lost += int(newly_lost.sum())
            touched.append((rows, cols))
        return lost

    def _candidate_connected_metrics(
        self,
        layout: object,
        room_id: str,
        state: RoomUsabilityState,
        candidates: list[Footprint],
        usability: LayoutUsabilityConfig,
    ) -> FreeSpaceMetrics:
        free = state.crop.free.copy()
        for fp in candidates:
            if fp.room is not None and str(fp.room) != str(room_id):
                continue
            self._burn_footprint_into_crop(layout, state.crop.room_slice, fp, usability.clearance_m, free)
        free_traversable = free & state.traversable
        traversable_cells = state.current_metrics.traversable_cells
        free_cells = int(free_traversable.sum())
        largest = _largest_component_size(free_traversable)
        occupied_area = state.occupied_area + self._footprint_occupied_area(room_id, candidates)
        return FreeSpaceMetrics(
            free_ratio=(free_cells / traversable_cells) if traversable_cells else 0.0,
            connected_ratio=(largest / traversable_cells) if traversable_cells else 0.0,
            occupied_area_ratio=occupied_area / state.room_area,
            free_cells=free_cells,
            largest_component_cells=int(largest),
            traversable_cells=traversable_cells,
        )

    def _room_can_accept_more_large(
        self,
        layout: object,
        room_id: str,
        occupied: list[Footprint],
        usability: LayoutUsabilityConfig,
        summary: dict[str, Any],
    ) -> bool:
        state = self._room_usability_state_cached(
            layout,
            room_id,
            occupied,
            clearance_m=usability.clearance_m,
            compute_connected=False,
        )
        if state is None:
            return False
        metrics = state.current_metrics
        if metrics.occupied_area_ratio >= usability.space_budget:
            summary["space_rejections"] = summary.get("space_rejections", 0) + 1
            return False
        if metrics.free_ratio < usability.min_traversable_free_ratio:
            summary["space_rejections"] = summary.get("space_rejections", 0) + 1
            return False
        if usability.check_connected_during_sampling:
            connected = self._room_usability_state_cached(
                layout,
                room_id,
                occupied,
                clearance_m=usability.clearance_m,
                compute_connected=True,
            )
            if connected is None or connected.current_metrics.connected_ratio < usability.min_connected_free_ratio:
                summary["space_rejections"] = summary.get("space_rejections", 0) + 1
                return False
        return True

    def _room_usability_state_cached(
        self,
        layout: object,
        room_id: str,
        occupied: list[Footprint],
        *,
        clearance_m: float,
        compute_connected: bool,
    ) -> RoomUsabilityState | None:
        key = (
            id(layout),
            str(room_id),
            round(float(clearance_m), 4),
            bool(compute_connected),
            self._footprint_metrics_signature(str(room_id), occupied),
        )
        cached = self._room_usability_state_cache.get(key)
        if cached is not None:
            self._room_usability_state_cache[key] = cached
            return cached
        if compute_connected:
            base_key = (
                id(layout),
                str(room_id),
                round(float(clearance_m), 4),
                False,
                self._footprint_metrics_signature(str(room_id), occupied),
            )
            base_state = self._room_usability_state_cache.get(base_key)
            if base_state is not None:
                free_traversable = base_state.crop.free & base_state.traversable
                traversable_cells = base_state.current_metrics.traversable_cells
                free_cells = int(free_traversable.sum())
                largest = _largest_component_size(free_traversable)
                metrics = FreeSpaceMetrics(
                    free_ratio=(free_cells / traversable_cells) if traversable_cells else 0.0,
                    connected_ratio=(largest / traversable_cells) if traversable_cells else 0.0,
                    occupied_area_ratio=base_state.occupied_area / base_state.room_area,
                    free_cells=free_cells,
                    largest_component_cells=int(largest),
                    traversable_cells=traversable_cells,
                )
                state = RoomUsabilityState(
                    crop=base_state.crop,
                    traversable=base_state.traversable,
                    room_area=base_state.room_area,
                    occupied_area=base_state.occupied_area,
                    current_metrics=metrics,
                )
                self._store_room_usability_state(key, state)
                return state
        crop = room_free_mask_crop(
            layout,
            room_id,
            occupied,
            clearance_m=clearance_m,
        )
        if crop is None:
            return None
        traversable = crop.room & traversable_mask_crop(layout, crop.room_slice, crop.free.shape)
        free_traversable = crop.free & traversable
        traversable_cells = int(traversable.sum())
        free_cells = int(free_traversable.sum())
        largest = _largest_component_size(free_traversable) if compute_connected else free_cells
        room_area = max(float(crop.room.sum()) * float(layout.map_resolution) ** 2, 1e-8)
        occupied_area = self._footprint_occupied_area(room_id, occupied)
        metrics = FreeSpaceMetrics(
            free_ratio=(free_cells / traversable_cells) if traversable_cells else 0.0,
            connected_ratio=(largest / traversable_cells) if traversable_cells else 0.0,
            occupied_area_ratio=occupied_area / room_area,
            free_cells=free_cells,
            largest_component_cells=int(largest),
            traversable_cells=traversable_cells,
        )
        state = RoomUsabilityState(
            crop=crop,
            traversable=traversable,
            room_area=room_area,
            occupied_area=occupied_area,
            current_metrics=metrics,
        )
        self._store_room_usability_state(key, state)
        return state

    def _footprint_metrics_signature(self, room_id: str, footprints: list[Footprint]) -> tuple[tuple[Any, ...], ...]:
        key = (id(footprints), str(room_id))
        item_ids = tuple(
            id(fp)
            for fp in footprints
            if fp.room is None or str(fp.room) == str(room_id)
        )
        cached = self._footprint_signature_cache.get(key)
        if cached is not None and cached[0] is footprints and cached[1] == item_ids:
            return cached[2]
        values: list[tuple[Any, ...]] = []
        for fp in footprints:
            if fp.room is not None and str(fp.room) != str(room_id):
                continue
            values.append(
                (
                    str(fp.name),
                    str(fp.owner or ""),
                    str(fp.room or ""),
                    str(fp.category),
                    str(fp.source),
                    float(fp.center_xy[0]),
                    float(fp.center_xy[1]),
                    float(fp.extent_xy[0]),
                    float(fp.extent_xy[1]),
                    float(fp.yaw),
                )
            )
        signature = tuple(sorted(values))
        if len(self._footprint_signature_cache) >= _FOOTPRINT_SIGNATURE_CACHE_MAX:
            self._footprint_signature_cache.clear()
        self._footprint_signature_cache[key] = (footprints, item_ids, signature)
        return signature

    def _store_room_usability_state(self, key: tuple[Any, ...], state: RoomUsabilityState) -> None:
        self._room_usability_state_cache[key] = state
        while len(self._room_usability_state_cache) > _ROOM_USABILITY_STATE_CACHE_MAX:
            self._room_usability_state_cache.pop(next(iter(self._room_usability_state_cache)))

    def _footprint_occupied_area(self, room_id: str, footprints: list[Footprint]) -> float:
        area = 0.0
        for fp in footprints:
            if fp.room != room_id:
                continue
            if fp.source in PROTECTED_ZONE_SOURCES or fp.category in FLOOR_LAYER_CATEGORIES:
                continue
            area += fp.area
        return area

    def _footprint_crop_rows_cols(
        self,
        layout: object,
        room_slice: tuple[slice, slice],
        target_shape: tuple[int, int],
        footprint: Footprint,
        clearance_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = footprint_rows_cols(layout, footprint.center_xy, footprint.extent_xy, clearance_m=clearance_m)
        row_offset = int(room_slice[0].start or 0)
        col_offset = int(room_slice[1].start or 0)
        rows = rows - row_offset
        cols = cols - col_offset
        rows = rows[(rows >= 0) & (rows < int(target_shape[0]))]
        cols = cols[(cols >= 0) & (cols < int(target_shape[1]))]
        return rows, cols

    def _burn_footprint_into_crop(
        self,
        layout: object,
        room_slice: tuple[slice, slice],
        footprint: Footprint,
        clearance_m: float,
        mask: np.ndarray,
    ) -> bool:
        rows, cols = self._footprint_crop_rows_cols(layout, room_slice, mask.shape, footprint, clearance_m)
        if not rows.size or not cols.size:
            return False
        mask[np.ix_(rows, cols)] = False
        return True

    def _protected_zone_overlap_reason(self, candidates: list[Footprint], occupied: list[Footprint]) -> str | None:
        """Check if candidates overlap with protected zones in occupied list.

        Uses spatial indexing when occupied list is large for better performance.
        """
        # Use spatial index for large occupied lists (>100 footprints)
        if len(occupied) > 100:
            spatial_index = build_spatial_index(occupied, cell_size=2.0)
            for candidate in candidates:
                # Query only nearby footprints
                nearby = spatial_index.query_footprint(candidate, clearance=0.0)
                for other in nearby:
                    if other.room and candidate.room and other.room != candidate.room:
                        continue
                    if other.owner and other.owner == candidate.owner:
                        continue
                    candidate_protected = candidate.source in {"door_clearance", "openable_clearance"}
                    other_protected = other.source in {"door_clearance", "openable_clearance"}
                    if candidate_protected == other_protected:
                        continue
                    if aabb_overlap_ratio(candidate, other) > 0.0:
                        if candidate_protected:
                            return candidate.source
                        return other.source
            return None

        # Original implementation for small lists
        for candidate in candidates:
            for other in occupied:
                if other.room and candidate.room and other.room != candidate.room:
                    continue
                if other.owner and other.owner == candidate.owner:
                    continue
                candidate_protected = candidate.source in {"door_clearance", "openable_clearance"}
                other_protected = other.source in {"door_clearance", "openable_clearance"}
                if candidate_protected == other_protected:
                    continue
                if aabb_overlap_ratio(candidate, other) > 0.0:
                    if candidate_protected:
                        return candidate.source
                    return other.source
        return None

    def _protected_zones_for_placed_object(
        self,
        name: str,
        category: str,
        room_id: str,
        asset: ModelAsset,
        xy: tuple[float, float],
        yaw: float,
        scale: tuple[float, float, float] | list[float] | None = None,
    ) -> list[Footprint]:
        return protected_zones_for_object(
            name=name,
            category=category,
            room=room_id,
            asset=asset,
            center_xy=xy,
            yaw=yaw,
            scale=scale,
            classifier=self.classifier,
            policy=self.rules.placement_policy,
        )

    def _allowed_large_categories_for_room(self, room_type: str) -> set[str]:
        mapping = self.rules.placement_policy.get("allowed_large_by_room") or {}
        if room_type in mapping:
            return {str(item) for item in mapping.get(room_type) or []}
        fallback = self.rules.placement_policy.get("fallback_large_by_room") or {}
        if room_type in fallback:
            return {str(item) for item in fallback.get(room_type) or []}
        return set(self.classifier.categories_for_role("large_floor")) | set(self.classifier.categories_for_role("support_surface"))

    def _filter_small_categories_for_room(self, room_type: str, categories: list[str]) -> list[str]:
        small_categories = [
            str(category)
            for category in categories
            if getattr(self.classifier.get(str(category)), "role", "") == "small_prop"
        ]
        mapping = self.rules.receptacle_policy.get("allowed_small_by_room") or {}
        if room_type not in mapping:
            return small_categories
        allowed = {str(item) for item in mapping.get(room_type) or []}
        return [category for category in small_categories if category in allowed]

    def _is_category_allowed_for_room(self, category: str, room_type: str, role: str) -> bool:
        if role in {"large_floor", "support_surface"}:
            return str(category) in self._allowed_large_categories_for_room(str(room_type))
        if role == "small_prop":
            mapping = self.rules.receptacle_policy.get("allowed_small_by_room") or {}
            if room_type in mapping:
                return str(category) in {str(item) for item in mapping.get(room_type) or []}
        return True

    def _fast_overlap_check(
        self,
        candidate: Footprint,
        occupied: list[Footprint],
        *,
        max_overlap: float = 0.0,
        spatial_index: SpatialIndex | None = None,
    ) -> bool:
        """Fast collision check using spatial index when available.

        Args:
            candidate: Candidate footprint to check
            occupied: List of occupied footprints
            max_overlap: Maximum allowed overlap ratio
            spatial_index: Optional pre-built spatial index for occupied

        Returns:
            True if candidate has overlap > max_overlap with any occupied footprint
        """
        if spatial_index is not None:
            # Use spatial index for fast query
            nearby = spatial_index.query_footprint(candidate, clearance=0.0)
            for other in nearby:
                if other.room and candidate.room and other.room != candidate.room:
                    continue
                if aabb_overlap_ratio(candidate, other) > max_overlap:
                    return True
            return False

        # Fallback: linear scan (for small lists)
        room_id = candidate.room
        for other in occupied:
            if other.room and room_id and other.room != room_id:
                continue
            if aabb_overlap_ratio(candidate, other) > max_overlap:
                return True
        return False

    def _large_candidate_stays_in_room(
        self,
        layout: object,
        room_id: str,
        xy: tuple[float, float],
        extent_xy: tuple[float, float],
        *,
        min_room_fraction: float = 0.98,
    ) -> bool:
        room = layout.room_mask(room_id)
        if room is None:
            return False
        rows, cols = footprint_rows_cols(layout, xy, extent_xy)
        if rows.size == 0 or cols.size == 0:
            return False
        if rows.min() < 0 or cols.min() < 0 or rows.max() >= room.shape[0] or cols.max() >= room.shape[1]:
            return False
        patch = np.asarray(room, dtype=bool)[np.ix_(rows, cols)]
        return bool(patch.size and float(patch.mean()) >= float(min_room_fraction))

    def _is_spawnable_large(self, category: str) -> bool:
        info = self.classifier.get(category)
        return bool(info and info.spawnable_large and self.catalog.has_category(category))

    def _is_density_large(self, category: str) -> bool:
        info = self.classifier.get(category)
        return bool(info and info.role in {"large_floor", "support_surface"})

    def _is_rewrite_candidate(self, record: Any, mode: str) -> bool:
        role = getattr(self.classifier.get(record.category), "role", "")
        if role not in {"large_floor", "support_surface"}:
            return False
        protected = set(str(item) for item in self.rules.placement_policy.get("rewrite_protected_categories") or [])
        if record.category in protected:
            return False
        movable = set(str(item) for item in self.rules.placement_policy.get("rewrite_large_categories") or [])
        if movable:
            return record.category in movable
        return record.category in self._allowed_large_categories_for_room(record.room_type or "")

    def _is_jitter_candidate(self, record: Any) -> bool:
        if record.fixed_base:
            return False
        jitterable = set(str(item) for item in self.rules.placement_policy.get("jitter_large_categories") or [])
        if jitterable:
            return record.category in jitterable
        return self._is_rewrite_candidate(record, "balanced")

    def _is_model_swap_allowed_in_rooms(self, record: Any, role: str) -> bool:
        if not record.rooms:
            return False
        if role in {"large_floor", "support_surface"}:
            return all(record.category in self._allowed_large_categories_for_room(room_type_from_instance(room)) for room in record.rooms)
        if role == "small_prop":
            allowed_by_room = self.rules.receptacle_policy.get("allowed_small_by_room") or {}
            for room in record.rooms:
                room_type = room_type_from_instance(room)
                if room_type in allowed_by_room and record.category not in {str(item) for item in allowed_by_room.get(room_type) or []}:
                    return False
        return True

    def _is_room_relayout_soft_obstacle(self, record: Any) -> bool:
        asset = self.catalog.get_model(record.category, record.model)
        if asset is None:
            return False
        role = getattr(self.classifier.get(record.category), "role", "")
        if role not in {"large_floor", "support_surface"}:
            return False
        if record.fixed_base:
            return False
        if is_openable_clearance_object(record.category, asset, self.classifier):
            return False
        protected = set(str(item) for item in self.rules.placement_policy.get("room_relayout_hard_obstacle_categories") or [])
        if record.category in protected:
            return False
        return True

    def _footprints_overlap_any(self, fp: Footprint, others: list[Footprint]) -> bool:
        max_overlap = float(self.rules.placement_policy.get("max_large_overlap_ratio", 0.08))
        for other in others:
            if fp.room and other.room and fp.room != other.room:
                continue
            if other.owner and fp.owner and other.owner == fp.owner:
                continue
            if aabb_overlap_ratio(fp, other) > max_overlap:
                return True
        return False

    def _sample_model_for_room(self, category: str, room_id: str, layout: object, rng: random.Random) -> ModelAsset | None:
        room = layout.room_mask(room_id)
        if room is None:
            return None
        rows, cols = room.nonzero()
        if rows.size == 0:
            return None
        room_w = (cols.max() - cols.min() + 1) * float(layout.map_resolution)
        room_h = (rows.max() - rows.min() + 1) * float(layout.map_resolution)
        max_dim = max(0.35, min(room_w, room_h) * 0.75)
        candidates = [asset for asset in self.catalog.get_models(category) if asset.largest_xy <= max_dim and asset.joint_count == 0]
        if not candidates:
            candidates = [asset for asset in self.catalog.get_models(category) if asset.joint_count == 0]
        if not candidates:
            return None
        return rng.choice(candidates)

    def _sample_small_model(self, category: str, rng: random.Random) -> ModelAsset | None:
        info = self.classifier.get(category)
        if info is None or not info.spawnable_small:
            return None
        models = [asset for asset in self.catalog.get_models(category) if asset.largest_xy <= 0.7 and asset.height <= 0.7 and asset.joint_count == 0]
        if not models:
            return None
        return rng.choice(models)

    def _unique_name(self, scene: dict[str, Any], category: str, model: str, seed: int, index: int) -> str:
        existing = set(parse_scene_objects(scene))
        base = f"{category}_{model}_aug_{seed:04d}_{index:03d}"
        name = base
        suffix = 0
        while name in existing:
            suffix += 1
            name = f"{base}_{suffix}"
        return name


def _skip(summary: dict[str, Any], reason: str) -> None:
    skipped = summary.setdefault("skipped", {})
    skipped[str(reason)] = skipped.get(str(reason), 0) + 1


def _small_operation_maps(summary: dict[str, Any]) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    operation_by_name: dict[str, str] = {}
    parent_by_name: dict[str, str] = {}
    op_index: dict[str, int] = {}
    for index, op in enumerate(summary.get("operations") or []):
        if not isinstance(op, dict) or not op.get("name"):
            continue
        name = str(op["name"])
        raw = str(op.get("op") or "")
        if raw == "add_small":
            operation_by_name[name] = "added_small"
            op_index[name] = int(index)
        elif raw == "relocate_small":
            operation_by_name.setdefault(name, "added_small")
            op_index.setdefault(name, int(index))
        elif raw == "remove_small":
            operation_by_name[name] = "removed_small"
        if raw in {"add_small", "relocate_small"} and op.get("parent"):
            parent_by_name[name] = str(op["parent"])
    return operation_by_name, parent_by_name, op_index


def _weighted_choice(candidates: list[tuple[str, float]], rng: random.Random) -> str | None:
    if not candidates:
        return None
    total = sum(max(0.0, float(weight)) for _, weight in candidates)
    if total <= 0:
        return rng.choice([category for category, _ in candidates])
    target = rng.random() * total
    acc = 0.0
    for category, weight in candidates:
        acc += max(0.0, float(weight))
        if acc >= target:
            return category
    return candidates[-1][0]


def _relative_bbox_delta(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    deltas = []
    for left, right in zip(a, b):
        denom = max(abs(left), 1e-6)
        deltas.append(abs(right - left) / denom)
    return max(deltas)


def _task_relevant_names(scene: dict[str, Any]) -> set[str]:
    task = (scene.get("metadata") or {}).get("task") or {}
    names = set()
    for value in (task.get("inst_to_name") or {}).values():
        if value:
            names.add(str(value))
    init_info = ((scene.get("objects_info") or {}).get("init_info") or {})
    for name, info in init_info.items():
        args = (info.get("args") or {}) if isinstance(info, dict) else {}
        if args.get("bddl_object_scope"):
            names.add(str(name))
    aug_meta = (scene.get("metadata") or {}).get("scoper.benchmark.behavior.scene_geometry") or {}
    operation_log = aug_meta.get("operation_log") or []
    if isinstance(operation_log, list):
        for op in operation_log:
            if isinstance(op, dict) and op.get("name") and op.get("task_relevant"):
                names.add(str(op["name"]))
    return names


def _room_adjacency(layout: object, *, dilation_px: int = 2) -> dict[str, tuple[str, ...]]:
    room_ids = {int(value): str(key) for key, value in getattr(layout, "room_ins_name_to_ins_id", {}).items()}
    room_map = np.asarray(layout.room_ins_map)
    edges: set[tuple[str, str]] = set()
    for gap in range(1, max(1, int(dilation_px)) + 1):
        _add_room_adjacency_edges(room_map[:-gap, :], room_map[gap:, :], room_ids, edges)
        _add_room_adjacency_edges(room_map[:, :-gap], room_map[:, gap:], room_ids, edges)
    out: dict[str, set[str]] = {room: set() for room in room_ids.values()}
    for left, right in edges:
        out.setdefault(left, set()).add(right)
        out.setdefault(right, set()).add(left)
    return {room: tuple(sorted(neighbors)) for room, neighbors in sorted(out.items())}


def _add_room_adjacency_edges(
    left: np.ndarray,
    right: np.ndarray,
    room_ids: dict[int, str],
    edges: set[tuple[str, str]],
) -> None:
    mask = (left != right) & (left != 0) & (right != 0)
    if not np.any(mask):
        return
    pairs = np.stack((left[mask].astype(np.int64), right[mask].astype(np.int64)), axis=1)
    for raw_a, raw_b in np.unique(pairs, axis=0).tolist():
        a = room_ids.get(int(raw_a))
        b = room_ids.get(int(raw_b))
        if not a or not b or a == b:
            continue
        edges.add(tuple(sorted((a, b))))


def _sample_connected_room_subset(
    adjacency: dict[str, tuple[str, ...]],
    eligible_rooms: tuple[str, ...],
    requested_size: int,
    rng: random.Random,
) -> tuple[str, ...]:
    eligible = set(eligible_rooms)
    starts = list(eligible_rooms)
    rng.shuffle(starts)
    starts.sort(key=lambda room: len([neighbor for neighbor in adjacency.get(room, ()) if neighbor in eligible]), reverse=True)
    best: list[str] = []
    for start in starts:
        selected = [start]
        selected_set = {start}
        frontier = [neighbor for neighbor in adjacency.get(start, ()) if neighbor in eligible]
        rng.shuffle(frontier)
        while len(selected) < requested_size and frontier:
            room = frontier.pop(0)
            if room in selected_set:
                continue
            selected.append(room)
            selected_set.add(room)
            neighbors = [neighbor for neighbor in adjacency.get(room, ()) if neighbor in eligible and neighbor not in selected_set]
            rng.shuffle(neighbors)
            frontier.extend(neighbors)
        if len(selected) >= requested_size:
            return tuple(sorted(selected))
        if len(selected) > len(best):
            best = selected
    if not best:
        best = [eligible_rooms[0]]
    return tuple(sorted(best[:requested_size]))


def _rooms_are_connected(rooms: set[str], adjacency: dict[str, tuple[str, ...]]) -> bool:
    if not rooms:
        return True
    start = next(iter(rooms))
    seen = {start}
    queue = [start]
    while queue:
        room = queue.pop(0)
        for neighbor in adjacency.get(room, ()):
            if neighbor in rooms and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen == rooms


def _without_object_footprints(occupied: list[Footprint], name: str) -> list[Footprint]:
    target = str(name)
    return [fp for fp in occupied if fp.name != target and fp.owner != target]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _record_joint_count(record: Any, asset: ModelAsset | None) -> int:
    lengths = [
        len(value)
        for value in (getattr(record, "joint_pos", ()), getattr(record, "joint_vel", ()))
        if isinstance(value, tuple) and value
    ]
    if lengths:
        return max(lengths)
    return int(asset.joint_count) if asset is not None else 0


def _transform_group_offset(anchor_xy: tuple[float, float], yaw: float, offset_xy: tuple[float, float]) -> tuple[float, float]:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    ox, oy = float(offset_xy[0]), float(offset_xy[1])
    return (
        float(anchor_xy[0]) + ox * c - oy * s,
        float(anchor_xy[1]) + ox * s + oy * c,
    )


def _group_anchor_from_footprint_center(
    center_xy: tuple[float, float],
    yaw: float,
    center_offset_xy: tuple[float, float],
) -> tuple[float, float]:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    ox, oy = float(center_offset_xy[0]), float(center_offset_xy[1])
    return (
        float(center_xy[0]) - (ox * c - oy * s),
        float(center_xy[1]) - (ox * s + oy * c),
    )


def _rewrite_priority(category: str) -> int:
    category = str(category)
    primary_priority = {
        "sofa",
        "bed",
    }
    storage_and_tables = {
        "bottom_cabinet",
        "bottom_cabinet_no_top",
        "bookcase",
        "shelf",
        "wardrobe",
        "desk",
        "breakfast_table",
        "coffee_table",
        "console_table",
        "pedestal_table",
        "nightstand",
    }
    seating_priority = {
        "straight_chair",
        "swivel_chair",
        "armchair",
    }
    medium_priority = {
        "carpet",
        "ottoman",
        "bench",
        "floor_lamp",
        "standing_tv",
        "standing_loudspeaker",
        "stand",
        "locker",
    }
    if category in primary_priority:
        return 0
    if category in storage_and_tables:
        return 1
    if category in seating_priority:
        return 2
    if category in medium_priority:
        return 3
    return 4


__all__ = [
    "SceneVariantGenerator",
    "VariantConfig",
    "VariantResult",
    "_room_adjacency",
    "_rooms_are_connected",
]
