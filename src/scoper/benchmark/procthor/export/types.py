from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

@dataclass(frozen=True)
class EpisodeKeyInput:
    schema_version: str
    backend: str
    task_id: str
    task_hash: str
    source_split: str
    scene_id: str
    layout_id: str
    execution_profile: str
    rollout_arm: str
    intervention_hash: str
    policy_id: str
    policy_hash: str
    protocol_hash: str
    base_seed: int
    repeat_id: int

@dataclass(frozen=True)
class TaskPathResolution:
    json_pointer: str
    locator_kind: str
    normalized_locator: str
    content_sha256: str
    dataset_name: str | None
    dataset_revision: str | None

@dataclass(frozen=True)
class TaskIdentityResult:
    runtime_task_template: Mapping[str, Any]
    identity_task: Mapping[str, Any]
    task_hash: str
    path_resolutions: tuple[TaskPathResolution, ...]
    frozen_house: Mapping[str, Any]

@dataclass(frozen=True)
class MaterializedTask:
    identity: TaskIdentityResult
    runtime_task: Mapping[str, Any]
    cache_entries: tuple[Mapping[str, Any], ...]

@dataclass(frozen=True)
class RunContext:
    run_id: str
    backend: str
    manifest: Mapping[str, Any]
    protocol: Mapping[str, Any]
    policy: Mapping[str, Any]
    capture_config_document: Mapping[str, Any]

@dataclass(frozen=True)
class EpisodeContext:
    run_id: str
    episode_id: str
    episode_key: str
    episode_key_payload: Mapping[str, Any]
    task: MaterializedTask
    source_split: str
    task_split: str
    rollout_purpose: str
    execution_profile: str
    seeds: Mapping[str, int]

@dataclass(frozen=True)
class CaptureConfig:
    root: Path
    profile: str
    keyframe_encoding: str
    stream_encoding: str
    stream_jpeg_quality: int
    fsync: bool
    full_world_boundaries: bool
    stream_fps: float | None = None

@dataclass(frozen=True)
class FrameMetadata:
    frame_kind: str
    step_id: int
    action_id: str | None
    stream_id: str | None
    stream_obs_id: str | None
    logical_time_s: float
    monotonic_time_ns: int
    wall_time_utc: str
    source: str
    camera: Mapping[str, Any]
    capture: Mapping[str, Any]

@dataclass(frozen=True)
class SkippedEpisode:
    episode_id: str
    episode_key: str
    manifest_path: Path
    manifest_sha256: str
    reason: str
__all__ = ['CaptureConfig', 'EpisodeContext', 'EpisodeKeyInput', 'FrameMetadata', 'MaterializedTask', 'RunContext', 'SkippedEpisode', 'TaskIdentityResult', 'TaskPathResolution']
