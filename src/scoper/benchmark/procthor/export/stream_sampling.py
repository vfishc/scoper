from __future__ import annotations
import math
from typing import Any, Mapping
from scoper.benchmark.procthor.export.schema import SchemaValidationError
STREAM_SAMPLING_SCHEMA_VERSION = 'procthor-stream-sampling.v1'
STREAM_SAMPLING_MODE = 'first_visual_event_at_or_after_logical_tick'
TRANSLATION_METERS_PER_SECOND = 0.4
ANGULAR_RADIANS_PER_SECOND = 0.8
ACTUATION_SECONDS = 0.6
METADATA_QUERY_SECONDS = 0.0
COMPATIBILITY_ACTION_SECONDS = 0.6
_METADATA_QUERY_ACTIONS = {'GetReachablePositions', 'GetSpawnCoordinatesAboveReceptacle'}
_COMPATIBILITY_ACTIONS = {'SetObjectPoses', 'SetObjectStates', 'Teleport', 'TeleportFull', 'TeleportObject'}

def stream_sampling_config(stream_fps: float) -> dict[str, Any]:
    fps = validate_stream_fps(stream_fps)
    return {'schema_version': STREAM_SAMPLING_SCHEMA_VERSION, 'mode': STREAM_SAMPLING_MODE, 'fps': fps, 'phase_seconds': 0.0, 'timing_model': {'translation_meters_per_second': TRANSLATION_METERS_PER_SECOND, 'angular_radians_per_second': ANGULAR_RADIANS_PER_SECOND, 'actuation_seconds': ACTUATION_SECONDS, 'metadata_query_seconds': METADATA_QUERY_SECONDS, 'compatibility_action_seconds': COMPATIBILITY_ACTION_SECONDS}}

def validate_stream_fps(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(float(value))) or (not 0.0 < float(value) <= 60.0):
        raise SchemaValidationError('numeric_config_invalid', 'stream FPS must be finite and in (0, 60]')
    return float(value)

def select_logical_stream_sample(logical_time: float, *, stream_fps: float | None, next_sample_time: float) -> tuple[bool, float]:
    if stream_fps is None:
        return (True, float(next_sample_time))
    period = 1.0 / validate_stream_fps(stream_fps)
    due = float(logical_time) + 1e-09 >= float(next_sample_time)
    if not due:
        return (False, float(next_sample_time))
    elapsed_ticks = math.floor((float(logical_time) - float(next_sample_time) + 1e-09) / period)
    return (True, float(next_sample_time) + (elapsed_ticks + 1) * period)

def logical_action_duration(record: Mapping[str, Any]) -> float:
    action = str(record.get('executed_action') or record.get('action') or '')
    if action in _METADATA_QUERY_ACTIONS:
        return METADATA_QUERY_SECONDS
    if action in _COMPATIBILITY_ACTIONS:
        return COMPATIBILITY_ACTION_SECONDS
    before = record.get('pose_before')
    after = record.get('pose_after')
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        before_pos = before.get('position') if isinstance(before.get('position'), Mapping) else before
        after_pos = after.get('position') if isinstance(after.get('position'), Mapping) else after
        try:
            distance = math.sqrt(sum(((float(after_pos[axis]) - float(before_pos[axis])) ** 2 for axis in ('x', 'y', 'z'))))
        except (KeyError, TypeError, ValueError):
            distance = None
        if distance is not None and distance > 0.0:
            return distance / TRANSLATION_METERS_PER_SECOND
    if 'Rotate' in action or 'Look' in action:
        degrees = record.get('degrees')
        if isinstance(degrees, (int, float)) and (not isinstance(degrees, bool)):
            return abs(float(degrees)) * math.pi / 180.0 / ANGULAR_RADIANS_PER_SECOND
    move = record.get('move_magnitude')
    if not isinstance(move, (int, float)) or isinstance(move, bool):
        move = record.get('moveMagnitude')
    if isinstance(move, (int, float)) and (not isinstance(move, bool)) and (float(move) > 0.0):
        return float(move) / TRANSLATION_METERS_PER_SECOND
    return ACTUATION_SECONDS
__all__ = ['ACTUATION_SECONDS', 'ANGULAR_RADIANS_PER_SECOND', 'COMPATIBILITY_ACTION_SECONDS', 'METADATA_QUERY_SECONDS', 'STREAM_SAMPLING_MODE', 'STREAM_SAMPLING_SCHEMA_VERSION', 'TRANSLATION_METERS_PER_SECOND', 'logical_action_duration', 'select_logical_stream_sample', 'stream_sampling_config', 'validate_stream_fps']
