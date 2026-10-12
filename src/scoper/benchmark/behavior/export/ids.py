from __future__ import annotations
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from scoper.benchmark.behavior.export.types import EpisodeKeyInput, TrajectoryConfigError
SHA256_RE = re.compile('^[0-9a-f]{64}$')
RUN_ID_RE = re.compile('^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
SOURCE_SPLIT_RE = re.compile('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
SEED_NAMESPACES = ('environment', 'scheduler', 'model', 'data_sampling', 'renderer')
ROLLOUT_ARMS = ('clean', 'disturbance', 'failure', 'mixed', 'learner')

def _pointer_token(value: object) -> str:
    return str(value).replace('~', '~0').replace('/', '~1')

def validate_json_value(value: Any, *, pointer: str='') -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrajectoryConfigError('non_finite_json', 'NaN and Infinity are forbidden', pointer=pointer or '/')
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_value(item, pointer=f'{pointer}/{index}')
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrajectoryConfigError('non_string_json_key', 'JSON object keys must be strings', pointer=pointer or '/')
            validate_json_value(item, pointer=f'{pointer}/{_pointer_token(key)}')
        return
    raise TrajectoryConfigError('unsupported_json_type', f'unsupported JSON value type {type(value).__name__}', pointer=pointer or '/')

def canonical_json_bytes(value: Any) -> bytes:
    validate_json_value(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

def canonical_json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b'\n'

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))

def _reject_constant(value: str) -> None:
    raise TrajectoryConfigError('non_finite_json', f'forbidden JSON constant {value}')

def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrajectoryConfigError('duplicate_json_key', f'duplicate JSON object key {key!r}')
        result[key] = value
    return result

def strict_json_loads(data: str | bytes, *, source: str='<json>') -> Any:
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='strict')
        value = json.loads(data, object_pairs_hook=_strict_object_pairs, parse_constant=_reject_constant)
    except TrajectoryConfigError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrajectoryConfigError('invalid_json', f'invalid JSON in {source}: {exc}') from exc
    validate_json_value(value)
    return value

def strict_json_load(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TrajectoryConfigError('json_read_error', f'cannot read {path}: {exc}') from exc
    return strict_json_loads(raw, source=str(path))

def json_deep_copy(value: Any) -> Any:
    return strict_json_loads(canonical_json_bytes(value))

def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrajectoryConfigError('episode_key_invalid', f'{field} must be a non-empty string')
    return value

def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise TrajectoryConfigError('episode_key_invalid', f'{field} must be a lowercase sha256 hex')
    return value

def validate_episode_key_input(value: EpisodeKeyInput) -> None:
    if not isinstance(value, EpisodeKeyInput):
        raise TrajectoryConfigError('episode_key_invalid', 'value must be EpisodeKeyInput')
    if value.schema_version != 'episode-key.v1':
        raise TrajectoryConfigError('episode_key_invalid', 'schema_version must be episode-key.v1')
    if value.backend != 'scoper.benchmark.behavior':
        raise TrajectoryConfigError('episode_key_invalid', 'backend must be scoper.benchmark.behavior')
    for field in ('task_id', 'source_split', 'scene_id', 'policy_id'):
        _require_non_empty_string(getattr(value, field), field)
    if not isinstance(value.layout_id, str):
        raise TrajectoryConfigError('episode_key_invalid', 'layout_id must be a string')
    if value.execution_profile != 'semantic_compatibility':
        raise TrajectoryConfigError('episode_key_invalid', 'execution_profile must be semantic_compatibility in v1')
    if value.rollout_arm not in ROLLOUT_ARMS:
        raise TrajectoryConfigError('episode_key_invalid', 'rollout_arm is not a contract value')
    for field in ('task_hash', 'intervention_hash', 'policy_hash', 'protocol_hash'):
        _require_sha256(getattr(value, field), field)
    if isinstance(value.base_seed, bool) or not isinstance(value.base_seed, int) or (not 0 <= value.base_seed < 2 ** 31):
        raise TrajectoryConfigError('episode_key_invalid', 'base_seed must be in [0, 2**31)')
    if isinstance(value.repeat_id, bool) or not isinstance(value.repeat_id, int) or value.repeat_id < 0:
        raise TrajectoryConfigError('episode_key_invalid', 'repeat_id must be a non-negative integer')

def build_episode_identity(value: EpisodeKeyInput) -> tuple[str, str]:
    validate_episode_key_input(value)
    digest = sha256_json(asdict(value))
    return (f'ebk1_{digest}', f'ep-{digest[:24]}')

def episode_key_payload(value: EpisodeKeyInput) -> Mapping[str, Any]:
    validate_episode_key_input(value)
    return asdict(value)

def derive_seed(episode_key: str, namespace: str) -> int:
    if not isinstance(episode_key, str) or re.fullmatch('ebk1_[0-9a-f]{64}', episode_key) is None:
        raise TrajectoryConfigError('episode_key_invalid', 'episode_key must be ebk1_ plus 64 lowercase hex')
    if namespace not in SEED_NAMESPACES:
        raise TrajectoryConfigError('seed_namespace_invalid', f'unknown seed namespace {namespace!r}')
    digest = hashlib.sha256(episode_key.encode('utf-8') + b'\x00' + namespace.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % 2 ** 31

def derive_all_seeds(episode_key: str) -> dict[str, int]:
    return {namespace: derive_seed(episode_key, namespace) for namespace in SEED_NAMESPACES}

def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise TrajectoryConfigError('run_id_invalid', 'run_id does not match the v1 portable name pattern')
    return run_id

def ordinal_id(prefix: str, ordinal: int, *, width: int) -> str:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise TrajectoryConfigError('ordinal_invalid', 'episode-local ordinals start at 1')
    return f'{prefix}-{ordinal:0{width}d}'

def record_id(ordinal: int) -> str:
    return ordinal_id('rec', ordinal, width=8)

def frame_id(ordinal: int) -> str:
    return ordinal_id('frm', ordinal, width=8)

def state_snapshot_id(ordinal: int) -> str:
    return ordinal_id('state', ordinal, width=8)

def model_call_id(ordinal: int) -> str:
    return ordinal_id('mcall', ordinal, width=8)

def binding_id(ordinal: int) -> str:
    return ordinal_id('bind', ordinal, width=8)

def decision_id(step_id: int) -> str:
    return _step_id('dec', step_id)

def action_id(step_id: int) -> str:
    return _step_id('act', step_id)

def stream_id(step_id: int) -> str:
    return _step_id('stream', step_id)

def low_level_id(step_id: int, ordinal: int) -> str:
    _validate_step_id(step_id)
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise TrajectoryConfigError('ordinal_invalid', 'low-level ordinal must start at 1')
    return f'll-{step_id:06d}-{ordinal:06d}'

def stream_observation_id(step_id: int, ordinal: int) -> str:
    _validate_step_id(step_id)
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise TrajectoryConfigError('ordinal_invalid', 'stream observation ordinal must start at 1')
    return f'sobs-{step_id:06d}-{ordinal:06d}'

def _validate_step_id(step_id: int) -> None:
    if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 1:
        raise TrajectoryConfigError('step_id_invalid', 'action/decision step_id must start at 1')

def _step_id(prefix: str, step_id: int) -> str:
    _validate_step_id(step_id)
    return f'{prefix}-{step_id:06d}'

def fact_id(canonical_fact_key: Mapping[str, Any]) -> str:
    return f'fact-{sha256_json(canonical_fact_key)[:16]}'

def derived_sample_id(*, kind: str, exporter_version: str, label_policy_version: str, source_episode_id: str, source_record_ids: Iterable[str], source_frame_ids: Iterable[str]) -> str:
    for field, value in (('kind', kind), ('exporter_version', exporter_version), ('label_policy_version', label_policy_version), ('source_episode_id', source_episode_id)):
        _require_non_empty_string(value, field)
    payload = {'kind': kind, 'exporter_version': exporter_version, 'label_policy_version': label_policy_version, 'source_episode_id': source_episode_id, 'source_record_ids': list(source_record_ids), 'source_frame_ids': list(source_frame_ids)}
    return f'{kind}-{sha256_json(payload)[:24]}'
__all__ = ['ROLLOUT_ARMS', 'RUN_ID_RE', 'SEED_NAMESPACES', 'SHA256_RE', 'SOURCE_SPLIT_RE', 'action_id', 'binding_id', 'build_episode_identity', 'canonical_json_bytes', 'canonical_json_line', 'decision_id', 'derive_all_seeds', 'derive_seed', 'derived_sample_id', 'episode_key_payload', 'fact_id', 'frame_id', 'json_deep_copy', 'low_level_id', 'model_call_id', 'record_id', 'sha256_bytes', 'sha256_file', 'sha256_json', 'state_snapshot_id', 'stream_id', 'stream_observation_id', 'strict_json_load', 'strict_json_loads', 'validate_episode_key_input', 'validate_json_value', 'validate_run_id']
