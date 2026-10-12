from __future__ import annotations
import hashlib
import json
from dataclasses import asdict
from typing import Any, Mapping, Sequence
from scoper.benchmark.procthor.export.schema import BACKEND, DERIVED_KINDS, EPISODE_KEY_RE, ROLLOUT_ARMS, SCHEMA_VERSIONS, SEED_NAMESPACES, SHA256_RE, SchemaValidationError, validate_json_value
from scoper.benchmark.procthor.export.types import EpisodeKeyInput

def canonical_json_bytes(value: Any) -> bytes:
    validate_json_value(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError('sha256_bytes expects bytes')
    return hashlib.sha256(value).hexdigest()

def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))

def episode_key_payload(value: EpisodeKeyInput) -> dict[str, Any]:
    if not isinstance(value, EpisodeKeyInput):
        raise SchemaValidationError('episode_key_input_invalid', 'value must be EpisodeKeyInput')
    payload = asdict(value)
    _validate_episode_key_payload(payload)
    return payload

def build_episode_identity(value: EpisodeKeyInput) -> tuple[str, str]:
    payload = episode_key_payload(value)
    digest = sha256_json(payload)
    return ('ebk1_' + digest, 'ep-' + digest[:24])

def derive_seed(episode_key: str, namespace: str) -> int:
    if not isinstance(episode_key, str) or EPISODE_KEY_RE.fullmatch(episode_key) is None:
        raise SchemaValidationError('episode_key_invalid', 'episode_key must be ebk1_ followed by lowercase SHA256')
    if namespace not in SEED_NAMESPACES:
        raise SchemaValidationError('seed_namespace_invalid', 'namespace must be one of {}'.format(', '.join(SEED_NAMESPACES)))
    digest = hashlib.sha256(episode_key.encode('utf-8') + b'\x00' + namespace.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % 2 ** 31

def build_derived_sample_id(*, kind: str, exporter_version: str, label_policy_version: str, source_episode_id: str, source_record_ids: Sequence[str], source_frame_ids: Sequence[str]) -> str:
    if kind not in DERIVED_KINDS:
        raise SchemaValidationError('derived_kind_invalid', 'unsupported derived kind')
    for name, item in (('exporter_version', exporter_version), ('label_policy_version', label_policy_version), ('source_episode_id', source_episode_id)):
        if not isinstance(item, str) or not item:
            raise SchemaValidationError('derived_identity_invalid', '{} must be a non-empty string'.format(name))
    payload = {'exporter_version': exporter_version, 'kind': kind, 'label_policy_version': label_policy_version, 'source_episode_id': source_episode_id, 'source_frame_ids': list(source_frame_ids), 'source_record_ids': list(source_record_ids)}
    validate_json_value(payload)
    return '{}-{}'.format(kind, sha256_json(payload)[:24])

def build_fact_id(canonical_fact_key: Mapping[str, Any]) -> str:
    return 'fact-' + sha256_json(dict(canonical_fact_key))[:16]

def record_id(ordinal: int) -> str:
    return 'rec-{:08d}'.format(_positive_ordinal(ordinal, 'record ordinal'))

def decision_id(step_id: int) -> str:
    return 'dec-{:06d}'.format(_positive_ordinal(step_id, 'step_id'))

def action_id(step_id: int) -> str:
    return 'act-{:06d}'.format(_positive_ordinal(step_id, 'step_id'))

def low_level_id(step_id: int, ordinal: int) -> str:
    return 'll-{:06d}-{:06d}'.format(_positive_ordinal(step_id, 'step_id'), _positive_ordinal(ordinal, 'low-level ordinal'))

def stream_id(step_id: int) -> str:
    return 'stream-{:06d}'.format(_positive_ordinal(step_id, 'step_id'))

def stream_obs_id(step_id: int, ordinal: int) -> str:
    return 'sobs-{:06d}-{:06d}'.format(_positive_ordinal(step_id, 'step_id'), _positive_ordinal(ordinal, 'stream observation ordinal'))

def frame_id(ordinal: int) -> str:
    return 'frm-{:08d}'.format(_positive_ordinal(ordinal, 'frame ordinal'))

def state_snapshot_id(ordinal: int) -> str:
    return 'state-{:08d}'.format(_positive_ordinal(ordinal, 'state ordinal'))

def intervention_id(ordinal: int) -> str:
    return 'int-{:06d}'.format(_positive_ordinal(ordinal, 'intervention ordinal'))

def model_call_id(ordinal: int) -> str:
    return 'mcall-{:08d}'.format(_positive_ordinal(ordinal, 'model call ordinal'))

def verification_id(ordinal: int) -> str:
    return 'ver-{:06d}'.format(_positive_ordinal(ordinal, 'verification ordinal'))

def correction_id(ordinal: int) -> str:
    return 'corr-{:06d}'.format(_positive_ordinal(ordinal, 'correction ordinal'))

def binding_id(ordinal: int) -> str:
    return 'bind-{:08d}'.format(_positive_ordinal(ordinal, 'binding ordinal'))

def audit_id(ordinal: int) -> str:
    return 'audit-{:08d}'.format(_positive_ordinal(ordinal, 'audit ordinal'))

def _validate_episode_key_payload(payload: Mapping[str, Any]) -> None:
    expected_keys = {'schema_version', 'backend', 'task_id', 'task_hash', 'source_split', 'scene_id', 'layout_id', 'execution_profile', 'rollout_arm', 'intervention_hash', 'policy_id', 'policy_hash', 'protocol_hash', 'base_seed', 'repeat_id'}
    if set(payload) != expected_keys:
        raise SchemaValidationError('episode_key_input_invalid', 'EpisodeKeyInput field set differs from contract')
    if payload['schema_version'] != SCHEMA_VERSIONS['episode_key']:
        raise SchemaValidationError('episode_key_input_invalid', 'schema_version must be episode-key.v1')
    if payload['backend'] != BACKEND:
        raise SchemaValidationError('episode_key_input_invalid', 'backend must be eb_procthor')
    for key in ('task_id', 'scene_id', 'policy_id'):
        if not isinstance(payload[key], str) or not payload[key]:
            raise SchemaValidationError('episode_key_input_invalid', '{} must be a non-empty string'.format(key))
    for key in ('task_hash', 'intervention_hash', 'policy_hash', 'protocol_hash'):
        if not isinstance(payload[key], str) or SHA256_RE.fullmatch(payload[key]) is None:
            raise SchemaValidationError('episode_key_input_invalid', '{} must be lowercase SHA256'.format(key))
    if payload['source_split'] not in ('train', 'val', 'test'):
        raise SchemaValidationError('episode_key_input_invalid', 'source_split must be train, val, or test')
    if payload['layout_id'] != '':
        raise SchemaValidationError('episode_key_input_invalid', 'ProcTHOR v1 layout_id must be empty')
    if payload['execution_profile'] not in ('native_ai2thor', 'semantic_compatibility'):
        raise SchemaValidationError('episode_key_input_invalid', 'unsupported execution_profile')
    if payload['rollout_arm'] not in ROLLOUT_ARMS:
        raise SchemaValidationError('episode_key_input_invalid', 'unsupported rollout_arm')
    base_seed = payload['base_seed']
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or (not 0 <= base_seed < 2 ** 31):
        raise SchemaValidationError('episode_key_input_invalid', 'base_seed must be in [0, 2**31)')
    repeat_id_value = payload['repeat_id']
    if isinstance(repeat_id_value, bool) or not isinstance(repeat_id_value, int) or repeat_id_value < 0:
        raise SchemaValidationError('episode_key_input_invalid', 'repeat_id must be a non-negative integer')

def _positive_ordinal(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaValidationError('ordinal_invalid', '{} must be a positive integer'.format(name))
    return value
__all__ = ['action_id', 'audit_id', 'binding_id', 'build_derived_sample_id', 'build_episode_identity', 'build_fact_id', 'canonical_json_bytes', 'correction_id', 'decision_id', 'derive_seed', 'episode_key_payload', 'frame_id', 'intervention_id', 'low_level_id', 'model_call_id', 'record_id', 'sha256_bytes', 'sha256_json', 'state_snapshot_id', 'stream_id', 'stream_obs_id', 'verification_id']
