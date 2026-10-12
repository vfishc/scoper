from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from scoper.benchmark.procthor.export.ids import canonical_json_bytes, sha256_bytes, sha256_json
from scoper.benchmark.procthor.export.schema import strict_json_loads
METHOD_SAMPLE_VERSION = 'method-sample.v1'
METHOD_MEDIA_MANIFEST_VERSION = 'method-media-manifest.v1'
METHOD_DATASET_MANIFEST_VERSION = 'method-dataset-manifest.v1'
METHOD_DATASET_CONFIG_VERSION = 'method-dataset-config.v1'
METHOD_REJECT_VERSION = 'method-sample-rejection.v1'
METHOD_CONTRACT_GOLDEN_VERSION = 'method-contract-golden.v2'
METHOD_CONTRACT_GOLDEN_SHA256 = 'eab1afa6a2870160004485661e1febeba8cc6a669cc61978dbb09ff121791068'
METHOD_CONTRACT_GOLDEN_BYTES = 8143
SAMPLE_KINDS = ('verifier_transition', 'step_decision', 'recovery_decision', 'stream_gate', 'preference', 'memory_update', 'memory_reconcile', 'memory_route')
DOMAINS = ('wild', 'scoper.benchmark.behavior', 'eb_procthor')
SPLITS = ('train', 'calibration', 'dev', 'test')
OUTCOMES = ('success', 'failure', 'ambiguous', 'unknown')
OBSERVABILITY = ('observable', 'partially_observable', 'unobservable', 'unknown')
LABEL_PROVENANCE = ('human_gold', 'dataset_native', 'state_derived', 'rule_derived', 'model_proposed')
EFFECT_VOCABULARY = ('picked_up', 'released', 'relation_added_inside', 'relation_added_on_top', 'relation_removed_inside', 'relation_removed_on_top', 'opened', 'closed', 'toggled_on', 'toggled_off', 'cleaned', 'dirtied', 'sliced', 'heated', 'cooled', 'filled', 'saturated', 'relocated', 'no_effect')
EFFECT_VOCABULARY_SHA256 = sha256_json(list(EFFECT_VOCABULARY))
PRIVATE_DENY_KEYS = ('object_id', 'objectId', 'private_id', 'role_map', 'role_bindings', 'goal_predicates', 'predicate_records', 'scheduler_truth', 'state_snapshot', 'normalized_state', 'private_pose', 'oracle_hint', 'repair_plan_template', 'env.core')
STABLE_REJECT_CODES = ('method_schema_invalid', 'method_identity_mismatch', 'method_group_identity_mismatch', 'method_duplicate_sample_id', 'method_group_split_leakage', 'method_private_key_forbidden', 'method_geometry_forbidden', 'method_future_action_forbidden', 'method_reasoning_label_forbidden', 'method_machine_locator_forbidden', 'method_embedded_media_forbidden', 'method_raw_object_handle_forbidden', 'method_model_input_key_forbidden', 'method_source_not_allowed', 'method_quality_gate_failed', 'method_nontrain_source_forbidden', 'method_split_mapping_invalid', 'method_media_invalid', 'method_media_missing', 'method_media_tampered', 'method_manifest_invalid', 'method_manifest_tampered', 'method_sample_join_invalid', 'source_effect_conflict', 'required_frame_missing', 'legacy_callback_rate_stream_ineligible', 'legacy_logical_time_inconsistent_stream_ineligible', 'unverified_stream_sampling_ineligible', 'evidence_replay_dependency_invalid')
from scoper.harness.support.method_contract import MethodContractError

def method_golden_path() -> Path:
    return Path(__file__).resolve().parents[2] / 'benchmark/procthor/docs/spec_assets' / 'method_contract_golden.json'

def load_and_validate_method_golden(path: Path | None=None) -> Mapping[str, Any]:
    source = method_golden_path() if path is None else Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise MethodContractError('method_contract_golden_invalid', f'cannot read {source}: {exc}') from exc
    if len(raw) != METHOD_CONTRACT_GOLDEN_BYTES:
        raise MethodContractError('method_contract_golden_invalid', 'golden byte length does not match the contract')
    if sha256_bytes(raw) != METHOD_CONTRACT_GOLDEN_SHA256:
        raise MethodContractError('method_contract_golden_invalid', 'golden sha256 does not match the contract')
    try:
        parsed = strict_json_loads(raw, source=str(source))
    except ValueError as exc:
        raise MethodContractError('method_contract_golden_invalid', str(exc)) from exc
    if not isinstance(parsed, Mapping):
        raise MethodContractError('method_contract_golden_invalid', 'golden root must be a mapping')
    if raw != canonical_json_bytes(parsed) + b'\n':
        raise MethodContractError('method_contract_golden_invalid', 'golden bytes are not canonical JSON plus newline')
    if parsed.get('schema_version') != METHOD_CONTRACT_GOLDEN_VERSION:
        raise MethodContractError('method_contract_golden_invalid', 'golden schema version is not frozen v2')
    return parsed
__all__ = ['DOMAINS', 'EFFECT_VOCABULARY', 'EFFECT_VOCABULARY_SHA256', 'LABEL_PROVENANCE', 'METHOD_CONTRACT_GOLDEN_BYTES', 'METHOD_CONTRACT_GOLDEN_SHA256', 'METHOD_CONTRACT_GOLDEN_VERSION', 'METHOD_DATASET_CONFIG_VERSION', 'METHOD_DATASET_MANIFEST_VERSION', 'METHOD_MEDIA_MANIFEST_VERSION', 'METHOD_REJECT_VERSION', 'METHOD_SAMPLE_VERSION', 'MethodContractError', 'OBSERVABILITY', 'OUTCOMES', 'PRIVATE_DENY_KEYS', 'SAMPLE_KINDS', 'SPLITS', 'STABLE_REJECT_CODES', 'load_and_validate_method_golden', 'method_golden_path']
