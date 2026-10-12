from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from scoper.benchmark.behavior.export.types import BundleQualityError, TrajectoryConfigError

@dataclass(frozen=True)
class LeakageFinding:
    error_code: str
    json_pointer: str
    key: str
    reason: str
_PRIVATE_KEYS = {'object_id', 'objectid', 'object_ids', 'objectids', 'private_id', 'private_ids', 'goal_predicates', 'scheduler_truth', 'state_snapshot', 'state_snapshot_id', 'state_snapshots', 'role_bindings', 'resolved_target', 'selected_object_id', 'visible_object_ids', 'target_object_ids', 'held_object_ids', 'held_object_ids_before', 'source_json', 'template_file', 'mask_path', 'camera_pose_truth', 'camera_intrinsics_truth', 'agent_pose_truth', 'evaluator_truth', 'full_task_spec', 'teacher_private_context', 'source_split', 'task_split', 'lineage_id', 'task_hash'}
_PRIVATE_PREFIXES = ('privileged', 'private_')
_KNOWN_EXPOSURE_PROFILES = {'visual_minimal', 'structured_agent', 'privileged_oracle'}
_CREDENTIAL_KEYS = {'api_key', 'apikey', 'authorization', 'auth_header', 'access_token', 'refresh_token', 'secret', 'password', 'credential', 'credentials', 'credential_file'}
_CREDENTIAL_VALUE_RE = re.compile('(?:sk-[A-Za-z0-9_-]{12,}|bearer\\s+[A-Za-z0-9._~+/=-]{8,}|AKIA[0-9A-Z]{16}|authorization\\s*[:=])', re.IGNORECASE)
_RAW_OBJECT_VALUE_RE = re.compile('^[A-Za-z][A-Za-z0-9_.:-]*\\|[^|\\s]+(?:\\|[^|\\s]+)*$')

def find_public_leakage(value: Any, *, exposure_profile: str='visual_minimal', root_pointer: str='') -> tuple[LeakageFinding, ...]:
    if exposure_profile not in _KNOWN_EXPOSURE_PROFILES:
        raise TrajectoryConfigError('exposure_profile_invalid', f'unknown exposure profile {exposure_profile!r}')
    findings: list[LeakageFinding] = []
    _walk_public(value, pointer=root_pointer, findings=findings, parent_key=None)
    return tuple(findings)

def assert_no_public_leakage(value: Any, *, exposure_profile: str='visual_minimal', root_pointer: str='', allow_privileged_oracle: bool=False) -> None:
    findings = find_public_leakage(value, exposure_profile=exposure_profile, root_pointer=root_pointer)
    if not findings:
        return
    if exposure_profile == 'privileged_oracle' and allow_privileged_oracle:
        return
    first = findings[0]
    raise BundleQualityError(first.error_code, first.reason, pointer=first.json_pointer)

def audit_sensitive_metadata(value: Any, *, repo_root: Path | None=None) -> tuple[LeakageFinding, ...]:
    findings: list[LeakageFinding] = []
    prefixes = [str(Path.home().resolve())]
    if repo_root is not None:
        prefixes.append(str(Path(repo_root).resolve()))

    def walk(item: Any, pointer: str, key: str | None) -> None:
        if isinstance(item, Mapping):
            for child_key, child in item.items():
                if not isinstance(child_key, str):
                    findings.append(LeakageFinding('metadata_key_invalid', pointer or '/', str(child_key), 'non-string metadata key'))
                    continue
                child_pointer = f'{pointer}/{_escape(child_key)}'
                normalized = child_key.casefold()
                if normalized in _CREDENTIAL_KEYS or normalized.endswith(('_api_key', '_access_token', '_password')):
                    findings.append(LeakageFinding('credential_leakage', child_pointer, child_key, 'credential-bearing metadata key is forbidden'))
                if normalized in {'env', 'environment', 'environ', 'environment_variables'} and isinstance(child, Mapping):
                    findings.append(LeakageFinding('environment_dump_forbidden', child_pointer, child_key, 'environment variable dumps are forbidden'))
                walk(child, child_pointer, child_key)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f'{pointer}/{index}', key)
        elif isinstance(item, str):
            if _CREDENTIAL_VALUE_RE.search(item):
                findings.append(LeakageFinding('credential_leakage', pointer or '/', key or '', 'credential-like value is forbidden'))
            for prefix in prefixes:
                if prefix not in {'', '/'} and prefix in item:
                    findings.append(LeakageFinding('machine_locator_leakage', pointer or '/', key or '', 'machine-local absolute prefix is forbidden'))
                    break
    walk(value, '', None)
    return tuple(findings)

def assert_safe_metadata(value: Any, *, repo_root: Path | None=None) -> None:
    findings = audit_sensitive_metadata(value, repo_root=repo_root)
    if findings:
        first = findings[0]
        raise TrajectoryConfigError(first.error_code, first.reason, pointer=first.json_pointer)

def _walk_public(value: Any, *, pointer: str, findings: list[LeakageFinding], parent_key: str | None) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                findings.append(LeakageFinding('public_key_invalid', pointer or '/', str(key), 'public mapping key must be a string'))
                continue
            child_pointer = f'{pointer}/{_escape(key)}'
            normalized = key.casefold()
            if normalized in _PRIVATE_KEYS or normalized.startswith(_PRIVATE_PREFIXES):
                findings.append(LeakageFinding('public_private_key_forbidden', child_pointer, key, f'private-only key {key!r} is forbidden in public data'))
            if normalized in _CREDENTIAL_KEYS or normalized.endswith(('_api_key', '_token', '_password')):
                findings.append(LeakageFinding('public_credential_forbidden', child_pointer, key, f'credential-bearing key {key!r} is forbidden in public data'))
            _walk_public(item, pointer=child_pointer, findings=findings, parent_key=normalized)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_public(item, pointer=f'{pointer}/{index}', findings=findings, parent_key=parent_key)
        return
    if isinstance(value, str):
        if _CREDENTIAL_VALUE_RE.search(value):
            findings.append(LeakageFinding('public_credential_forbidden', pointer or '/', parent_key or '', 'credential-like string is forbidden in public data'))
        if parent_key and any((token in parent_key for token in ('entity', 'target', 'object'))):
            if _RAW_OBJECT_VALUE_RE.fullmatch(value):
                findings.append(LeakageFinding('public_raw_object_handle_forbidden', pointer or '/', parent_key, 'raw simulator object handle is forbidden in public data'))

def _escape(value: str) -> str:
    return value.replace('~', '~0').replace('/', '~1')
__all__ = ['LeakageFinding', 'assert_no_public_leakage', 'assert_safe_metadata', 'audit_sensitive_metadata', 'find_public_leakage']
