from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.procthor.export.schema import TrajectoryExportError, validate_json_value
_PRIVATE_KEYS = {'objectid', 'object_id', 'object_ids', 'private_id', 'private_ids', 'private_entity_ids', 'goal_predicates', 'scheduler_truth', 'state_snapshot', 'state_snapshot_id', 'role_bindings', 'metadata', 'privileged', 'native_truth', 'effective_truth', 'native_state', 'effective_state', 'house_path', 'original_procthor_split', 'selected_object_id', 'target_object_ids', 'visible_object_ids', 'resolved_target', 'mask_path', 'instance_bbox_refs', 'segmentation_refs'}
_CREDENTIAL_KEYS = {'api_key', 'apikey', 'access_key', 'access_token', 'authorization', 'cookie', 'credential', 'credentials', 'password', 'secret', 'set_cookie', 'token'}
_ENV_DUMP_KEYS = {'env', 'environ', 'environment_variables', 'process_environment'}
_PATH_KEYS = {'path', 'file', 'filename', 'directory', 'dir', 'root', 'metadata_path', 'config_path'}
_AI2THOR_OBJECT_ID_RE = re.compile('^[A-Za-z][A-Za-z0-9_]*(?:\\|\\S+)+$')
_WINDOWS_PATH_RE = re.compile('^(?:[A-Za-z]:[\\\\/]|\\\\\\\\|//)')
_CREDENTIAL_VALUE_RES = (re.compile('(?i)^bearer\\s+\\S+'), re.compile('^sk-[A-Za-z0-9_-]{12,}$'), re.compile('^AKIA[0-9A-Z]{16}$'), re.compile('(?i)[?&](?:x-amz-signature|signature|token|api_key)=([^&]+)'))

@dataclass(frozen=True)
class LeakageFinding:
    error_code: str
    json_pointer: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {'error_code': self.error_code, 'json_pointer': self.json_pointer, 'message': self.message}

class LeakageError(TrajectoryExportError):

    def __init__(self, finding: LeakageFinding):
        super().__init__(finding.error_code, finding.message, finding.json_pointer)
        self.finding = finding

def audit_public_value(value: Any, *, root_pointer: str='', forbidden_absolute_prefixes: Sequence[Path | str]=()) -> tuple[LeakageFinding, ...]:
    validate_json_value(value, json_pointer=root_pointer)
    prefixes = tuple((str(Path(prefix).expanduser().absolute()) for prefix in forbidden_absolute_prefixes if str(prefix)))
    findings: list[LeakageFinding] = []
    _walk(value, root_pointer, prefixes, findings)
    return tuple(sorted(findings, key=lambda item: (item.json_pointer, item.error_code)))

def assert_no_public_leakage(value: Any, *, root_pointer: str='', forbidden_absolute_prefixes: Sequence[Path | str]=()) -> None:
    findings = audit_public_value(value, root_pointer=root_pointer, forbidden_absolute_prefixes=forbidden_absolute_prefixes)
    if findings:
        raise LeakageError(findings[0])

def audit_operational_value(value: Any, *, root_pointer: str='', forbidden_absolute_prefixes: Sequence[Path | str]=()) -> tuple[LeakageFinding, ...]:
    validate_json_value(value, json_pointer=root_pointer)
    prefixes = tuple((str(Path(prefix).expanduser().absolute()) for prefix in forbidden_absolute_prefixes if str(prefix)))
    findings: list[LeakageFinding] = []
    _walk_operational(value, root_pointer, prefixes, findings, path_like=False)
    return tuple(sorted(findings, key=lambda item: (item.json_pointer, item.error_code)))

def assert_no_operational_leakage(value: Any, *, root_pointer: str='', forbidden_absolute_prefixes: Sequence[Path | str]=()) -> None:
    findings = audit_operational_value(value, root_pointer=root_pointer, forbidden_absolute_prefixes=forbidden_absolute_prefixes)
    if findings:
        raise LeakageError(findings[0])

def audit_public_private_sidecars(index_records: Sequence[Mapping[str, Any]], public_records: Sequence[Mapping[str, Any]], privileged_records: Sequence[Mapping[str, Any]], *, forbidden_absolute_prefixes: Sequence[Path | str]=()) -> tuple[LeakageFinding, ...]:
    public_by_id = _unique_records(public_records, 'public')
    private_by_id = _unique_records(privileged_records, 'privileged')
    findings: list[LeakageFinding] = []
    seen_index: set[str] = set()
    for index, record in enumerate(index_records):
        record_id = record.get('record_id')
        pointer = '/trajectory/{}'.format(index)
        if not isinstance(record_id, str) or record_id in seen_index:
            findings.append(LeakageFinding('sidecar_record_invalid', pointer + '/record_id', 'index record_id is missing or duplicated'))
            continue
        seen_index.add(record_id)
        for flag, records, name in ((record.get('public_payload'), public_by_id, 'public'), (record.get('privileged_payload'), private_by_id, 'privileged')):
            present = record_id in records
            if flag is not present:
                findings.append(LeakageFinding('sidecar_presence_mismatch', pointer + '/{}_payload'.format(name), '{} sidecar presence differs from index flag'.format(name)))
        if record_id in public_by_id:
            findings.extend(audit_public_value(public_by_id[record_id], root_pointer='/public/{}'.format(record_id), forbidden_absolute_prefixes=forbidden_absolute_prefixes))
    for name, records in (('public', public_by_id), ('privileged', private_by_id)):
        for record_id in records:
            if record_id not in seen_index:
                findings.append(LeakageFinding('sidecar_orphan_record', '/{}/{}'.format(name, record_id), 'sidecar record has no index record'))
    return tuple(sorted(findings, key=lambda item: (item.json_pointer, item.error_code)))

def _walk(value: Any, pointer: str, prefixes: tuple[str, ...], findings: list[LeakageFinding]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = '{}/{}'.format(pointer, _escape_pointer(key))
            lowered = key.casefold().replace('-', '_')
            if lowered in _PRIVATE_KEYS or lowered.endswith('_object_id') or lowered.endswith('_object_ids') or lowered.startswith('privileged_'):
                findings.append(LeakageFinding('public_private_key_forbidden', child, "private key '{}' is forbidden in public/model input".format(key)))
            if lowered in _CREDENTIAL_KEYS or lowered.endswith(('_api_key', '_access_token', '_password', '_secret')):
                findings.append(LeakageFinding('public_credential_key_forbidden', child, "credential-bearing key '{}' is forbidden".format(key)))
            if lowered in _ENV_DUMP_KEYS:
                findings.append(LeakageFinding('public_environment_dump_forbidden', child, 'environment dumps are forbidden'))
            _walk(item, child, prefixes, findings)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk(item, '{}/{}'.format(pointer, index), prefixes, findings)
        return
    if not isinstance(value, str):
        return
    if value.startswith('data:image/'):
        findings.append(LeakageFinding('public_inline_image_forbidden', pointer or '/', 'model/public JSON must reference saved frames instead of inline image bytes'))
    if _AI2THOR_OBJECT_ID_RE.fullmatch(value) is not None:
        findings.append(LeakageFinding('public_object_id_value_forbidden', pointer or '/', 'raw AI2-THOR object id is forbidden in public/model input'))
    if value.startswith(('/', '~/', 'file:')) or _WINDOWS_PATH_RE.match(value):
        findings.append(LeakageFinding('public_absolute_path_forbidden', pointer or '/', 'machine-local path is forbidden in public/model input'))
    elif any((prefix in value for prefix in prefixes)):
        findings.append(LeakageFinding('public_absolute_path_forbidden', pointer or '/', 'machine-local path prefix is forbidden in public/model input'))
    if any((pattern.search(value) is not None for pattern in _CREDENTIAL_VALUE_RES)):
        findings.append(LeakageFinding('public_credential_value_forbidden', pointer or '/', 'credential-like value is forbidden in public/model input'))

def _walk_operational(value: Any, pointer: str, prefixes: tuple[str, ...], findings: list[LeakageFinding], path_like: bool) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = '{}/{}'.format(pointer, _escape_pointer(key))
            lowered = key.casefold().replace('-', '_')
            if lowered in _CREDENTIAL_KEYS or lowered.endswith(('_api_key', '_access_token', '_password', '_secret')):
                findings.append(LeakageFinding('operational_credential_key_forbidden', child, "credential-bearing key '{}' is forbidden".format(key)))
            if lowered in _ENV_DUMP_KEYS:
                findings.append(LeakageFinding('operational_environment_dump_forbidden', child, 'environment dumps are forbidden'))
            if lowered == 'json_pointer':
                continue
            child_path_like = lowered in _PATH_KEYS or lowered.endswith(('_path', '_file', '_dir', '_root'))
            _walk_operational(item, child, prefixes, findings, path_like=child_path_like)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_operational(item, '{}/{}'.format(pointer, index), prefixes, findings, path_like=path_like)
        return
    if not isinstance(value, str):
        return
    contains_prefix = any((prefix in value for prefix in prefixes))
    if path_like and (value.startswith(('/', '~/', 'file:')) or _WINDOWS_PATH_RE.match(value)):
        findings.append(LeakageFinding('operational_absolute_path_forbidden', pointer or '/', 'machine-local path is forbidden in run/provenance data'))
    elif contains_prefix:
        findings.append(LeakageFinding('operational_absolute_path_forbidden', pointer or '/', 'machine-local path prefix is forbidden in run/provenance data'))
    if any((pattern.search(value) is not None for pattern in _CREDENTIAL_VALUE_RES)):
        findings.append(LeakageFinding('operational_credential_value_forbidden', pointer or '/', 'credential-like value is forbidden in run/provenance data'))

def _unique_records(records: Sequence[Mapping[str, Any]], sidecar: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        record_id = record.get('record_id')
        if not isinstance(record_id, str) or not record_id:
            raise LeakageError(LeakageFinding('sidecar_record_invalid', '/{}/{}/record_id'.format(sidecar, index), 'sidecar record_id must be a non-empty string'))
        if record_id in result:
            raise LeakageError(LeakageFinding('sidecar_record_invalid', '/{}/{}/record_id'.format(sidecar, index), 'duplicate sidecar record_id'))
        result[record_id] = record
    return result

def _escape_pointer(value: str) -> str:
    return value.replace('~', '~0').replace('/', '~1')
__all__ = ['LeakageError', 'LeakageFinding', 'assert_no_operational_leakage', 'assert_no_public_leakage', 'audit_operational_value', 'audit_public_private_sidecars', 'audit_public_value']
