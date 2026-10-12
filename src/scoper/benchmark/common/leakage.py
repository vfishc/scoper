from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from scoper.benchmark.common.canonical import validate_json_value
from scoper.harness.support.method_contract import MethodContractError, PRIVATE_DENY_KEYS

@dataclass(frozen=True)
class MethodLeakageFinding:
    error_code: str
    json_pointer: str
    key: str
    reason: str

def _normalize_key(value: str) -> str:
    return re.sub('[^a-z0-9]', '', value.casefold())
_PRIVATE_KEYS = {_normalize_key(key) for key in PRIVATE_DENY_KEYS}
_PRIVATE_KEYS.update({'objectids', 'privateids', 'evaluatortruth', 'goaltruth', 'oraclecontext', 'privilegedpayload', 'resolvedtarget', 'selectedobjectid', 'symbolicstate', 'teacherprivatecontext'})
_GEOMETRY_KEYS = {'agentpose', 'agentposition', 'bbox', 'boundingbox', 'camerahorizon', 'cameraintrinsics', 'camerapose', 'coordinates', 'depth', 'distance', 'exactpose', 'geometry', 'heading', 'horizon', 'location', 'mask', 'maskpath', 'observationpose', 'pitch', 'pose', 'position', 'rotation', 'roompose', 'roll', 'segmentation', 'segmentationmask', 'worldcoordinates', 'x', 'xyz', 'y', 'yaw', 'z'}
_FUTURE_ACTION_KEYS = {'futureaction', 'futureteacheraction', 'interrupttoolcall', 'nextaction', 'nextteacheraction', 'nexttoolcall', 'recommendedaction', 'recommendedtoolcall', 'teacheraction', 'teacherfutureaction'}
_CREDENTIAL_KEYS = {'apikey', 'authorization', 'credential', 'credentials', 'password', 'refreshtoken', 'secret', 'token'}
_RAW_OBJECT_VALUE_RE = re.compile('^[A-Za-z][A-Za-z0-9_.:-]*\\|[^|\\s]+(?:\\|[^|\\s]+)*$')
_WINDOWS_ABSOLUTE_RE = re.compile('^[A-Za-z]:[\\\\/]')

def _escape(value: str) -> str:
    return value.replace('~', '~0').replace('/', '~1')

def _finding(code: str, pointer: str, key: str, reason: str) -> MethodLeakageFinding:
    return MethodLeakageFinding(code, pointer or '/', key, reason)

def find_method_leakage(value: Any, *, root_pointer: str='/model_input', extra_private_keys: Sequence[str]=(), include_geometry: bool=True, include_future_action: bool=True) -> tuple[MethodLeakageFinding, ...]:
    try:
        validate_json_value(value, json_pointer=root_pointer)
    except ValueError as exc:
        raise MethodContractError('method_schema_invalid', str(exc), pointer=root_pointer) from exc
    private_keys = _PRIVATE_KEYS | {_normalize_key(key) for key in extra_private_keys}
    findings: list[MethodLeakageFinding] = []

    def walk(item: Any, pointer: str, parent_key: str | None) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item, key=lambda candidate: str(candidate).encode('ascii', errors='backslashreplace')):
                if not isinstance(key, str):
                    findings.append(_finding('method_schema_invalid', pointer, str(key), 'mapping key must be a string'))
                    continue
                child_pointer = f'{pointer}/{_escape(key)}'
                normalized = _normalize_key(key)
                if normalized in private_keys or normalized.startswith(('private', 'privileged')):
                    findings.append(_finding('method_private_key_forbidden', child_pointer, key, f'private-only key {key!r} is forbidden'))
                elif normalized == 'core' and parent_key == 'env':
                    findings.append(_finding('method_private_key_forbidden', child_pointer, key, 'private-only env.core path is forbidden'))
                elif include_geometry and normalized in _GEOMETRY_KEYS:
                    findings.append(_finding('method_geometry_forbidden', child_pointer, key, f'exact geometry key {key!r} is forbidden'))
                elif include_future_action and normalized in _FUTURE_ACTION_KEYS:
                    findings.append(_finding('method_future_action_forbidden', child_pointer, key, f'future teacher/action key {key!r} is forbidden'))
                elif normalized in _CREDENTIAL_KEYS or normalized.endswith(('apikey', 'password', 'accesstoken')):
                    findings.append(_finding('method_private_key_forbidden', child_pointer, key, f'credential-bearing key {key!r} is forbidden'))
                walk(item[key], child_pointer, normalized)
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f'{pointer}/{index}', parent_key)
            return
        if isinstance(item, str):
            if item.startswith('data:image/'):
                findings.append(_finding('method_embedded_media_forbidden', pointer, parent_key or '', 'embedded image data is forbidden; use ImageRef'))
            if _RAW_OBJECT_VALUE_RE.fullmatch(item):
                findings.append(_finding('method_raw_object_handle_forbidden', pointer, parent_key or '', 'raw simulator object handle is forbidden'))
            locator_key = parent_key and any((token in parent_key for token in ('path', 'file', 'locator')))
            if item.startswith('file://') or (locator_key and (item.startswith('/') or _WINDOWS_ABSOLUTE_RE.match(item))):
                findings.append(_finding('method_machine_locator_forbidden', pointer, parent_key or '', 'machine-local locator is forbidden'))
    walk(value, root_pointer.rstrip('/') or '', None)
    findings.sort(key=lambda finding: (finding.json_pointer.encode('utf-8'), finding.error_code))
    return tuple(findings)

def assert_no_method_leakage(value: Any, *, root_pointer: str='/model_input', extra_private_keys: Sequence[str]=(), include_geometry: bool=True, include_future_action: bool=True) -> None:
    findings = find_method_leakage(value, root_pointer=root_pointer, extra_private_keys=extra_private_keys, include_geometry=include_geometry, include_future_action=include_future_action)
    if findings:
        first = findings[0]
        raise MethodContractError(first.error_code, first.reason, pointer=first.json_pointer)

def strip_geometry(value: Any, *, root_pointer: str='/model_input') -> tuple[Any, tuple[str, ...]]:
    removed: list[str] = []

    def project(item: Any, pointer: str) -> Any:
        if isinstance(item, Mapping):
            output: dict[str, Any] = {}
            for key in sorted(item, key=lambda candidate: str(candidate).encode('ascii', errors='backslashreplace')):
                if not isinstance(key, str):
                    raise MethodContractError('method_schema_invalid', 'mapping key must be a string', pointer=pointer)
                child_pointer = f'{pointer}/{_escape(key)}'
                if _normalize_key(key) in _GEOMETRY_KEYS:
                    removed.append(child_pointer)
                    continue
                output[key] = project(item[key], child_pointer)
            return output
        if isinstance(item, list):
            return [project(child, f'{pointer}/{index}') for index, child in enumerate(item)]
        return item
    projected = project(value, root_pointer.rstrip('/') or '')
    return (projected, tuple(sorted(removed, key=lambda pointer: pointer.encode('utf-8'))))
__all__ = ['MethodLeakageFinding', 'assert_no_method_leakage', 'find_method_leakage', 'strip_geometry']
