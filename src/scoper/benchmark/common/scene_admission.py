from __future__ import annotations
from typing import Any, Mapping
from scoper.benchmark.common.canonical import sha256_json, validate_json_value
STATIC_SCENE_ADMISSION_SCHEMA_VERSION = 'static_scene_admission.v1'
STATIC_SCENE_ADMISSION_STATUSES = frozenset({'admitted', 'rejected', 'invalid'})
STATIC_SCENE_ADMISSION_CHECK_STATES = frozenset({'pass', 'fail', 'not_run'})
_REQUIRED_FIELDS = frozenset({'schema_version', 'backend', 'task_id', 'task_sha256', 'scene_id', 'layout_id_or_null', 'implementation_id', 'geometry_source', 'status', 'failure_code_or_null', 'checks', 'diagnostics'})

def build_static_scene_admission_result(*, backend: str, task: Mapping[str, Any], scene_id: str, layout_id_or_null: str | None, implementation_id: str, geometry_source: str, status: str, failure_code_or_null: str | None, checks: Mapping[str, str], diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    result = {'schema_version': STATIC_SCENE_ADMISSION_SCHEMA_VERSION, 'backend': str(backend), 'task_id': str(task.get('id') or ''), 'task_sha256': sha256_json(task), 'scene_id': str(scene_id), 'layout_id_or_null': None if layout_id_or_null is None else str(layout_id_or_null), 'implementation_id': str(implementation_id), 'geometry_source': str(geometry_source), 'status': str(status), 'failure_code_or_null': failure_code_or_null, 'checks': {str(name): str(value) for name, value in sorted(checks.items())}, 'diagnostics': dict(diagnostics)}
    validate_static_scene_admission_result(result)
    return result

def validate_static_scene_admission_result(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != _REQUIRED_FIELDS:
        raise ValueError('static scene admission result fields are invalid')
    if value.get('schema_version') != STATIC_SCENE_ADMISSION_SCHEMA_VERSION:
        raise ValueError('static scene admission schema version is invalid')
    for field in ('backend', 'task_id', 'scene_id', 'implementation_id', 'geometry_source'):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f'static scene admission {field} must be a non-empty string')
    task_sha256 = value.get('task_sha256')
    if not isinstance(task_sha256, str) or len(task_sha256) != 64:
        raise ValueError('static scene admission task_sha256 is invalid')
    layout_id = value.get('layout_id_or_null')
    if layout_id is not None and (not isinstance(layout_id, str) or not layout_id):
        raise ValueError('static scene admission layout_id_or_null is invalid')
    status = value.get('status')
    if status not in STATIC_SCENE_ADMISSION_STATUSES:
        raise ValueError('static scene admission status is invalid')
    failure_code = value.get('failure_code_or_null')
    if status == 'admitted':
        if failure_code is not None:
            raise ValueError('admitted static scene result cannot have a failure code')
    elif not isinstance(failure_code, str) or not failure_code:
        raise ValueError('non-admitted static scene result requires a failure code')
    checks = value.get('checks')
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError('static scene admission checks must be a non-empty mapping')
    for name, state in checks.items():
        if not isinstance(name, str) or not name or state not in STATIC_SCENE_ADMISSION_CHECK_STATES:
            raise ValueError('static scene admission check is invalid')
    if status == 'admitted' and any((state != 'pass' for state in checks.values())):
        raise ValueError('admitted static scene result requires every check to pass')
    if status == 'rejected' and 'fail' not in checks.values():
        raise ValueError('rejected static scene result requires a failed check')
    diagnostics = value.get('diagnostics')
    if not isinstance(diagnostics, Mapping):
        raise ValueError('static scene admission diagnostics must be a mapping')
    validate_json_value(dict(value))
__all__ = ['STATIC_SCENE_ADMISSION_CHECK_STATES', 'STATIC_SCENE_ADMISSION_SCHEMA_VERSION', 'STATIC_SCENE_ADMISSION_STATUSES', 'build_static_scene_admission_result', 'validate_static_scene_admission_result']
