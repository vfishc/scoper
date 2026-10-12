from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from scoper.benchmark.behavior.scheduler import normalize_scheduler_specs
from scoper.benchmark.behavior.scheduler.disturbance import DisturbanceConfig
from scoper.benchmark.behavior.scheduler.failure_injection import FailureConfig
from scoper.benchmark.behavior.task_normalization import materialize_tidy_policy_initial_violations
from scoper.benchmark.behavior.export.ids import canonical_json_bytes, json_deep_copy, sha256_bytes, sha256_file, strict_json_load
from scoper.benchmark.behavior.export.types import MaterializedTask, TaskIdentityResult, TaskPathResolution, TrajectoryConfigError
_SOURCE_JSON_POINTER = '/source/source_json'
_PATH_KEYS = {'path', 'file', 'filename', 'directory', 'dir', 'root', 'source_json', 'template_file', 'metadata_path', 'config_path'}
_WINDOWS_DRIVE_RE = re.compile('^[A-Za-z]:[\\\\/]')
_WINDOWS_UNC_RE = re.compile('^(?:\\\\\\\\|//)[^\\\\/]+[\\\\/][^\\\\/]+')
_NORMALIZED_URI_RE = re.compile('^(?:repo-json|external-json)://sha256/[0-9a-f]{64}(?:/.*)?$')

def normalize_behavior_task_for_identity(runtime_task: Mapping[str, Any], *, repo_root: Path) -> TaskIdentityResult:
    if not isinstance(runtime_task, Mapping):
        raise TrajectoryConfigError('task_identity_invalid', 'task must be a mapping')
    root = _resolve_repo_root(repo_root)
    try:
        runtime_copy = materialize_tidy_policy_initial_violations(runtime_task)
    except ValueError as exc:
        raise TrajectoryConfigError('task_identity_invalid', str(exc), pointer='/family_spec/tidy/violations') from exc
    identity = json_deep_copy(runtime_copy)
    assets_root = root / 'datasets' / 'behavior-1k-assets'
    if (assets_root / 'objects').is_dir():
        from scoper.benchmark.behavior.render.asset_resolver import resolve_task_render_assets
        resolved = resolve_task_render_assets(identity, assets_root=assets_root)
        runtime_copy = json_deep_copy(resolved.task)
        identity = json_deep_copy(resolved.task)
    resolutions: list[TaskPathResolution] = []
    frozen_assets: dict[str, Any] = {}
    source = identity.get('source')
    if not isinstance(source, MutableMapping):
        raise TrajectoryConfigError('task_identity_invalid', 'task.source must be a mapping', pointer='/source')
    if 'source_json' in source and source['source_json'] is not None:
        locator = source['source_json']
        if not isinstance(locator, str) or not locator:
            raise TrajectoryConfigError('task_path_invalid', 'source.source_json must be null or a non-empty string', pointer=_SOURCE_JSON_POINTER)
        resolution, parsed = _normalize_json_locator(locator, repo_root=root)
        source['source_json'] = resolution.normalized_locator
        resolutions.append(resolution)
        frozen_assets[_SOURCE_JSON_POINTER] = parsed
    template = identity.get('template')
    if template is not None:
        if not isinstance(template, Mapping):
            raise TrajectoryConfigError('task_identity_invalid', 'task.template must be a mapping', pointer='/template')
        if 'template_file' in template:
            template_file = template['template_file']
            if template_file is not None and (not isinstance(template_file, str) or not template_file or '/' in template_file or ('\\' in template_file)):
                raise TrajectoryConfigError('task_path_invalid', 'template.template_file must be null or a logical basename', pointer='/template/template_file')
    _audit_machine_locators(identity, repo_root=root, home_root=Path.home().resolve())
    resolutions.sort(key=lambda item: item.json_pointer)
    task_hash = sha256_bytes(canonical_json_bytes(identity))
    return TaskIdentityResult(runtime_task_template=runtime_copy, identity_task=identity, task_hash=task_hash, path_resolutions=tuple(resolutions), frozen_json_assets=frozen_assets)

def portable_task_for_freeze(identity: TaskIdentityResult, *, repo_root: Path) -> Mapping[str, Any]:
    task = json_deep_copy(identity.runtime_task_template)
    root = _resolve_repo_root(repo_root)
    for resolution in identity.path_resolutions:
        if resolution.locator_kind != 'repo_json':
            raise TrajectoryConfigError('external_task_asset_not_publishable', 'freezer input references an external JSON asset', pointer=resolution.json_pointer)
        prefix = f'repo-json://sha256/{resolution.content_sha256}/'
        relative = resolution.normalized_locator.removeprefix(prefix)
        if relative == resolution.normalized_locator or not relative:
            raise TrajectoryConfigError('task_path_invalid', 'malformed normalized repo JSON locator')
        resolved = (root / relative).resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise TrajectoryConfigError('task_path_invalid', 'repo JSON locator escaped repo root')
        _set_json_pointer(task, resolution.json_pointer, resolved.relative_to(root).as_posix())
    return task

def materialize_behavior_task_assets(identity: TaskIdentityResult, *, initializing_run_root: Path, published_run_root: Path) -> MaterializedTask:
    initializing = Path(initializing_run_root)
    published = Path(published_run_root)
    if initializing.parent != published.parent or initializing.name != published.name + '.initializing':
        raise TrajectoryConfigError('asset_cache_root_invalid', 'initializing root must be the published run sibling with .initializing suffix')
    cache_dir = initializing / '.asset_cache' / 'json'
    cache_dir.mkdir(parents=True, exist_ok=True)
    runtime_task = json_deep_copy(identity.runtime_task_template)
    entries: list[Mapping[str, Any]] = []
    resolution_by_pointer = {item.json_pointer: item for item in identity.path_resolutions}
    if set(resolution_by_pointer) != set(identity.frozen_json_assets):
        raise TrajectoryConfigError('asset_cache_identity_mismatch', 'path resolutions and frozen JSON assets do not match')
    for pointer in sorted(identity.frozen_json_assets):
        resolution = resolution_by_pointer[pointer]
        content = canonical_json_bytes(identity.frozen_json_assets[pointer])
        if sha256_bytes(content) != resolution.content_sha256:
            raise TrajectoryConfigError('asset_cache_hash_mismatch', 'frozen JSON asset hash changed', pointer=pointer)
        relative_path = Path('.asset_cache') / 'json' / f'{resolution.content_sha256}.json'
        destination = initializing / relative_path
        _write_content_addressed_file(destination, content)
        _set_json_pointer(runtime_task, pointer, str(published / relative_path))
        entries.append({'json_pointer': pointer, 'asset_kind': 'json', 'content_sha256': resolution.content_sha256, 'relative_path': relative_path.as_posix(), 'bytes': len(content)})
    entries.sort(key=lambda item: (item['json_pointer'], item['content_sha256']))
    return MaterializedTask(identity=identity, runtime_task=runtime_task, cache_entries=tuple(entries))

def validate_resolved_protocol(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    value = json_deep_copy(protocol)
    if not isinstance(value, Mapping) or value.get('schema_version') != 'benchmark-protocol.v1':
        raise TrajectoryConfigError('protocol_schema_invalid', 'schema_version must be benchmark-protocol.v1')
    for pointer, field_value in (('/benchmark/id', _nested(value, 'benchmark', 'id')), ('/benchmark/version', _nested(value, 'benchmark', 'version')), ('/assets/behavior_assets_version', _nested(value, 'assets', 'behavior_assets_version')), ('/assets/scene_profile_version', _nested(value, 'assets', 'scene_profile_version')), ('/tool_schema/version', _nested(value, 'tool_schema', 'version'))):
        if not isinstance(field_value, str) or not field_value:
            raise TrajectoryConfigError('protocol_schema_invalid', 'required protocol string is missing', pointer=pointer)
    for pointer in ('/assets/behavior_assets_version', '/assets/scene_profile_version'):
        field_value = _pointer_get(value, pointer)
        if field_value.casefold() in {'latest', 'current', 'dev'}:
            raise TrajectoryConfigError('protocol_schema_invalid', 'asset version is mutable', pointer=pointer)
    data_policy = value.get('data_policy')
    if not isinstance(data_policy, MutableMapping):
        raise TrajectoryConfigError('protocol_schema_invalid', 'data_policy must be a mapping')
    for field, sha_values in (('nontrain_lineage_ids', False), ('nontrain_task_hashes', True)):
        entries = data_policy.get(field)
        if not isinstance(entries, list) or any((not isinstance(item, str) or not item for item in entries)):
            raise TrajectoryConfigError('protocol_schema_invalid', f'data_policy.{field} must be a string list')
        if len(entries) != len(set(entries)):
            raise TrajectoryConfigError('protocol_schema_invalid', f'data_policy.{field} contains duplicates')
        if sha_values and any((re.fullmatch('[0-9a-f]{64}', item) is None for item in entries)):
            raise TrajectoryConfigError('protocol_schema_invalid', f'data_policy.{field} contains non-sha256 values')
        data_policy[field] = sorted(entries)
    arms = value.get('rollout_arms')
    if not isinstance(arms, Mapping) or set(arms) != {'clean', 'disturbance', 'failure', 'mixed', 'learner'}:
        raise TrajectoryConfigError('rollout_arm_schema_invalid', 'rollout_arms must contain exactly five v1 arms')
    expected_booleans = {'clean': (False, False), 'disturbance': (True, False), 'failure': (False, True), 'mixed': (True, True)}
    arm_fields = {'include_disturbances', 'include_failures', 'additional_disturbances', 'additional_failures', 'additional_human_events'}
    for arm_name, arm in arms.items():
        if not isinstance(arm, Mapping) or set(arm) != arm_fields:
            raise TrajectoryConfigError('rollout_arm_schema_invalid', f'arm {arm_name!r} fields are not exact')
        if not isinstance(arm['include_disturbances'], bool) or not isinstance(arm['include_failures'], bool):
            raise TrajectoryConfigError('rollout_arm_schema_invalid', f'arm {arm_name!r} include flags must be bool')
        if arm_name in expected_booleans and (arm['include_disturbances'], arm['include_failures']) != expected_booleans[arm_name]:
            raise TrajectoryConfigError('rollout_arm_schema_invalid', f'arm {arm_name!r} include flags are fixed')
        for field in ('additional_disturbances', 'additional_failures', 'additional_human_events'):
            if not isinstance(arm[field], list) or any((not isinstance(item, Mapping) for item in arm[field])):
                raise TrajectoryConfigError('rollout_arm_schema_invalid', f'arm {arm_name!r}.{field} must be object list')
        _validate_scheduler_entries(arm['additional_disturbances'], kind='disturbance')
        _validate_scheduler_entries(arm['additional_failures'], kind='failure')
        _validate_scheduler_entries(arm['additional_human_events'], kind='human_event')
    if arms['clean']['additional_disturbances'] or arms['clean']['additional_failures']:
        raise TrajectoryConfigError('rollout_arm_schema_invalid', 'clean arm cannot add disturbances or failures')
    if arms['disturbance']['additional_failures']:
        raise TrajectoryConfigError('rollout_arm_schema_invalid', 'disturbance arm cannot add failures')
    if arms['failure']['additional_disturbances']:
        raise TrajectoryConfigError('rollout_arm_schema_invalid', 'failure arm cannot add disturbances')
    return value

def build_effective_interventions(identity_task: Mapping[str, Any], resolved_protocol: Mapping[str, Any], rollout_arm: str) -> list[Mapping[str, Any]]:
    protocol = validate_resolved_protocol(resolved_protocol)
    if rollout_arm not in protocol['rollout_arms']:
        raise TrajectoryConfigError('rollout_arm_schema_invalid', f'unknown rollout arm {rollout_arm!r}')
    disturbances, failures, human_events, _ignored_seed = normalize_scheduler_specs(identity_task)
    _validate_scheduler_entries(disturbances, kind='disturbance')
    _validate_scheduler_entries(failures, kind='failure')
    _validate_scheduler_entries(human_events, kind='human_event')
    arm = protocol['rollout_arms'][rollout_arm]
    groups = (('disturbance', disturbances if arm['include_disturbances'] else [], arm['additional_disturbances']), ('failure', failures if arm['include_failures'] else [], arm['additional_failures']), ('human_event', human_events, arm['additional_human_events']))
    effective: list[Mapping[str, Any]] = []
    counts: dict[str, int] = {}
    for kind, task_entries, protocol_entries in groups:
        counts[kind] = len(task_entries) + len(protocol_entries)
        for source, entries in (('task', task_entries), ('protocol', protocol_entries)):
            for ordinal, entry in enumerate(entries):
                effective.append({'kind': kind, 'source': source, 'source_ordinal': ordinal, 'spec': _strip_scheduler_seeds(entry)})
    if rollout_arm == 'disturbance' and (not counts['disturbance']):
        raise TrajectoryConfigError('rollout_arm_empty', 'disturbance arm has no effective disturbance')
    if rollout_arm == 'failure' and (not counts['failure']):
        raise TrajectoryConfigError('rollout_arm_empty', 'failure arm has no effective failure')
    if rollout_arm == 'mixed' and (not counts['disturbance'] or not counts['failure']):
        raise TrajectoryConfigError('rollout_arm_empty', 'mixed arm requires disturbance and failure entries')
    return effective

def _resolve_repo_root(repo_root: Path) -> Path:
    try:
        root = Path(repo_root).resolve(strict=True)
    except OSError as exc:
        raise TrajectoryConfigError('task_path_invalid', f'repo root is invalid: {exc}') from exc
    if not root.is_dir():
        raise TrajectoryConfigError('task_path_invalid', 'repo root must be a directory')
    return root

def _normalize_json_locator(locator: str, *, repo_root: Path) -> tuple[TaskPathResolution, Any]:
    raw_path = Path(locator).expanduser()
    is_absolute_input = raw_path.is_absolute()
    candidate = raw_path if is_absolute_input else repo_root / raw_path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise TrajectoryConfigError('task_path_invalid', f'cannot resolve source JSON: {exc}', pointer=_SOURCE_JSON_POINTER) from exc
    if not is_absolute_input and (not resolved.is_relative_to(repo_root)):
        raise TrajectoryConfigError('task_path_invalid', 'relative source JSON escaped repo root', pointer=_SOURCE_JSON_POINTER)
    if not resolved.is_file():
        raise TrajectoryConfigError('task_path_invalid', 'source JSON must resolve to a regular file', pointer=_SOURCE_JSON_POINTER)
    if not os.access(resolved, os.R_OK):
        raise TrajectoryConfigError('task_path_invalid', 'source JSON is not readable', pointer=_SOURCE_JSON_POINTER)
    try:
        parsed = strict_json_load(resolved)
    except TrajectoryConfigError as exc:
        raise TrajectoryConfigError('task_path_invalid', str(exc), pointer=_SOURCE_JSON_POINTER) from exc
    canonical = canonical_json_bytes(parsed)
    content_sha256 = sha256_bytes(canonical)
    if resolved.is_relative_to(repo_root):
        relative = resolved.relative_to(repo_root).as_posix()
        locator_kind = 'repo_json'
        normalized = f'repo-json://sha256/{content_sha256}/{relative}'
    else:
        if not is_absolute_input:
            raise TrajectoryConfigError('task_path_invalid', 'external source JSON must use an absolute input locator')
        locator_kind = 'external_json'
        normalized = f'external-json://sha256/{content_sha256}'
    return (TaskPathResolution(json_pointer=_SOURCE_JSON_POINTER, locator_kind=locator_kind, normalized_locator=normalized, content_sha256=content_sha256), parsed)

def _audit_machine_locators(value: Any, *, repo_root: Path, home_root: Path, pointer: str='') -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _audit_machine_locators(item, repo_root=repo_root, home_root=home_root, pointer=f'{pointer}/{index}')
        return
    if not isinstance(value, Mapping):
        return
    prefixes = tuple((prefix for prefix in (str(repo_root), str(home_root)) if prefix and prefix not in {'/', '.'}))
    for key, item in value.items():
        child_pointer = f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}"
        if isinstance(item, str):
            if any((prefix in item for prefix in prefixes)):
                raise TrajectoryConfigError('unregistered_task_path', 'identity contains a machine-local absolute prefix', pointer=child_pointer)
            normalized_key = key.casefold()
            path_like_key = normalized_key in _PATH_KEYS or normalized_key.endswith(('_path', '_file', '_dir', '_root'))
            if path_like_key and (not _NORMALIZED_URI_RE.fullmatch(item)) and _looks_machine_local(item):
                raise TrajectoryConfigError('unregistered_task_path', 'unregistered filesystem-like task value', pointer=child_pointer)
        else:
            _audit_machine_locators(item, repo_root=repo_root, home_root=home_root, pointer=child_pointer)

def _looks_machine_local(value: str) -> bool:
    lowered = value.casefold()
    return bool(value.startswith(('/', '~/')) or lowered.startswith('file:') or _WINDOWS_DRIVE_RE.match(value) or _WINDOWS_UNC_RE.match(value))

def _set_json_pointer(document: Any, pointer: str, value: Any) -> None:
    if not pointer.startswith('/'):
        raise TrajectoryConfigError('json_pointer_invalid', f'invalid JSON pointer {pointer!r}')
    parts = [part.replace('~1', '/').replace('~0', '~') for part in pointer[1:].split('/')]
    current = document
    for part in parts[:-1]:
        if not isinstance(current, MutableMapping) or part not in current:
            raise TrajectoryConfigError('json_pointer_invalid', f'missing JSON pointer parent {pointer!r}')
        current = current[part]
    if not isinstance(current, MutableMapping) or parts[-1] not in current:
        raise TrajectoryConfigError('json_pointer_invalid', f'missing JSON pointer {pointer!r}')
    current[parts[-1]] = value

def _write_content_addressed_file(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or sha256_file(path) != sha256_bytes(content) or path.stat().st_size != len(content):
            raise TrajectoryConfigError('asset_cache_conflict', f'asset cache conflict at {path}')
        return
    temp = path.with_name(path.name + '.tmp')
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 420)
    except FileExistsError as exc:
        raise TrajectoryConfigError('asset_cache_conflict', f'stale asset cache temp file {temp}') from exc
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temp.unlink()
        except OSError:
            pass
        raise

def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current

def _pointer_get(value: Mapping[str, Any], pointer: str) -> Any:
    current: Any = value
    for part in pointer.strip('/').split('/'):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current

def _validate_scheduler_entries(entries: list[Mapping[str, Any]], *, kind: str) -> None:
    for index, entry in enumerate(entries):
        try:
            if kind == 'disturbance':
                DisturbanceConfig(**_config_kwargs(entry, DisturbanceConfig, extra_field='extra'))
            elif kind == 'failure':
                FailureConfig(**_config_kwargs(entry, FailureConfig, extra_field='extra'))
            else:
                if not isinstance(entry.get('id'), str) or not entry['id']:
                    raise ValueError('human event requires non-empty id')
                if entry.get('type') not in {'qa', 'insert'}:
                    raise ValueError('human event type must be qa or insert')
                activation = entry.get('activation')
                if activation is not None and (not isinstance(activation, Mapping)):
                    raise ValueError('human event activation must be a mapping')
        except (TypeError, ValueError) as exc:
            raise TrajectoryConfigError('rollout_arm_schema_invalid', f'invalid {kind} entry {index}: {exc}') from exc

def _config_kwargs(value: Mapping[str, Any], dataclass_type: type, *, extra_field: str) -> dict[str, Any]:
    allowed = dataclass_type.__dataclass_fields__
    explicit_extra = value.get(extra_field)
    if explicit_extra is not None and (not isinstance(explicit_extra, Mapping)):
        raise ValueError(f'{extra_field} must be a mapping')
    result = {key: json_deep_copy(item) for key, item in value.items() if key in allowed}
    extras = {key: json_deep_copy(item) for key, item in value.items() if key not in allowed and key not in {'seed', 'rng_seed', 'random_seed'}}
    if extras:
        current = result.get(extra_field)
        merged = dict(current) if isinstance(current, Mapping) else {}
        merged.update(extras)
        result[extra_field] = merged
    return result

def _strip_scheduler_seeds(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_scheduler_seeds(item) for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = key.casefold()
            if normalized in {'seed', 'rng_seed', 'random_seed'}:
                continue
            if normalized.endswith('_seed'):
                raise TrajectoryConfigError('unregistered_scheduler_seed', f'unregistered scheduler seed field {key!r}')
            result[key] = _strip_scheduler_seeds(item)
        return result
    return json_deep_copy(value)
__all__ = ['materialize_behavior_task_assets', 'normalize_behavior_task_for_identity', 'portable_task_for_freeze', 'build_effective_interventions', 'validate_resolved_protocol']
