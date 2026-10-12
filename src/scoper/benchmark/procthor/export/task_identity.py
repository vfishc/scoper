from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from scoper.benchmark.procthor.online_task_spec import TaskSpecValidationError, validate_online_task_spec
from scoper.benchmark.procthor.procthor_dataset_assets import ProcTHORDatasetAssetError, is_prior_house_uri, load_prior_house, make_prior_house_uri, parse_prior_house_uri
from scoper.benchmark.procthor.export.ids import canonical_json_bytes, sha256_bytes, sha256_json
from scoper.benchmark.procthor.export.schema import BACKEND, TASK_BANK_HASH_SCHEMA_VERSION, TASK_SET_HASH_SCHEMA_VERSION, SchemaValidationError, strict_json_loads, validate_json_value, validate_task_set_envelope
from scoper.benchmark.procthor.export.types import MaterializedTask, TaskIdentityResult, TaskPathResolution
_PATH_KEYS = {'path', 'file', 'filename', 'directory', 'dir', 'root', 'house_path', 'metadata_path', 'config_path'}
_WINDOWS_PATH_RE = re.compile('^(?:[A-Za-z]:[\\\\/]|\\\\\\\\|//)')
_NORMALIZED_LOCATOR_PREFIXES = ('prior-json://sha256/', 'repo-json://sha256/', 'external-json://sha256/')

def normalize_procthor_task_for_identity(runtime_task: Mapping[str, Any], *, repo_root: Path, dataset_name: str, dataset_revision: str) -> TaskIdentityResult:
    return _normalize_procthor_task_for_identity(runtime_task, repo_root=repo_root, dataset_name=dataset_name, dataset_revision=dataset_revision, validate_schema=True)

def _normalize_procthor_task_for_identity_unchecked(runtime_task: Mapping[str, Any], *, repo_root: Path, dataset_name: str, dataset_revision: str) -> TaskIdentityResult:
    return _normalize_procthor_task_for_identity(runtime_task, repo_root=repo_root, dataset_name=dataset_name, dataset_revision=dataset_revision, validate_schema=False)

def _normalize_procthor_task_for_identity(runtime_task: Mapping[str, Any], *, repo_root: Path, dataset_name: str, dataset_revision: str, validate_schema: bool) -> TaskIdentityResult:
    if validate_schema:
        try:
            validate_online_task_spec(runtime_task)
        except TaskSpecValidationError as exc:
            raise SchemaValidationError('task_schema_invalid', 'online task validation failed: {}'.format(exc)) from exc
    validate_json_value(dict(runtime_task))
    runtime_template = _json_deep_copy(runtime_task)
    identity_task = _json_deep_copy(runtime_task)
    root = Path(repo_root).expanduser()
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise SchemaValidationError('task_path_invalid', 'repo_root cannot be resolved: {}'.format(exc)) from exc
    if not root.is_dir():
        raise SchemaValidationError('task_path_invalid', 'repo_root must be a directory')
    source = identity_task.get('source')
    if not isinstance(source, dict):
        raise SchemaValidationError('task_identity_invalid', 'task source must be an object')
    locator = source.get('house_path')
    if not isinstance(locator, str) or not locator:
        raise SchemaValidationError('task_path_invalid', '/source/house_path must be a non-empty string')
    if is_prior_house_uri(locator):
        resolution, house = _normalize_prior_locator(locator, source_split=source.get('split'), dataset_name=dataset_name, dataset_revision=dataset_revision)
    else:
        resolution, house = _normalize_local_locator(locator, root=root)
    source['house_path'] = resolution.normalized_locator
    _audit_machine_locators(identity_task, repo_root=root)
    task_hash = sha256_json(identity_task)
    return TaskIdentityResult(runtime_task_template=runtime_template, identity_task=identity_task, task_hash=task_hash, path_resolutions=(resolution,), frozen_house=_json_deep_copy(house))

def materialize_procthor_task_assets(identity: TaskIdentityResult, *, initializing_run_root: Path, published_run_root: Path) -> MaterializedTask:
    if not isinstance(identity, TaskIdentityResult):
        raise SchemaValidationError('task_materialization_invalid', 'identity must be TaskIdentityResult')
    initializing = Path(initializing_run_root).expanduser().absolute()
    published = Path(published_run_root).expanduser().absolute()
    if initializing.name != published.name + '.initializing' or initializing.parent != published.parent:
        raise SchemaValidationError('task_materialization_invalid', 'initializing and published run roots must be sibling NAME.initializing/NAME paths')
    if len(identity.path_resolutions) != 1 or identity.path_resolutions[0].json_pointer != '/source/house_path':
        raise SchemaValidationError('task_materialization_invalid', 'ProcTHOR v1 requires exactly one house resolution')
    house_bytes = canonical_json_bytes(dict(identity.frozen_house))
    resolution = identity.path_resolutions[0]
    actual_hash = sha256_bytes(house_bytes)
    if actual_hash != resolution.content_sha256:
        raise SchemaValidationError('asset_content_hash_mismatch', 'frozen house differs from task identity')
    relative_path = Path('.asset_cache') / 'houses' / (actual_hash + '.json')
    initializing_path = initializing / relative_path
    initializing_path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_atomic_bytes(initializing_path, house_bytes)
    runtime_task = _json_deep_copy(identity.runtime_task_template)
    runtime_task['source']['house_path'] = str(published / relative_path)
    entry = {'json_pointer': resolution.json_pointer, 'asset_kind': 'house', 'content_sha256': actual_hash, 'relative_path': relative_path.as_posix(), 'bytes': len(house_bytes), 'dataset_name': resolution.dataset_name, 'dataset_revision': resolution.dataset_revision}
    return MaterializedTask(identity=identity, runtime_task=runtime_task, cache_entries=(entry,))

def build_task_bank_hash(task_identities: Mapping[str, TaskIdentityResult]) -> str:
    tasks = _sorted_task_hashes(task_identities)
    return sha256_json({'schema_version': TASK_BANK_HASH_SCHEMA_VERSION, 'backend': BACKEND, 'tasks': tasks})

def build_task_set_hash(task_set: Mapping[str, Any], task_identities: Mapping[str, TaskIdentityResult]) -> str:
    validate_task_set_envelope(task_set)
    missing = [task_id for task_id in task_set['task_ids'] if task_id not in task_identities]
    if missing:
        raise SchemaValidationError('task_set_missing_task', 'task set references missing task ids: {}'.format(', '.join(missing)))
    tasks = sorted(({'task_id': task_id, 'task_hash': task_identities[task_id].task_hash} for task_id in task_set['task_ids']), key=lambda item: item['task_id'])
    envelope_identity = {'schema_version': task_set['schema_version'], 'name': task_set['name'], 'version': task_set['version'], 'lineage_id': task_set['lineage_id'], 'benchmark_split': task_set['benchmark_split']}
    return sha256_json({'schema_version': TASK_SET_HASH_SCHEMA_VERSION, 'backend': BACKEND, 'task_set': envelope_identity, 'tasks': tasks})

def build_task_selection_hash(task_ids: Sequence[str], task_identities: Mapping[str, TaskIdentityResult]) -> str:
    missing = [task_id for task_id in task_ids if task_id not in task_identities]
    if missing:
        raise SchemaValidationError('task_selection_missing_task', 'missing task identities: {}'.format(', '.join(missing)))
    return sha256_json(sorted(({'task_id': task_id, 'task_hash': task_identities[task_id].task_hash} for task_id in task_ids), key=lambda item: item['task_id']))

def build_task_order_hash(task_ids: Sequence[str]) -> str:
    if not all((isinstance(task_id, str) and task_id for task_id in task_ids)):
        raise SchemaValidationError('task_order_invalid', 'task order must contain non-empty string ids')
    return sha256_json(list(task_ids))

def _normalize_prior_locator(locator: str, *, source_split: Any, dataset_name: str, dataset_revision: str) -> tuple[TaskPathResolution, Mapping[str, Any]]:
    if dataset_name != 'procthor-10k':
        raise SchemaValidationError('task_path_invalid', 'prior ProcTHOR house requires dataset_name=procthor-10k')
    if not isinstance(dataset_revision, str) or not dataset_revision:
        raise SchemaValidationError('dataset_revision_required', 'prior house identity requires a frozen dataset revision')
    try:
        split, index = parse_prior_house_uri(locator)
        canonical_uri = make_prior_house_uri(split, index)
    except ProcTHORDatasetAssetError as exc:
        raise SchemaValidationError('task_path_invalid', str(exc)) from exc
    if source_split != split:
        raise SchemaValidationError('split_metadata_conflict', "prior URI split '{}' differs from task source split '{}'".format(split, source_split))
    try:
        loaded = load_prior_house(canonical_uri, dataset_name=dataset_name, revision=dataset_revision)
    except (ProcTHORDatasetAssetError, OSError, ValueError) as exc:
        raise SchemaValidationError('task_path_invalid', 'cannot load prior house: {}'.format(exc)) from exc
    house = _strict_house_copy(loaded, source=canonical_uri)
    content_hash = sha256_json(house)
    normalized = 'prior-json://sha256/{}/procthor-10k/{}/{}'.format(content_hash, split, index)
    return (TaskPathResolution(json_pointer='/source/house_path', locator_kind='prior_json', normalized_locator=normalized, content_sha256=content_hash, dataset_name=dataset_name, dataset_revision=dataset_revision), house)

def _normalize_local_locator(locator: str, *, root: Path) -> tuple[TaskPathResolution, Mapping[str, Any]]:
    input_path = Path(locator).expanduser()
    input_absolute = input_path.is_absolute()
    candidate = input_path if input_absolute else root / input_path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SchemaValidationError('task_path_invalid', 'house path cannot be resolved: {}'.format(exc)) from exc
    if not resolved.is_file():
        raise SchemaValidationError('task_path_invalid', 'house path must name a regular file')
    inside_repo = _is_relative_to(resolved, root)
    if not input_absolute and (not inside_repo):
        raise SchemaValidationError('task_path_invalid', 'relative house path resolves outside repo_root')
    try:
        text = resolved.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise SchemaValidationError('task_path_invalid', 'house JSON is not readable UTF-8: {}'.format(exc)) from exc
    parsed = strict_json_loads(text, source=str(resolved))
    house = _strict_house_copy(parsed, source=str(resolved))
    content_hash = sha256_json(house)
    if inside_repo:
        relative = resolved.relative_to(root).as_posix()
        normalized = 'repo-json://sha256/{}/{}'.format(content_hash, relative)
        kind = 'repo_json'
    else:
        if not input_absolute:
            raise SchemaValidationError('task_path_invalid', 'external house input must be absolute')
        normalized = 'external-json://sha256/{}'.format(content_hash)
        kind = 'external_json'
    return (TaskPathResolution(json_pointer='/source/house_path', locator_kind=kind, normalized_locator=normalized, content_sha256=content_hash, dataset_name=None, dataset_revision=None), house)

def _strict_house_copy(value: Any, *, source: str) -> Mapping[str, Any]:
    try:
        copied = strict_json_loads(canonical_json_bytes(value).decode('utf-8'), source=source)
    except (SchemaValidationError, TypeError, ValueError) as exc:
        raise SchemaValidationError('task_path_invalid', 'house is not strict JSON: {}'.format(exc)) from exc
    if not isinstance(copied, dict):
        raise SchemaValidationError('task_path_invalid', 'house JSON must be an object')
    return copied

def _audit_machine_locators(value: Any, *, repo_root: Path, pointer: str='') -> None:
    home = Path.home().expanduser().resolve()
    absolute_prefixes = tuple((prefix for prefix in (str(repo_root), str(home)) if prefix and prefix != os.path.sep))
    if isinstance(value, dict):
        for key, item in value.items():
            child_pointer = '{}/{}'.format(pointer, key.replace('~', '~0').replace('/', '~1'))
            if isinstance(item, str):
                if any((prefix in item for prefix in absolute_prefixes)):
                    raise SchemaValidationError('unregistered_task_path', 'identity contains a machine-local absolute prefix', child_pointer)
                lowered = key.lower()
                path_key = lowered in _PATH_KEYS or lowered.endswith(('_path', '_file', '_dir', '_root'))
                if path_key and (not item.startswith(_NORMALIZED_LOCATOR_PREFIXES)):
                    if item.startswith(('/', '~/', 'file:')) or _WINDOWS_PATH_RE.match(item) is not None:
                        raise SchemaValidationError('unregistered_task_path', 'unregistered path-like task field', child_pointer)
            _audit_machine_locators(item, repo_root=repo_root, pointer=child_pointer)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _audit_machine_locators(item, repo_root=repo_root, pointer='{}/{}'.format(pointer, index))

def _exclusive_atomic_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise SchemaValidationError('asset_cache_conflict', 'cannot verify existing cache file: {}'.format(exc)) from exc
        if existing != content or sha256_bytes(existing) != path.stem:
            raise SchemaValidationError('asset_cache_conflict', 'existing content-addressed asset is corrupt')
        return
    temporary = path.with_name(path.name + '.tmp')
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 420)
    except FileExistsError as exc:
        raise SchemaValidationError('asset_cache_conflict', 'stale exclusive cache temporary exists') from exc
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise

def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _json_deep_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return strict_json_loads(canonical_json_bytes(dict(value)).decode('utf-8'))

def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def _sorted_task_hashes(task_identities: Mapping[str, TaskIdentityResult]) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for task_id, identity in task_identities.items():
        if not isinstance(task_id, str) or not task_id:
            raise SchemaValidationError('task_bank_hash_invalid', 'task id must be non-empty')
        if not isinstance(identity, TaskIdentityResult):
            raise SchemaValidationError('task_bank_hash_invalid', 'task identity has wrong type')
        if identity.identity_task.get('id') != task_id:
            raise SchemaValidationError('task_bank_hash_invalid', 'task identity id differs from mapping key')
        tasks.append({'task_id': task_id, 'task_hash': identity.task_hash})
    return sorted(tasks, key=lambda item: item['task_id'])
__all__ = ['build_task_bank_hash', 'build_task_order_hash', 'build_task_selection_hash', 'build_task_set_hash', 'materialize_procthor_task_assets', 'normalize_procthor_task_for_identity']
