from __future__ import annotations
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from scoper.benchmark.behavior.export.ids import canonical_json_bytes, json_deep_copy, sha256_bytes, sha256_json
from scoper.benchmark.behavior.paths import repo_root
from scoper.benchmark.behavior.scene.doors import is_door_like
ASSET_PREFLIGHT_SCHEMA = 'behavior-render-asset-preflight.v1'
ASSET_FAILURE_CODE = 'ADMISSION_MISSING_REQUIRED_ASSET'
_SCENE_PROVIDED_CATEGORIES = frozenset({'floors', 'floor', 'walls', 'wall', 'ceilings', 'ceiling'})
_ABILITY_TAGS = {'openable': frozenset({'openable'}), 'toggleable': frozenset({'toggleable', 'togglebutton'})}

@dataclass(frozen=True)
class AssetCatalog:
    root: Path
    revision: str
    catalog_hash: str
    models_by_category: Mapping[str, tuple[str, ...]]
    scope: str = 'full'

@dataclass(frozen=True)
class AssetPreflightResult:
    task: Mapping[str, Any]
    provenance: Mapping[str, Any]
    valid: bool

class AssetPreflightError(ValueError):

    def __init__(self, result: AssetPreflightResult) -> None:
        self.result = result
        failures = result.provenance.get('failures') or []
        reason = failures[0].get('reason') if failures else 'asset_preflight_failed'
        super().__init__(f'{ASSET_FAILURE_CODE}: {reason}')

def default_assets_root() -> Path:
    data_path = os.environ.get('OMNIGIBSON_DATA_PATH')
    if data_path is not None:
        root = Path(data_path).expanduser()
        if not root.is_absolute():
            root = repo_root() / 'OmniGibson' / 'omnigibson' / root
        return root / 'behavior-1k-assets'
    return repo_root() / 'datasets' / 'behavior-1k-assets'

def load_asset_catalog(assets_root: Path | None=None) -> AssetCatalog:
    return _load_asset_catalog_cached(str(Path(assets_root or default_assets_root()).resolve()))

def load_fixed_task_catalog(task, assets_root, scene_layout_mode, load_doors):
    root = Path(assets_root or default_assets_root()).resolve()
    layout = _effective_scene_layout_mode(scene_layout_mode, task)
    scene_ids = _full_scene_object_ids(task) if layout == 'source_json_full' else set()
    requested = set()
    for name, role in task.get('roles', {}).items():
        category, model = (role.get('category'), role.get('model'))
        if _role_exemption(name, role, category=category, scene_layout_mode=layout, full_scene_object_ids=scene_ids, load_doors=load_doors):
            continue
        if not isinstance(category, str) or not category or (not isinstance(model, str)) or (model in ('', 'unknown')):
            raise ValueError('fixed_asset_preflight_requires_explicit_category_and_model:' + name)
        if any((Path(value).name != value or value in ('.', '..') for value in (category, model))):
            raise ValueError('invalid_fixed_asset_path:' + name)
        requested.add((category, model))
    models = {}
    facts = []
    for category, model in sorted(requested):
        usd = root / 'objects' / category / model / 'usd' / (model + '.encrypted.usd')
        models.setdefault(category, [])
        if usd.is_file():
            models[category].append(model)
            facts.append({'category': category, 'model': model, 'bytes': usd.stat().st_size})
    version = root / 'VERSION'
    revision = version.read_text().strip() if version.is_file() else 'unknown'
    digest = sha256_json({'scope': 'fixed_task', 'revision': revision, 'assets': facts})
    return AssetCatalog(root, revision, digest, {key: tuple(value) for key, value in models.items()}, 'fixed_task')

@lru_cache(maxsize=8)
def _load_asset_catalog_cached(root_value: str) -> AssetCatalog:
    root = Path(root_value)
    objects_root = root / 'objects'
    if not objects_root.is_dir():
        raise FileNotFoundError(f'BEHAVIOR objects catalog is unavailable: {objects_root}')
    models: dict[str, tuple[str, ...]] = {}
    facts: list[dict[str, Any]] = []
    for category_path in sorted((path for path in objects_root.iterdir() if path.is_dir()), key=lambda path: path.name):
        category_models = []
        for model_path in sorted((path for path in category_path.iterdir() if path.is_dir()), key=lambda path: path.name):
            usd = model_path / 'usd' / f'{model_path.name}.encrypted.usd'
            if not usd.is_file():
                continue
            category_models.append(model_path.name)
            facts.append({'category': category_path.name, 'model': model_path.name, 'usd': usd.relative_to(root).as_posix(), 'bytes': usd.stat().st_size})
        models[category_path.name] = tuple(category_models)
    revision_path = root / 'VERSION'
    revision = revision_path.read_text(encoding='utf-8').strip() if revision_path.is_file() else 'unknown'
    catalog_hash = sha256_bytes(canonical_json_bytes({'schema_version': 'behavior-asset-catalog.v1', 'revision': revision, 'assets': facts}))
    return AssetCatalog(root=root, revision=revision, catalog_hash=catalog_hash, models_by_category=models)

def resolve_task_render_assets(task: Mapping[str, Any], *, catalog: AssetCatalog | None=None, assets_root: Path | None=None, scene_layout_mode: str='auto', load_doors: bool=False) -> AssetPreflightResult:
    if os.environ.get('EB_BEHAVIOR_SKIP_ASSET_AUDIT') == '1':
        from scoper.benchmark.behavior.render.asset_selection import pin_spawn_models
        resolved, provenance = pin_spawn_models(task, assets_root=assets_root or default_assets_root())
        return AssetPreflightResult(resolved, provenance, True)
    if catalog is None:
        catalog = load_fixed_task_catalog(task, assets_root, scene_layout_mode, load_doors) if os.environ.get('EB_BEHAVIOR_FIXED_ASSET_PREFLIGHT') == '1' else load_asset_catalog(assets_root)
    resolved = json_deep_copy(task)
    roles = resolved.get('roles') if isinstance(resolved.get('roles'), Mapping) else {}
    effective_layout = _effective_scene_layout_mode(scene_layout_mode, resolved)
    full_scene_ids = _full_scene_object_ids(resolved) if effective_layout == 'source_json_full' else set()
    selection_identity = sha256_json(_task_without_renderer_asset_metadata(resolved))
    rows = []
    failures = []
    for role_name in sorted(roles):
        role = roles[role_name]
        requested_category = role.get('category')
        requested_model = role.get('model')
        category = _resolve_category(role, catalog)
        exemption = _role_exemption(role_name, role, category=category, scene_layout_mode=effective_layout, full_scene_object_ids=full_scene_ids, load_doors=load_doors)
        row = {'role': role_name, 'requested_synset': role.get('synset'), 'resolved_synset': role.get('synset'), 'requested_category': requested_category, 'resolved_category': category, 'requested_model': requested_model, 'resolved_model': None, 'required_abilities': sorted((str(value) for value in role.get('abilities') or ())), 'requires_dataset_object': exemption is None, 'exemption': exemption, 'available_model_count': 0, 'available_model_sample': [], 'status': 'exempt' if exemption else 'pending', 'reason': None}
        if exemption is not None:
            rows.append(row)
            continue
        available = tuple(catalog.models_by_category.get(str(category), ())) if category else ()
        compatible = tuple((model for model in available if _model_supports_abilities(catalog, str(category), model, row['required_abilities'])))
        row['available_model_count'] = len(available)
        row['available_model_sample'] = list(available[:8])
        if not category:
            reason = 'resolved_category_missing'
        elif category not in catalog.models_by_category:
            reason = 'category_not_found'
        elif not available:
            reason = 'category_has_no_loadable_models'
        elif isinstance(requested_model, str) and requested_model and (requested_model != 'unknown'):
            if requested_model not in available:
                reason = 'requested_model_not_found'
            elif requested_model not in compatible:
                reason = 'requested_model_ability_incompatible'
            else:
                reason = None
                selected = requested_model
        elif not compatible:
            reason = 'no_ability_compatible_model'
        else:
            reason = None
            selected = compatible[int(sha256_json({'task': selection_identity, 'role': role_name, 'catalog': catalog.catalog_hash})[:16], 16) % len(compatible)]
        if reason is not None:
            row['status'] = 'fail'
            row['reason'] = reason
            failures.append(json_deep_copy(row))
        else:
            row['status'] = 'pass'
            row['resolved_model'] = selected
            role['category'] = category
            role['model'] = selected
        rows.append(row)
    provenance = {'schema_version': ASSET_PREFLIGHT_SCHEMA, 'task_id': resolved.get('id'), 'asset_catalog_revision': catalog.revision, 'asset_catalog_hash': catalog.catalog_hash, 'scene_layout_mode': effective_layout, 'load_doors': bool(load_doors), 'status': 'pass' if not failures else 'fail', 'failure_code': None if not failures else ASSET_FAILURE_CODE, 'retryable': False, 'roles': rows, 'failures': failures}
    if catalog.scope != 'full':
        provenance['asset_catalog_scope'] = catalog.scope
    resolved['renderer_asset_provenance'] = json_deep_copy(provenance)
    return AssetPreflightResult(task=resolved, provenance=provenance, valid=not failures)

def require_task_render_assets(task: Mapping[str, Any], **kwargs: Any) -> AssetPreflightResult:
    result = resolve_task_render_assets(task, **kwargs)
    if not result.valid:
        raise AssetPreflightError(result)
    return result

def _resolve_category(role: Mapping[str, Any], catalog: AssetCatalog) -> str | None:
    explicit = role.get('category')
    if isinstance(explicit, str) and explicit:
        return explicit
    synset = role.get('synset')
    if not isinstance(synset, str) or not synset:
        return None
    try:
        from scoper.benchmark.behavior.bddl_adapter import BDDLAdapter
        adapter = BDDLAdapter()
        categories = adapter.categories_for_synset(synset) if adapter.is_valid_synset(synset) else []
    except Exception:
        categories = []
    available = sorted((str(category) for category in categories if str(category) in catalog.models_by_category))
    if available:
        return available[0]
    return str(sorted(categories)[0]) if categories else synset.split('.', 1)[0]

def _role_exemption(role_name: str, role: Mapping[str, Any], *, category: str | None, scene_layout_mode: str, full_scene_object_ids: set[str], load_doors: bool) -> str | None:
    instance = str(role.get('bddl_inst') or role_name)
    if category in _SCENE_PROVIDED_CATEGORIES:
        return 'scene_provided_structure'
    if not load_doors and is_door_like(str(category or ''), instance):
        return 'door_loading_policy_excluded'
    object_id = str(role.get('object_id') or '')
    if scene_layout_mode == 'source_json_full' and object_id in full_scene_object_ids and (not bool(role.get('requires_spawn'))):
        return 'source_scene_object_alias'
    if str(category) == 'derived_faucet' or (str(role.get('source_kind') or '') == 'derived_scene_part' and str(role.get('object_type') or '') == 'Faucet'):
        return 'derived_faucet_alias'
    return None

def _model_supports_abilities(catalog: AssetCatalog, category: str, model: str, abilities: list[str]) -> bool:
    required = {ability for ability in abilities if ability in _ABILITY_TAGS}
    if not required:
        return True
    metadata_path = catalog.root / 'objects' / category / model / 'misc' / 'metadata.json'
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    flattened = set()
    for tags in (metadata.get('link_tags') or {}).values():
        if isinstance(tags, list):
            flattened.update((str(tag) for tag in tags))
    for link in (metadata.get('meta_links') or {}).values():
        if isinstance(link, Mapping):
            flattened.update((str(tag) for tag in link))
    if metadata.get('openable_joint_ids'):
        flattened.add('openable')
    return all((bool(flattened & _ABILITY_TAGS[ability]) for ability in required))

def _effective_scene_layout_mode(requested: str, task: Mapping[str, Any]) -> str:
    mode = str(requested or 'auto')
    if mode == 'auto':
        return 'source_json_full' if (task.get('source') or {}).get('source_json') else 'task_roles'
    return mode

def _full_scene_object_ids(task: Mapping[str, Any]) -> set[str]:
    source_json = (task.get('source') or {}).get('source_json')
    if not source_json:
        return set()
    try:
        scene = json.loads(Path(source_json).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return set()
    return {str(name) for name in (scene.get('objects_info') or {}).get('init_info') or {}}

def _task_without_renderer_asset_metadata(task: Mapping[str, Any]) -> Mapping[str, Any]:
    value = json_deep_copy(task)
    value.pop('renderer_asset_provenance', None)
    return value
__all__ = ['ASSET_FAILURE_CODE', 'ASSET_PREFLIGHT_SCHEMA', 'AssetCatalog', 'AssetPreflightError', 'AssetPreflightResult', 'load_asset_catalog', 'require_task_render_assets', 'resolve_task_render_assets']
