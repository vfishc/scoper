from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from scoper.benchmark.behavior.paths import data_dir
TEMPLATE_DISABLED_FAILURE_CODE = 'TEMPLATE_DISABLED_BY_ACTIVE_SET'
TEMPLATE_ACTIVATION_CONFIG = data_dir() / 'task_sets' / 'template_activation.json'

class TemplateActivationError(ValueError):
    pass

def load_template_activation(profile: str | None=None, *, path: Path | None=None) -> dict[str, Any]:
    config_path = path or TEMPLATE_ACTIVATION_CONFIG
    if not config_path.exists():
        raise TemplateActivationError('Template activation config is missing: {}'.format(config_path))
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise TemplateActivationError('Template activation config is invalid JSON: {}'.format(exc)) from exc
    if not isinstance(config, Mapping):
        raise TemplateActivationError('Template activation config must be a JSON object: {}'.format(config_path))
    profiles = config.get('profiles')
    if not isinstance(profiles, Mapping) or not profiles:
        raise TemplateActivationError("Template activation config must define non-empty 'profiles'")
    profile_name = str(profile or config.get('default_profile') or '').strip()
    if not profile_name:
        raise TemplateActivationError("Template activation config must define 'default_profile'")
    profile_config = profiles.get(profile_name)
    if not isinstance(profile_config, Mapping):
        raise TemplateActivationError('Unknown template activation profile: {}'.format(profile_name))
    enabled = _string_list(profile_config.get('enabled_template_ids'), 'enabled_template_ids')
    disabled = _disabled_map(profile_config.get('disabled_template_ids'))
    _validate_unique(enabled, 'enabled_template_ids')
    _validate_unique(list(disabled), 'disabled_template_ids')
    overlap = sorted(set(enabled).intersection(disabled))
    if overlap:
        raise TemplateActivationError('Template activation enabled/disabled overlap: {}'.format(', '.join(overlap)))
    if not enabled:
        raise TemplateActivationError("Template activation profile '{}' has no enabled templates".format(profile_name))
    return {'profile': profile_name, 'enabled_template_ids': tuple(enabled), 'disabled_template_ids': dict(disabled), 'config_path': str(config_path)}

def load_active_template_ids(profile: str | None=None, *, path: Path | None=None) -> tuple[str, ...]:
    return tuple(load_template_activation(profile, path=path)['enabled_template_ids'])

def load_disabled_template_ids(profile: str | None=None, *, path: Path | None=None) -> dict[str, str]:
    return dict(load_template_activation(profile, path=path)['disabled_template_ids'])

def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TemplateActivationError("Template activation '{}' must be a list".format(field_name))
    out = []
    for item in value:
        item_str = str(item).strip()
        if not item_str:
            raise TemplateActivationError("Template activation '{}' contains an empty id".format(field_name))
        out.append(item_str)
    return out

def _disabled_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TemplateActivationError("Template activation 'disabled_template_ids' must be an object of id -> reason")
    out = {}
    for key, reason in value.items():
        key_str = str(key).strip()
        if not key_str:
            raise TemplateActivationError("Template activation 'disabled_template_ids' contains an empty id")
        out[key_str] = str(reason).strip()
    return out

def _validate_unique(items: list[str], field_name: str) -> None:
    seen = set()
    duplicates = []
    for item in items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        raise TemplateActivationError("Template activation '{}' contains duplicate ids: {}".format(field_name, ', '.join(sorted(duplicates))))
__all__ = ['TEMPLATE_ACTIVATION_CONFIG', 'TEMPLATE_DISABLED_FAILURE_CODE', 'TemplateActivationError', 'load_active_template_ids', 'load_disabled_template_ids', 'load_template_activation']
