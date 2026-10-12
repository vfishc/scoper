from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Set
from scoper.benchmark.procthor.procthor_dataset_assets import is_prior_house_uri, parse_prior_house_uri
REQUIRED_HOUSE_REGISTRY_FIELDS = ('house_id', 'split', 'source', 'house_path')

class HouseRegistryError(ValueError):
    pass

def load_house_registry(path: str, validate_paths: bool=True) -> List[dict]:
    entries = _read_jsonl(path)
    validate_house_registry(entries, validate_paths=validate_paths)
    return entries

def validate_house_registry(entries: Sequence[Mapping[str, Any]], validate_paths: bool=True) -> None:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise HouseRegistryError('House registry must be a sequence of entries.')
    seen_house_ids: Set[str] = set()
    for index, entry in enumerate(entries):
        validate_house_registry_entry(entry, validate_paths=validate_paths, context='registry entry {}'.format(index))
        house_id = entry['house_id']
        if house_id in seen_house_ids:
            raise HouseRegistryError("Duplicate house_id '{}' in house registry.".format(house_id))
        seen_house_ids.add(house_id)

def validate_house_registry_entry(entry: Mapping[str, Any], validate_paths: bool=True, context: str='registry entry') -> None:
    if not isinstance(entry, Mapping):
        raise HouseRegistryError('{} must be a mapping.'.format(context))
    _require_fields(entry, REQUIRED_HOUSE_REGISTRY_FIELDS, context)
    for field in REQUIRED_HOUSE_REGISTRY_FIELDS:
        _require_non_empty_string(entry, field, context)
    if entry['source'] != 'procthor':
        raise HouseRegistryError("{} field 'source' must be 'procthor'.".format(context))
    metadata = entry.get('metadata')
    if metadata is not None and (not isinstance(metadata, Mapping)):
        raise HouseRegistryError("{} optional field 'metadata' must be a mapping.".format(context))
    house_path = entry['house_path']
    if is_prior_house_uri(house_path):
        parse_prior_house_uri(house_path)
        return
    if validate_paths and (not Path(house_path).is_file()):
        raise HouseRegistryError("{} field 'house_path' does not exist: {}.".format(context, house_path))

def _read_jsonl(path: str) -> List[dict]:
    entries: List[dict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HouseRegistryError("Invalid JSON in house registry '{}' at line {}: {}.".format(path, line_number, exc)) from exc
        if not isinstance(entry, dict):
            raise HouseRegistryError("House registry '{}' line {} must be a JSON object.".format(path, line_number))
        entries.append(entry)
    return entries

def _require_fields(mapping: Mapping[str, Any], required_fields: Iterable[str], context: str) -> None:
    missing = [field for field in required_fields if field not in mapping]
    if missing:
        raise HouseRegistryError('{} missing required fields: {}.'.format(context, ', '.join(missing)))

def _require_non_empty_string(mapping: Mapping[str, Any], field: str, context: str) -> None:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise HouseRegistryError("{} field '{}' must be a non-empty string.".format(context, field))
__all__ = ['HouseRegistryError', 'REQUIRED_HOUSE_REGISTRY_FIELDS', 'load_house_registry', 'validate_house_registry', 'validate_house_registry_entry']
