from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set
REQUIRED_TAXONOMY_FIELDS = ('version', 'groups')

class ObjectTaxonomyError(ValueError):
    pass

class ObjectTaxonomy:

    def __init__(self, groups: Mapping[str, Iterable[str]], version: str='0.1', excluded_object_types: Optional[Iterable[str]]=None, aliases: Optional[Mapping[str, str]]=None):
        self.version = version
        self.groups: Dict[str, Set[str]] = {group_name: set(object_types) for group_name, object_types in groups.items()}
        self.excluded_object_types = set(excluded_object_types or [])
        self.aliases = dict(aliases or {})

    @classmethod
    def from_mapping(cls, taxonomy: Mapping[str, Any]) -> 'ObjectTaxonomy':
        validate_object_taxonomy(taxonomy)
        return cls(groups=taxonomy['groups'], version=taxonomy['version'], excluded_object_types=taxonomy.get('excluded_object_types', []), aliases=taxonomy.get('aliases', {}))

    def normalize_type(self, object_type: str) -> str:
        return self.aliases.get(object_type, object_type)

    def has_group(self, group_name: str) -> bool:
        return group_name in self.groups

    def object_type_in_group(self, object_type: str, group_name: str) -> bool:
        normalized = self.normalize_type(object_type)
        if normalized in self.excluded_object_types:
            return False
        return normalized in self.groups.get(group_name, set())

    def is_excluded(self, object_type: str) -> bool:
        return self.normalize_type(object_type) in self.excluded_object_types

    def group_object_types(self, group_name: str) -> List[str]:
        return sorted(self.groups.get(group_name, set()))

def load_object_taxonomy(path: str) -> ObjectTaxonomy:
    try:
        taxonomy = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ObjectTaxonomyError("Invalid object taxonomy JSON '{}': {}.".format(path, exc)) from exc
    if not isinstance(taxonomy, Mapping):
        raise ObjectTaxonomyError("Object taxonomy '{}' must be a JSON object.".format(path))
    return ObjectTaxonomy.from_mapping(taxonomy)

def validate_object_taxonomy(taxonomy: Mapping[str, Any]) -> None:
    if not isinstance(taxonomy, Mapping):
        raise ObjectTaxonomyError('Object taxonomy must be a mapping.')
    _require_fields(taxonomy, REQUIRED_TAXONOMY_FIELDS, 'object taxonomy')
    if not isinstance(taxonomy['version'], str) or not taxonomy['version']:
        raise ObjectTaxonomyError("Object taxonomy field 'version' must be a string.")
    if not isinstance(taxonomy['groups'], Mapping) or not taxonomy['groups']:
        raise ObjectTaxonomyError("Object taxonomy field 'groups' must be a non-empty mapping.")
    for group_name, object_types in taxonomy['groups'].items():
        if not isinstance(group_name, str) or not group_name:
            raise ObjectTaxonomyError('Taxonomy group names must be non-empty strings.')
        _require_string_list(object_types, "taxonomy group '{}'".format(group_name))
    _require_optional_string_list(taxonomy, 'excluded_object_types', 'object taxonomy')
    aliases = taxonomy.get('aliases', {})
    if aliases is not None:
        if not isinstance(aliases, Mapping):
            raise ObjectTaxonomyError("Object taxonomy field 'aliases' must be a mapping.")
        for alias, canonical in aliases.items():
            if not isinstance(alias, str) or not alias:
                raise ObjectTaxonomyError('Taxonomy alias keys must be strings.')
            if not isinstance(canonical, str) or not canonical:
                raise ObjectTaxonomyError('Taxonomy alias values must be strings.')

def filter_objects_by_taxonomy_group(objects: Sequence[Mapping[str, Any]], taxonomy: ObjectTaxonomy, group_name: str) -> List[Mapping[str, Any]]:
    return [obj for obj in objects if taxonomy.object_type_in_group(_object_type(obj), group_name)]

def _object_type(obj: Mapping[str, Any]) -> str:
    object_type = obj.get('object_type') or obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    object_id = obj.get('object_id') or obj.get('objectId')
    if isinstance(object_id, str) and object_id:
        return object_id.split('|', 1)[0]
    raise ObjectTaxonomyError('Object is missing object_type/object_id.')

def _require_fields(mapping: Mapping[str, Any], required_fields: Iterable[str], context: str) -> None:
    missing = [field for field in required_fields if field not in mapping]
    if missing:
        raise ObjectTaxonomyError('{} missing required fields: {}.'.format(context, ', '.join(missing)))

def _require_optional_string_list(mapping: Mapping[str, Any], field: str, context: str) -> None:
    if field in mapping and mapping[field] is not None:
        _require_string_list(mapping[field], "{} field '{}'".format(context, field))

def _require_string_list(value: Any, context: str) -> None:
    if not isinstance(value, list):
        raise ObjectTaxonomyError('{} must be a list.'.format(context))
    if not all((isinstance(item, str) and item for item in value)):
        raise ObjectTaxonomyError('{} must contain only non-empty strings.'.format(context))
__all__ = ['ObjectTaxonomy', 'ObjectTaxonomyError', 'filter_objects_by_taxonomy_group', 'load_object_taxonomy', 'validate_object_taxonomy']
