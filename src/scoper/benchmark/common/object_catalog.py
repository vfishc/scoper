from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

@dataclass(frozen=True)
class ObjectCatalogEntry:
    alias: str
    object_id: str
    object_type: str
    visible: bool
    pickupable: bool
    openable: bool
    is_open: Optional[bool]
    toggleable: bool
    is_toggled: Optional[bool]
    sliceable: bool
    is_sliced: Optional[bool]
    receptacle: bool
    distance: Optional[float]
    parent_receptacles: List[str]
    position: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)

class ObjectCatalog:

    def __init__(self, alias_by_object_id: Optional[Mapping[str, str]]=None, next_index_by_type: Optional[Mapping[str, int]]=None):
        self._alias_by_object_id = dict(alias_by_object_id or {})
        self._alias_to_object_id = {alias: object_id for object_id, alias in self._alias_by_object_id.items()}
        self._next_index_by_type = dict(next_index_by_type or {})
        if not self._next_index_by_type:
            self._rebuild_type_indices()
        self.entries: List[ObjectCatalogEntry] = []
        self.entries_by_alias: Dict[str, ObjectCatalogEntry] = {}
        self.entries_by_object_id: Dict[str, ObjectCatalogEntry] = {}

    @classmethod
    def from_metadata(cls, metadata_or_objects: Any, previous_catalog: Optional['ObjectCatalog']=None) -> 'ObjectCatalog':
        catalog = cls(alias_by_object_id=previous_catalog._alias_by_object_id if previous_catalog is not None else None, next_index_by_type=previous_catalog._next_index_by_type if previous_catalog is not None else None)
        catalog.update_from_metadata(metadata_or_objects)
        return catalog

    def update_from_metadata(self, metadata_or_objects: Any) -> 'ObjectCatalog':
        objects = _extract_objects(metadata_or_objects)
        entries = []
        for obj in objects:
            object_id = _get_required_str(obj, 'objectId')
            object_type = _get_object_type(obj, object_id)
            alias = self._get_or_assign_alias(object_id, object_type)
            entries.append(_entry_from_metadata(alias, object_id, object_type, obj))
        self.entries = entries
        self.entries_by_alias = {entry.alias: entry for entry in entries}
        self.entries_by_object_id = {entry.object_id: entry for entry in entries}
        return self

    def resolve_alias(self, alias: str) -> Optional[str]:
        if alias in self.entries_by_alias:
            return self.entries_by_alias[alias].object_id
        return self._alias_to_object_id.get(alias)

    def get(self, alias: str) -> Optional[ObjectCatalogEntry]:
        return self.entries_by_alias.get(alias)

    def aliases(self) -> List[str]:
        return [entry.alias for entry in self.entries]

    def to_dicts(self) -> List[dict]:
        return [entry.to_dict() for entry in self.entries]

    def to_prompt_context(self, allowed_object_types: Optional[Iterable[str]]=None) -> dict:
        entries = self._prompt_entries(allowed_object_types)
        return {'navigable_objects': [entry.alias for entry in entries], 'pickupable_objects': [entry.alias for entry in entries if entry.pickupable], 'openable_objects': [entry.alias for entry in entries if entry.openable], 'receptacles': [entry.alias for entry in entries if entry.receptacle], 'toggleable_objects': [entry.alias for entry in entries if entry.toggleable], 'sliceable_objects': [entry.alias for entry in entries if entry.sliceable], 'visible_objects': [entry.alias for entry in entries if entry.visible]}

    def _prompt_entries(self, allowed_object_types: Optional[Iterable[str]]) -> Sequence[ObjectCatalogEntry]:
        if allowed_object_types is None:
            return self.entries
        allowed = set(allowed_object_types)
        return [entry for entry in self.entries if entry.object_type in allowed]

    def _get_or_assign_alias(self, object_id: str, object_type: str) -> str:
        if object_id in self._alias_by_object_id:
            return self._alias_by_object_id[object_id]
        next_index = self._next_index_by_type.get(object_type, 0) + 1
        while True:
            alias = object_type if next_index == 1 else '{}_{}'.format(object_type, next_index)
            if alias not in self._alias_to_object_id:
                break
            next_index += 1
        self._next_index_by_type[object_type] = next_index
        self._alias_by_object_id[object_id] = alias
        self._alias_to_object_id[alias] = object_id
        return alias

    def _rebuild_type_indices(self) -> None:
        for object_id, alias in self._alias_by_object_id.items():
            object_type = object_id.split('|', 1)[0]
            if alias == object_type:
                index = 1
            elif alias.startswith(object_type + '_'):
                suffix = alias.rsplit('_', 1)[-1]
                index = int(suffix) if suffix.isdigit() else 1
            else:
                index = 1
            self._next_index_by_type[object_type] = max(self._next_index_by_type.get(object_type, 0), index)

def build_object_catalog(metadata_or_objects: Any, previous_catalog: Optional[ObjectCatalog]=None) -> ObjectCatalog:
    return ObjectCatalog.from_metadata(metadata_or_objects, previous_catalog=previous_catalog)

def _extract_objects(metadata_or_objects: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(metadata_or_objects, Mapping):
        objects = metadata_or_objects.get('objects')
        if objects is None:
            raise ValueError("Metadata mapping must contain an 'objects' field.")
        return objects
    if isinstance(metadata_or_objects, list) or isinstance(metadata_or_objects, tuple):
        return metadata_or_objects
    raise ValueError('Expected AI2-THOR metadata or a list of object metadata.')

def _entry_from_metadata(alias: str, object_id: str, object_type: str, obj: Mapping[str, Any]) -> ObjectCatalogEntry:
    return ObjectCatalogEntry(alias=alias, object_id=object_id, object_type=object_type, visible=bool(obj.get('visible', False)), pickupable=bool(obj.get('pickupable', False)), openable=bool(obj.get('openable', False)), is_open=_optional_bool(obj.get('isOpen')), toggleable=bool(obj.get('toggleable', False)), is_toggled=_optional_bool(obj.get('isToggled')), sliceable=bool(obj.get('sliceable', False)), is_sliced=_optional_bool(obj.get('isSliced')), receptacle=bool(obj.get('receptacle', False)), distance=_optional_float(obj.get('distance')), parent_receptacles=list(obj.get('parentReceptacles') or []), position=dict(obj.get('position') or {}))

def _get_required_str(obj: Mapping[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError("Object metadata is missing required string field '{}'.".format(key))
    return value

def _get_object_type(obj: Mapping[str, Any], object_id: str) -> str:
    object_type = obj.get('objectType')
    if isinstance(object_type, str) and object_type:
        return object_type
    return object_id.split('|', 1)[0]

def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)

def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
__all__ = ['ObjectCatalog', 'ObjectCatalogEntry', 'build_object_catalog']
