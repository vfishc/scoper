from __future__ import annotations
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from scoper.resources import resolve_resource
from typing import Any
from scoper.benchmark.behavior.scene_geometry.paths import repo_root
ABILITY_COLUMNS = ('breakable', 'fillable', 'flammable', 'openable', 'toggleable', 'cookable', 'heatSource', 'coldSource', 'sliceable', 'diceable', 'slicer', 'assembleable', 'meltable', 'particleRemover', 'particleApplier', 'particleSource', 'needsOrientation', 'particleSink', 'sceneObject', 'waterCook', 'mixingTool')

@dataclass(frozen=True)
class CategoryTaxonomyInfo:
    category: str
    synset: str | None
    object_type: str | None
    abilities: dict[str, bool]
    approved: bool | None

class TaxonomyIndex:

    def __init__(self, category_info: dict[str, CategoryTaxonomyInfo], synset_rows: dict[str, dict[str, str]], *, object_taxonomy_available: bool=False, object_taxonomy_synset_count: int=0, object_taxonomy_category_count: int=0):
        self.category_info = category_info
        self.synset_rows = synset_rows
        self.object_taxonomy_available = bool(object_taxonomy_available)
        self.object_taxonomy_synset_count = int(object_taxonomy_synset_count)
        self.object_taxonomy_category_count = int(object_taxonomy_category_count)

    @classmethod
    def build(cls, root: str | Path | None=None) -> 'TaxonomyIndex':
        base = Path(root) if root is not None else repo_root()
        mapping_path = resolve_resource('bddl3/bddl/generated_data/category_mapping.csv') if root is None else base / 'bddl3/bddl/generated_data/category_mapping.csv'
        synsets_path = resolve_resource('bddl3/bddl/generated_data/synsets.csv') if root is None else base / 'bddl3/bddl/generated_data/synsets.csv'
        synset_rows = _read_csv_by_key(synsets_path, 'synset')
        category_rows = _read_csv_by_key(mapping_path, 'category')
        object_taxonomy = _load_object_taxonomy(base)
        taxonomy_category_synsets: dict[str, str] = {}
        taxonomy_synset_abilities: dict[str, dict[str, bool]] = {}
        taxonomy_synset_count = 0
        if object_taxonomy is not None:
            for synset in object_taxonomy.get_all_synsets():
                taxonomy_synset_count += 1
                try:
                    taxonomy_synset_abilities[str(synset)] = {str(ability): True for ability in object_taxonomy.get_abilities(synset) or {}}
                    for category in object_taxonomy.get_categories(synset):
                        taxonomy_category_synsets.setdefault(str(category), str(synset))
                except Exception:
                    continue
        info: dict[str, CategoryTaxonomyInfo] = {}
        for category in sorted(set(category_rows) | set(taxonomy_category_synsets)):
            row = category_rows.get(category, {})
            synset = _empty_to_none(row.get('synset')) or taxonomy_category_synsets.get(category)
            synset_row = synset_rows.get(synset or '', {})
            abilities = {ability: _truthy(synset_row.get(ability)) for ability in ABILITY_COLUMNS if ability in synset_row}
            for ability, enabled in taxonomy_synset_abilities.get(synset or '', {}).items():
                abilities[ability] = bool(abilities.get(ability)) or bool(enabled)
            approved = None
            if 'approved' in row:
                approved = _truthy(row.get('approved'))
            info[category] = CategoryTaxonomyInfo(category=category, synset=synset, object_type=_empty_to_none(synset_row.get('objectType')), abilities=abilities, approved=approved)
        return cls(info, synset_rows, object_taxonomy_available=object_taxonomy is not None, object_taxonomy_synset_count=taxonomy_synset_count, object_taxonomy_category_count=len(taxonomy_category_synsets))

    def get(self, category: str) -> CategoryTaxonomyInfo | None:
        return self.category_info.get(str(category))

    def synset_for_category(self, category: str) -> str | None:
        info = self.get(category)
        return info.synset if info else None

    def object_type_for_category(self, category: str) -> str | None:
        info = self.get(category)
        return info.object_type if info else None

    def abilities_for_category(self, category: str) -> dict[str, bool]:
        info = self.get(category)
        return dict(info.abilities) if info else {}

    def summary(self) -> dict[str, Any]:
        object_types: dict[str, int] = {}
        for info in self.category_info.values():
            key = info.object_type or 'unknown'
            object_types[key] = object_types.get(key, 0) + 1
        return {'category_count': len(self.category_info), 'synset_count': len(self.synset_rows), 'object_taxonomy_available': self.object_taxonomy_available, 'object_taxonomy_synset_count': self.object_taxonomy_synset_count, 'object_taxonomy_category_count': self.object_taxonomy_category_count, 'object_types': dict(sorted(object_types.items()))}

def _read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return {str(row.get(key) or ''): dict(row) for row in reader if row.get(key)}

def _truthy(value: Any) -> bool:
    text = str(value or '').strip().lower()
    return text in {'1', 'true', 'yes', 'y'}

def _empty_to_none(value: Any) -> str | None:
    text = str(value or '').strip()
    return text or None

def _load_object_taxonomy(base: Path) -> Any | None:
    bddl_root = base / 'bddl3'
    if str(bddl_root) not in sys.path:
        sys.path.insert(0, str(bddl_root))
    try:
        from bddl.object_taxonomy import ObjectTaxonomy
        return ObjectTaxonomy()
    except Exception:
        return None
__all__ = ['ABILITY_COLUMNS', 'CategoryTaxonomyInfo', 'TaxonomyIndex']
