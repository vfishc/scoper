from __future__ import annotations
import random
from typing import Any, Mapping
from scoper.benchmark.behavior.bddl_adapter import BDDLAdapter
_BDDL_TO_EGO_ABILITIES: dict[str, tuple[str, ...]] = {'rigidBody': ('pickupable',), 'sceneObject': ('support_surface',), 'openable': ('openable', 'container'), 'fillable': ('fillable', 'container'), 'cookable': ('pickupable', 'cookable', 'heatable'), 'heatSource': ('heatSource', 'toggleable'), 'sliceable': ('pickupable', 'sliceable')}
_DEFAULT_EXTENTS: dict[str, list[float]] = {'rigidBody': [0.15, 0.15, 0.15], 'cookable': [0.12, 0.12, 0.12], 'sliceable': [0.12, 0.12, 0.12], 'fillable': [0.2, 0.2, 0.18], 'sceneObject': [0.8, 0.8, 0.9], 'openable': [0.7, 0.7, 1.6], 'heatSource': [0.7, 0.7, 0.9]}
_CURATED_SYNSETS: dict[str, list[str]] = {'rigidBody': ['bowl.n.01', 'mug.n.04', 'plate.n.04'], 'cookable': ['apple.n.01', 'potato.n.01', 'avocado.n.01', 'bacon.n.01', 'asparagus.n.02'], 'sceneObject': ['breakfast_table.n.01', 'shelf.n.01', 'countertop.n.01', 'coffee_table.n.01', 'desk.n.01'], 'openable': ['electric_refrigerator.n.01', 'cabinet.n.01', 'cabinet.n.03'], 'heatSource': ['stove.n.01', 'oven.n.01'], 'fillable': ['bowl.n.01', 'saucepan.n.01']}

def enumerate_role_bindings(template: Mapping[str, Any], adapter: BDDLAdapter, max_per_template: int=50, rng_seed: int=0) -> list[dict]:
    rng = random.Random(rng_seed)
    roles = template['roles']
    candidates: dict[str, list[str]] = {}
    for role_name, role_spec in roles.items():
        syns = _candidates_for_role(role_spec, adapter)
        if not syns:
            return []
        candidates[role_name] = syns
    bindings: list[dict] = []
    role_names = list(roles.keys())
    attempts = 0
    max_attempts = max_per_template * 10
    while len(bindings) < max_per_template and attempts < max_attempts:
        attempts += 1
        binding: dict[str, dict] = {}
        used_synsets: set[str] = set()
        ok = True
        for role_name in role_names:
            role_spec = roles[role_name]
            pool = [s for s in candidates[role_name] if s not in used_synsets]
            if not pool:
                ok = False
                break
            synset = rng.choice(pool)
            used_synsets.add(synset)
            primary_ability = _primary_ability(role_spec)
            binding[role_name] = {'synset': synset, 'bddl_inst': '{}_1'.format(synset), 'abilities': list(_BDDL_TO_EGO_ABILITIES.get(primary_ability, ('pickupable',))), 'extent': list(role_spec.get('default_extent') or _DEFAULT_EXTENTS.get(primary_ability, [0.2, 0.2, 0.2])), 'category': _category(synset, adapter)}
        if ok and binding not in bindings:
            bindings.append(binding)
    return bindings

def _candidates_for_role(role_spec: Mapping[str, Any], adapter: BDDLAdapter) -> list[str]:
    primary = _primary_ability(role_spec)
    if primary in _CURATED_SYNSETS:
        pool = list(_CURATED_SYNSETS[primary])
    else:
        pool = [s for s in adapter.taxonomy.get_all_synsets() if adapter.taxonomy.has_ability(s, primary)]
    ancestor = role_spec.get('synset_descendant_of')
    if ancestor:
        descendants = set(adapter.descendants_of(ancestor))
        pool = [s for s in pool if s in descendants]
    for extra in (role_spec.get('abilities_required') or [])[1:]:
        pool = [s for s in pool if adapter.taxonomy.has_ability(s, extra)]
    pool = [s for s in pool if _has_taxonomy_category(s, adapter)]
    return pool

def _primary_ability(role_spec: Mapping[str, Any]) -> str:
    abilities = role_spec.get('abilities_required') or []
    return abilities[0] if abilities else 'rigidBody'

def _category(synset: str, adapter: BDDLAdapter) -> str:
    if not adapter.is_valid_synset(synset):
        return synset.split('.')[0]
    cats = adapter.categories_for_synset(synset)
    return cats[0] if cats else synset.split('.')[0]

def _has_taxonomy_category(synset: str, adapter: BDDLAdapter) -> bool:
    return not adapter.is_valid_synset(synset) or bool(adapter.categories_for_synset(synset))
__all__ = ['enumerate_role_bindings']
