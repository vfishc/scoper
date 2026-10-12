from __future__ import annotations
from typing import Any, List, Mapping, Tuple
_EB_TO_BDDL: dict[str, Tuple[str | None, bool]] = {'object_on_receptacle': ('ontop', False), 'object_in_receptacle': ('inside', False), 'object_not_in_receptacle': ('inside', True), 'object_in_any_receptacle': (None, False), 'object_not_in_any_receptacle': (None, False), 'object_held': ('grasped', False), 'object_not_held': ('grasped', True), 'object_open': ('open', False), 'object_closed': ('open', True), 'object_toggled_on': ('toggled_on', False), 'object_toggled_off': ('toggled_on', True), 'object_filled': ('filled', False), 'object_soaked': ('saturated', False), 'object_cooked': ('cooked', False), 'object_hot': ('hot', False), 'object_cool': (None, False), 'object_dirty': (None, False), 'object_clean': (None, True), 'object_sliced': (None, False), 'agent_near': (None, False)}
_SUBSTANCE_PREDICATES = ('object_filled', 'object_soaked')
DEFAULT_SUBSTANCE = 'water'

class BDDLAdapterError(ValueError):
    pass

class BDDLAdapter:

    def __init__(self) -> None:
        from bddl.object_taxonomy import ObjectTaxonomy
        from bddl.predicates import TOKEN_TO_PREDICATE
        self.taxonomy = ObjectTaxonomy()
        self._token_to_pred = TOKEN_TO_PREDICATE

    def is_valid_synset(self, synset: str) -> bool:
        return self.taxonomy.is_valid_synset(synset)

    def synset_has_ability(self, synset: str, ability: str) -> bool:
        return self.taxonomy.has_ability(synset, ability)

    def synsets_with_ability(self, ability: str) -> List[str]:
        return sorted((s for s in self.taxonomy.get_all_synsets() if self.taxonomy.has_ability(s, ability)))

    def descendants_of(self, synset: str) -> List[str]:
        return list(self.taxonomy.get_descendants(synset))

    def categories_for_synset(self, synset: str) -> List[str]:
        return list(self.taxonomy.get_categories(synset))

    def supports_predicate(self, eb_type: str) -> bool:
        return eb_type in _EB_TO_BDDL

    def is_bddl_predicate(self, eb_type: str) -> bool:
        entry = _EB_TO_BDDL.get(eb_type)
        return bool(entry and entry[0] is not None)

    def to_bddl_token(self, eb_type: str) -> Tuple[str, bool]:
        if eb_type not in _EB_TO_BDDL:
            raise BDDLAdapterError("Unknown EB predicate type '{}'.".format(eb_type))
        token, negate = _EB_TO_BDDL[eb_type]
        if token is None:
            raise BDDLAdapterError("'{}' is an EB-private predicate, no BDDL token.".format(eb_type))
        return (token, negate)

    def to_bddl_predicate(self, eb_type: str):
        token, _ = self.to_bddl_token(eb_type)
        return self._token_to_pred[token]

    def split_goal_predicates(self, goal: Mapping[str, Any]) -> Tuple[list, list]:
        bddl_preds: list = []
        eb_preds: list = []
        for predicate in goal['predicates']:
            target = bddl_preds if self.is_bddl_predicate(predicate['type']) else eb_preds
            target.append(predicate)
        return (bddl_preds, eb_preds)

    def compile_goal(self, bddl_preds: list, role_map: Mapping[str, str]) -> list:
        from bddl.condition_evaluation import compile_state, create_scope
        parsed: list = []
        instances: set[str] = set()
        for predicate in bddl_preds:
            token, negate = self.to_bddl_token(predicate['type'])
            args = self._resolve_args(predicate, role_map, instances)
            atom = [token, *args]
            parsed.append(['not', atom] if negate else atom)
        object_map = self._object_map_from_instances(instances)
        scope = create_scope(object_map)
        return compile_state(parsed, scope=scope, object_map=object_map)

    @staticmethod
    def _resolve_args(predicate: Mapping[str, Any], role_map: Mapping[str, str], instances: set[str]) -> List[str]:
        names: List[str] = []
        args = predicate.get('args', {})
        if 'object' in args:
            names.append(_resolve_role(args['object'], role_map))
        if 'receptacle' in args:
            names.append(_resolve_role(args['receptacle'], role_map))
        if 'substance' in args:
            names.append(str(args['substance']))
        elif predicate['type'] in _SUBSTANCE_PREDICATES:
            names.append(DEFAULT_SUBSTANCE)
        for name in names:
            instances.add(name)
        return names

    @staticmethod
    def _object_map_from_instances(instances: set[str]) -> dict[str, List[str]]:
        omap: dict[str, List[str]] = {}
        for inst in instances:
            parts = inst.rsplit('_', 1)
            synset = parts[0] if len(parts) == 2 and parts[1].isdigit() else inst
            omap.setdefault(synset, []).append(inst)
        return omap

def _resolve_role(name: str, role_map: Mapping[str, str]) -> str:
    return role_map[name] if name in role_map else name
__all__ = ['BDDLAdapter', 'BDDLAdapterError', 'DEFAULT_SUBSTANCE']
