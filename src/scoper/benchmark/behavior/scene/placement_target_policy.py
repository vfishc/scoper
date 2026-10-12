from typing import Any, Mapping

def supports_generic_surface_placement(candidate: Mapping[str, Any]) -> bool:
    category = str(candidate.get('category') or '').lower()
    object_type = str(candidate.get('object_type') or '').lower()
    return 'sink' not in category and 'faucet' not in category and (object_type not in {'sink', 'sinkbasin', 'faucet'})

def role_requires_generic_placement(template: Mapping[str, Any], role_name: str) -> bool:
    predicates = list(template.get('goal_predicates') or []) + list(template.get('initial_state_constraints') or [])
    for predicate in predicates:
        if predicate.get('type') not in {'object_in_receptacle', 'object_on_receptacle', 'object_in_any_receptacle', 'ontop', 'on_top'}:
            continue
        args = predicate.get('args') or {}
        if role_name == (predicate.get('receptacle_role') or args.get('receptacle')):
            return True
        if role_name in (predicate.get('receptacle_roles') or []):
            return True
    return False
