from __future__ import annotations
import copy
from typing import Any, Mapping, MutableMapping
TIDY_POLICY_TEXT = 'Apply these rules only to relevant issues you actually observe: if water is running, turn it off; if a storage container is open, close it; and if a dirty dish needs putting away, wash it before storing it. Some issue types may not be present.'

def conditional_tidy_policy_instruction(scope_label: Any=None) -> str:
    scope = str(scope_label or 'specified area').strip() or 'specified area'
    return f'In the {scope}, briefly inspect and make the area tidy and safe. {TIDY_POLICY_TEXT} Do not repeatedly search just to find an example of every rule. After handling the applicable issues you observed, stop.'

def materialize_tidy_policy_initial_violations(runtime_task: Mapping[str, Any]) -> Mapping[str, Any]:
    task = copy.deepcopy(dict(runtime_task))
    family_spec = task.get('family_spec')
    tidy = family_spec.get('tidy') if isinstance(family_spec, Mapping) else None
    if not isinstance(tidy, MutableMapping) or tidy.get('violations') != '__SAMPLED__':
        return task
    violations = []
    init = list(task.get('init') or [])
    for predicate in (task.get('goal') or {}).get('predicates') or []:
        if not isinstance(predicate, Mapping) or predicate.get('type') != 'object_toggled_off':
            continue
        args = predicate.get('args')
        role = args.get('object') if isinstance(args, Mapping) else None
        if not isinstance(role, str) or role not in (task.get('roles') or {}):
            continue
        violations.append({'type': 'faucet_on', 'object_role': role, 'urgency': 'immediate'})
        if not any((isinstance(item, Mapping) and item.get('predicate') in {'toggled_on', 'toggled_off'} and (list(item.get('args') or [])[:1] == [role]) for item in init)):
            init.append({'predicate': 'toggled_on', 'args': [role]})
    if not violations:
        raise ValueError('legacy tidy violations placeholder cannot be materialized from task goals')
    tidy['violations'] = violations
    tidy['policy_text'] = TIDY_POLICY_TEXT
    scope = family_spec.get('scope') if isinstance(family_spec, Mapping) else None
    scope_label = scope.get('label') if isinstance(scope, Mapping) else None
    task['instruction'] = conditional_tidy_policy_instruction(scope_label)
    task['init'] = init
    reset_plan = task.get('reset_plan')
    if isinstance(reset_plan, MutableMapping):
        reset_plan['state_and_relation_init'] = copy.deepcopy(init)
    generation = task.get('generation')
    if isinstance(generation, MutableMapping):
        generation['family_spec_materialized'] = copy.deepcopy(family_spec)
        notes = list(generation.get('binding_notes') or [])
        note = 'legacy_tidy_violation_materialized'
        if note not in notes:
            notes.append(note)
        instruction_note = 'legacy_tidy_conditional_policy_instruction'
        if instruction_note not in notes:
            notes.append(instruction_note)
        generation['binding_notes'] = notes
    return task
__all__ = ['TIDY_POLICY_TEXT', 'conditional_tidy_policy_instruction', 'materialize_tidy_policy_initial_violations']
