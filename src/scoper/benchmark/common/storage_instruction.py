from copy import deepcopy
import re
from scoper.benchmark.common.public_object_names import public_object_type
from scoper.benchmark.common.binding import normalize_reference
from scoper.benchmark.common.instruction_contract import compose_instructions, placement_preposition
VERSION = 'storage-instruction.v2'

def repair_storage_instructions(task):
    result = deepcopy(task)
    roles = result.get('roles') or {}
    backend = result.get('backend')
    family = result.get('family_spec') or {}
    children = (family.get('composite') or {}).get('subtasks') or []
    if children:
        changed = False
        for child in children:
            view = dict(child, roles=roles, backend=backend, instruction=child.get('instruction_fragment', ''), goal={'predicates': child.get('goal_predicates', [])})
            fixed = repair_storage_instructions(view)
            if fixed == view:
                continue
            changed = True
            child['instruction_fragment'] = fixed['instruction']
            if 'family_spec' in child:
                child['family_spec'] = fixed['family_spec']
        if changed:
            result['instruction'] = compose_instructions([c['instruction_fragment'] for c in children])
        return result

    def destination(role):
        metadata = roles[role]
        typ = metadata.get('object_type') or metadata.get('category')
        if not typ:
            raise ValueError('Storage destination lacks a concrete type: ' + role)
        name = public_object_type(normalize_reference(typ))
        predicates = (result.get('goal') or {}).get('predicates', [])
        on_surface = any((p.get('type') == 'object_on_receptacle' and (p.get('receptacle_role') or (p.get('args') or {}).get('receptacle')) == role for p in predicates))
        prep = placement_preposition(backend, 'object_on_receptacle' if on_surface else 'object_in_receptacle', typ)
        return (name, prep)
    text = result.get('instruction', '')
    storage = (family.get('put_away') or {}).get('storage_roles') or []
    if len(storage) == 1:
        name, prep = destination(storage[0])
        had_cabinet = 'storage cabinet' in text.lower()
        if had_cabinet and roles[storage[0]].get('object_type') in {'Shelf', 'ShelvingUnit', 'CounterTop', 'DiningTable', 'SideTable'}:
            text = re.sub('\\s*Leave the (?:storage )?cabinet closed when finished\\.', '', text, flags=re.I)
        text = re.sub('\\b(?:into|in|on) the storage cabinet\\b', prep + ' the ' + name, text, flags=re.I)
        text = re.sub('\\bthe storage cabinet\\b', 'the ' + name, text, flags=re.I)
        if had_cabinet:
            text = re.sub('\\b(?:into|in|on) the cabinet\\b', prep + ' the ' + name, text, flags=re.I)
        text = re.sub('\\b(?:into|in|on) (?:the )?storage\\b', prep + ' the ' + name, text, flags=re.I)
        text = re.sub('\\b(?:into|in|on) (?:the )?(?:cleaning|dish|food) storage\\b', prep + ' the ' + name, text, flags=re.I)
    recover = family.get('recover') or {}
    incidents = ([recover['incident']] if isinstance(recover.get('incident'), dict) else []) + list(recover.get('incidents') or [])
    incidents += list((family.get('tidy') or {}).get('violations') or [])
    incidents += list((family.get('observation_conditioned') or {}).get('latent_payloads') or [])
    dirty_destinations = {i.get('support_roles', {}).get('storage') for i in incidents if i.get('type', i.get('issue_type')) in {'dirty_dish', 'dirty_dish_on_floor_or_surface'} and i.get('support_roles', {}).get('storage')}
    if len(dirty_destinations) > 1:
        raise ValueError('Multiple hidden dish destinations require an observable routing policy')
    for incident in incidents:
        storage_role = (incident.get('support_roles') or {}).get('storage')
        if incident.get('type', incident.get('issue_type')) not in {'dirty_dish', 'dirty_dish_on_floor_or_surface'} or not storage_role:
            continue
        name, prep = destination(storage_role)
        old = 'wash them and put them away'
        replacement = f'wash them and put them {prep} the {name}'
        text = text.replace(old, replacement)
        for phrase in ['clean it and put it away', 'wash it before storing it', 'wash it and store it']:
            text = text.replace(phrase, f'clean it and put it {prep} the {name}')
        text = text.replace('clean dirty washable items before putting them away', f'clean dirty washable items and put them {prep} the {name}')
        if isinstance(incident.get('policy_text'), str):
            incident['policy_text'] = incident['policy_text'].replace(old, replacement)
    tidy = family.get('tidy') or {}
    if len(dirty_destinations) == 1 and isinstance(tidy.get('policy_text'), str):
        name, prep = destination(next(iter(dirty_destinations)))
        tidy['policy_text'] = tidy['policy_text'].replace('wash it before storing it', f'clean it and put it {prep} the {name}')
    result['instruction'] = text
    return result

def repair_cold_storage_rule(task, eligible_types):
    result = deepcopy(task)
    children = ((result.get('family_spec') or {}).get('composite') or {}).get('subtasks') or []
    if children:
        for child in children:
            view = dict(child, roles=result.get('roles', {}), backend=result.get('backend'), instruction=child.get('instruction_fragment', ''))
            child['instruction_fragment'] = repair_cold_storage_rule(view, eligible_types)['instruction']
        result['instruction'] = compose_instructions([c['instruction_fragment'] for c in children])
        return result
    family = result.get('family_spec') or {}
    policy = family.get('put_away') or {}
    text = result.get('instruction', '')
    if policy.get('semantic_category') != 'cold_food' or 'food that should be kept cold' not in text:
        return result
    allowed = set(eligible_types)
    for role in policy.get('target_roles') or []:
        if result['roles'][role].get('object_type') not in allowed:
            raise ValueError('Cold-food task target is outside the public template rule: ' + role)
    names = sorted({public_object_type(normalize_reference(t)) for t in allowed})
    text = text.replace('food that should be kept cold', 'food matching the cold-storage category rule')
    if 'inspect:' in text:
        text = re.sub('In the [^.]+?, put away', 'In the specified observation rooms, put away', text)
    text += ' For this benchmark, the cold-storage category rule includes: ' + '; '.join(names) + '. This is the general category rule, not a list of items present in this scene. Find the matching food items by observation; do not move nonmatching items into the fridge. Items already inside the fridge need not be taken out and placed again.'
    result['instruction'] = text
    return result
