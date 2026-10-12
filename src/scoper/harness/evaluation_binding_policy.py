from collections.abc import Mapping
EVIDENCE_BINDING_COMPILER_VERSION = 'task-binding-compiler.v7'
REFINED_BINDING_COMPILER_VERSION = 'task-binding-compiler.v8'
CONCRETE_BINDING_COMPILER_VERSION = 'task-binding-compiler.v9'
EXACT_BINDING_COMPILER_VERSION = 'task-binding-compiler.v10'
GOAL_ROLE_BINDING_COMPILER_VERSION = 'task-binding-compiler.v11'
PUBLIC_CONTRACT_BINDING_COMPILER_VERSION = 'task-binding-compiler.v12'
GOAL_ROLE_BINDING_COMPILER_VERSIONS = frozenset({GOAL_ROLE_BINDING_COMPILER_VERSION, PUBLIC_CONTRACT_BINDING_COMPILER_VERSION})
EXACT_BINDING_COMPILER_VERSIONS = frozenset({EXACT_BINDING_COMPILER_VERSION, *GOAL_ROLE_BINDING_COMPILER_VERSIONS})
CONCRETE_BINDING_COMPILER_VERSIONS = frozenset({CONCRETE_BINDING_COMPILER_VERSION, *EXACT_BINDING_COMPILER_VERSIONS})
EVIDENCE_BINDING_COMPILER_VERSIONS = frozenset({EVIDENCE_BINDING_COMPILER_VERSION, REFINED_BINDING_COMPILER_VERSION, *CONCRETE_BINDING_COMPILER_VERSIONS})
NON_BINDABLE_GENERIC_TARGETS = frozenset({'dish', 'dishware', 'storage', 'storage container', 'dish storage', 'dishware storage', 'storage dish', 'storage dishware'})
SCENE_NAVIGATION_NAMES = ('bed', 'sofa', 'couch', 'chair', 'armchair', 'stool', 'bench', 'desk', 'dining table', 'coffee table', 'side table', 'end table', 'ottoman', 'wardrobe', 'cabinet', 'drawer', 'shelf', 'bookshelf', 'dresser', 'nightstand', 'window', 'door', 'curtain', 'blinds', 'mirror', 'painting', 'wall', 'floor', 'ceiling', 'rug', 'carpet', 'bathtub', 'toilet', 'shower', 'sink', 'fridge', 'oven', 'microwave', 'stove', 'dishwasher', 'washing machine', 'dryer', 'radiator', 'air conditioner', 'fireplace', 'tv stand', 'floor lamp', 'desk lamp', 'ceiling light', 'light switch', 'countertop', 'kitchen island', 'potted plant', 'vase', 'television', 'computer', 'monitor', 'recycling bin', 'garbage can', 'towel rack', 'paper towel holder', 'toilet paper holder')
EVALUATION_CONCRETE_ALIASES = (('butter knife',), ('oven',), ('straight chair',), ('armchair', 'arm chair'), ('tv stand', 't v stand', 'television stand'), ('pen',), ('apple pie',), ('bar soap',), ('bookend',), ('bottle of apple juice',), ('bottle of tomato juice',), ('bottle of tomato paste',), ('license plate',), ('pencil case',), ('pencil holder',), ('teapot',), ('plant pot',), ('potato masher',), ('breakfast table',), ('commercial kitchen table',), ('base cabinet',), ('wall cabinet',))

def concrete_term_aliases(terms):
    values = set(terms)
    for group in EVALUATION_CONCRETE_ALIASES:
        if values.intersection(group):
            values.update(group)
    return tuple(sorted(values))

def uses_evidence_binding(spec):
    return spec.compiler_version in EVIDENCE_BINDING_COMPILER_VERSIONS

def spec_requires_observation(spec):
    return any((role.source_policy in {'observation', 'task_generic_witness'} and role.activation != 'concrete_reference' for role in spec.roles.values()))

def unobserved_scene_binding_enabled(spec):
    return spec.compiler_version in GOAL_ROLE_BINDING_COMPILER_VERSIONS and (not spec_requires_observation(spec))

def task_uses_evidence_binding(task):
    return isinstance(task, Mapping) and (task.get('task_binding_spec') or {}).get('compiler_version') in EVIDENCE_BINDING_COMPILER_VERSIONS

def task_uses_refined_binding(task):
    return isinstance(task, Mapping) and (task.get('task_binding_spec') or {}).get('compiler_version') in {REFINED_BINDING_COMPILER_VERSION, *CONCRETE_BINDING_COMPILER_VERSIONS}

def is_nonconcrete_target(reference):
    from scoper.harness.binding import normalize_reference
    value = normalize_reference(reference)
    return is_generic_target(value) or value.endswith(' storage') or value in {'storage receptacle', 'storage area', 'storage space', 'storage location'}

def normalize_concrete_reference(reference):
    from scoper.harness.binding import normalize_reference
    value = normalize_reference(reference)
    aliases = {'sink faucet': 'faucet', 'sink tap': 'tap', 'kitchen faucet': 'faucet', 'kitchen tap': 'tap', 'kitchen counter': 'countertop', 'counter': 'countertop', 'storage cabinet': 'cabinet', 'storage cupboard': 'cupboard', 'sinkbasin': 'sink basin', 'diningtable': 'dining table', 'sidetable': 'side table'}
    for prefix in ('interior of the ', 'interior of ', 'inside the ', 'inside ', 'on top of the ', 'on top of '):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return aliases.get(value, value)
QA_FINISH_RULES = 'Calling stop ends physical task execution. Any pending questions then form an answer-only finish phase: use answer(question_id, value), not navigation, scan or object actions. Task actions cannot resume after stop. Answers and stop use the same total episode step budget; no extra steps are granted. Task success and QA accuracy are scored separately. Do not wait for confirmation of success.'

def require_observation_binding(task):
    config = task.get('evaluation_binding') or {}
    if not isinstance(config, Mapping):
        raise ValueError('evaluation_binding must be an object')
    value = config.get('require_observation', False)
    if type(value) is not bool:
        raise ValueError('evaluation_binding.require_observation must be a boolean')
    return value

def terminal_failed_tool(transitions):
    if transitions and transitions[-1].get('success') is False:
        return transitions[-1]
    return None

def scan_public_feedback(completed, requested):
    if completed >= requested:
        return 'Scan completed.'
    if completed:
        return 'Scan partially completed; use the available views. Some views could not be reached.'
    return 'No scan views completed. Move to a clearer nearby position before scanning again.'

def candidate_role_pairs(spec, role_map, candidate_ids):
    reverse = {str(oid): role for role, oid in role_map.items() if role in spec.roles}
    if not uses_evidence_binding(spec):
        return [(oid, reverse.get(oid)) for oid in sorted(candidate_ids)]
    roles_by_id = {}
    for role, oid in role_map.items():
        if role in spec.roles:
            roles_by_id.setdefault(str(oid), []).append(role)
    return [(oid, role) for oid in sorted(candidate_ids) for role in sorted(roles_by_id.get(oid, [])) or [None]]
CLOSED_REFERENCE_PROMPT_VERSION = 'closed-public-reference.v1'
CLOSED_REFERENCE_ARGUMENT_DESCRIPTION = 'Use one complete category name or explicitly listed alias from the provided public object vocabulary. Copy the listed name without adding unlisted position, color, state or relation words. For example, use faucet if faucet is listed; do not invent right side faucet. Multiword names are valid when listed in full. Do not use private IDs or generic dish/dishware/storage targets.'

def public_reference_rules():
    from scoper.harness.vocabulary import SHARED_VOCABULARY_REGISTRY
    from scoper.harness.public_object_names import public_object_type
    groups = ['/'.join((public_object_type(alias) for alias in family.public_aliases)) for family in SHARED_VOCABULARY_REGISTRY.families]
    groups.extend(('/'.join((public_object_type(alias) for alias in group)) for group in EVALUATION_CONCRETE_ALIASES))
    groups.extend(SCENE_NAVIGATION_NAMES)
    return 'Reference rules (language support, not a list of objects present in this scene):\nUse only a complete category name or explicitly listed alias from the provided public object vocabulary. You may attempt a listed name before seeing the object; a name being supported does not mean that the object exists or the action is ready. Common concrete names and aliases: ' + '; '.join(dict.fromkeys((public_object_type(group) for group in groups))) + '.\ndish, dishware, storage, storage container and combinations such as dish storage or storage dish, cleaning storage and food storage describe task concepts, not tool targets. Observe to identify the relevant object category and condition, then output its listed category or alias. Do not add unlisted position, color, state or relation words to an object argument. For example, use faucet if faucet is listed; do not invent right side faucet. Multiword names remain valid when listed in full. Prior observation is optional by default. For room navigation, use the public room label specified in the instruction. Do not invent distinguishing names for unrelated same-category background objects; binding prioritizes the relevant task or subtask target using the public task context. A conditional instruction does not establish that its object or abnormal state exists. After an object moves or drops, find it again; its previous location is not its new location.\n' + ACTION_SELF_CHECK_RULES
ACTION_SELF_CHECK_RULES = 'Action prerequisites and visual self-check: Requesting a tool does not establish that it succeeded. Before picking up, opening, closing, toggling, or otherwise interacting with an object, navigate to that specific object and check that you are close enough. Navigating to its room alone does not establish that you have reached the object. Use tools only on suitable objects. Before put_down, you need to be holding the intended object and be near the intended receptacle; open the receptacle first when needed. After every action, including navigation, use the available images to assess whether the intended change occurred. Outcomes and failure reasons are not supplied. If placement did not achieve the goal, inspect whether the object is still held or is elsewhere. If it is no longer held, find it, navigate to it, pick it up, then navigate to the destination before attempting placement again. Do not assume every unsuccessful placement releases the object. If repeated requests make no visible progress, obtain another view or change your approach instead of repeating the same request indefinitely. Stop only when visual evidence supports completion.'

def match_public_rooms(reference, rooms):
    from scoper.harness.binding import normalize_reference
    aliases = {'livingroom': 'living room', 'diningroom': 'dining room', 'bed room': 'bedroom'}

    def norm(value):
        text = normalize_reference(value)
        return aliases.get(text, text)
    query = norm(reference)
    matches = [room for room in rooms if query and query in {norm(room.get('room_type', '')), norm(room.get('label', ''))}]
    if not matches and query == 'office':
        matches = [room for room in rooms if norm(room.get('room_type', '')).endswith('office')]
    return matches

def public_room_records(rooms):
    import re
    from collections import Counter
    from scoper.harness.binding import normalize_reference
    records = [dict(room) for room in rooms]

    def natural_key(room):
        return tuple(((0, int(p)) if p.isdigit() else (1, p) for p in re.split('(\\d+)', str(room.get('room_id', '')))))
    records.sort(key=natural_key)
    names = [normalize_reference(r.get('room_type', 'room')) for r in records]
    counts = Counter(names)
    indexes = Counter()
    for record, name in zip(records, names):
        indexes[name] += 1
        record['label'] = name if counts[name] == 1 else f'{name} {indexes[name]}'
    return records

def reference_tool_schema(item):
    import copy
    result = copy.deepcopy(item)
    name = result.get('name')
    if name == 'navigate_to':
        from scoper.harness.direct_tool_guidance import COMMON
        result['description'] = COMMON['navigate_to'] + ' Prior observation is not required by default.'
    props = result.get('parameters', {}).get('properties', {})
    if 'room' in props:
        props['room']['description'] = 'Natural public room label from the instruction, including its number when specified. Copy the supplied label; do not invent a room or number. Do not use simulator identifiers.'
    for key in ('target', 'object_ref', 'source'):
        if key in props:
            props[key]['description'] = CLOSED_REFERENCE_ARGUMENT_DESCRIPTION
    if name == 'navigate_to' and 'object_ref' in props:
        props['object_ref']['description'] += ' A natural room label from the instruction is also valid, including its number when specified.'
    return result

def public_destination_roles(task, focus_role):
    from scoper.harness.binding import reference_forms, normalize_reference
    if not focus_role:
        return ()
    spec_roles = (task.get('task_binding_spec') or {}).get('roles', {})
    result = set()
    for view in scoped_role_tasks(task, focus_role):
        instruction = ' ' + normalize_reference(view.get('instruction', '')) + ' '
        for pred in (view.get('goal') or {}).get('predicates', []):
            args = pred.get('args') or {}
            if pred.get('object_role', args.get('object')) != focus_role:
                continue
            roles = [pred.get('receptacle_role', args.get('receptacle')), *(pred.get('receptacle_roles') or [])]
            for role in roles:
                terms = spec_roles.get(role, {}).get('public_terms', [])
                if any((' ' + form + ' ' in instruction for term in terms for form in reference_forms(term))):
                    result.add(role)
    return tuple(sorted(result))

def is_generic_target(normalized):
    words = normalized.split()
    while words and words[0] in {'dirty', 'clean', 'open', 'closed', 'running'}:
        words.pop(0)
    return ' '.join(words) in NON_BINDABLE_GENERIC_TARGETS

def _references(value, role_name):
    if isinstance(value, Mapping):
        return any((_references(v, role_name) for v in value.values()))
    if isinstance(value, (list, tuple)):
        return any((_references(v, role_name) for v in value))
    return value == role_name

def scoped_role_tasks(task, role_name):
    children = (task.get('family_spec') or {}).get('composite', {}).get('subtasks', [])
    if not children:
        return [task]
    views = []
    for child in children:
        if _references(child.get('family_spec', {}), role_name) or _references(child.get('goal_predicates', []), role_name):
            view = dict(task)
            view.update(category=child.get('category'), family_spec=child.get('family_spec', {}), instruction=child.get('instruction_fragment', ''))
            view['goal'] = {'predicates': child.get('goal_predicates', [])}
            views.extend(scoped_role_tasks(view, role_name))
    return views

def public_sequence_roles(task):
    roles = (task.get('task_binding_spec') or {}).get('roles', {})
    family = task.get('family_spec') or {}
    children = (family.get('composite') or {}).get('subtasks') or []
    if children:
        return tuple(dict.fromkeys((role for child in children for role, spec in roles.items() if spec.get('source_policy') == 'task_explicit' and (_references(child.get('family_spec', {}), role) or _references(child.get('goal_predicates', []), role)))))
    primary = (family.get('mid_execution') or {}).get('primary_task') or {}
    if primary.get('type') == 'container_re_closed':
        return tuple((role for role in primary.get('target_objects', []) if role in roles))
    return ()

def public_placed_roles(task, history, role_map):
    sequence = public_sequence_roles(task)
    if not sequence:
        return ()
    destinations = {}
    for predicate in (task.get('goal') or {}).get('predicates', []):
        args = predicate.get('args') or {}
        role = predicate.get('object_role', args.get('object'))
        targets = [predicate.get('receptacle_role', args.get('receptacle')), *(predicate.get('receptacle_roles') or [])]
        if role:
            destinations.setdefault(role, set()).update((str(role_map[r]) for r in targets if r in role_map))
    completed = set()
    held_role = None
    for transition in history:
        tool = transition.get('tool_name')
        if tool not in {'pick_up', 'put_down', 'drop'}:
            continue
        raw = transition.get('raw_execution') or {}
        target = str(raw.get('object_id') or raw.get('target_object_id') or raw.get('target') or '')
        if tool == 'pick_up' and transition.get('success') is True:
            held_role = next((r for r in sequence if str(role_map.get(r, '')) == target and r not in completed), None)
        elif tool == 'put_down':
            if transition.get('success') is True and held_role and (target in destinations.get(held_role, set())):
                completed.add(held_role)
            held_role = None
        elif tool == 'drop':
            held_role = None
    return tuple((role for role in sequence if role in completed))

def evidence_source_policy(task, role_name, legacy_policy, public_terms=()):
    policies = []
    local_name = role_name.rsplit('__', 1)[-1]
    for view in scoped_role_tasks(task, role_name):
        category = str(view.get('category') or '')
        if category in {'recover_multi_incident', 'mid_execution_discovery', 'tidy_by_policy', 'observation_conditioned_room_inspection', 'observation_conditioned_task'}:
            policies.append(('observation', 'visual_evidence'))
        elif category == 'recover_and_store_incident_object' and (not local_name.startswith('primary_')):
            policies.append(('observation', 'visual_evidence'))
        else:
            policy = legacy_policy(view, local_name)
            if policy[0] == 'task_explicit':
                from scoper.harness.binding import normalize_reference, reference_forms
                text = ' ' + normalize_reference(view.get('instruction', '')) + ' '
                grounded = any((' ' + form + ' ' in text for term in public_terms if not is_generic_target(normalize_reference(term)) for form in reference_forms(term)))
                if not grounded:
                    policy = ('observation', 'visual_evidence')
            policies.append(policy)
    if not policies:
        return ('observation', 'visual_evidence')
    if ('task_explicit', 'episode_start') in policies:
        return ('task_explicit', 'episode_start')
    if any((p[0] in {'observation', 'task_generic_witness'} for p in policies)):
        return ('observation', 'visual_evidence')
    return policies[0]
