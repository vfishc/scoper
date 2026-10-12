import re
VERSION = 'public-instruction-contract.v3'
OBSERVATION_CATEGORIES = frozenset({'explore_then_act', 'put_away_cross_room', 'put_away_visual_judgment', 'put_away_by_category', 'recover_and_store_incident_object', 'recover_multi_incident', 'tidy_by_policy', 'observation_conditioned_room_inspection', 'observation_conditioned_task', 'mid_execution_discovery'})

def room_navigation_arguments(tool_name, arguments, rooms):
    from scoper.benchmark.common.evaluation_binding_policy import match_public_rooms, public_room_records
    if tool_name != 'navigate_to':
        return None
    ref = arguments.get('object_ref')
    if not isinstance(ref, str):
        return None
    matches = match_public_rooms(ref, public_room_records(rooms))
    if not matches:
        return None
    return {'room': ref}

def with_room_scope(instruction, labels):
    labels = list(dict.fromkeys(labels))
    if not labels or any((not isinstance(x, str) or not x.strip() for x in labels)):
        raise ValueError('Verified room labels are required')
    text = str(instruction)
    text = re.sub('^For objects or conditions that need to be identified by observation, inspect: [^.]+\\.\\s*Objects already named in the instruction can be navigated to directly, without first navigating to a room\\.\\s*', '', text)
    if len(labels) > 1:
        text = re.sub('Inspect the ([^.]+?) and resolve', 'Inspect all of the specified rooms and resolve', text)
    text = re.sub('^Work in [^.]+\\.\\s*', '', text, flags=re.I)
    text = re.sub('\\bsomewhere in the house\\b|\\bin another room\\b', 'in the specified rooms', text, flags=re.I)
    text = re.sub('\\bIn the house\\b|\\bIn the whole house\\b', 'In the specified rooms', text, flags=re.I)
    return 'For objects or conditions that need to be identified by observation, inspect: ' + '; '.join(labels) + '. Objects already named in the instruction can be navigated to directly, without first navigating to a room. ' + text

def requires_observation_rooms(category, unidentified_roles):
    return category in OBSERVATION_CATEGORIES and bool(unidentified_roles)

def placement_preposition(backend, predicate_type, object_type):
    if predicate_type == 'object_on_receptacle':
        return 'on'
    if backend == 'behavior':
        return 'in'
    if backend != 'procthor':
        raise ValueError('Unknown backend')
    if object_type in {'Cabinet', 'Drawer', 'Fridge', 'Microwave', 'GarbageCan', 'Bowl', 'Cup', 'Mug', 'Sink', 'SinkBasin'}:
        return 'in'
    if object_type in {'Shelf', 'ShelvingUnit', 'DiningTable', 'ArmChair', 'Chair', 'CounterTop', 'Sofa', 'SideTable', 'CoffeeTable', 'Desk', 'Plate', 'TVStand', 'Bed', 'Dresser'}:
        return 'on'
    raise ValueError('Unreviewed destination type: ' + str(object_type))

def explicit_completion(instruction, requirements):
    text = str(instruction)
    additions = []
    for item in requirements:
        name, state = (item['name'], item['state'])
        pattern = '(?<!\\w)' + re.escape(name) + '(?!\\w)'
        if not re.search(pattern, instruction, re.I):
            raise ValueError('Completion target is not explicitly named')
        if state in ('in', 'on'):
            text = re.sub('\\b(?:on or in|in or on|on|in) the (' + re.escape(name) + ')(?!\\w)', state + ' the \\1', text, flags=re.I)
        elif state in ('open', 'closed'):
            additions.append(f'Leave the {name} {state} when finished.')
        else:
            raise ValueError('Unsupported public completion state')
    return ' '.join([text, *dict.fromkeys(additions)]).strip()

def compose_instructions(children):
    if not children or any((not str(text).strip() for text in children)):
        raise ValueError('Every child needs an instruction')
    return 'Complete these tasks in order: ' + ' '.join((f'({i + 1}) {text}' for i, text in enumerate(children))) + ' Stop only after completing all subtasks; a subtask instruction to stop means finish that subtask and continue to the next one.'
