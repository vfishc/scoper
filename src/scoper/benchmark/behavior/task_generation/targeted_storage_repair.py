from copy import deepcopy
from scoper.benchmark.behavior.task_generation.asset_revision import refresh_instructions, role_requirements
from scoper.benchmark.behavior.task_generation.scene_matcher import serialize_role_binding, candidate_satisfies_requirement
from scoper.benchmark.behavior.task_generation.profile_task_generator import role_abilities, translate_goal_predicates, available_spawn_support_slot, surface_occupants_for_support, spawn_support_can_hold

def replace_reference(value, old, new):
    if isinstance(value, dict):
        return {k: replace_reference(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_reference(v, old, new) for v in value]
    return new if value == old else value

def native_binding(name, candidate):
    role = serialize_role_binding(name, candidate)
    pos = candidate['position']
    role['center'] = [pos['x'], pos['z'], pos['y']]
    role['bddl_inst'] = candidate['object_id']
    role['abilities'] = role_abilities(candidate)
    role['init_states'] = {}
    return role

def split_child_destination(task, child_index, local_role, new_global_role):
    result = deepcopy(task)
    namespace = result['generation']['role_namespace'][child_index]
    old = namespace[local_role]
    namespace[local_role] = new_global_role
    result['roles'][new_global_role] = deepcopy(result['roles'][old])
    children = result['family_spec']['composite']['subtasks']
    child = children[child_index]
    children[child_index] = replace_reference(child, old, new_global_role)
    owner = child['subtask_id']
    result['goal']['predicates'] = [replace_reference(p, old, new_global_role) if p.get('owner_subtask_id') == owner else p for p in result['goal']['predicates']]
    return result

def set_room_scope(family, room_ids, rooms):
    from scoper.benchmark.common.evaluation_binding_policy import public_room_records
    labels = {r['room_id']: r['label'] for r in public_room_records(rooms)}
    ids = list(dict.fromkeys(room_ids))
    assert ids and all((i in labels for i in ids))
    family['scope'] = {'type': 'single_room' if len(ids) == 1 else 'multi_room', 'room_ids': ids, 'label': ' and the '.join((labels[i] for i in ids))}

def regenerate_child_goals(task, index, template):
    namespace = task['generation']['role_namespace'][index]
    selected = {n: task['roles'][g] for n, g in namespace.items()}
    child = task['family_spec']['composite']['subtasks'][index]
    predicates = translate_goal_predicates(template['goal_predicates'], {'roles': selected}, template)
    for local, global_role in namespace.items():
        predicates = replace_reference(predicates, local, global_role)
    for p in predicates:
        p['owner_subtask_id'] = child['subtask_id']
    child['goal_predicates'] = predicates
    task['goal']['predicates'] = [deepcopy(p) for c in task['family_spec']['composite']['subtasks'] for p in c['goal_predicates']]

def move_child_items_to_room(task, index, room_id, profile):
    namespace = task['generation']['role_namespace'][index]
    moves = []
    names = [g for n, g in namespace.items() if n.startswith(('target_', 'distractor_')) and task['roles'][g].get('requires_spawn')]
    for name in dict.fromkeys(names):
        role = task['roles'][name]
        placement = role.get('spawn_placement') or {}
        current_support = task['roles'].get(placement.get('support_role'), {})
        if current_support.get('room_id') == room_id:
            role['room_id'] = room_id
            placement['anchor_role'] = namespace['storage_0']
            continue
        supports = [(n, r) for n, r in task['roles'].items() if r.get('auxiliary_spawn_support') and r.get('room_id') == room_id and spawn_support_can_hold(r, role)]
        for support_name, support in supports:
            blockers = [r for n, r in task['roles'].items() if n != name and r.get('requires_spawn')]
            slot, center, offset = available_spawn_support_slot(role, support_name, support, {}, blockers, occupied_items=surface_occupants_for_support(profile, support))
            if center is None:
                continue
            before = deepcopy(role)
            role['center'] = center
            role['room_id'] = room_id
            role['spawn_placement'] = {'mode': 'on_support_surface', 'support_role': support_name, 'anchor_role': namespace['storage_0'], 'offset': offset, 'slot': slot, 'clearance': 0.02}
            moves.append({'role': name, 'before': before, 'after': deepcopy(role)})
            break
        else:
            raise ValueError('NO_SAFE_SAME_ROOM_SUPPORT:' + name)
    return moves

def validate_native_choice(task, name, candidate, profile, templates):
    rules = role_requirements(task, templates)[name]
    if not all((candidate_satisfies_requirement(candidate, r, profile) for r in rules)):
        raise ValueError('REPLACEMENT_VIOLATES_TEMPLATE:' + name)
