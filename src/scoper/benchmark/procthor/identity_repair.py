from copy import deepcopy
import json
import math
import re
from scoper.benchmark.common.task_identity_policy import conflicts
from scoper.benchmark.procthor.task_templates import role_candidates, _binding_satisfies_role_constraints
from scoper.benchmark.procthor.task_generator import _instantiate_instruction, _humanize_object_type

def _child(task, role, templates):
    match = re.match('task(\\d+)__(.*)', role)
    if match:
        index, local = (int(match[1]), match[2])
        return (index, local, templates[index])
    return (None, role, templates[0])

def _choices(task, role, profile, templates, taxonomy):
    objects = {o['object_id']: o for o in profile['objects']}
    old = objects[task['roles'][role]['object_id']]
    index, local, template = _child(task, role, templates)
    requirement = deepcopy(template.get('requires', {}).get(local))
    if requirement is None:
        if local.startswith('container_'):
            requirement = {'taxonomy_group': 'policy_open_containers', 'must_be_openable': True}
        elif local.startswith('storage_'):
            requirement = {'taxonomy_group': 'dishware_storage', 'must_be_receptacle': True, 'must_not_be_pickupable': True}
        else:
            return []
    required_affordances = set()
    state_affordances = {'object_closed': 'openable', 'object_open': 'openable', 'object_clean': 'dirtyable', 'object_dirty': 'dirtyable', 'object_sliced': 'sliceable', 'object_toggled_on': 'toggleable', 'object_toggled_off': 'toggleable', 'object_cooked': 'cookable'}

    def collect(node):
        if isinstance(node, dict):
            if node.get('object_role') == role:
                capability = state_affordances.get(node.get('type'))
                if capability:
                    required_affordances.add(capability)
                if node.get('type') == 'set_open_state':
                    required_affordances.add('openable')
                if node.get('type') == 'set_dirty_state':
                    required_affordances.add('dirtyable')
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)
    collect(task.get('goal', {}))
    collect(task.get('init', {}))
    collect(task.get('family_spec', {}))
    bound = {}
    prefix = '' if index is None else f'task{index}__'
    for name, value in task['roles'].items():
        if (index is None or name.startswith(prefix)) and value.get('object_id') in objects:
            bound[name[len(prefix):]] = objects[value['object_id']]
    used = {v.get('object_id') for v in task['roles'].values()}
    candidates = []
    families = ({'Bowl', 'Plate', 'Cup', 'Mug'}, {'Book', 'Pencil', 'Pen'}, {'Apple', 'Potato', 'Tomato', 'Lettuce', 'Bread', 'Egg'}, {'SoapBar', 'SoapBottle', 'SprayBottle', 'DishSponge'})
    family = next((g for g in families if old['object_type'] in g), None)
    for obj in role_candidates(profile, requirement, taxonomy):
        if any((not obj.get(capability) for capability in required_affordances)):
            continue
        if obj['object_id'] in used or obj['object_type'] == old['object_type']:
            continue
        keep_room = old.get('pickupable') or local.startswith(('storage_', 'container_'))
        if keep_room and (not old.get('room_id') or obj.get('room_id') != old['room_id']):
            continue
        if bool(obj.get('pickupable')) != bool(old.get('pickupable')):
            continue
        if old.get('pickupable') and (family is None or obj['object_type'] not in family):
            continue
        if not _binding_satisfies_role_constraints(template, dict(bound, **{local: obj})):
            continue
        for other in requirement.get('different_from', []):
            if other in bound and bound[other]['object_id'] == obj['object_id']:
                break
        else:
            a, b = (old.get('position', {}), obj.get('position', {}))
            distance = math.hypot(a.get('x', 0) - b.get('x', 0), a.get('z', 0) - b.get('z', 0))
            candidates.append((distance, obj['object_id'], obj))
    return [o for _, _, o in sorted(candidates)[:24]]

def _update_public_text(task, original, templates):
    children = task.get('family_spec', {}).get('composite', {}).get('subtasks', [])
    groups = list(enumerate(children)) if children else [(None, task)]
    for index, child in groups:
        prefix = '' if index is None else f'task{index}__'
        namespaces = task.get('metadata', {}).get('composite_generation', {}).get('role_namespace', [])
        namespace = namespaces[index] if index is not None and index < len(namespaces) else {r[len(prefix):]: r for r in task['roles'] if index is None or r.startswith(prefix)}
        selected = {r: task['roles'][global_role] for r, global_role in namespace.items()}
        changes = [(r, original['roles'][namespace[r]], v) for r, v in selected.items() if original['roles'][namespace[r]] != v]
        if not changes:
            continue
        template = templates[0 if index is None else index]
        scope = child.get('family_spec', {}).get('scope', {})
        texts = template.get('instruction_templates') or []
        if not texts:
            raise ValueError('NO_SOURCE_INSTRUCTION_TEMPLATE')
        instruction = _instantiate_instruction(texts[0], selected, scope_room_label=scope.get('label'))
        child['instruction_fragment' if children else 'instruction'] = instruction
        replacements = {}
        for role, old, new in changes:
            old_name = _humanize_object_type(old['object_type'])
            if sum((v['object_type'] == old['object_type'] for v in selected.values())) > 1:
                raise ValueError('AMBIGUOUS_SCOPED_TEXT_REPAIR')
            replacements[old_name] = _humanize_object_type(new['object_type'])

        def visit(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, str) and key in {'message', 'question', 'answer', 'text', 'instruction'}:
                        for old, new in replacements.items():
                            value = re.sub('\\b' + re.escape(old) + '\\b', new, value, flags=re.I)
                        node[key] = value
                    elif isinstance(value, (dict, list)):
                        visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)
        visit(child.get('family_spec', {}))
    if children:
        task['instruction'] = 'Complete these tasks in order: ' + ' '.join((f"({i}) {c['instruction_fragment']}" for i, c in enumerate(children, 1)))

def propose_repairs(original, profile, templates, taxonomy, *, limit=12, max_states=1500):
    initial = deepcopy(original)
    for key in ('qa_overlay', 'task_binding_spec', 'binding_spec', 'task_binding_spec_sha256'):
        initial.pop(key, None)
    queue = [initial]
    seen, results = (set(), [])
    examined = 0
    while queue and examined < max_states and (len(results) < limit):
        task = queue.pop(0)
        key = json.dumps(task['roles'], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        examined += 1
        problems = conflicts(task)
        if not problems:
            _update_public_text(task, original, templates)
            changes = [{'role': r, 'before': original['roles'][r], 'after': v} for r, v in task['roles'].items() if v != original['roles'][r]]
            changed = {c['role']: c['after'] for c in changes}

            def update_family(node):
                if isinstance(node, dict):
                    role = node.get('object_role')
                    if isinstance(role, str) and role in changed and ('object_family' in node):
                        node['object_family'] = _humanize_object_type(changed[role]['object_type'])
                    for value in node.values():
                        update_family(value)
                elif isinstance(node, list):
                    for value in node:
                        update_family(value)
            update_family(task.get('family_spec', {}))
            results.append({'task': task, 'changes': changes})
            continue
        for role in problems[0]['roles']:
            if task['roles'][role] != original['roles'][role]:
                continue
            obj = next((o for o in profile['objects'] if o['object_id'] == task['roles'][role]['object_id']))
            if not obj.get('pickupable') and any((o.get('object_role') == role for o in original.get('init', {}).get('object_pose_overrides', []))):
                continue
            for obj in _choices(task, role, profile, templates, taxonomy):
                candidate = deepcopy(task)
                candidate['roles'][role] = {'object_id': obj['object_id'], 'object_type': obj['object_type']}
                if len(conflicts(candidate)) >= len(problems):
                    continue
                queue.append(candidate)
    return (results, {'states_examined': examined, 'candidates': len(results), 'layout_preserved': True, 'fresh_oracle_required': True})
