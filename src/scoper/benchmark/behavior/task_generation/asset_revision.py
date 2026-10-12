from copy import deepcopy
import hashlib
import re
import random
import json
from functools import lru_cache
from pathlib import Path
from scoper.benchmark.common.asset_diversity import choose_asset
from scoper.benchmark.common.task_identity_policy import conflicts, physical_identity
from scoper.benchmark.behavior.task_generation.scene_matcher import candidate_satisfies_requirement, serialize_role_binding

@lru_cache(maxsize=8192)
def model_extent(path):
    try:
        values = json.loads(Path(path).read_text())['bbox_size']
        if len(values) == 3 and all((isinstance(v, (int, float)) and v > 0 for v in values)):
            return tuple(values)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None

def fit_extent(native, available):
    if not native or not available or len(available) != 3:
        return None
    scale = min(1.0, *(a / n for a, n in zip(available, native)))
    if scale < 0.7:
        return None
    return [n * scale for n in native]

def apply_model_proportions(task, assets_root):
    task = deepcopy(task)
    for name, role in task.get('roles', {}).items():
        if not role.get('requires_spawn'):
            continue
        native = model_extent(str(Path(assets_root) / 'objects' / role['category'] / role['model'] / 'misc/metadata.json'))
        extent = fit_extent(native, role.get('extent'))
        if extent is None:
            raise ValueError('ASSET_DOES_NOT_FIT_WITHOUT_DISTORTION:' + name)
        dz = (extent[2] - role['extent'][2]) / 2
        role['extent'] = extent
        if isinstance(role.get('center'), list):
            role['center'][2] += dz
        placement = role.get('spawn_placement') or {}
        if isinstance(placement.get('offset'), list):
            placement['offset'][2] += dz
    return task

def template_ids(task, templates):
    children = ((task.get('family_spec') or {}).get('composite') or {}).get('subtasks') or []
    if not children:
        return [task['template']['template_id']]
    result = []
    for child in children:
        source = child.get('source_task_id', '')
        found = [t for t in templates if t in source]
        found = [t for t in found if not any((t != other and t in other for other in found))]
        if len(found) != 1:
            raise ValueError('CHILD_TEMPLATE_NOT_UNIQUE:' + source)
        result.append(found[0])
    return result

def role_requirements(task, templates):
    ids = template_ids(task, templates)
    namespaces = (task.get('generation') or {}).get('role_namespace')
    if not namespaces:
        namespaces = [{r: r for r in task['roles']}]
    result = {}
    for tid, namespace in zip(ids, namespaces):
        for local, global_role in namespace.items():
            requirement = templates[tid].get('requires', {}).get(local)
            if requirement is None and local in {'target_object_2', 'secondary_object'}:
                requirement = templates[tid].get('requires', {}).get('target_object')
            if requirement is not None:
                result.setdefault(global_role, []).append(requirement)
    return result

def refresh_instructions(task, templates):
    from scoper.benchmark.behavior.scene.concrete_categories import apply_concrete_public_names
    apply_concrete_public_names(task)
    from scoper.benchmark.behavior.task_generation.profile_task_generator import render_instruction, effective_instruction_for_template
    ids = template_ids(task, templates)
    children = ((task.get('family_spec') or {}).get('composite') or {}).get('subtasks') or []
    namespaces = (task.get('generation') or {}).get('role_namespace') or [{r: r for r in task['roles']}]
    for i, tid in enumerate(ids):
        child = children[i] if children else task
        selected = {local: task['roles'][name] for local, name in namespaces[i].items()}
        scope = (child.get('family_spec') or {}).get('scope') or {}
        text = render_instruction(templates[tid], {'roles': selected}, random.Random(0), scope_room_label=scope.get('label'))
        text = effective_instruction_for_template(templates[tid], {'roles': selected}, text, roles=selected)
        child['instruction_fragment' if children else 'instruction'] = text
    if children:
        task['instruction'] = 'Complete these tasks in order: ' + ' '.join((f'({i + 1}) ' + c['instruction_fragment'] for i, c in enumerate(children)))

def required_symbolic_capabilities(task, name, rules):
    required = set()
    if any(('dirtyable' in str(rule.get('taxonomy_group', '')) for rule in rules)):
        required.add('soakable')

    def visit(node):
        if isinstance(node, dict):
            refs = [v for v in node.values() if isinstance(v, str)]
            args = node.get('args') or {}
            if name in refs or (isinstance(args, dict) and name in args.values()):
                kind = ' '.join((str(node.get(k, '')) for k in ('type', 'predicate', 'state')))
                if any((word in kind for word in ('dirty', 'clean', 'soak'))):
                    required.add('soakable')
                if any((word in kind for word in ('slice', 'cut'))):
                    required.add('cuttable')
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    for key in ('goal', 'init', 'family_spec', 'disturbance_spec'):
        visit(task.get(key, {}))
    return required

def propose_asset_revision(original, profile, templates, *, assets_root, seed, usage, revise_roles=None, excluded_assets=(), failed_receptacles=None):
    kwargs = dict(assets_root=assets_root, seed=seed, usage=usage, revise_roles=revise_roles, excluded_assets=excluded_assets, failed_receptacles=failed_receptacles)
    try:
        return _propose_asset_revision(original, profile, templates, **kwargs)
    except ValueError as exc:
        if not str(exc).startswith('NO_CLOSE_COMPATIBLE_ASSET:'):
            raise
        failure = exc
    requirements = role_requirements(original, templates)
    selected = set(revise_roles if revise_roles is not None else [n for n, r in original['roles'].items() if r.get('requires_spawn') and r.get('object_type') == 'Bowl'])
    options = [n for n, r in sorted(original['roles'].items()) if n not in selected and n in requirements and r.get('requires_spawn') and ('distractor' not in n) and (not r.get('auxiliary_spawn_support')) and ('target' in n or 'object' in n)]
    for extra in options[:6]:
        try:
            return _propose_asset_revision(original, profile, templates, **dict(kwargs, revise_roles=selected | {extra}))
        except ValueError as exc:
            if not str(exc).startswith(('NO_CLOSE_COMPATIBLE_ASSET:', 'DECLARED_IDENTITY_CONFLICT_REMAINS')):
                raise
    raise failure

def _propose_asset_revision(original, profile, templates, *, assets_root, seed, usage, revise_roles=None, excluded_assets=(), failed_receptacles=None):
    task = deepcopy(original)
    for key in ('qa_overlay', 'task_binding_spec', 'task_binding_spec_sha256', 'renderer_asset_provenance'):
        task.pop(key, None)
    requirements = role_requirements(task, templates)
    for name in failed_receptacles or ():
        if name not in requirements:
            role = task['roles'][name]
            requirements[name] = [{'object_type_allowlist': [role['object_type']], 'must_be_receptacle': True}]
    for name in revise_roles or ():
        role = task['roles'][name]
        if name not in requirements and role.get('requires_spawn'):
            requirements[name] = [{'object_type_allowlist': [role['object_type']], 'must_be_pickupable': True}]
    changes = []
    seen = {}
    problems = conflicts(task)
    conflicting = {name for problem in problems for name in problem['roles']}
    if revise_roles is None:
        revise_roles = {name for name, role in original['roles'].items() if role.get('requires_spawn') and role.get('object_type') == 'Bowl' and (name in requirements)}
    revise_roles = set(revise_roles)
    for problem in problems:
        pair = problem['roles']
        movable = [n for n in pair if original['roles'][n].get('requires_spawn')]
        revise_roles.add((movable or pair)[-1])
    failed_receptacles = failed_receptacles or {}
    for name, old in original['roles'].items():
        if old.get('requires_spawn') or old.get('auxiliary_spawn_support') or name not in requirements:
            continue
        runtime_repair = name in failed_receptacles
        if not runtime_repair and (name not in revise_roles or name not in conflicting):
            continue
        if 'receptacle' not in name and 'storage' not in name:
            continue
        if runtime_repair:
            if any((n != name and physical_identity(r, n) == physical_identity(old, name) for n, r in task['roles'].items())):
                raise ValueError('RECEPTACLE_SHARED_IDENTITY:' + name)
            initial = json.dumps({'init': task.get('init'), 'mutations': task.get('scene_mutations'), 'placements': [{k: v for k, v in (r.get('spawn_placement') or {}).items() if k != 'anchor_role'} for r in task['roles'].values()]})
            if any((json.dumps(v) in initial for v in (name, old.get('object_id'), old.get('bddl_inst')) if v)):
                raise ValueError('RECEPTACLE_INITIAL_DEPENDENCY:' + name)
        excluded_ids = set(failed_receptacles.get(name, ())) | {old.get('object_id'), old.get('bddl_inst')}
        old_caps = set(old.get('abilities', [])) & {'openable', 'toggleable'}
        choices = []
        for obj in profile.get('objects', []):
            if obj.get('object_id') in excluded_ids or obj.get('requires_spawn'):
                continue
            if not all((candidate_satisfies_requirement(obj, r, profile) for r in requirements[name])):
                continue
            if old.get('room_id') and obj.get('room_id') != old['room_id']:
                continue
            if not old_caps <= set(obj.get('abilities', [])):
                continue
            if obj.get('object_type') in {r.get('object_type') for n, r in task['roles'].items() if n != name and n in requirements}:
                continue
            if obj.get('object_type') in {'SinkBasin', 'Faucet'}:
                continue
            choices.append(obj)
        if choices:
            origin = old.get('center') or [0, 0, 0]

            def distance(obj):
                pos = obj.get('position') or {}
                return (pos.get('x', origin[0]) - origin[0]) ** 2 + (pos.get('z', origin[1]) - origin[1]) ** 2
            choices.sort(key=lambda x: (distance(x), x['object_id']))
            obj = choices[0]
            new = serialize_role_binding(name, obj)
            pos = obj.get('position') or {}
            if all((k in pos for k in ('x', 'y', 'z'))):
                new['center'] = [pos['x'], pos['z'], pos['y']]
            new['bddl_inst'] = obj['object_id']
            new['abilities'] = deepcopy(obj.get('abilities', []))
            new['init_states'] = deepcopy(old.get('init_states', {}))
            task['roles'][name] = new
            changes.append({'role': name, 'before': old, 'after': new})
        elif runtime_repair:
            raise ValueError('NO_COMPATIBLE_RECEPTACLE:' + name)
    for name, old in sorted(original['roles'].items()):
        if not old.get('requires_spawn') or name not in requirements:
            continue
        if name not in revise_roles:
            continue
        identity = physical_identity(old, name)
        if identity in seen:
            task['roles'][name] = deepcopy(seen[identity])
            continue
        rules = requirements[name]
        for alias, r in original['roles'].items():
            if alias != name and physical_identity(r, alias) == identity:
                rules = rules + requirements.get(alias, [])
        forbidden = {r.get('object_type') for n, r in task['roles'].items() if n != name and physical_identity(r, n) != identity and (n in requirements)}
        eligible = []
        prototypes = {}
        extents = {}
        geometry = {}
        from scoper.benchmark.behavior.task_generation.profile_task_generator import role_abilities
        needed_caps = required_symbolic_capabilities(task, name, rules)
        for proto in profile.get('spawnable_catalog', []):
            if proto.get('object_type') in forbidden:
                continue
            if not all((candidate_satisfies_requirement(proto, r, profile) for r in rules)):
                continue
            if not needed_caps <= set(role_abilities(proto)):
                continue
            category = proto['category']
            prototypes[category] = proto
            for p in (Path(assets_root) / 'objects' / category).glob('*/usd/*.encrypted.usd'):
                model = p.parent.parent.name
                if p.name == model + '.encrypted.usd':
                    if (name, category, model) in excluded_assets:
                        continue
                    native = model_extent(str(p.parent.parent / 'misc/metadata.json'))
                    available = old.get('extent')
                    scale = min(1.5, *(a / n for a, n in zip(available, native))) if native and available else 0
                    extent = [n * scale for n in native] if native and scale >= 0.7 else None
                    ratios = [a / b for a, b in zip(extent, old['extent'])] if extent else []
                    if extent is not None and min(ratios) >= 0.7:
                        eligible.append(dict(object_type=proto['object_type'], category=category, model=model))
                        extents[category, model] = extent
                        geometry[category, model] = max((abs(1 - r) for r in ratios))
        if not eligible:
            raise ValueError('NO_CLOSE_COMPATIBLE_ASSET:' + name)
        alternatives = [v for v in eligible if v['object_type'] != old.get('object_type')]
        if alternatives:
            eligible = alternatives
        best = min((geometry[v['category'], v['model']] for v in eligible))
        eligible = [v for v in eligible if geometry[v['category'], v['model']] <= best + 0.1]
        selected = choose_asset(eligible, seed=(seed, name), usage=usage, forbidden_types=forbidden)
        proto = prototypes[selected['category']]
        new = deepcopy(old)
        new.update(selected)
        new['extent'] = extents[selected['category'], selected['model']]
        dz = (new['extent'][2] - old['extent'][2]) / 2
        if isinstance(new.get('center'), list):
            new['center'][2] += dz
        placement = new.get('spawn_placement') or {}
        if isinstance(placement.get('offset'), list):
            placement['offset'][2] += dz
        for key in ('synset', 'abilities', 'catalog_object_id'):
            if key in proto:
                new[key] = deepcopy(proto[key])
        new['abilities'] = role_abilities(proto)
        new['catalog_object_id'] = proto['object_id']
        suffix = int(hashlib.sha256(str(identity).encode()).hexdigest()[:7], 16)
        new['bddl_inst'] = new['synset'] + '_' + str(suffix)
        task['roles'][name] = new
        seen[identity] = new
        changes.append({'role': name, 'before': old, 'after': new})
    for name, old in original['roles'].items():
        identity = physical_identity(old, name)
        if identity in seen:
            replacement = seen[identity]
            new = deepcopy(old)
            for key in ('object_type', 'category', 'model', 'extent', 'synset', 'abilities', 'catalog_object_id', 'bddl_inst'):
                if key in replacement:
                    new[key] = deepcopy(replacement[key])
            dz = (new['extent'][2] - old['extent'][2]) / 2
            if isinstance(new.get('center'), list):
                new['center'][2] += dz
            placement = new.get('spawn_placement') or {}
            if isinstance(placement.get('offset'), list):
                placement['offset'][2] += dz
            task['roles'][name] = new
    changes = [{'role': name, 'before': old, 'after': deepcopy(task['roles'][name])} for name, old in original['roles'].items() if old != task['roles'][name]]

    def failures(node):
        if isinstance(node, dict):
            if node.get('mode') == 'nav_wrong_target':
                target = node.get('target_role', 'target_object')
                prefix = target.rsplit('__', 1)[0] + '__' if '__' in target else ''
                role = prefix + 'same_type_distractor'
                if role in task['roles']:
                    node['targets'] = [role]
            for v in list(node.values()):
                failures(v)
        elif isinstance(node, list):
            for v in node:
                failures(v)
    failures(task.get('disturbance_spec', {}))
    failures(task.get('family_spec', {}))
    if conflicts(task):
        raise ValueError('DECLARED_IDENTITY_CONFLICT_REMAINS')
    overrides = task.get('binding_policy', {}).get('roles', {})
    for name, old in original['roles'].items():
        if old.get('category') != task['roles'][name].get('category'):
            overrides.get(name, {}).pop('public_terms', None)
    refresh_instructions(task, templates)
    labels = {}
    for change in changes:
        old_label = change['before'].get('category', '').replace('_', ' ')
        new_label = change['after'].get('category', '').replace('_', ' ')
        if old_label and old_label != new_label:
            labels.setdefault(old_label, set()).add(new_label)

    def refresh_text(node):
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if isinstance(v, str) and k in {'question', 'message', 'text', 'answer', 'evaluator_answer', 'expected_answer'}:
                    for old_label, values in labels.items():
                        pattern = '\\b' + re.escape(old_label) + '\\b'
                        if re.search(pattern, v, re.I):
                            if len(values) != 1:
                                raise ValueError('AMBIGUOUS_SCOPED_TEXT_REPAIR')
                            v = re.sub(pattern, next(iter(values)), v, flags=re.I)
                    node[k] = v
                elif isinstance(v, (dict, list)):
                    refresh_text(v)
        elif isinstance(node, list):
            for v in node:
                refresh_text(v)
    refresh_text(task.get('family_spec', {}))
    return (task, changes)
