from __future__ import annotations
import random
from typing import Any, Mapping
from scoper.benchmark.behavior.bddl_adapter import BDDLAdapter
from scoper.benchmark.behavior.task_generation.role_binder import enumerate_role_bindings
from scoper.benchmark.behavior.task_spec import validate_behavior_task_spec
from scoper.benchmark.common import attach_compiled_task_binding_spec
_FLOOR_ROLE = {'synset': 'floor.n.01', 'bddl_inst': 'floors_1', 'category': 'floors', 'center': [5.0, 3.0, -0.02], 'extent': [10.0, 6.0, 0.04]}
_FIXED_CENTERS = [[6.5, 4.5, 0.45], [2.5, 4.5, 0.45], [6.5, 1.5, 0.45], [2.5, 1.5, 0.45]]

def generate_task_bank(templates: list[dict], adapter: BDDLAdapter, scene_model: str='Rs_int', max_per_template: int=10, rng_seed: int=0) -> list[dict]:
    tasks = []
    for template in templates:
        if template.get('requires_bddl_goal'):
            continue
        bindings = enumerate_role_bindings(template, adapter, max_per_template=max_per_template, rng_seed=rng_seed)
        rng = random.Random(rng_seed)
        for i, binding in enumerate(bindings):
            spec = _build_task_spec(template, binding, scene_model, i, rng)
            if spec is None:
                continue
            from scoper.benchmark.common.storage_instruction import repair_storage_instructions
            spec = attach_compiled_task_binding_spec(repair_storage_instructions(spec))
            try:
                validate_behavior_task_spec(spec)
                tasks.append(spec)
            except Exception:
                pass
    return tasks

def _build_task_spec(template: dict, binding: dict, scene_model: str, index: int, rng: random.Random) -> dict | None:
    roles: dict[str, Any] = {'floor': dict(_FLOOR_ROLE)}
    center_pool = [list(c) for c in _FIXED_CENTERS]
    rng.shuffle(center_pool)
    for i, (role_name, role_data) in enumerate(binding.items()):
        entry: dict[str, Any] = {'synset': role_data['synset'], 'bddl_inst': role_data['bddl_inst'], 'category': role_data['category'], 'abilities': list(role_data['abilities']), 'extent': list(role_data['extent'])}
        if 'pickupable' not in role_data['abilities'] and center_pool:
            entry['center'] = center_pool.pop(0)
            entry['center'][2] = _surface_z(role_data['extent'])
        roles[role_name] = entry
    init = []
    for constraint in template.get('init_constraints') or []:
        args = list(constraint.get('args', []))
        if all((a in binding or a == 'floor' for a in args)):
            entry = {'predicate': constraint['predicate'], 'args': args}
            if constraint.get('negate'):
                neg_map = {'open': 'closed', 'toggled_on': 'toggled_off'}
                entry['predicate'] = neg_map.get(constraint['predicate'], constraint['predicate'])
            init.append(entry)
    goal_predicates = []
    for gp in template['goal_predicates']:
        args = dict(gp.get('args', {}))
        if not all((v in binding for v in args.values())):
            return None
        goal_predicates.append({'type': gp['type'], 'args': args})
    tmpl = rng.choice(template['instruction_templates'])
    try:
        instruction = tmpl.format(**{k: v['category'] for k, v in binding.items()})
    except KeyError:
        instruction = tmpl
    task_id = 'behavior_{}_{:06d}'.format(template['template_id'], index)
    return {'id': task_id, 'version': '1.0', 'backend': 'behavior', 'category': template['category'], 'source': {'type': 'template', 'scene_model': scene_model, 'template_id': template['template_id']}, 'instruction': instruction, 'roles': roles, 'init': init, 'goal': {'type': 'predicates', 'require_model_stop': False, 'bddl_goal': None, 'predicates': goal_predicates}}

def _surface_z(extent: list[float]) -> float:
    return round(extent[2] / 2.0, 3)
__all__ = ['generate_task_bank']
