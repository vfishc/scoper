import os
from itertools import combinations
POLICY_VERSION = 'task-identity-distinct-concrete.v5'

def physical_identity(role, name):
    object_id = str(role.get('object_id') or '')
    if role.get('requires_spawn') or object_id.startswith('spawnable:'):
        if role.get('bddl_inst'):
            return 'instance:' + str(role['bddl_inst'])
        if object_id and (not object_id.startswith('spawnable:')):
            return 'object:' + object_id
        return 'spawn-role:' + str(name)
    return 'object:' + (object_id or str(role.get('bddl_inst') or name))

def enabled():
    return os.environ.get('EB_TASK_BANK_DISTINCT_REFERENCES', '1') != '0'

def conflicts(task):
    from scoper.benchmark.common import compile_task_binding_spec
    from scoper.benchmark.common.binding import normalize_reference
    from scoper.benchmark.common.evaluation_binding_policy import is_nonconcrete_target, concrete_term_aliases
    from scoper.benchmark.common.binding_compiler import _public_terms
    spec = compile_task_binding_spec(task, compiler_version='task-binding-compiler.v9')
    roles = task.get('roles') or {}
    candidates = []
    for name, role in roles.items():
        if not isinstance(role, dict) or not any((role.get(k) for k in ('object_type', 'category', 'synset'))):
            continue
        identity = physical_identity(role, name)
        rule = spec.roles.get(name)
        if rule is None:
            continue
        terms = {normalize_reference(t) for t in concrete_term_aliases(rule.public_terms if rule else _public_terms(role, {})) if not is_nonconcrete_target(normalize_reference(t))}
        candidates.append((name, identity, terms, rule is not None))
    result = []
    for left, right in combinations(candidates, 2):
        overlap = left[2] & right[2]
        if left[1] != right[1] and overlap and (left[3] or right[3]):
            result.append({'roles': [left[0], right[0]], 'terms': sorted(overlap)})
    return result

def assert_distinct(task):
    problems = conflicts(task)
    if problems:
        raise ValueError('TASK_PUBLIC_IDENTITY_CONFLICT: ' + repr(problems))
    return task
