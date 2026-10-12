from __future__ import annotations
from collections import defaultdict
from typing import Any, Mapping, Sequence
from scoper.benchmark.common.evaluation_binding_policy import EVIDENCE_BINDING_COMPILER_VERSION, EVIDENCE_BINDING_COMPILER_VERSIONS, evidence_source_policy
from scoper.benchmark.common.binding import TASK_BINDING_SIDECAR_SCHEMA, TASK_BINDING_SPEC_SCHEMA, BindingCollectionSpec, BindingRoleSpec, TaskBindingSpec, build_binding_sidecar, instruction_scene_reference_category, normalize_reference, reference_forms, task_spec_sha256, validate_task_binding_spec
DEFAULT_BINDING_COMPILER_VERSION = 'task-binding-compiler.v6'
_NON_ACTIONABLE_ROLE_PATTERNS = ('distractor', 'floor', 'relocate_target', 'same_type_distractor', 'source_anchor', 'spawn_support')
_OBSERVATION_CATEGORIES = frozenset({'mid_execution_discovery', 'observation_conditioned_room_inspection', 'observation_conditioned_task', 'tidy_by_policy'})
_PUBLIC_ALIASES = {'cabinet': ('cabinet', 'storage cabinet'), 'carving knife': ('cutting tool', 'knife'), 'countertop': ('counter', 'countertop'), 'drawer': ('drawer', 'storage drawer'), 'electric refrigerator': ('fridge', 'refrigerator'), 'faucet': ('faucet', 'tap'), 'fridge': ('fridge', 'refrigerator'), 'furniture sink': ('sink', 'sink basin'), 'shelving unit': ('shelf', 'shelving unit'), 'sink basin': ('sink', 'sink basin'), 'soap bottle': ('cleaning supply', 'soap bottle')}

def actionable_binding_roles(task: Mapping[str, Any], *, retain_goal_roles: bool=False) -> tuple[str, ...]:
    roles = task.get('roles') or {}
    if not isinstance(roles, Mapping):
        raise ValueError('task roles must be a mapping')
    goal_roles = set()
    if retain_goal_roles:

        def collect(value):
            if isinstance(value, str) and value in roles:
                goal_roles.add(value)
            elif isinstance(value, Mapping):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
        for predicate in (task.get('goal') or {}).get('predicates', []):
            if not str(predicate.get('type', '')).startswith('object_not_'):
                collect(predicate.get('args') or predicate)
    result = []
    for role_name in roles:
        lowered = str(role_name).lower()
        if role_name not in goal_roles and any((pattern in lowered for pattern in _NON_ACTIONABLE_ROLE_PATTERNS)):
            continue
        result.append(str(role_name))
    return tuple(sorted(result, key=str.encode))

def compile_task_binding_spec(task: Mapping[str, Any], *, compiler_version: str=DEFAULT_BINDING_COMPILER_VERSION) -> TaskBindingSpec:
    task_id = str(task.get('id') or task.get('task_id') or '')
    if not task_id:
        raise ValueError('task requires id')
    task_roles = task.get('roles') or {}
    if not isinstance(task_roles, Mapping):
        raise ValueError('task roles must be a mapping')
    from scoper.benchmark.common.evaluation_binding_policy import GOAL_ROLE_BINDING_COMPILER_VERSIONS, PUBLIC_CONTRACT_BINDING_COMPILER_VERSION
    role_names = actionable_binding_roles(task, retain_goal_roles=compiler_version in GOAL_ROLE_BINDING_COMPILER_VERSIONS)
    collection_by_role, collections = _compile_collections(task, role_names)
    compiled_roles: dict[str, BindingRoleSpec] = {}
    for role_name in role_names:
        role = task_roles.get(role_name)
        if not isinstance(role, Mapping):
            raise ValueError(f'task role {role_name} must be a mapping')
        if compiler_version == PUBLIC_CONTRACT_BINDING_COMPILER_VERSION and normalize_reference(role.get('category')) == 'desk' and (normalize_reference(role.get('object_type')) == 'dining table'):
            role = dict(role, object_type='Desk')
        override = _role_override(task, role_name)
        public_terms = _public_terms(role, override, task=task, role_name=role_name)
        if compiler_version in EVIDENCE_BINDING_COMPILER_VERSIONS:
            from scoper.benchmark.common.evaluation_binding_policy import concrete_term_aliases
            public_terms = concrete_term_aliases(public_terms)
        if role_name in collection_by_role:
            public_terms = tuple(sorted({*public_terms, *_public_collection_terms(task, role)}, key=str.encode))
        from scoper.benchmark.common.evaluation_binding_policy import CONCRETE_BINDING_COMPILER_VERSIONS, is_nonconcrete_target
        if compiler_version in CONCRETE_BINDING_COMPILER_VERSIONS:
            public_terms = concrete_term_aliases(tuple((term for term in _public_terms(role, override) if not is_nonconcrete_target(term))))
            if not public_terms:
                raise ValueError(f'role {role_name} has no concrete public target name')
        if compiler_version in GOAL_ROLE_BINDING_COMPILER_VERSIONS:
            from scoper.benchmark.common.public_object_names import public_object_type
            public_terms = tuple(sorted({public_object_type(term) for term in public_terms}))
        source_policy, activation = _source_policy(task, role_name, override, public_terms=public_terms, instruction_grounded=compiler_version == DEFAULT_BINDING_COMPILER_VERSION)
        if compiler_version in EVIDENCE_BINDING_COMPILER_VERSIONS:
            source_policy, activation = evidence_source_policy(task, role_name, lambda view, local_role: _source_policy(view, local_role, {}, public_terms=public_terms, instruction_grounded=True), public_terms=public_terms)
        generic_terms = tuple(sorted({str(value).strip() for value in override.get('generic_terms', ()) if str(value).strip()}, key=str.encode))
        generic_terms = tuple(sorted({*generic_terms, *_family_generic_terms(task, role_name, source_policy=source_policy, instruction_grounded=compiler_version == DEFAULT_BINDING_COMPILER_VERSION)}, key=str.encode))
        if generic_terms and source_policy != 'task_generic_witness':
            if compiler_version in EVIDENCE_BINDING_COMPILER_VERSIONS:
                generic_terms = ()
            else:
                raise ValueError(f'task role {role_name} has generic_terms without task_generic_witness')
        if compiler_version in CONCRETE_BINDING_COMPILER_VERSIONS:
            generic_terms = ()
        allow_unseen = bool(override.get('allow_unseen_navigation', source_policy == 'task_explicit'))
        if compiler_version in EVIDENCE_BINDING_COMPILER_VERSIONS:
            allow_unseen = source_policy == 'task_explicit'
            from scoper.benchmark.common.evaluation_binding_policy import require_observation_binding
            if not require_observation_binding(task) and source_policy in {'observation', 'task_generic_witness'}:
                activation = 'concrete_reference'
                allow_unseen = True
        compiled_roles[role_name] = BindingRoleSpec(source_policy=source_policy, public_terms=public_terms, generic_terms=generic_terms, activation=str(activation if compiler_version in EVIDENCE_BINDING_COMPILER_VERSIONS else override.get('activation') or activation), allow_unseen_navigation=allow_unseen, collection_id=collection_by_role.get(role_name))
    spec = TaskBindingSpec(schema_version=TASK_BINDING_SPEC_SCHEMA, task_id=task_id, task_spec_sha256=task_spec_sha256(task), compiler_version=compiler_version, roles=compiled_roles, collections=collections)
    validate_task_binding_spec(task, spec, required_roles=role_names)
    return spec

def attach_compiled_task_binding_spec(task: Mapping[str, Any], *, compiler_version: str=DEFAULT_BINDING_COMPILER_VERSION) -> dict[str, Any]:
    value = dict(task)
    value.pop('task_binding_spec', None)
    value['task_binding_spec'] = compile_task_binding_spec(value, compiler_version=compiler_version).to_dict()
    return value

def load_binding_sidecar(value: Mapping[str, Any], *, bank_sha256: str | None=None) -> dict[str, TaskBindingSpec]:
    if not isinstance(value, Mapping) or value.get('schema_version') != TASK_BINDING_SIDECAR_SCHEMA:
        raise ValueError('unsupported binding sidecar schema')
    if bank_sha256 is not None and value.get('bank_sha256') != bank_sha256:
        raise ValueError('binding sidecar bank_sha256 differs')
    specs = value.get('specs')
    if not isinstance(specs, list) or len(specs) != int(value.get('task_count') or -1):
        raise ValueError('binding sidecar specs do not match task_count')
    loaded = [TaskBindingSpec.from_dict(spec) for spec in specs]
    if [spec.task_id for spec in loaded] != sorted((spec.task_id for spec in loaded), key=str.encode):
        raise ValueError('binding sidecar specs must be sorted')
    if len({spec.task_id for spec in loaded}) != len(loaded):
        raise ValueError('binding sidecar contains duplicate task ids')
    return {spec.task_id: spec for spec in loaded}

def compile_binding_sidecar(tasks: Sequence[Mapping[str, Any]], *, bank_sha256: str, compiler_version: str=DEFAULT_BINDING_COMPILER_VERSION) -> dict[str, Any]:
    specs = [compile_task_binding_spec(task, compiler_version=compiler_version) for task in tasks]
    return build_binding_sidecar(bank_sha256=bank_sha256, task_count=len(tasks), compiler_version=compiler_version, specs=specs)

def _source_policy(task: Mapping[str, Any], role_name: str, override: Mapping[str, Any], *, public_terms: Sequence[str], instruction_grounded: bool) -> tuple[str, str]:
    if override.get('source_policy'):
        return (str(override['source_policy']), str(override.get('activation') or 'episode_start'))
    category = str(task.get('category') or '')
    lowered = role_name.lower()
    if category in _OBSERVATION_CATEGORIES:
        if instruction_grounded and _observation_conditioned_generic_terms(task, role_name):
            return ('task_generic_witness', 'visual_evidence')
        return ('observation', 'visual_evidence')
    if category in {'put_away_visual_judgment', 'put_away_by_category'} and lowered.startswith('target_'):
        if _family_semantic_category(task, role_name):
            return ('task_generic_witness', 'visual_evidence')
        return ('observation', 'visual_evidence')
    if category == 'recover_and_store_incident_object' and role_name in _incident_support_roles(task) and _instruction_mentions_any(task.get('instruction'), public_terms):
        return ('task_explicit', 'episode_start')
    if category == 'recover_and_store_incident_object' and lowered.startswith('incident_'):
        return ('observation', 'visual_evidence')
    if category == 'human_interactive_execution' and (not lowered.startswith('primary_')):
        return ('public_event', 'public_event')
    return ('task_explicit', 'episode_start')

def _incident_support_roles(task: Mapping[str, Any]) -> frozenset[str]:
    family = task.get('family_spec') or {}
    recover = family.get('recover') if isinstance(family, Mapping) else None
    incident = recover.get('incident') if isinstance(recover, Mapping) else None
    support = incident.get('support_roles') if isinstance(incident, Mapping) else None
    if not isinstance(support, Mapping):
        return frozenset()
    return frozenset((str(role_name) for role_name in support.values() if isinstance(role_name, str) and role_name))

def _instruction_mentions_any(instruction: Any, public_terms: Sequence[str]) -> bool:
    normalized_instruction = normalize_reference(instruction)
    if not normalized_instruction:
        return False
    padded_instruction = f' {normalized_instruction} '
    return any((f' {form} ' in padded_instruction for term in public_terms for form in sorted(reference_forms(term), key=str.encode) if form))

def _role_override(task: Mapping[str, Any], role_name: str) -> Mapping[str, Any]:
    policy = task.get('binding_policy') or {}
    if not isinstance(policy, Mapping):
        return {}
    roles = policy.get('roles') or {}
    if not isinstance(roles, Mapping):
        return {}
    value = roles.get(role_name) or {}
    return value if isinstance(value, Mapping) else {}

def _public_terms(role: Mapping[str, Any], override: Mapping[str, Any], *, task: Mapping[str, Any] | None=None, role_name: str | None=None) -> tuple[str, ...]:
    explicit = override.get('public_terms')
    if isinstance(explicit, Sequence) and (not isinstance(explicit, (str, bytes))):
        terms = {str(value).strip() for value in explicit if str(value).strip()}
    else:
        terms = set()
        for key in ('object_type', 'category', 'synset'):
            raw = str(role.get(key) or '').split('.n.', 1)[0]
            normalized = normalize_reference(raw)
            if normalized:
                terms.add(normalized)
                terms.update(_PUBLIC_ALIASES.get(normalized, ()))
    if task is not None and role_name is not None:
        terms.update(_family_explicit_terms(task, role_name))
    if not terms:
        raise ValueError('binding role has no public natural-language term')
    return tuple(sorted(terms, key=str.encode))

def _family_put_away(task: Mapping[str, Any]) -> Mapping[str, Any] | None:
    family = task.get('family_spec')
    if not isinstance(family, Mapping):
        return None
    value = family.get('put_away')
    return value if isinstance(value, Mapping) else None

def _family_semantic_category(task: Mapping[str, Any], role_name: str) -> str | None:
    put_away = _family_put_away(task)
    if put_away is None:
        return None
    targets = put_away.get('target_roles')
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        return None
    if role_name not in {str(value) for value in targets}:
        return None
    category = normalize_reference(put_away.get('semantic_category'))
    return category or None

def _observation_conditioned_generic_terms(task: Mapping[str, Any], role_name: str) -> tuple[str, ...]:
    if str(task.get('category') or '') != 'observation_conditioned_room_inspection':
        return ()
    family = task.get('family_spec')
    observation = family.get('observation_conditioned') if isinstance(family, Mapping) else None
    payloads = observation.get('latent_payloads') if isinstance(observation, Mapping) else None
    if not isinstance(payloads, Sequence) or isinstance(payloads, (str, bytes)):
        return ()
    issue_terms = {'dirty_dish': 'dishware', 'faucet_on': 'faucet', 'misplaced_object': 'dishware', 'open_container': 'storage container'}
    terms: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get('object_role') or '') == role_name:
            term = issue_terms.get(str(payload.get('issue_type') or ''))
            if term:
                terms.add(term)
        support_roles = payload.get('support_roles')
        if isinstance(support_roles, Mapping):
            for support_kind, support_role in support_roles.items():
                if str(support_role) == role_name and str(support_kind) == 'storage':
                    terms.add('storage')
    instruction = task.get('instruction')
    return tuple(sorted((term for term in terms if instruction_scene_reference_category(term, instruction) == term), key=str.encode))

def _family_generic_terms(task: Mapping[str, Any], role_name: str, *, source_policy: str, instruction_grounded: bool) -> tuple[str, ...]:
    if source_policy != 'task_generic_witness':
        return ()
    observation_terms = _observation_conditioned_generic_terms(task, role_name) if instruction_grounded else ()
    if observation_terms:
        return observation_terms
    category = _family_semantic_category(task, role_name)
    if not category:
        return ()
    terms = {category}
    instruction = normalize_reference(task.get('instruction'))
    if instruction:
        for phrase in (f'clean {category}',):
            if f' {phrase} ' in f' {instruction} ':
                terms.add(phrase)
    return tuple(sorted(terms, key=str.encode))

def _family_explicit_terms(task: Mapping[str, Any], role_name: str) -> set[str]:
    put_away = _family_put_away(task)
    if put_away is None:
        return set()
    storage_roles = put_away.get('storage_roles')
    if not isinstance(storage_roles, Sequence) or isinstance(storage_roles, (str, bytes)):
        return set()
    if role_name not in {str(value) for value in storage_roles}:
        return set()
    instruction = normalize_reference(task.get('instruction'))
    if not instruction:
        return set()
    padded = f' {instruction} '
    phrases = {'storage', 'storage container', 'storage receptacle', 'storage cabinet', 'storage drawer', 'storage shelf', 'shelf', 'drawer', 'cabinet', 'countertop'}
    matched = {phrase for phrase in phrases if f' {phrase} ' in padded}
    tokens = instruction.split()
    for index, token in enumerate(tokens):
        if token != 'storage':
            continue
        matched.add('storage')
        if index > 0 and tokens[index - 1] not in {'at', 'by', 'for', 'from', 'in', 'into', 'on', 'to', 'with'}:
            matched.add(f'{tokens[index - 1]} storage')
    return matched

def _public_collection_terms(task: Mapping[str, Any], role: Mapping[str, Any]) -> set[str]:
    instruction = normalize_reference(task.get('instruction'))
    if not instruction:
        return set()
    padded = f' {instruction} '
    terms: set[str] = set()
    for noun in ('item', 'object'):
        if f' {noun} ' in padded:
            terms.add(noun)
    category = normalize_reference(role.get('object_type') or role.get('category') or role.get('synset'))
    if category:
        category_head = category.split()[-1]
        raw_instruction = ' '.join(str(task.get('instruction') or '').lower().replace('_', ' ').split())
        for noun in ('item', 'object'):
            phrase = f'{category_head} {noun}'
            plural_phrase = f'{phrase}s'
            if f' {phrase} ' in padded or f' {phrase} ' in f' {raw_instruction} ' or f' {plural_phrase} ' in f' {raw_instruction} ':
                terms.add(phrase)
    return terms

def _compile_collections(task: Mapping[str, Any], role_names: Sequence[str]) -> tuple[dict[str, str], dict[str, BindingCollectionSpec]]:
    task_roles = task.get('roles') or {}
    actionable_roles = set(role_names)
    collection_by_role: dict[str, str] = {}
    collections: dict[str, BindingCollectionSpec] = {}
    put_away = _family_put_away(task)
    declared_targets = put_away.get('target_roles') if put_away is not None else None
    if isinstance(declared_targets, Sequence) and (not isinstance(declared_targets, (str, bytes))):
        role_order = tuple((str(role_name) for role_name in declared_targets if str(role_name) in actionable_roles))
        if len(role_order) >= 2 and len(set(role_order)) == len(role_order):
            collection_id = 'family_target_collection'
            collections[collection_id] = BindingCollectionSpec(role_order=role_order)
            collection_by_role.update({role_name: collection_id for role_name in role_order})
    grouped: dict[str, list[str]] = defaultdict(list)
    for role_name in role_names:
        if role_name in collection_by_role:
            continue
        if not _looks_like_collection_member(role_name):
            continue
        role = task_roles[role_name]
        object_type = normalize_reference(role.get('object_type') or role.get('category') or role.get('synset'))
        grouped[object_type].append(role_name)
    for index, (_, members) in enumerate(sorted(grouped.items(), key=lambda item: item[0].encode())):
        if len(members) < 2:
            continue
        role_order = tuple(sorted(members, key=_role_sequence_key))
        collection_id = f'collection_{index + 1:02d}'
        collections[collection_id] = BindingCollectionSpec(role_order=role_order)
        collection_by_role.update({role_name: collection_id for role_name in role_order})
    return (collection_by_role, collections)

def _looks_like_collection_member(role_name: str) -> bool:
    lowered = role_name.lower()
    return lowered.startswith('target_') or lowered in {'target_object', 'target_object_2'}

def _role_sequence_key(role_name: str) -> tuple[str, int, bytes]:
    head, _, suffix = role_name.rpartition('_')
    if suffix.isdigit():
        return (head, int(suffix), role_name.encode())
    return (role_name, -1, role_name.encode())
__all__ = ['DEFAULT_BINDING_COMPILER_VERSION', 'actionable_binding_roles', 'attach_compiled_task_binding_spec', 'compile_task_binding_spec', 'compile_binding_sidecar', 'load_binding_sidecar']
