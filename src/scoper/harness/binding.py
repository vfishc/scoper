from __future__ import annotations
import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from scoper.harness.canonical import sha256_json
from scoper.harness.vocabulary import SHARED_VOCABULARY_VERSION, SHARED_VOCABULARY_REGISTRY, classify_public_reference, resolve_public_family
TASK_BINDING_SPEC_SCHEMA = 'task-binding-spec.v1'
TASK_BINDING_SIDECAR_SCHEMA = 'task-binding-sidecar.v1'
TASK_BINDING_MANIFEST_SCHEMA = 'task-binding-manifest.v1'
BENEVOLENT_BINDING_POLICY_ID = 'task-benevolent-binding.v1'
SOURCE_POLICIES = ('observation', 'public_event', 'task_explicit', 'task_generic_witness')
BINDING_STATUSES = ('ambiguous', 'conflict', 'not_interaction_ready', 'rejected', 'success', 'unresolved')
NAVIGATION_TOOLS = frozenset({'navigate_to'})
LOOK_TOOLS = frozenset({'look_for', 'scan'})
INTERACTION_TOOLS = frozenset({'clean', 'close', 'cool', 'heat', 'open', 'operation', 'pick_up', 'put_down', 'slice', 'toggle_off', 'toggle_on', 'wash'})
_PRONOUNS = frozenset({'it', 'that', 'that one', 'the object', 'the target', 'them'})
_ARTICLES = frozenset({'a', 'an', 'the', 'this', 'that', 'one'})
_GENERIC_TERMS = frozenset({'cleaning supply', 'container', 'destination', 'dish', 'dishware', 'food', 'item', 'object', 'receptacle', 'storage', 'storage container', 'surface', 'target', 'thing'})
_CONTROLLED_SYNONYMS = {'basin': 'sink', 'cabinetry': 'cabinet', 'cold storage': 'fridge', 'counter top': 'countertop', 'cup': 'mug', 'dining table': 'table', 'drawer unit': 'drawer', 'electric refrigerator': 'fridge', 'faucet': 'tap', 'fridge': 'refrigerator', 'knife': 'cutting tool', 'refrigerator': 'fridge', 'sink basin': 'sink', 'soap bottle': 'soap', 'tap': 'faucet'}
_CONTROLLED_STATE_MODIFIERS = frozenset({'clean', 'closed', 'cold', 'dirty', 'empty', 'filled', 'hot', 'open', 'saturated', 'sliced'})
_CONTROLLED_COLLECTION_MODIFIERS = frozenset({'another', 'first', 'next', 'second', 'third'})
_RELATION_PREPOSITIONS = frozenset({'at', 'by', 'in', 'inside', 'near', 'on'})
_INSTRUCTION_SCENE_CATEGORY_RULES = (('storage container', ('storage container', 'storage left open'), ('cabinet', 'cupboard', 'drawer', 'fridge', 'refrigerator', 'storage', 'storage container')), ('dishware', ('dishware', 'dirty dish'), ('bowl', 'cup', 'dish', 'dishware', 'mug', 'plate')), ('faucet', ('faucet', 'running water', 'tap', 'water source'), ('faucet', 'tap', 'water source')), ('storage', ('storage',), ('cabinet', 'cupboard', 'drawer', 'fridge', 'refrigerator', 'shelf', 'shelving unit', 'storage')))

def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    normalized = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sorted(normalized, key=str.encode))

def normalize_reference(value: Any) -> str:
    text = re.sub('(?<!^)(?=[A-Z])', ' ', str(value or ''))
    tokens = [token for token in re.split('[^a-z0-9]+', text.lower().replace('_', ' ')) if token and token not in _ARTICLES]
    if tokens:
        final = tokens[-1]
        if final.endswith('ies') and len(final) > 3:
            tokens[-1] = final[:-3] + 'y'
        elif final.endswith('s') and (not final.endswith('ss')):
            tokens[-1] = final[:-1]
    return ' '.join(tokens)

def reference_forms(value: Any, taxonomy_aliases: Mapping[str, Sequence[str]]=()) -> frozenset[str]:
    normalized = normalize_reference(value)
    if not normalized:
        return frozenset()
    forms = {normalized}
    for left, right in _CONTROLLED_SYNONYMS.items():
        if normalized == normalize_reference(left):
            forms.add(normalize_reference(right))
        if normalized == normalize_reference(right):
            forms.add(normalize_reference(left))
    aliases = taxonomy_aliases if isinstance(taxonomy_aliases, Mapping) else {}
    for canonical, values in aliases.items():
        group = {normalize_reference(canonical), *(normalize_reference(value) for value in values)}
        if normalized in group:
            forms.update((item for item in group if item))
    family = resolve_public_family(normalized)
    if family is not None:
        forms.update((normalize_reference(value) for value in SHARED_VOCABULARY_REGISTRY.aliases(family)))
    return frozenset(forms)

def instruction_scene_reference_category(reference: Any, instruction: Any) -> str | None:
    normalized_reference = normalize_reference(reference)
    normalized_instruction = normalize_reference(instruction)
    if not normalized_reference or not normalized_instruction:
        return None
    reference_tokens = normalized_reference.split()
    while reference_tokens and reference_tokens[0] in _CONTROLLED_STATE_MODIFIERS:
        reference_tokens.pop(0)
    state_free_reference = ' '.join(reference_tokens)
    for category, cues, members in _INSTRUCTION_SCENE_CATEGORY_RULES:
        if not any((_contains_complete_phrase(normalized_instruction, normalize_reference(cue)) for cue in cues)):
            continue
        if any((state_free_reference == normalize_reference(member) or state_free_reference.endswith(' ' + normalize_reference(member)) for member in members)):
            return category
    return None

def instruction_scene_reference_relevant(reference: Any, instruction: Any) -> bool:
    normalized_reference = normalize_reference(reference)
    normalized_instruction = normalize_reference(instruction)
    return bool(normalized_reference and normalized_instruction and (instruction_scene_reference_category(reference, instruction) is not None or _contains_complete_phrase(normalized_instruction, normalized_reference)))

def normalize_instruction_grounded_memory_proposal(proposal: Mapping[str, Any], instruction: Any, *, vocabulary_version: str | None=None) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(proposal))
    if normalized.get('mode') != 'observe':
        return normalized
    mutations = normalized.get('fact_mutations')
    if not isinstance(mutations, list):
        return normalized
    if vocabulary_version == SHARED_VOCABULARY_VERSION:
        return normalized
    for mutation in mutations:
        if not isinstance(mutation, dict) or mutation.get('relation_or_property') != 'scene_observation':
            continue
        category = instruction_scene_reference_category(mutation.get('public_entity_ref'), instruction)
        if category is not None:
            mutation['public_entity_ref'] = category
    return normalized

@dataclass(frozen=True)
class BindingRoleSpec:
    source_policy: str
    public_terms: tuple[str, ...]
    generic_terms: tuple[str, ...] = ()
    activation: str = 'episode_start'
    allow_unseen_navigation: bool = False
    collection_id: str | None = None

    def __post_init__(self) -> None:
        if self.source_policy not in SOURCE_POLICIES:
            raise ValueError(f'unsupported binding source policy: {self.source_policy}')
        if not isinstance(self.activation, str) or not self.activation:
            raise ValueError('BindingRoleSpec.activation must be non-empty text')
        if not self.public_terms:
            raise ValueError('BindingRoleSpec.public_terms must not be empty')
        if self.public_terms != _sorted_unique(self.public_terms):
            raise ValueError('BindingRoleSpec.public_terms must be sorted and unique')
        if self.generic_terms != _sorted_unique(self.generic_terms):
            raise ValueError('BindingRoleSpec.generic_terms must be sorted and unique')
        if self.source_policy != 'task_generic_witness' and self.generic_terms:
            raise ValueError('generic_terms require task_generic_witness policy')
        if self.collection_id is not None and (not str(self.collection_id).strip()):
            raise ValueError('BindingRoleSpec.collection_id must be non-empty when present')

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['public_terms'] = list(self.public_terms)
        value['generic_terms'] = list(self.generic_terms)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'BindingRoleSpec':
        expected = {'source_policy', 'public_terms', 'generic_terms', 'activation', 'allow_unseen_navigation', 'collection_id'}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError('BindingRoleSpec fields are not exact')
        return cls(source_policy=str(value['source_policy']), public_terms=_string_tuple(value['public_terms'], 'public_terms'), generic_terms=_string_tuple(value['generic_terms'], 'generic_terms'), activation=str(value['activation']), allow_unseen_navigation=bool(value['allow_unseen_navigation']), collection_id=None if value['collection_id'] is None else str(value['collection_id']))

@dataclass(frozen=True)
class BindingCollectionSpec:
    role_order: tuple[str, ...]
    selection_rule: str = 'next_publicly_unprocessed'

    def __post_init__(self) -> None:
        if not self.role_order or len(set(self.role_order)) != len(self.role_order):
            raise ValueError('BindingCollectionSpec.role_order must be non-empty and unique')
        if self.selection_rule != 'next_publicly_unprocessed':
            raise ValueError('unsupported collection selection rule')

    def to_dict(self) -> dict[str, Any]:
        return {'role_order': list(self.role_order), 'selection_rule': self.selection_rule}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'BindingCollectionSpec':
        if not isinstance(value, Mapping) or set(value) != {'role_order', 'selection_rule'}:
            raise ValueError('BindingCollectionSpec fields are not exact')
        return cls(role_order=_string_tuple(value['role_order'], 'role_order'), selection_rule=str(value['selection_rule']))

@dataclass(frozen=True)
class TaskBindingSpec:
    schema_version: str
    task_id: str
    task_spec_sha256: str
    compiler_version: str
    roles: Mapping[str, BindingRoleSpec]
    collections: Mapping[str, BindingCollectionSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != TASK_BINDING_SPEC_SCHEMA:
            raise ValueError('unsupported TaskBindingSpec schema')
        if not self.task_id or not self.compiler_version:
            raise ValueError('TaskBindingSpec identity fields must be non-empty')
        _require_sha256(self.task_spec_sha256, 'task_spec_sha256')
        if not self.roles:
            raise ValueError('TaskBindingSpec.roles must not be empty')
        if list(self.roles) != sorted(self.roles, key=str.encode):
            raise ValueError('TaskBindingSpec.roles must be sorted')
        if list(self.collections) != sorted(self.collections, key=str.encode):
            raise ValueError('TaskBindingSpec.collections must be sorted')
        for role_name, role in self.roles.items():
            if role.collection_id is not None:
                collection = self.collections.get(role.collection_id)
                if collection is None or role_name not in collection.role_order:
                    raise ValueError(f'role {role_name} has invalid collection_id')
        for collection_id, collection in self.collections.items():
            for role_name in collection.role_order:
                role = self.roles.get(role_name)
                if role is None or role.collection_id != collection_id:
                    raise ValueError(f'collection {collection_id} references an invalid role')

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': self.schema_version, 'task_id': self.task_id, 'task_spec_sha256': self.task_spec_sha256, 'compiler_version': self.compiler_version, 'roles': {name: role.to_dict() for name, role in self.roles.items()}, 'collections': {name: collection.to_dict() for name, collection in self.collections.items()}}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'TaskBindingSpec':
        expected = {'schema_version', 'task_id', 'task_spec_sha256', 'compiler_version', 'roles', 'collections'}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError('TaskBindingSpec fields are not exact')
        raw_roles = value['roles']
        raw_collections = value['collections']
        if not isinstance(raw_roles, Mapping) or not isinstance(raw_collections, Mapping):
            raise ValueError('TaskBindingSpec roles and collections must be mappings')
        return cls(schema_version=str(value['schema_version']), task_id=str(value['task_id']), task_spec_sha256=str(value['task_spec_sha256']), compiler_version=str(value['compiler_version']), roles={str(name): BindingRoleSpec.from_dict(role) for name, role in sorted(raw_roles.items(), key=lambda item: str(item[0]).encode())}, collections={str(name): BindingCollectionSpec.from_dict(collection) for name, collection in sorted(raw_collections.items(), key=lambda item: str(item[0]).encode())})

def task_spec_sha256(task: Mapping[str, Any]) -> str:
    value = dict(task)
    value.pop('binding_spec', None)
    value.pop('task_binding_spec', None)
    return sha256_json(value)

def validate_task_binding_spec(task: Mapping[str, Any], spec: TaskBindingSpec, *, required_roles: Iterable[str] | None=None) -> None:
    task_id = str(task.get('id') or task.get('task_id') or '')
    if spec.task_id != task_id:
        raise ValueError('TaskBindingSpec task_id differs from task')
    if spec.task_spec_sha256 != task_spec_sha256(task):
        raise ValueError('TaskBindingSpec task_spec_sha256 differs from task')
    task_roles = task.get('roles') or {}
    if not isinstance(task_roles, Mapping):
        raise ValueError('task roles must be a mapping')
    missing = set(required_roles or ()) - set(spec.roles)
    if missing:
        raise ValueError(f'TaskBindingSpec misses required roles: {sorted(missing)}')
    raw_identifiers: set[str] = set()
    for role in task_roles.values():
        if not isinstance(role, Mapping):
            continue
        for key in ('object_id', 'bddl_inst', 'raw_object_id'):
            if role.get(key):
                raw_identifiers.add(normalize_reference(role[key]))
    for role_name, role_spec in spec.roles.items():
        if role_name not in task_roles:
            raise ValueError(f'TaskBindingSpec contains unknown role: {role_name}')
        for term in (*role_spec.public_terms, *role_spec.generic_terms):
            normalized = normalize_reference(term)
            role_value = task_roles[role_name]
            natural_role_terms = {normalize_reference(role_value.get('object_type')), normalize_reference(role_value.get('category')), normalize_reference(str(role_value.get('synset') or '').split('.n.', 1)[0])}
            role_handle_collision = normalized == normalize_reference(role_name) and normalized not in natural_role_terms
            if normalized in raw_identifiers or role_handle_collision:
                raise ValueError(f'TaskBindingSpec exposes role or raw object identifier: {term}')

@dataclass(frozen=True)
class BindingCandidate:
    private_id: str
    canonical_public_ref: str
    object_type: str
    aliases: tuple[str, ...] = ()
    source: str = 'observation'
    role: str | None = None
    visible: bool = False
    interaction_ready: bool = False
    activated: bool = True
    scene_observed: bool = False
    last_seen_step: int | None = None
    room_id: str | None = None

    def __post_init__(self) -> None:
        if not self.private_id or not self.canonical_public_ref:
            raise ValueError('BindingCandidate identity fields must be non-empty')
        if self.aliases != _sorted_unique(self.aliases):
            raise ValueError('BindingCandidate.aliases must be sorted and unique')

    @property
    def public_forms(self) -> frozenset[str]:
        forms: set[str] = set()
        for value in (self.canonical_public_ref, self.object_type, *self.aliases):
            forms.update(reference_forms(value))
        return frozenset(forms)

@dataclass(frozen=True)
class BindingFocus:
    private_id: str
    canonical_public_ref: str
    role: str | None = None
    interaction_ready: bool = False
    high_level_step: int | None = None

@dataclass(frozen=True)
class BindingRequest:
    reference: str
    tool_name: str
    spec: TaskBindingSpec
    candidates: tuple[BindingCandidate, ...]
    task_instruction: str = ''
    public_phase_role_order: tuple[str, ...] = ()
    recent_focus: BindingFocus | None = None
    local_interaction_context_active: bool = False
    bypass_interaction_readiness: bool = False
    publicly_processed_roles: tuple[str, ...] = ()
    benevolent_role_order: tuple[str, ...] = ()
    role_handles: tuple[str, ...] = ()
    raw_object_ids: tuple[str, ...] = ()
    taxonomy_aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    vocabulary_version: str | None = None
    public_destination_roles: tuple[str, ...] = ()

@dataclass(frozen=True)
class PublicBindingState:
    safe_human_messages: tuple[Mapping[str, Any], ...] = ()
    effective_facts: tuple[Mapping[str, Any], ...] = ()
    scene_observations: tuple[Mapping[str, Any], ...] = ()
    active_public_event_roles: tuple[str, ...] = ()
    scene_observation_roles: tuple[str, ...] = ()
    publicly_processed_collection_roles: tuple[str, ...] = ()

def derive_public_binding_state(spec: TaskBindingSpec, *, safe_human_messages: Sequence[Mapping[str, Any]]=(), effective_facts: Sequence[Mapping[str, Any]]=(), scene_observations: Sequence[Mapping[str, Any]]=(), task_instruction: str='') -> PublicBindingState:
    messages = tuple((copy.deepcopy(dict(item)) for item in safe_human_messages))
    facts = tuple((copy.deepcopy(dict(item)) for item in effective_facts))
    observations = tuple((copy.deepcopy(dict(item)) for item in scene_observations))
    insert_texts = tuple((normalize_reference(item.get('instruction')) for item in messages if item.get('type') == 'insert' and isinstance(item.get('instruction'), str)))
    active_event_roles = []
    for role_name, role_spec in spec.roles.items():
        if role_spec.source_policy != 'public_event':
            continue
        terms = (*role_spec.public_terms, *role_spec.generic_terms)
        if any((_contains_complete_phrase(instruction, normalize_reference(term)) for instruction in insert_texts for term in terms)):
            active_event_roles.append(role_name)
    observed_forms = {form for item in observations if item.get('relation_or_property') == 'scene_observation' and item.get('status', 'current') == 'current' for form in reference_forms(item.get('subject_public_role'))}
    observed_categories = {normalize_reference(item.get('subject_public_role')) for item in observations if item.get('relation_or_property') == 'scene_observation' and item.get('status', 'current') == 'current'}
    scene_observation_roles = []
    for role_name, role_spec in spec.roles.items():
        if role_spec.source_policy not in {'observation', 'task_generic_witness'}:
            continue
        role_forms = {form for term in (*role_spec.public_terms, *role_spec.generic_terms) for form in reference_forms(term)}
        role_categories = {category for term in (*role_spec.public_terms, *role_spec.generic_terms) if (category := instruction_scene_reference_category(term, task_instruction))}
        if observed_forms & role_forms or observed_categories & role_categories:
            scene_observation_roles.append(role_name)
    return PublicBindingState(safe_human_messages=messages, effective_facts=facts, scene_observations=observations, active_public_event_roles=_sorted_unique(active_event_roles), scene_observation_roles=_sorted_unique(scene_observation_roles), publicly_processed_collection_roles=())

def scene_observation_activates_binding_role(state: PublicBindingState, role_name: str | None, tool_name: str) -> bool:
    return bool(role_name and tool_name in NAVIGATION_TOOLS | {'look_for'} and (role_name in state.scene_observation_roles))

def equivalent_unique_task_witness_calls(spec: TaskBindingSpec, left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if left.get('tool_name') != right.get('tool_name'):
        return False
    left_arguments = left.get('arguments')
    right_arguments = right.get('arguments')
    if not isinstance(left_arguments, Mapping) or not isinstance(right_arguments, Mapping):
        return False
    left_target = left_arguments.get('target')
    right_target = right_arguments.get('target')
    if not isinstance(left_target, str) or not isinstance(right_target, str):
        return False
    if {key: value for key, value in left_arguments.items() if key != 'target'} != {key: value for key, value in right_arguments.items() if key != 'target'}:
        return False
    left_role = _unique_task_witness_role(spec, left_target)
    right_role = _unique_task_witness_role(spec, right_target)
    return left_role is not None and left_role == right_role

def _unique_task_witness_role(spec: TaskBindingSpec, reference: str) -> str | None:
    matches = _task_witness_roles(spec, reference)
    return matches[0] if len(matches) == 1 else None

def public_task_witness_role_count(spec: TaskBindingSpec, reference: str) -> int:
    return len(_task_witness_roles(spec, reference))

def _task_witness_roles(spec: TaskBindingSpec, reference: str) -> list[str]:
    normalized = normalize_reference(reference)
    state_base = _controlled_state_base(normalized)
    query_head = _reference_head_noun(normalized)
    matches = []
    for role_name, role_spec in spec.roles.items():
        role_forms: set[str] = set()
        for term in (*role_spec.public_terms, *role_spec.generic_terms):
            role_forms.update(reference_forms(term))
        role_heads = {head for form in role_forms if (head := _reference_head_noun(form)) is not None}
        if normalized in role_forms or (state_base is not None and state_base in role_forms) or (query_head is not None and query_head in role_heads):
            matches.append(role_name)
    return matches

def _contains_complete_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return bool(re.search('(?:^| ){}(?: |$)'.format(re.escape(phrase)), text))

@dataclass(frozen=True)
class BindingResult:
    status: str
    reason: str
    source: str | None = None
    match_kind: str | None = None
    canonical_public_ref: str | None = None
    private_id: str | None = None
    role: str | None = None
    navigation_class: str | None = None

    def __post_init__(self) -> None:
        if self.status not in BINDING_STATUSES:
            raise ValueError(f'unsupported BindingResult status: {self.status}')
        if self.status == 'success' and (self.private_id is None or self.canonical_public_ref is None):
            raise ValueError('successful BindingResult requires a target')

    @property
    def success(self) -> bool:
        return self.status == 'success'

    def public_record(self) -> dict[str, Any]:
        return {'status': self.status, 'reason': self.reason, 'source': self.source, 'match_kind': self.match_kind, 'canonical_public_ref': self.canonical_public_ref, 'navigation_class': self.navigation_class}

def resolve_binding(request: BindingRequest) -> BindingResult:
    from dataclasses import replace
    from scoper.harness.evaluation_binding_policy import uses_evidence_binding, is_generic_target
    evidence_binding = uses_evidence_binding(request.spec)
    reference = str(request.reference or '').strip()
    normalized = normalize_reference(reference)
    from scoper.harness.evaluation_binding_policy import REFINED_BINDING_COMPILER_VERSION, CONCRETE_BINDING_COMPILER_VERSIONS, normalize_concrete_reference, is_nonconcrete_target
    if request.spec.compiler_version in {REFINED_BINDING_COMPILER_VERSION, *CONCRETE_BINDING_COMPILER_VERSIONS}:
        normalized = normalize_concrete_reference(normalized)
    if not normalized:
        return BindingResult('unresolved', 'empty_reference')
    if request.spec.compiler_version in CONCRETE_BINDING_COMPILER_VERSIONS and is_nonconcrete_target(normalized):
        return BindingResult('rejected', 'generic_target_requires_concrete_observation')
    if evidence_binding and is_generic_target(normalized):
        return BindingResult('rejected', 'generic_target_requires_concrete_observation')
    if evidence_binding:
        request = replace(request, candidates=tuple((replace(candidate, activated=True) if candidate.role in request.spec.roles and request.spec.roles[candidate.role].activation == 'concrete_reference' else candidate for candidate in request.candidates)))
        request = replace(request, candidates=tuple((candidate for candidate in request.candidates if candidate.activated or (candidate.role in request.spec.roles and request.spec.roles[candidate.role].source_policy == 'task_explicit'))))
    forbidden = {normalize_reference(value) for value in (*request.role_handles, *request.raw_object_ids) if normalize_reference(value)}
    declared_public_terms = {normalize_reference(term) for role_spec in request.spec.roles.values() for term in (*role_spec.public_terms, *role_spec.generic_terms)}
    forbidden.difference_update(declared_public_terms)
    from scoper.harness.evaluation_binding_policy import PUBLIC_CONTRACT_BINDING_COMPILER_VERSION
    if request.spec.compiler_version == PUBLIC_CONTRACT_BINDING_COMPILER_VERSION and reference not in {*request.role_handles, *request.raw_object_ids}:
        base = _controlled_state_base(normalized)
        if base != normalized and base in declared_public_terms:
            forbidden.discard(normalized)
    if normalized in forbidden:
        return BindingResult('rejected', 'private_reference_forbidden')
    strict_vocabulary = request.vocabulary_version == SHARED_VOCABULARY_VERSION
    if evidence_binding:
        strict_vocabulary = False
    vocabulary_class = classify_public_reference(normalized)
    if strict_vocabulary and request.tool_name in INTERACTION_TOOLS and (vocabulary_class in {'forbidden_operation', 'query_only'}):
        return BindingResult('rejected', 'broad_reference_not_operation_target')
    if strict_vocabulary and request.tool_name in INTERACTION_TOOLS and (vocabulary_class == 'unknown') and (resolve_public_family(_reference_head_noun(normalized)) is None):
        return BindingResult('unresolved', 'unknown_reference_not_operation_target')
    candidates = tuple(sorted(request.candidates, key=_candidate_sort_key))
    focus_result = _resolve_focus(request, normalized, candidates)
    if focus_result is not None:
        return focus_result
    task_result = _resolve_task_witness(request, normalized, candidates)
    deferred_task_conflict = task_result if task_result is not None and task_result.status == 'conflict' else None
    if task_result is not None and task_result.status != 'conflict':
        return task_result
    alias_matches = [candidate for candidate in candidates if _candidate_alias_is_public(request, candidate) and (normalized == normalize_reference(candidate.canonical_public_ref) or normalized in {normalize_reference(alias) for alias in candidate.aliases})]
    alias_result = _select_candidates(request, alias_matches, 'public_alias', 'public_alias')
    if alias_result is not None:
        return alias_result
    public_candidates = [candidate for candidate in candidates if candidate.activated and candidate.source in {'observation', 'public_event'}]
    observed_matches = _lexical_matches(normalized, public_candidates, request.taxonomy_aliases)
    observed_result = _select_candidates(request, observed_matches, 'public_evidence', 'lexical')
    if observed_result is not None:
        return observed_result
    memory_candidates = [candidate for candidate in candidates if candidate.activated and candidate.source == 'memory']
    memory_matches = _lexical_matches(normalized, memory_candidates, request.taxonomy_aliases)
    memory_result = _select_candidates(request, memory_matches, 'memory', 'memory_alias')
    if memory_result is not None:
        return memory_result
    from scoper.harness.evaluation_binding_policy import unobserved_scene_binding_enabled
    if unobserved_scene_binding_enabled(request.spec):
        scene_matches = _lexical_matches(normalized, [c for c in candidates if c.activated and c.source == 'scene'], request.taxonomy_aliases)
        scene_result = _select_candidates(request, scene_matches, 'scene_reference', 'lexical')
        if scene_result is not None:
            return scene_result
    room_matches = [candidate for candidate in candidates if candidate.source == 'room' and normalized in candidate.public_forms]
    room_result = _select_candidates(request, room_matches, 'room', 'room')
    if room_result is not None:
        return room_result
    if deferred_task_conflict is not None:
        if evidence_binding:
            return BindingResult('unresolved', 'no_public_match_observe_or_disambiguate')
        return deferred_task_conflict
    if evidence_binding:
        return BindingResult('unresolved', 'no_public_match_observe_or_disambiguate')
    return BindingResult('unresolved', 'no_public_match')

def _resolve_focus(request: BindingRequest, normalized: str, candidates: Sequence[BindingCandidate]) -> BindingResult | None:
    focus = request.recent_focus
    if focus is None:
        return None
    focus_forms = reference_forms(focus.canonical_public_ref, request.taxonomy_aliases)
    if normalized not in _PRONOUNS and normalized not in focus_forms:
        return None
    matches = [candidate for candidate in candidates if candidate.private_id == focus.private_id]
    if len(matches) == 0:
        return None
    from scoper.harness.evaluation_binding_policy import uses_evidence_binding
    if uses_evidence_binding(request.spec) and len(matches) > 1:
        matching_roles = [candidate for candidate in matches if candidate.role == focus.role]
        if matching_roles:
            matches = matching_roles
        elif len({candidate.private_id for candidate in matches}) == 1:
            matches = matches[:1]
    if len(matches) != 1:
        return BindingResult('ambiguous', 'recent_focus_not_unique')
    candidate = matches[0]
    blocked = _interaction_gate(request, candidate, 'recent_focus', 'focus')
    if blocked is not None:
        return blocked
    return _success(candidate, source='recent_focus', match_kind='focus', request=request)

def _resolve_task_witness(request: BindingRequest, normalized: str, candidates: Sequence[BindingCandidate]) -> BindingResult | None:
    matches: list[BindingCandidate] = []
    concrete_conflicts: list[BindingCandidate] = []
    generic = normalized in {normalize_reference(term) for term in _GENERIC_TERMS}
    state_base = _controlled_state_base(normalized)
    collection_base = _controlled_collection_base(normalized)
    declared_forms: set[str] = set()
    for declared_role in request.spec.roles.values():
        for term in (*declared_role.public_terms, *declared_role.generic_terms):
            declared_forms.update(reference_forms(term, request.taxonomy_aliases))
    declared_heads = {head for form in declared_forms if (head := _reference_head_noun(form)) is not None}
    query_scene_category = instruction_scene_reference_category(normalized, request.task_instruction)
    query_head = _reference_head_noun(normalized)
    unique_head_role = _unique_task_witness_role(request.spec, normalized)
    relation_subject_tokens = []
    for token in normalized.split():
        if token in _RELATION_PREPOSITIONS:
            break
        relation_subject_tokens.append(token)
    relation_subject = ' '.join(relation_subject_tokens)
    relation_qualified = relation_subject != normalized
    generic_relation_subject = relation_subject in {normalize_reference(term) for term in _GENERIC_TERMS}
    state_qualified = bool(normalized.split() and normalized.split()[0] in _CONTROLLED_STATE_MODIFIERS)
    reference_declared_anywhere = bool(normalized in declared_forms or state_base in declared_forms or (query_head is not None and query_head in declared_heads))
    unactivated_look_witnesses: set[str] = set()
    if request.tool_name == 'look_for':
        for role_name, role_spec in request.spec.roles.items():
            if role_spec.source_policy not in {'observation', 'task_generic_witness'}:
                continue
            role_candidates = [candidate for candidate in candidates if candidate.role == role_name]
            if len(role_candidates) != 1:
                continue
            candidate = role_candidates[0]
            if candidate.activated:
                continue
            role_forms: set[str] = set()
            for term in (*role_spec.public_terms, *role_spec.generic_terms):
                role_forms.update(reference_forms(term, request.taxonomy_aliases))
            if normalized in role_forms:
                unactivated_look_witnesses.add(candidate.private_id)
        if len(unactivated_look_witnesses) > 1:
            return BindingResult('ambiguous', 'multiple_unactivated_look_witnesses', 'task_policy', 'task_witness')
    for role_name, role_spec in request.spec.roles.items():
        role_candidates = [candidate for candidate in candidates if candidate.role == role_name]
        if not role_candidates:
            continue
        public_forms: set[str] = set()
        for term in role_spec.public_terms:
            public_forms.update(reference_forms(term, request.taxonomy_aliases))
        generic_forms: set[str] = set()
        for term in role_spec.generic_terms:
            generic_forms.update(reference_forms(term, request.taxonomy_aliases))
        matched = normalized in public_forms
        generic_matched = normalized in generic_forms
        category_matched = bool(query_scene_category and any((candidate.scene_observed for candidate in role_candidates)) and (query_scene_category in {category for term in (*role_spec.public_terms, *role_spec.generic_terms) if (category := instruction_scene_reference_category(term, request.task_instruction))}))
        collection_matched = bool(role_spec.collection_id and collection_base != normalized and (collection_base in public_forms or collection_base in generic_forms))
        head_matched = bool(relation_qualified and (not generic_relation_subject) and (unique_head_role == role_name) and (query_head is not None) and (query_head in {head for form in (*public_forms, *generic_forms) if (head := _reference_head_noun(form)) is not None}))
        state_matched = bool(state_base and state_base != normalized and (state_base in public_forms or state_base in generic_forms))
        if role_spec.source_policy in {'observation', 'public_event', 'task_generic_witness'}:
            allow_unactivated_look = any((candidate.private_id in unactivated_look_witnesses for candidate in role_candidates))
            if not any((candidate.activated for candidate in role_candidates)) and (not allow_unactivated_look):
                continue
            if not (matched or generic_matched or collection_matched or state_matched or head_matched or category_matched):
                continue
            matches.extend((candidate for candidate in role_candidates if (candidate.activated or candidate.private_id in unactivated_look_witnesses) and (not (state_matched or (head_matched and state_qualified)) or candidate.visible or candidate.last_seen_step is not None)))
            continue
        if generic_matched and role_spec.source_policy != 'task_generic_witness':
            continue
        if matched or generic_matched or collection_matched or head_matched or category_matched:
            matches.extend((candidate for candidate in role_candidates if not (head_matched and state_qualified) or candidate.visible or candidate.last_seen_step is not None))
        elif state_matched:
            matches.extend((candidate for candidate in role_candidates if candidate.visible or candidate.last_seen_step is not None))
        elif not generic and (not reference_declared_anywhere) and _reference_looks_concrete(normalized):
            candidate_forms = set().union(*(candidate.public_forms for candidate in role_candidates))
            if candidate_forms and _reference_type_conflicts(normalized, public_forms | candidate_forms):
                concrete_conflicts.extend(role_candidates)
    if matches:
        from scoper.harness.evaluation_binding_policy import EXACT_BINDING_COMPILER_VERSIONS, normalize_concrete_reference
        if request.spec.compiler_version in EXACT_BINDING_COMPILER_VERSIONS:
            exact = [c for c in matches if c.role in request.spec.roles and any((normalized == normalize_concrete_reference(term) for term in request.spec.roles[c.role].public_terms))]
            if exact:
                matches = exact
        from scoper.harness.evaluation_binding_policy import uses_evidence_binding
        if uses_evidence_binding(request.spec):
            contextual = [candidate for candidate in matches if candidate.role in request.public_destination_roles]
            if len({candidate.private_id for candidate in contextual}) == 1:
                matches = contextual
            else:
                focus = request.recent_focus
                ready_matches = [c for c in matches if focus and c.private_id == focus.private_id]
                if request.tool_name in INTERACTION_TOOLS and len({c.private_id for c in ready_matches}) == 1:
                    matches = ready_matches
                else:
                    matches = _select_public_phase_witness(request, matches)
        else:
            matches = _select_public_phase_witness(request, matches)
            matches = _select_collection_witness(request, matches)
        return _select_candidates(request, matches, 'task_policy', 'task_witness')
    if concrete_conflicts:
        return BindingResult('conflict', 'reference_conflicts_with_task')
    return None

def _select_public_phase_witness(request: BindingRequest, candidates: Sequence[BindingCandidate]) -> list[BindingCandidate]:
    phase_order = tuple(request.public_phase_role_order)
    if len(phase_order) < 2:
        return list(candidates)
    candidate_roles = {str(candidate.role or '') for candidate in candidates}
    ordered_matches = tuple((role for role in phase_order if role in candidate_roles))
    if len(ordered_matches) < 2:
        return list(candidates)
    processed = set(request.publicly_processed_roles)
    for role_name in ordered_matches:
        if role_name in processed:
            continue
        selected = [candidate for candidate in candidates if candidate.role == role_name]
        if selected:
            return selected
    return []

def derive_public_phase_role_order(task: Mapping[str, Any]) -> tuple[str, ...]:
    from scoper.harness.evaluation_binding_policy import task_uses_evidence_binding, public_sequence_roles
    if task_uses_evidence_binding(task):
        sequence = public_sequence_roles(task)
        if sequence:
            return sequence
    instruction = normalize_reference(task.get('instruction'))
    if not instruction:
        return ()
    family_spec = task.get('family_spec')
    if not isinstance(family_spec, Mapping):
        return ()
    mid_execution = family_spec.get('mid_execution')
    if not isinstance(mid_execution, Mapping):
        return ()
    primary = mid_execution.get('primary_task')
    if not isinstance(primary, Mapping) or primary.get('scenario') != 'staging_interrupt':
        return ()
    staging_role = str(primary.get('staging_receptacle_role') or '')
    final_role = str(primary.get('receptacle_role') or '')
    if not staging_role or not final_role or staging_role == final_role:
        return ()
    roles = task.get('roles')
    if not isinstance(roles, Mapping) or staging_role not in roles or final_role not in roles:
        return ()
    tokens = instruction.split()
    stage_positions = [index for index, token in enumerate(tokens) if token in {'stage', 'staging'}]
    final_positions = [index for index, token in enumerate(tokens) if token in {'put', 'place', 'return'}]
    if not stage_positions or not any((position > stage_positions[0] for position in final_positions)):
        return ()
    return (staging_role, final_role)

def _select_collection_witness(request: BindingRequest, candidates: Sequence[BindingCandidate]) -> list[BindingCandidate]:
    collection_ids = {request.spec.roles[candidate.role].collection_id for candidate in candidates if candidate.role in request.spec.roles}
    collection_ids.discard(None)
    if len(collection_ids) != 1:
        return list(candidates)
    collection = request.spec.collections[str(next(iter(collection_ids)))]
    processed = set(request.publicly_processed_roles)
    for role_name in collection.role_order:
        if role_name in processed:
            continue
        selected = [candidate for candidate in candidates if candidate.role == role_name]
        if selected:
            return selected
    return []

def _select_candidates(request: BindingRequest, candidates: Sequence[BindingCandidate], source: str, match_kind: str) -> BindingResult | None:
    if not candidates:
        return None
    unique = {candidate.private_id: candidate for candidate in candidates}
    benevolent = _select_benevolent_candidate(request, tuple(unique.values()))
    if benevolent is not None:
        unique = {benevolent.private_id: benevolent}
        match_kind = 'benevolent_task_progress'
    if len(unique) != 1:
        return BindingResult('ambiguous', 'multiple_public_matches', source, match_kind)
    candidate = next(iter(unique.values()))
    role_spec = request.spec.roles.get(candidate.role or '')
    if request.tool_name in NAVIGATION_TOOLS:
        from scoper.harness.evaluation_binding_policy import unobserved_scene_binding_enabled
        if not candidate.visible and candidate.last_seen_step is None:
            if not unobserved_scene_binding_enabled(request.spec) and (not candidate.scene_observed) and (role_spec is None or not role_spec.allow_unseen_navigation):
                return BindingResult('unresolved', 'unseen_navigation_not_allowed', source, match_kind)
        return _success(candidate, source=source, match_kind=match_kind, request=request)
    if request.tool_name in LOOK_TOOLS:
        from scoper.harness.evaluation_binding_policy import unobserved_scene_binding_enabled
        unique_task_witness = request.tool_name == 'look_for' and source == 'task_policy' and (match_kind == 'task_witness')
        if not candidate.visible and candidate.source not in {'memory', 'observation'} and (not unique_task_witness) and (not unobserved_scene_binding_enabled(request.spec)):
            return BindingResult('unresolved', 'look_for_requires_public_evidence', source, match_kind)
        return _success(candidate, source=source, match_kind=match_kind, request=request)
    blocked = _interaction_gate(request, candidate, source, match_kind)
    if blocked is not None:
        return blocked
    return _success(candidate, source=source, match_kind=match_kind, request=request)

def _select_benevolent_candidate(request: BindingRequest, candidates: Sequence[BindingCandidate]) -> BindingCandidate | None:
    if len(candidates) < 2:
        return None
    for role_name in request.benevolent_role_order:
        matches = [candidate for candidate in candidates if candidate.role == role_name]
        if len({candidate.private_id for candidate in matches}) == 1:
            return matches[0]
    return None

def derive_benevolent_role_order(task: Mapping[str, Any], *, predicate_satisfied: Sequence[bool]=(), held_roles: Sequence[str]=()) -> tuple[str, ...]:
    goal = task.get('goal')
    predicates = goal.get('predicates') if isinstance(goal, Mapping) else None
    if not isinstance(predicates, list) or not predicates:
        return ()
    statuses = tuple((bool(value) for value in predicate_satisfied))
    if statuses and len(statuses) != len(predicates):
        return ()
    pending = [predicate for index, predicate in enumerate(predicates) if isinstance(predicate, Mapping) and (not (statuses and statuses[index]))]
    held = {str(role) for role in held_roles if isinstance(role, str) and role}
    ordered: list[str] = []

    def add(role: str) -> None:
        if role and role not in ordered:
            ordered.append(role)
    role_groups = [_predicate_role_order(predicate) for predicate in pending]
    for roles in role_groups:
        if held.intersection(roles):
            for role in roles:
                if role not in held:
                    add(role)
    for roles in role_groups:
        for role in roles:
            add(role)
    return tuple(ordered)

def _predicate_role_order(predicate: Mapping[str, Any]) -> tuple[str, ...]:
    singular = ('object_role', 'target_role', 'role', 'receptacle_role', 'destination_role', 'storage_role', 'source_role')
    plural = tuple((key for key in predicate if key.endswith('_roles')))
    result: list[str] = []
    for key in (*singular, *sorted(plural, key=str.encode)):
        value = predicate.get(key)
        values = value if isinstance(value, list) else [value]
        for role in values:
            if isinstance(role, str) and role and (role not in result):
                result.append(role)
    return tuple(result)

def _interaction_gate(request: BindingRequest, candidate: BindingCandidate, source: str, match_kind: str) -> BindingResult | None:
    if request.tool_name not in INTERACTION_TOOLS:
        return None
    if request.bypass_interaction_readiness:
        return None
    from scoper.harness.evaluation_binding_policy import uses_evidence_binding, spec_requires_observation
    require_visible = not uses_evidence_binding(request.spec) or spec_requires_observation(request.spec)
    if not request.local_interaction_context_active:
        reason = 'navigation_context_required'
    elif require_visible and (not candidate.visible) or not candidate.interaction_ready:
        reason = 'target_not_interaction_ready'
    else:
        return None
    return BindingResult('not_interaction_ready', reason, source, match_kind, canonical_public_ref=candidate.canonical_public_ref)

def _success(candidate: BindingCandidate, *, source: str, match_kind: str, request: BindingRequest) -> BindingResult:
    navigation_class = None
    if request.tool_name in NAVIGATION_TOOLS and candidate.source != 'room':
        if candidate.visible:
            navigation_class = 'seen_object_navigation'
        elif candidate.last_seen_step is None:
            navigation_class = 'unseen_object_exploration'
        else:
            navigation_class = 'stale_object_navigation'
    return BindingResult(status='success', reason='matched', source=source, match_kind=match_kind, canonical_public_ref=candidate.canonical_public_ref, private_id=candidate.private_id, role=candidate.role, navigation_class=navigation_class)

def _lexical_matches(normalized: str, candidates: Sequence[BindingCandidate], taxonomy_aliases: Mapping[str, Sequence[str]]) -> list[BindingCandidate]:
    query_forms = reference_forms(normalized, taxonomy_aliases)
    exact = [candidate for candidate in candidates if query_forms.intersection(set().union(*(reference_forms(value, taxonomy_aliases) for value in (candidate.canonical_public_ref, candidate.object_type, *candidate.aliases))))]
    if exact:
        return exact
    query_tokens = set(normalized.split())
    scored: list[tuple[int, BindingCandidate]] = []
    for candidate in candidates:
        best = 0
        for form in candidate.public_forms:
            tokens = set(form.split())
            if tokens and (tokens <= query_tokens or query_tokens <= tokens):
                best = max(best, len(tokens.intersection(query_tokens)))
        if best:
            scored.append((best, candidate))
    if not scored:
        return []
    best_score = max((score for score, _ in scored))
    return [candidate for score, candidate in scored if score == best_score]

def _reference_looks_concrete(normalized: str) -> bool:
    return bool(normalized and normalized not in {normalize_reference(term) for term in _GENERIC_TERMS})

def _controlled_state_base(reference: str) -> str:
    tokens = reference.split()
    while tokens and tokens[0] in _CONTROLLED_STATE_MODIFIERS:
        tokens.pop(0)
    return ' '.join(tokens)

def _controlled_collection_base(reference: str) -> str:
    tokens = reference.split()
    while tokens and tokens[0] in _CONTROLLED_COLLECTION_MODIFIERS:
        tokens.pop(0)
    return ' '.join(tokens)

def _reference_head_noun(reference: str) -> str | None:
    tokens = reference.split()
    for index, token in enumerate(tokens):
        if token in _RELATION_PREPOSITIONS:
            tokens = tokens[:index]
            break
    while tokens and tokens[0] in _CONTROLLED_STATE_MODIFIERS:
        tokens.pop(0)
    return tokens[-1] if tokens else None

def _reference_type_conflicts(reference: str, forms: Iterable[str]) -> bool:
    query_head = _reference_head_noun(reference)
    if query_head is None:
        return False
    allowed_heads = {head for form in forms if (head := _reference_head_noun(normalize_reference(form))) is not None}
    return bool(allowed_heads and query_head not in allowed_heads)

def _candidate_sort_key(candidate: BindingCandidate) -> tuple[bytes, bytes]:
    return (candidate.canonical_public_ref.encode(), candidate.private_id.encode())

def _candidate_alias_is_public(request: BindingRequest, candidate: BindingCandidate) -> bool:
    from scoper.harness.evaluation_binding_policy import unobserved_scene_binding_enabled
    if candidate.source == 'scene':
        return candidate.activated and unobserved_scene_binding_enabled(request.spec)
    role_spec = request.spec.roles.get(candidate.role or '')
    if role_spec is not None and role_spec.source_policy in {'task_explicit', 'task_generic_witness'}:
        return True
    return candidate.activated and candidate.source in {'memory', 'observation', 'public_event', 'room'}

def build_binding_sidecar(*, bank_sha256: str, task_count: int, compiler_version: str, specs: Sequence[TaskBindingSpec]) -> dict[str, Any]:
    _require_sha256(bank_sha256, 'bank_sha256')
    ordered = sorted(specs, key=lambda spec: spec.task_id.encode())
    if len(ordered) != task_count or len({spec.task_id for spec in ordered}) != task_count:
        raise ValueError('sidecar specs do not exactly cover task_count')
    sidecar = {'schema_version': TASK_BINDING_SIDECAR_SCHEMA, 'bank_sha256': bank_sha256, 'task_count': int(task_count), 'compiler_version': str(compiler_version), 'specs': [spec.to_dict() for spec in ordered]}
    return sidecar

def binding_sidecar_manifest(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    if sidecar.get('schema_version') != TASK_BINDING_SIDECAR_SCHEMA:
        raise ValueError('unsupported binding sidecar schema')
    specs = sidecar.get('specs')
    if not isinstance(specs, list) or len(specs) != int(sidecar.get('task_count') or -1):
        raise ValueError('binding sidecar task count differs')
    loaded = [TaskBindingSpec.from_dict(value) for value in specs]
    if [spec.task_id for spec in loaded] != sorted((spec.task_id for spec in loaded), key=str.encode):
        raise ValueError('binding sidecar is not sorted')
    return {'schema_version': TASK_BINDING_MANIFEST_SCHEMA, 'bank_sha256': str(sidecar['bank_sha256']), 'task_count': int(sidecar['task_count']), 'compiler_version': str(sidecar['compiler_version']), 'sidecar_content_sha256': sha256_json(sidecar)}

def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any((not isinstance(item, str) for item in value)):
        raise ValueError(f'{name} must be a string list')
    return tuple(value)

def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any((character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'{name} must be sha256')
__all__ = ['BINDING_STATUSES', 'INTERACTION_TOOLS', 'LOOK_TOOLS', 'NAVIGATION_TOOLS', 'SOURCE_POLICIES', 'TASK_BINDING_MANIFEST_SCHEMA', 'TASK_BINDING_SIDECAR_SCHEMA', 'TASK_BINDING_SPEC_SCHEMA', 'BindingCandidate', 'BindingCollectionSpec', 'BindingFocus', 'BindingRequest', 'BindingResult', 'BindingRoleSpec', 'PublicBindingState', 'TaskBindingSpec', 'binding_sidecar_manifest', 'build_binding_sidecar', 'derive_public_binding_state', 'derive_public_phase_role_order', 'equivalent_unique_task_witness_calls', 'instruction_scene_reference_category', 'instruction_scene_reference_relevant', 'normalize_reference', 'normalize_instruction_grounded_memory_proposal', 'reference_forms', 'resolve_binding', 'scene_observation_activates_binding_role', 'task_spec_sha256', 'validate_task_binding_spec']
