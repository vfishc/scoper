from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
from scoper.harness.canonical import canonical_json_bytes
from scoper.harness.contracts import EFFECT_VOCABULARY, MethodContractError
from scoper.harness.runtime_schema import validate_memory_delta
from scoper.harness.validation import ensure_members, exact_mapping, integer, non_empty_string, optional_string, probability, public_json, public_semantic_name, sha256, string_sequence
from scoper.harness.evidence_memory import CANONICAL_PHYSICAL_ACTIONS, EvidenceFact, EvidenceMemory, MemoryCapacity, PendingObligation, RecoveryIncident, commit_effect_bindings, enforce_capacity, fact_id_for, validate_failure_type
from scoper.harness.effect_binding import EffectBinding, SCENE_OBSERVATION_RELATION, binding_fact_key, validate_effect_bindings, validate_fact_semantics
from scoper.harness.memory_router import RouterLimits
from scoper.harness.verifier import VerifierSelection
from scoper.harness.vocabulary import OBSERVE_MAX_FACT_MUTATIONS, OBSERVE_MAX_RETAINED_FACTS
PROPOSAL_FIELDS = frozenset({'schema_version', 'proposal_id', 'mode', 'base_revision', 'evidence_frame_ids', 'fact_mutations', 'incident_mutations', 'obligation_mutations', 'per_fact_status', 'relevant_memory_ids', 'abstain_reason_or_null', 'confidence', 'checkpoint_sha256'})
FACT_MUTATION_FIELDS = frozenset({'operation', 'public_entity_ref', 'relation_or_property', 'object_public_role_or_value', 'confidence'})
INCIDENT_MUTATION_FIELDS = frozenset({'operation', 'failure_type', 'public_roles', 'confidence'})
OBLIGATION_MUTATION_FIELDS = frozenset({'operation', 'type', 'public_roles', 'source_incident_id_or_null'})
PER_FACT_STATUS_FIELDS = frozenset({'fact_id', 'status'})

@dataclass(frozen=True)
class ControllerThresholds:
    commit_threshold: float = 0.7
    contradiction_threshold: float = 0.7
    incident_threshold: float = 0.7

    def __post_init__(self) -> None:
        probability(self.commit_threshold, pointer='/thresholds/commit_threshold')
        probability(self.contradiction_threshold, pointer='/thresholds/contradiction_threshold')
        probability(self.incident_threshold, pointer='/thresholds/incident_threshold')

@dataclass(frozen=True)
class VerificationCommitContext:
    canonical_action: str
    selected_outcome: str
    selected_effects: tuple[str, ...]
    selected_observability: str
    success_probability: float
    failure_probability: float
    effect_probabilities: tuple[tuple[str, float], ...]
    effect_bindings: tuple[EffectBinding, ...]
    public_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        action = non_empty_string(self.canonical_action, pointer='/verification/canonical_action')
        if action not in CANONICAL_PHYSICAL_ACTIONS:
            raise MethodContractError('method_memory_vocabulary_invalid', 'verification action is outside the frozen physical vocabulary', pointer='/verification/canonical_action')
        if self.selected_outcome not in {'success', 'failure', 'ambiguous'}:
            raise MethodContractError('method_verifier_output_invalid', 'selected outcome is invalid')
        effects = string_sequence(list(self.selected_effects), pointer='/verification/selected_effects')
        if effects != tuple((effect for effect in EFFECT_VOCABULARY if effect in effects)):
            raise MethodContractError('method_verifier_output_invalid', 'selected effects are not in vocabulary order')
        if self.selected_observability not in {'observable', 'partially_observable', 'unobservable'}:
            raise MethodContractError('method_verifier_output_invalid', 'selected observability is invalid')
        success = probability(self.success_probability, pointer='/verification/success_probability')
        failure = probability(self.failure_probability, pointer='/verification/failure_probability')
        if success + failure > 1.0 + 1e-05:
            raise MethodContractError('method_verifier_output_invalid', 'success and failure probabilities exceed one')
        probability_names = []
        for index, item in enumerate(self.effect_probabilities):
            if not isinstance(item, tuple) or len(item) != 2:
                raise MethodContractError('method_schema_invalid', 'effect probabilities must be key/value tuples')
            effect, value = item
            probability_names.append(effect)
            probability(value, pointer='/verification/effect_probabilities/{}'.format(index))
        if tuple(probability_names) != EFFECT_VOCABULARY:
            raise MethodContractError('method_verifier_output_invalid', 'effect probabilities must cover the frozen vocabulary')
        bindings = validate_effect_bindings(self.effect_bindings)
        if bindings != self.effect_bindings:
            raise MethodContractError('method_effect_binding_invalid', 'effect bindings are not canonical')
        roles = string_sequence(list(self.public_roles), pointer='/verification/public_roles', non_empty=True, sorted_utf8=True)
        role_set = set(roles)
        for binding in bindings:
            if binding.binding_status != 'known':
                continue
            if binding.subject_public_role not in role_set:
                raise MethodContractError('method_public_reference_invalid', 'binding subject is not a public role')
            if binding.relation_or_property in {'inside', 'on_top'} and isinstance(binding.object_public_role_or_value, str) and (binding.object_public_role_or_value not in role_set):
                raise MethodContractError('method_public_reference_invalid', 'binding object is not a public role')

    @property
    def selection(self) -> VerifierSelection:
        return VerifierSelection(self.selected_outcome, self.selected_effects, self.selected_observability, 'no_effect' in self.selected_effects and len(self.selected_effects) > 1)

    @property
    def effective_outcome(self) -> str:
        effect_conflict = 'no_effect' in self.selected_effects and len(self.selected_effects) > 1
        if effect_conflict or self.selected_observability == 'unobservable':
            return 'ambiguous'
        if self.selected_outcome != 'success':
            return self.selected_outcome
        required = tuple((binding for binding in self.effect_bindings if binding.required))
        if 'no_effect' in self.selected_effects or any((binding.binding_status != 'known' for binding in required)) or any((binding.effect not in self.selected_effects for binding in required)):
            return 'ambiguous'
        return 'success'

    def effect_probability(self, effect: str) -> float:
        try:
            return dict(self.effect_probabilities)[effect]
        except KeyError as exc:
            raise MethodContractError('method_verifier_output_invalid', 'binding effect lacks a probability') from exc

@dataclass(frozen=True)
class RoutedMemoryIds:
    fact_ids: tuple[str, ...] = ()
    incident_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, list[str]]:
        return {'fact_ids': list(self.fact_ids), 'incident_ids': list(self.incident_ids), 'obligation_ids': list(self.obligation_ids)}

@dataclass(frozen=True)
class ProposalContext:
    action_id: str
    step: int
    expected_evidence_frame_ids: tuple[str, ...]
    allowed_public_entity_refs: tuple[str, ...]
    active_roles: tuple[str, ...] = ()
    candidate_fact_ids: tuple[str, ...] = ()
    candidate_incident_ids: tuple[str, ...] = ()
    candidate_obligation_ids: tuple[str, ...] = ()
    mandatory_incident_ids: tuple[str, ...] = ()
    mandatory_obligation_ids: tuple[str, ...] = ()
    router_limits: RouterLimits = RouterLimits()
    source_incident_id_or_null: str | None = None
    selected_observability: str = 'observable'
    verification_or_null: VerificationCommitContext | None = None

    def __post_init__(self) -> None:
        non_empty_string(self.action_id, pointer='/context/action_id')
        integer(self.step, pointer='/context/step')
        string_sequence(list(self.expected_evidence_frame_ids), pointer='/context/expected_evidence_frame_ids')
        string_sequence(list(self.allowed_public_entity_refs), pointer='/context/allowed_public_entity_refs', sorted_utf8=True)
        string_sequence(list(self.active_roles), pointer='/context/active_roles')
        string_sequence(list(self.candidate_fact_ids), pointer='/context/candidate_fact_ids', sorted_utf8=True)
        for field, values in (('candidate_incident_ids', self.candidate_incident_ids), ('candidate_obligation_ids', self.candidate_obligation_ids)):
            string_sequence(list(values), pointer='/context/{}'.format(field), sorted_utf8=True)
        for field, values in (('mandatory_incident_ids', self.mandatory_incident_ids), ('mandatory_obligation_ids', self.mandatory_obligation_ids)):
            string_sequence(list(values), pointer='/context/{}'.format(field))
        candidate_sets = (set(self.candidate_fact_ids), set(self.candidate_incident_ids), set(self.candidate_obligation_ids))
        if any((candidate_sets[left] & candidate_sets[right] for left, right in ((0, 1), (0, 2), (1, 2)))):
            raise MethodContractError('method_public_reference_invalid', 'route candidate types overlap')
        ensure_members(self.mandatory_incident_ids, self.candidate_incident_ids, pointer='/context/mandatory_incident_ids')
        ensure_members(self.mandatory_obligation_ids, self.candidate_obligation_ids, pointer='/context/mandatory_obligation_ids')
        if not isinstance(self.router_limits, RouterLimits):
            raise MethodContractError('method_schema_invalid', 'router limits are invalid')
        optional_string(self.source_incident_id_or_null, pointer='/context/source_incident_id_or_null')
        if self.selected_observability not in {'observable', 'partially_observable', 'unobservable', 'unknown'}:
            raise MethodContractError('method_schema_invalid', 'selected observability is invalid')
        if self.verification_or_null is not None:
            if not isinstance(self.verification_or_null, VerificationCommitContext):
                raise MethodContractError('method_schema_invalid', 'verification context is invalid')
            ensure_members(self.verification_or_null.public_roles, self.allowed_public_entity_refs, pointer='/context/verification/public_roles')

@dataclass(frozen=True)
class ControllerDecision:
    accepted_by_harness: bool
    reason: str
    memory: EvidenceMemory
    proposal_id: str | None
    routed_memory_or_null: RoutedMemoryIds | None = None
    error_code_or_null: str | None = None
    proposal_mode_or_null: str | None = None
    parent_action_id_or_null: str | None = None
    base_revision_or_null: int | None = None

    @property
    def new_revision(self) -> int:
        return self.memory.revision

    def summary(self) -> dict[str, Any]:
        return {'accepted_by_harness': self.accepted_by_harness, 'new_revision': self.new_revision, 'reason': self.reason}

    @property
    def routed_memory_ids(self) -> tuple[str, ...]:
        if self.routed_memory_or_null is None:
            return ()
        return self.routed_memory_or_null.fact_ids + self.routed_memory_or_null.incident_ids + self.routed_memory_or_null.obligation_ids

    def accepted_mutation(self) -> dict[str, Any] | None:
        if not self.accepted_by_harness or self.reason == 'route_valid_no_commit':
            return None
        return {'accepted_by_harness': True, 'revision': self.memory.revision, 'facts': [fact.to_mapping() for fact in self.memory.facts], 'incidents': [incident.to_mapping() for incident in self.memory.incidents], 'pending_obligations': [obligation.to_mapping() for obligation in self.memory.pending_obligations]}

def _fact_mutation_key(value: Mapping[str, Any]) -> tuple[bytes, bytes, bytes]:
    return (str(value['public_entity_ref']).encode('utf-8'), str(value['relation_or_property']).encode('utf-8'), canonical_json_bytes(value['object_public_role_or_value']))

def validate_memory_proposal(value: Mapping[str, Any]) -> Mapping[str, Any]:
    public_json(value, pointer='/proposal', include_future_action=True)
    proposal = exact_mapping(value, PROPOSAL_FIELDS, pointer='/proposal')
    if proposal['schema_version'] != 'memory-proposer-io.v1':
        raise MethodContractError('method_schema_invalid', 'memory proposal schema is invalid')
    non_empty_string(proposal['proposal_id'], pointer='/proposal/proposal_id')
    mode = proposal['mode']
    if mode not in {'update', 'reconcile', 'route', 'observe'}:
        raise MethodContractError('method_schema_invalid', 'memory proposal mode is invalid')
    integer(proposal['base_revision'], pointer='/proposal/base_revision')
    frames = string_sequence(proposal['evidence_frame_ids'], pointer='/proposal/evidence_frame_ids', non_empty=mode != 'route')
    if not all((isinstance(proposal[field], list) for field in ('fact_mutations', 'incident_mutations', 'obligation_mutations', 'per_fact_status', 'relevant_memory_ids'))):
        raise MethodContractError('method_schema_invalid', 'proposal collections must be lists')
    probability(proposal['confidence'], pointer='/proposal/confidence')
    sha256(proposal['checkpoint_sha256'], pointer='/proposal/checkpoint_sha256')
    abstain = proposal['abstain_reason_or_null']
    if abstain is not None:
        non_empty_string(abstain, pointer='/proposal/abstain_reason_or_null')
    fact_keys: dict[tuple[bytes, bytes, bytes], list[str]] = {}
    for index, raw_mutation in enumerate(proposal['fact_mutations']):
        mutation = exact_mapping(raw_mutation, FACT_MUTATION_FIELDS, pointer='/proposal/fact_mutations/{}'.format(index))
        public_semantic_name(mutation['public_entity_ref'], pointer='/proposal/fact_mutations/{}/public_entity_ref'.format(index))
        validate_fact_semantics(mutation['relation_or_property'], mutation['object_public_role_or_value'], pointer='/proposal/fact_mutations/{}'.format(index), allow_scene_observation=mode == 'observe')
        fact_keys.setdefault(_fact_mutation_key(mutation), []).append(mutation['operation'])
    if any((len(operations) != 1 for operations in fact_keys.values())):
        raise MethodContractError('method_memory_mutation_conflict', 'one proposal mutates the same fact more than once')
    if mode in {'update', 'reconcile'}:
        validate_memory_delta({'schema_version': 'memory-delta.v1', 'base_revision': proposal['base_revision'], 'evidence_frame_ids': list(frames), 'proposal_mode': mode, 'fact_mutations': proposal['fact_mutations'], 'incident_mutations': proposal['incident_mutations'], 'obligation_mutations': proposal['obligation_mutations'], 'abstain_reason_or_null': abstain}, pointer='/proposal')
    elif mode == 'observe':
        if len(proposal['fact_mutations']) > OBSERVE_MAX_FACT_MUTATIONS or proposal['incident_mutations'] or proposal['obligation_mutations'] or proposal['per_fact_status'] or proposal['relevant_memory_ids']:
            raise MethodContractError('method_schema_invalid', 'observe may only contain at most eight scene-observation fact mutations')
        if any((mutation['relation_or_property'] != SCENE_OBSERVATION_RELATION or mutation['operation'] not in {'add', 'confirm', 'invalidate'} for mutation in proposal['fact_mutations'])):
            raise MethodContractError('method_schema_invalid', 'observe fact mutation is outside the advisory vocabulary')
    elif any((proposal[field] for field in ('fact_mutations', 'incident_mutations', 'obligation_mutations', 'per_fact_status'))):
        raise MethodContractError('method_schema_invalid', 'route proposal cannot mutate memory')
    status_ids = []
    for index, raw_status in enumerate(proposal['per_fact_status']):
        status = exact_mapping(raw_status, PER_FACT_STATUS_FIELDS, pointer='/proposal/per_fact_status/{}'.format(index))
        status_ids.append(non_empty_string(status['fact_id'], pointer='/proposal/per_fact_status'))
        if status['status'] not in {'confirm', 'contradict', 'unobservable', 'unknown'}:
            raise MethodContractError('method_schema_invalid', 'per-fact status is invalid')
    if status_ids != sorted(status_ids, key=lambda item: item.encode('utf-8')) or len(set(status_ids)) != len(status_ids):
        raise MethodContractError('method_schema_invalid', 'per-fact status IDs are not canonical')
    relevant_ids = string_sequence(proposal['relevant_memory_ids'], pointer='/proposal/relevant_memory_ids')
    if mode != 'route' and relevant_ids:
        raise MethodContractError('method_schema_invalid', 'non-route proposal has relevant IDs')
    if mode == 'update' and status_ids:
        raise MethodContractError('method_schema_invalid', 'update proposal has per-fact statuses')
    incident_keys = []
    for index, mutation in enumerate(proposal['incident_mutations']):
        exact_mapping(mutation, INCIDENT_MUTATION_FIELDS, pointer='/proposal/incident_mutations/{}'.format(index))
        roles = string_sequence(mutation['public_roles'], pointer='/proposal/incident_mutations/{}/public_roles'.format(index), sorted_utf8=True)
        failure_type = validate_failure_type(mutation['failure_type'], pointer='/proposal/incident_mutations/{}/failure_type'.format(index))
        incident_keys.append((failure_type, roles))
    if len(set(incident_keys)) != len(incident_keys):
        raise MethodContractError('method_memory_mutation_conflict', 'one proposal mutates the same incident more than once')
    obligation_keys = []
    for index, mutation in enumerate(proposal['obligation_mutations']):
        exact_mapping(mutation, OBLIGATION_MUTATION_FIELDS, pointer='/proposal/obligation_mutations/{}'.format(index))
        roles = string_sequence(mutation['public_roles'], pointer='/proposal/obligation_mutations/{}/public_roles'.format(index), non_empty=True, sorted_utf8=True)
        obligation_keys.append((mutation['type'], roles, mutation['source_incident_id_or_null']))
    if len(set(obligation_keys)) != len(obligation_keys):
        raise MethodContractError('method_memory_mutation_conflict', 'one proposal mutates the same obligation more than once')
    return proposal

def _rejected(memory: EvidenceMemory, reason: str, proposal_id: str | None, *, error_code: str | None=None) -> ControllerDecision:
    return ControllerDecision(False, reason, memory, proposal_id, error_code_or_null=error_code)

def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)

def _evaluate_route_proposal(memory: EvidenceMemory, proposal: Mapping[str, Any], context: ProposalContext) -> ControllerDecision:
    proposal_id = proposal['proposal_id']
    if context.verification_or_null is not None:
        return _rejected(memory, 'route_verification_context_forbidden', proposal_id)
    fact_ids = {fact.fact_id for fact in memory.facts}
    incident_ids = {incident.incident_id for incident in memory.incidents}
    obligation_ids = {obligation.obligation_id for obligation in memory.pending_obligations}
    try:
        ensure_members(context.candidate_fact_ids, fact_ids, pointer='/context/candidate_fact_ids')
        ensure_members(context.candidate_incident_ids, incident_ids, pointer='/context/candidate_incident_ids')
        ensure_members(context.candidate_obligation_ids, obligation_ids, pointer='/context/candidate_obligation_ids')
    except MethodContractError as exc:
        return _rejected(memory, 'route_candidate_pool_invalid', proposal_id, error_code=exc.code)
    open_incidents = sorted((incident for incident in memory.incidents if incident.status == 'open'), key=lambda incident: (-incident.created_step, incident.incident_id.encode('utf-8')))
    open_obligations = sorted((obligation for obligation in memory.pending_obligations if obligation.status == 'open'), key=lambda obligation: (obligation.created_step, obligation.obligation_id.encode('utf-8')))
    expected_incidents = tuple((incident.incident_id for incident in open_incidents[:context.router_limits.max_incidents]))
    expected_obligations = tuple((obligation.obligation_id for obligation in open_obligations[:context.router_limits.max_obligations]))
    if context.mandatory_incident_ids != expected_incidents or context.mandatory_obligation_ids != expected_obligations:
        return _rejected(memory, 'route_mandatory_context_invalid', proposal_id)
    all_candidates = context.candidate_fact_ids + context.candidate_incident_ids + context.candidate_obligation_ids
    try:
        ensure_members(proposal['relevant_memory_ids'], all_candidates, pointer='/proposal/relevant_memory_ids')
    except MethodContractError as exc:
        return _rejected(memory, 'route_reference_invalid', proposal_id, error_code=exc.code)
    proposed_facts = tuple((item for item in proposal['relevant_memory_ids'] if item in fact_ids))
    proposed_incidents = tuple((item for item in proposal['relevant_memory_ids'] if item in incident_ids))
    proposed_obligations = tuple((item for item in proposal['relevant_memory_ids'] if item in obligation_ids))
    active_roles = set(context.active_roles)
    active_fact_ids = tuple((fact.fact_id for fact in sorted((fact for fact in memory.facts if fact.fact_id in context.candidate_fact_ids and (fact.subject_public_role in active_roles or (isinstance(fact.object_public_role_or_value, str) and fact.object_public_role_or_value in active_roles))), key=lambda fact: (-fact.observed_step, fact.fact_id.encode('utf-8')))))
    routed = RoutedMemoryIds(_dedupe(active_fact_ids + proposed_facts)[:context.router_limits.max_facts], _dedupe(context.mandatory_incident_ids + proposed_incidents)[:context.router_limits.max_incidents], _dedupe(context.mandatory_obligation_ids + proposed_obligations)[:context.router_limits.max_obligations])
    return ControllerDecision(True, 'route_valid_no_commit', memory, proposal_id, routed_memory_or_null=routed)

def _commit_observe_from_verification(memory: EvidenceMemory, proposal_id: str, context: ProposalContext, *, reason: str, capacity: MemoryCapacity) -> ControllerDecision:
    verification = context.verification_or_null
    if verification is None:
        return _rejected(memory, 'verification_context_missing', proposal_id)
    ambiguous_confidence = max(0.0, 1.0 - verification.success_probability - verification.failure_probability)
    lifecycle = apply_verification_lifecycle(memory, selected_outcome='ambiguous', canonical_action=verification.canonical_action, confidence=ambiguous_confidence, action_id=context.action_id, evidence_frame_ids=context.expected_evidence_frame_ids, public_roles=verification.public_roles, step=context.step, capacity=capacity)
    return ControllerDecision(lifecycle.accepted_by_harness, reason if lifecycle.accepted_by_harness else lifecycle.reason, lifecycle.memory, proposal_id)

def _evaluate_update_proposal(memory: EvidenceMemory, proposal: Mapping[str, Any], context: ProposalContext, thresholds: ControllerThresholds, capacity: MemoryCapacity) -> ControllerDecision:
    proposal_id = proposal['proposal_id']
    verification = context.verification_or_null
    if verification is None:
        return _rejected(memory, 'verification_context_missing', proposal_id)
    if context.candidate_fact_ids:
        return _rejected(memory, 'update_context_invalid', proposal_id)
    if proposal['per_fact_status'] or proposal['relevant_memory_ids']:
        return _rejected(memory, 'update_proposal_invalid', proposal_id)
    source_incident = None
    source_obligation = None
    if context.source_incident_id_or_null is not None:
        source_incident = next((incident for incident in memory.incidents if incident.incident_id == context.source_incident_id_or_null and incident.status == 'open'), None)
        matching_obligations = sorted((obligation for obligation in memory.pending_obligations if obligation.status == 'open' and obligation.type == 'act' and (obligation.source_incident_id_or_null == context.source_incident_id_or_null)), key=lambda obligation: (obligation.created_step, obligation.obligation_id.encode('utf-8')))
        if source_incident is None or len(matching_obligations) != 1 or matching_obligations[0].public_roles != verification.public_roles:
            return _rejected(memory, 'recovery_context_invalid', proposal_id)
        source_obligation = matching_obligations[0]
    if proposal['abstain_reason_or_null'] is not None:
        return _commit_observe_from_verification(memory, proposal_id, context, reason='proposal_abstained_observe_commit', capacity=capacity)
    effective_outcome = verification.effective_outcome
    if effective_outcome == 'ambiguous':
        return _commit_observe_from_verification(memory, proposal_id, context, reason='ambiguous_observe_commit', capacity=capacity)
    if effective_outcome == 'failure':
        if proposal['fact_mutations'] or proposal['obligation_mutations']:
            return _rejected(memory, 'failure_fact_or_obligation_forbidden', proposal_id)
        expected_failure_type = source_incident.failure_type if source_incident is not None else 'visual_effect_not_achieved/{}'.format(verification.canonical_action)
        if verification.failure_probability < thresholds.incident_threshold or len(proposal['incident_mutations']) != 1:
            return _commit_observe_from_verification(memory, proposal_id, context, reason='failure_without_valid_incident_observe_commit', capacity=capacity)
        mutation = proposal['incident_mutations'][0]
        if mutation['operation'] != ('update' if source_incident is not None else 'create') or mutation['failure_type'] != expected_failure_type or tuple(mutation['public_roles']) != verification.public_roles or (mutation['confidence'] < thresholds.incident_threshold):
            return _rejected(memory, 'failure_incident_invalid', proposal_id)
        if source_incident is not None:
            incidents = tuple((replace(incident, confidence=verification.failure_probability, evidence_frame_ids=context.expected_evidence_frame_ids, last_updated_step=context.step, status='open', recovery_attempts=incident.recovery_attempts + 1) if incident.incident_id == source_incident.incident_id else incident for incident in memory.incidents))
            updated = EvidenceMemory(memory.episode_id, memory.revision + 1, memory.facts, incidents, memory.pending_obligations)
            return ControllerDecision(True, 'proposal_valid_commit', enforce_capacity(updated, capacity), proposal_id)
        lifecycle = apply_verification_lifecycle(memory, selected_outcome='failure', canonical_action=verification.canonical_action, confidence=verification.failure_probability, action_id=context.action_id, evidence_frame_ids=context.expected_evidence_frame_ids, public_roles=verification.public_roles, step=context.step, capacity=capacity)
        return ControllerDecision(lifecycle.accepted_by_harness, 'proposal_valid_commit' if lifecycle.accepted_by_harness else lifecycle.reason, lifecycle.memory, proposal_id)
    if verification.success_probability < thresholds.commit_threshold:
        return _commit_observe_from_verification(memory, proposal_id, context, reason='success_below_threshold_observe_commit', capacity=capacity)
    if source_incident is None:
        if proposal['incident_mutations'] or proposal['obligation_mutations']:
            return _rejected(memory, 'success_incident_or_obligation_forbidden', proposal_id)
    else:
        if len(proposal['incident_mutations']) != 1:
            return _rejected(memory, 'recovery_incident_invalid', proposal_id)
        incident_mutation = proposal['incident_mutations'][0]
        if incident_mutation['operation'] != 'resolve' or incident_mutation['failure_type'] != source_incident.failure_type or tuple(incident_mutation['public_roles']) != verification.public_roles or (incident_mutation['confidence'] < thresholds.incident_threshold):
            return _rejected(memory, 'recovery_incident_invalid', proposal_id)
        if len(proposal['obligation_mutations']) != 1:
            return _rejected(memory, 'recovery_obligation_invalid', proposal_id)
        obligation_mutation = proposal['obligation_mutations'][0]
        if obligation_mutation['operation'] != 'resolve' or obligation_mutation['type'] != 'act' or tuple(obligation_mutation['public_roles']) != source_obligation.public_roles or (obligation_mutation['source_incident_id_or_null'] != source_incident.incident_id):
            return _rejected(memory, 'recovery_obligation_invalid', proposal_id)
    expected_mutations: dict[str, str] = {}
    for binding in verification.effect_bindings:
        if binding.effect not in verification.selected_effects or binding.binding_status != 'known':
            continue
        identifier = fact_id_for(**binding_fact_key(binding))
        if identifier in expected_mutations:
            return _rejected(memory, 'effect_binding_fact_conflict', proposal_id, error_code='method_memory_mutation_conflict')
        if binding.effect in {'relation_removed_inside', 'relation_removed_on_top'}:
            operation = 'invalidate'
        else:
            operation = 'confirm' if any((fact.fact_id == identifier for fact in memory.facts)) else 'add'
        expected_mutations[identifier] = operation
    actual_mutations = {}
    for mutation in proposal['fact_mutations']:
        identifier = fact_id_for(mutation['public_entity_ref'], mutation['relation_or_property'], mutation['object_public_role_or_value'])
        actual_mutations[identifier] = mutation
    if set(actual_mutations) != set(expected_mutations):
        return _commit_observe_from_verification(memory, proposal_id, context, reason='binding_proposal_incomplete_observe_commit', capacity=capacity)
    for identifier, operation in expected_mutations.items():
        mutation = actual_mutations[identifier]
        if mutation['operation'] != operation or mutation['confidence'] < thresholds.commit_threshold:
            return _rejected(memory, 'binding_proposal_invalid', proposal_id)
    result = commit_effect_bindings(memory, verification.effect_bindings, verification.selection, success_probability=verification.success_probability, effect_probabilities=dict(verification.effect_probabilities), action_id=context.action_id, evidence_frame_ids=context.expected_evidence_frame_ids, step=context.step, capacity=capacity, observation_public_roles=verification.public_roles)
    result_memory = result.memory
    if result.accepted_by_harness and source_incident is not None:
        result_memory = EvidenceMemory(result.memory.episode_id, result.memory.revision, result.memory.facts, tuple((replace(incident, status='resolved', last_updated_step=context.step) if incident.incident_id == source_incident.incident_id else incident for incident in result.memory.incidents)), tuple((replace(obligation, status='resolved') if obligation.obligation_id == source_obligation.obligation_id else obligation for obligation in result.memory.pending_obligations))).validated()
    return ControllerDecision(result.accepted_by_harness, 'proposal_valid_commit' if result.accepted_by_harness else result.reason, result_memory, proposal_id)

def _evaluate_memory_proposal_uncorrelated(memory: EvidenceMemory, proposal_value: Mapping[str, Any], context: ProposalContext, *, thresholds: ControllerThresholds=ControllerThresholds(), capacity: MemoryCapacity=MemoryCapacity()) -> ControllerDecision:
    memory = memory.validated()
    proposal_id = proposal_value.get('proposal_id') if isinstance(proposal_value, Mapping) else None
    try:
        proposal = validate_memory_proposal(proposal_value)
    except MethodContractError as exc:
        return _rejected(memory, 'proposal_invalid', proposal_id, error_code=exc.code)
    proposal_id = proposal['proposal_id']
    if proposal['base_revision'] != memory.revision:
        return _rejected(memory, 'revision_mismatch', proposal_id)
    if tuple(proposal['evidence_frame_ids']) != context.expected_evidence_frame_ids:
        return _rejected(memory, 'evidence_frame_mismatch', proposal_id)
    if proposal['mode'] == 'update':
        return _evaluate_update_proposal(memory, proposal, context, thresholds, capacity)
    if proposal['mode'] == 'observe':
        if context.verification_or_null is not None or context.source_incident_id_or_null is not None:
            return _rejected(memory, 'observe_context_invalid', proposal_id)
        if proposal['abstain_reason_or_null'] is not None:
            return _rejected(memory, 'observe_empty', proposal_id)
        facts_by_id = {fact.fact_id: fact for fact in memory.facts}
        changed = False
        for mutation in proposal['fact_mutations']:
            identifier = fact_id_for(mutation['public_entity_ref'], mutation['relation_or_property'], mutation['object_public_role_or_value'])
            existing = facts_by_id.get(identifier)
            operation = mutation['operation']
            if mutation['confidence'] < thresholds.commit_threshold:
                return _rejected(memory, 'mutation_below_threshold', proposal_id)
            if operation in {'confirm', 'invalidate'} and existing is None:
                return _rejected(memory, 'fact_reference_dangling', proposal_id)
            if operation == 'invalidate':
                facts_by_id[identifier] = replace(existing, confidence=mutation['confidence'], source_frame_ids=tuple(proposal['evidence_frame_ids']), source_action_id=context.action_id, observed_step=context.step, status='contradicted')
            else:
                for fact_id, fact in tuple(facts_by_id.items()):
                    if fact_id != identifier and fact.status == 'current' and (fact.relation_or_property == SCENE_OBSERVATION_RELATION) and (fact.subject_public_role == mutation['public_entity_ref']):
                        facts_by_id[fact_id] = replace(fact, status='stale')
                facts_by_id[identifier] = EvidenceFact.create(subject_public_role=mutation['public_entity_ref'], relation_or_property=SCENE_OBSERVATION_RELATION, object_public_role_or_value=mutation['object_public_role_or_value'], confidence=mutation['confidence'], source_frame_ids=proposal['evidence_frame_ids'], source_action_id=context.action_id, observed_step=context.step)
            changed = True
        if not changed:
            return _rejected(memory, 'observe_empty', proposal_id)
        scene_facts = sorted((fact for fact in facts_by_id.values() if fact.relation_or_property == SCENE_OBSERVATION_RELATION), key=lambda fact: (-fact.observed_step, fact.fact_id.encode('utf-8')))
        retained_scene_ids = {fact.fact_id for fact in scene_facts[:OBSERVE_MAX_RETAINED_FACTS]}
        updated = EvidenceMemory(memory.episode_id, memory.revision + 1, tuple(sorted((fact for fact in facts_by_id.values() if fact.relation_or_property != SCENE_OBSERVATION_RELATION or fact.fact_id in retained_scene_ids), key=lambda fact: fact.fact_id.encode('utf-8'))), memory.incidents, memory.pending_obligations)
        return ControllerDecision(True, 'observe_valid_commit', enforce_capacity(updated, capacity), proposal_id)
    if context.verification_or_null is not None:
        return _rejected(memory, 'non_update_verification_context_forbidden', proposal_id)
    if proposal['abstain_reason_or_null'] is not None:
        return _rejected(memory, 'proposal_abstained', proposal_id)
    if proposal['mode'] == 'route':
        return _evaluate_route_proposal(memory, proposal, context)
    if context.source_incident_id_or_null is not None:
        return _rejected(memory, 'recovery_verification_required', proposal_id)
    if context.selected_observability in {'unobservable', 'unknown'} and any((proposal[field] for field in ('fact_mutations', 'incident_mutations', 'obligation_mutations'))):
        return _rejected(memory, 'unobservable_mutation_forbidden', proposal_id)
    facts_by_id = {fact.fact_id: fact for fact in memory.facts}
    incidents_by_id = {incident.incident_id: incident for incident in memory.incidents}
    obligations = list(memory.pending_obligations)
    statuses = {item['fact_id']: item['status'] for item in proposal['per_fact_status']}
    if proposal['mode'] == 'reconcile':
        if tuple(statuses) != context.candidate_fact_ids:
            return _rejected(memory, 'reconcile_candidate_mismatch', proposal_id)
        if any((fact_id not in facts_by_id for fact_id in statuses)):
            return _rejected(memory, 'reconcile_fact_dangling', proposal_id)
    try:
        ensure_members([mutation['public_entity_ref'] for mutation in proposal['fact_mutations']], context.allowed_public_entity_refs, pointer='/proposal/fact_mutations')
        ensure_members([mutation['object_public_role_or_value'] for mutation in proposal['fact_mutations'] if mutation['relation_or_property'] in {'inside', 'on_top'} and isinstance(mutation['object_public_role_or_value'], str)], context.allowed_public_entity_refs, pointer='/proposal/fact_mutations/object_public_role_or_value')
        ensure_members([role for mutation in proposal['incident_mutations'] + proposal['obligation_mutations'] for role in mutation['public_roles']], context.allowed_public_entity_refs, pointer='/proposal/public_roles')
    except MethodContractError as exc:
        return _rejected(memory, 'public_reference_invalid', proposal_id, error_code=exc.code)
    retiring_fact_ids = {fact_id_for(mutation['public_entity_ref'], mutation['relation_or_property'], mutation['object_public_role_or_value']) for mutation in proposal['fact_mutations'] if mutation['operation'] in {'invalidate', 'delete'}}
    for mutation in proposal['fact_mutations']:
        if mutation['operation'] not in {'add', 'confirm'}:
            continue
        identifier = fact_id_for(mutation['public_entity_ref'], mutation['relation_or_property'], mutation['object_public_role_or_value'])
        conflicts = (fact for fact in facts_by_id.values() if fact.status == 'current' and fact.fact_id != identifier and (fact.subject_public_role == mutation['public_entity_ref']) and (fact.relation_or_property == mutation['relation_or_property']))
        if any((fact.fact_id not in retiring_fact_ids for fact in conflicts)):
            return _rejected(memory, 'fact_conflict_unresolved', proposal_id, error_code='method_memory_mutation_conflict')
    changed = False
    for mutation in proposal['fact_mutations']:
        identifier = fact_id_for(mutation['public_entity_ref'], mutation['relation_or_property'], mutation['object_public_role_or_value'])
        operation = mutation['operation']
        status = statuses.get(identifier)
        if status in {'unobservable', 'unknown'}:
            return _rejected(memory, 'unobservable_mutation_forbidden', proposal_id)
        if proposal['mode'] == 'reconcile' and status is not None:
            expected_operation = {'confirm': 'confirm', 'contradict': 'invalidate'}.get(status)
            if operation != expected_operation:
                return _rejected(memory, 'reconcile_status_conflict', proposal_id)
        required_threshold = thresholds.contradiction_threshold if operation == 'invalidate' else thresholds.commit_threshold
        if mutation['confidence'] < required_threshold:
            return _rejected(memory, 'mutation_below_threshold', proposal_id)
        existing = facts_by_id.get(identifier)
        if operation in {'confirm', 'invalidate', 'delete'} and existing is None:
            return _rejected(memory, 'fact_reference_dangling', proposal_id)
        if operation == 'delete':
            del facts_by_id[identifier]
        elif operation == 'invalidate':
            facts_by_id[identifier] = replace(existing, confidence=mutation['confidence'], source_frame_ids=tuple(proposal['evidence_frame_ids']), source_action_id=context.action_id, observed_step=context.step, status='contradicted')
        else:
            facts_by_id[identifier] = EvidenceFact.create(subject_public_role=mutation['public_entity_ref'], relation_or_property=mutation['relation_or_property'], object_public_role_or_value=mutation['object_public_role_or_value'], confidence=mutation['confidence'], source_frame_ids=proposal['evidence_frame_ids'], source_action_id=context.action_id, observed_step=context.step)
        changed = True
    for mutation in proposal['incident_mutations']:
        if mutation['confidence'] < thresholds.incident_threshold:
            return _rejected(memory, 'mutation_below_threshold', proposal_id)
        operation = mutation['operation']
        if operation == 'create':
            incident = RecoveryIncident.create(episode_id=memory.episode_id, action_id=context.action_id, failure_type=mutation['failure_type'], confidence=mutation['confidence'], evidence_frame_ids=proposal['evidence_frame_ids'], step=context.step)
            existing = incidents_by_id.get(incident.incident_id)
            if existing is None:
                incidents_by_id[incident.incident_id] = incident
            else:
                incidents_by_id[incident.incident_id] = replace(existing, confidence=mutation['confidence'], evidence_frame_ids=tuple(proposal['evidence_frame_ids']), last_updated_step=context.step, status='open')
            changed = True
        else:
            target_id = context.source_incident_id_or_null
            target = incidents_by_id.get(target_id) if target_id is not None else None
            if target is None or target.failure_type != mutation['failure_type']:
                return _rejected(memory, 'incident_reference_dangling', proposal_id)
            incidents_by_id[target.incident_id] = replace(target, confidence=mutation['confidence'], evidence_frame_ids=tuple(proposal['evidence_frame_ids']), last_updated_step=context.step, status='resolved' if operation == 'resolve' else 'open', recovery_attempts=target.recovery_attempts + (operation == 'update'))
            changed = True
    for mutation in proposal['obligation_mutations']:
        source_incident_id = mutation['source_incident_id_or_null']
        if source_incident_id is not None and source_incident_id not in incidents_by_id:
            return _rejected(memory, 'obligation_incident_dangling', proposal_id)
        roles = tuple(mutation['public_roles'])
        if mutation['operation'] == 'create':
            matching = [obligation for obligation in obligations if obligation.status == 'open' and obligation.type == mutation['type'] and (obligation.public_roles == roles) and (obligation.source_incident_id_or_null == source_incident_id)]
            if not matching:
                candidate = PendingObligation.create(episode_id=memory.episode_id, obligation_type=mutation['type'], public_roles=roles, created_step=context.step, source_incident_id_or_null=source_incident_id)
                existing_obligation = next((item for item in obligations if item.obligation_id == candidate.obligation_id), None)
                if existing_obligation is None:
                    obligations.append(candidate)
                else:
                    obligations[obligations.index(existing_obligation)] = replace(existing_obligation, status='open')
                changed = True
        else:
            matched = False
            for index, obligation in enumerate(obligations):
                if obligation.status == 'open' and obligation.type == mutation['type'] and (obligation.public_roles == roles) and (obligation.source_incident_id_or_null == source_incident_id):
                    obligations[index] = replace(obligation, status='resolved')
                    matched = True
            if not matched:
                return _rejected(memory, 'obligation_reference_dangling', proposal_id)
            changed = True
    if not changed:
        return _rejected(memory, 'proposal_no_effect', proposal_id)
    updated = EvidenceMemory(memory.episode_id, memory.revision + 1, tuple(sorted(facts_by_id.values(), key=lambda item: item.fact_id.encode('utf-8'))), tuple(sorted(incidents_by_id.values(), key=lambda item: item.incident_id.encode('utf-8'))), tuple(sorted(obligations, key=lambda item: item.obligation_id.encode('utf-8'))))
    updated = enforce_capacity(updated, capacity)
    return ControllerDecision(True, 'proposal_valid_commit', updated, proposal_id)

def evaluate_memory_proposal(memory: EvidenceMemory, proposal_value: Mapping[str, Any], context: ProposalContext, *, thresholds: ControllerThresholds=ControllerThresholds(), capacity: MemoryCapacity=MemoryCapacity(), harness_version: str | None=None) -> ControllerDecision:
    if harness_version in {'18', '19', '20', 'scoper-1'} and isinstance(proposal_value, Mapping):
        from scoper.harness.control_prompts import bound_observation_proposal
        proposal_value = bound_observation_proposal(proposal_value)
    decision = _evaluate_memory_proposal_uncorrelated(memory, proposal_value, context, thresholds=thresholds, capacity=capacity)
    proposal_mode = proposal_value.get('mode') if isinstance(proposal_value, Mapping) else None
    if proposal_mode not in {'update', 'reconcile', 'route', 'observe'}:
        proposal_mode = None
    base_revision = proposal_value.get('base_revision') if isinstance(proposal_value, Mapping) else None
    if isinstance(base_revision, bool) or not isinstance(base_revision, int) or base_revision < 0:
        base_revision = None
    return replace(decision, proposal_mode_or_null=proposal_mode, parent_action_id_or_null=context.action_id, base_revision_or_null=base_revision)

def apply_verification_lifecycle(memory: EvidenceMemory, *, selected_outcome: str, canonical_action: str, confidence: float, action_id: str, evidence_frame_ids: Sequence[str], public_roles: Sequence[str], step: int, capacity: MemoryCapacity=MemoryCapacity()) -> ControllerDecision:
    memory = memory.validated()
    if selected_outcome not in {'success', 'failure', 'ambiguous'}:
        raise MethodContractError('method_schema_invalid', 'verification outcome is invalid')
    if selected_outcome == 'success':
        return _rejected(memory, 'verification_success_has_no_lifecycle_mutation', None)
    action = non_empty_string(canonical_action, pointer='/canonical_action')
    action_identifier = non_empty_string(action_id, pointer='/action_id')
    frames = string_sequence(list(evidence_frame_ids), pointer='/evidence_frame_ids', non_empty=True)
    roles = string_sequence(list(public_roles), pointer='/public_roles', non_empty=True, sorted_utf8=True)
    current_step = integer(step, pointer='/step')
    score = probability(confidence, pointer='/confidence')
    incidents = list(memory.incidents)
    obligations = list(memory.pending_obligations)
    changed = False
    if selected_outcome == 'failure':
        failure_type = 'visual_effect_not_achieved/{}'.format(action)
        incident = RecoveryIncident.create(episode_id=memory.episode_id, action_id=action_identifier, failure_type=failure_type, confidence=score, evidence_frame_ids=frames, step=current_step)
        existing = next((item for item in incidents if item.incident_id == incident.incident_id), None)
        if existing is None:
            incidents.append(incident)
            changed = True
        else:
            updated_incident = replace(existing, confidence=score, evidence_frame_ids=frames, last_updated_step=current_step, status='open')
            if updated_incident != existing:
                incidents[incidents.index(existing)] = updated_incident
                changed = True
        if not any((item.status == 'open' and item.type == 'act' and (item.public_roles == roles) and (item.source_incident_id_or_null == incident.incident_id) for item in obligations)):
            candidate = PendingObligation.create(episode_id=memory.episode_id, obligation_type='act', public_roles=roles, created_step=current_step, source_incident_id_or_null=incident.incident_id)
            existing_obligation = next((item for item in obligations if item.obligation_id == candidate.obligation_id), None)
            if existing_obligation is None:
                obligations.append(candidate)
            else:
                obligations[obligations.index(existing_obligation)] = replace(existing_obligation, status='open')
            changed = True
    elif not any((item.status == 'open' and item.type == 'observe' and (item.public_roles == roles) for item in obligations)):
        candidate = PendingObligation.create(episode_id=memory.episode_id, obligation_type='observe', public_roles=roles, created_step=current_step)
        existing_obligation = next((item for item in obligations if item.obligation_id == candidate.obligation_id), None)
        if existing_obligation is None:
            obligations.append(candidate)
        else:
            obligations[obligations.index(existing_obligation)] = replace(existing_obligation, status='open')
        changed = True
    if not changed:
        return _rejected(memory, 'verification_lifecycle_no_effect', None)
    updated = EvidenceMemory(memory.episode_id, memory.revision + 1, memory.facts, tuple(sorted(incidents, key=lambda item: item.incident_id.encode('utf-8'))), tuple(sorted(obligations, key=lambda item: item.obligation_id.encode('utf-8'))))
    updated = enforce_capacity(updated, capacity)
    return ControllerDecision(True, 'verification_lifecycle_commit', updated, None)
__all__ = ['ControllerDecision', 'ControllerThresholds', 'ProposalContext', 'RoutedMemoryIds', 'VerificationCommitContext', 'apply_verification_lifecycle', 'evaluate_memory_proposal', 'validate_memory_proposal']
