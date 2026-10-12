from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
from scoper.harness.canonical import sha256_json
from scoper.harness.contracts import MethodContractError
from scoper.harness.validation import exact_mapping, integer, non_empty_string, optional_string, probability, public_json, public_semantic_name, string_sequence
from scoper.harness.effect_binding import EffectBinding, SCENE_OBSERVATION_RELATION, binding_fact_key, validate_effect_bindings, validate_fact_semantics
from scoper.harness.verifier import VerifierSelection
FACT_FIELDS = frozenset({'fact_id', 'subject_public_role', 'relation_or_property', 'object_public_role_or_value', 'confidence', 'source_frame_ids', 'source_action_id', 'observed_step', 'status'})
INCIDENT_FIELDS = frozenset({'incident_id', 'action_id', 'failure_type', 'confidence', 'evidence_frame_ids', 'created_step', 'last_updated_step', 'status', 'recovery_attempts'})
OBLIGATION_FIELDS = frozenset({'obligation_id', 'type', 'public_roles', 'created_step', 'source_incident_id_or_null', 'status'})
MEMORY_FIELDS = frozenset({'episode_id', 'revision', 'facts', 'incidents', 'pending_obligations'})
CANONICAL_PHYSICAL_ACTIONS = ('pick_up', 'put_down', 'open', 'close', 'toggle_on', 'toggle_off', 'slice', 'clean', 'saturate', 'fill', 'heat', 'cool')
CANONICAL_FAILURE_TYPES = frozenset({'public_interaction_not_ready', 'unresolved_visual_effect'} | {'visual_effect_not_achieved/{}'.format(action) for action in CANONICAL_PHYSICAL_ACTIONS})

def validate_failure_type(value: Any, *, pointer: str='/incident/failure_type') -> str:
    failure_type = public_semantic_name(value, pointer=pointer)
    if failure_type not in CANONICAL_FAILURE_TYPES:
        raise MethodContractError('method_memory_vocabulary_invalid', 'incident failure type is outside the frozen vocabulary', pointer=pointer)
    return failure_type

def fact_id_for(subject_public_role: str, relation_or_property: str, object_public_role_or_value: Any) -> str:
    key = {'subject_public_role': public_semantic_name(subject_public_role, pointer='/fact/subject_public_role')}
    relation, object_value = validate_fact_semantics(relation_or_property, object_public_role_or_value, pointer='/fact', allow_scene_observation=relation_or_property == SCENE_OBSERVATION_RELATION)
    key['relation_or_property'] = relation
    key['object_public_role_or_value'] = object_value
    return 'fact-{}'.format(sha256_json(key)[:16])

def incident_id_for(episode_id: str, action_id: str, failure_type: str) -> str:
    key = {'episode_id': non_empty_string(episode_id, pointer='/incident/episode_id'), 'action_id': non_empty_string(action_id, pointer='/incident/action_id'), 'failure_type': validate_failure_type(failure_type)}
    return 'incident-{}'.format(sha256_json(key)[:16])

def obligation_id_for(episode_id: str, obligation_type: str, public_roles: Sequence[str], source_incident_id_or_null: str | None, created_step: int) -> str:
    roles = string_sequence(list(public_roles), pointer='/obligation/public_roles', non_empty=True, sorted_utf8=True)
    if obligation_type not in {'observe', 'act'}:
        raise MethodContractError('method_schema_invalid', 'obligation type is invalid')
    key = {'episode_id': non_empty_string(episode_id, pointer='/obligation/episode_id'), 'type': obligation_type, 'public_roles': list(roles), 'source_incident_id_or_null': optional_string(source_incident_id_or_null, pointer='/obligation/source_incident_id_or_null'), 'created_step': integer(created_step, pointer='/obligation/created_step')}
    return 'obl-{}'.format(sha256_json(key)[:16])

def facts_are_mutually_exclusive(*, existing_subject: str, existing_relation: str, existing_value: Any, incoming_subject: str, incoming_relation: str, incoming_value: Any) -> bool:
    if existing_subject == incoming_subject and existing_relation == incoming_relation:
        return existing_value != incoming_value
    return existing_relation == incoming_relation == 'held_by_agent' and existing_value is True and (incoming_value is True)

@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    subject_public_role: str
    relation_or_property: str
    object_public_role_or_value: Any
    confidence: float
    source_frame_ids: tuple[str, ...]
    source_action_id: str
    observed_step: int
    status: str = 'current'

    @classmethod
    def create(cls, *, subject_public_role: str, relation_or_property: str, object_public_role_or_value: Any, confidence: float, source_frame_ids: Sequence[str], source_action_id: str, observed_step: int, status: str='current') -> 'EvidenceFact':
        subject = public_semantic_name(subject_public_role, pointer='/fact/subject_public_role')
        relation, object_value = validate_fact_semantics(relation_or_property, object_public_role_or_value, pointer='/fact', allow_scene_observation=relation_or_property == SCENE_OBSERVATION_RELATION)
        frames = string_sequence(list(source_frame_ids), pointer='/fact/source_frame_ids', non_empty=True)
        action_id = non_empty_string(source_action_id, pointer='/fact/source_action_id')
        step = integer(observed_step, pointer='/fact/observed_step')
        if status not in {'current', 'stale', 'contradicted'}:
            raise MethodContractError('method_schema_invalid', 'fact status is invalid')
        fact = cls(fact_id_for(subject, relation, object_value), subject, relation, object_value, probability(confidence, pointer='/fact/confidence'), frames, action_id, step, status)
        public_json(fact.to_mapping(), pointer='/fact', include_future_action=True)
        return fact

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'EvidenceFact':
        item = exact_mapping(value, FACT_FIELDS, pointer='/fact')
        fact = cls.create(subject_public_role=item['subject_public_role'], relation_or_property=item['relation_or_property'], object_public_role_or_value=item['object_public_role_or_value'], confidence=item['confidence'], source_frame_ids=item['source_frame_ids'], source_action_id=item['source_action_id'], observed_step=item['observed_step'], status=item['status'])
        if item['fact_id'] != fact.fact_id:
            raise MethodContractError('method_memory_identity_invalid', 'fact_id differs from its fact key')
        return fact

    def to_mapping(self) -> dict[str, Any]:
        return {'fact_id': self.fact_id, 'subject_public_role': self.subject_public_role, 'relation_or_property': self.relation_or_property, 'object_public_role_or_value': self.object_public_role_or_value, 'confidence': self.confidence, 'source_frame_ids': list(self.source_frame_ids), 'source_action_id': self.source_action_id, 'observed_step': self.observed_step, 'status': self.status}

@dataclass(frozen=True)
class RecoveryIncident:
    incident_id: str
    action_id: str
    failure_type: str
    confidence: float
    evidence_frame_ids: tuple[str, ...]
    created_step: int
    last_updated_step: int
    status: str
    recovery_attempts: int

    @classmethod
    def create(cls, *, episode_id: str, action_id: str, failure_type: str, confidence: float, evidence_frame_ids: Sequence[str], step: int, status: str='open', recovery_attempts: int=0) -> 'RecoveryIncident':
        action = non_empty_string(action_id, pointer='/incident/action_id')
        failure = validate_failure_type(failure_type)
        frames = string_sequence(list(evidence_frame_ids), pointer='/incident/evidence_frame_ids', non_empty=True)
        created = integer(step, pointer='/incident/created_step')
        if status not in {'open', 'resolved'}:
            raise MethodContractError('method_schema_invalid', 'incident status is invalid')
        incident = cls(incident_id_for(episode_id, action, failure), action, failure, probability(confidence, pointer='/incident/confidence'), frames, created, created, status, integer(recovery_attempts, pointer='/incident/recovery_attempts'))
        public_json(incident.to_mapping(), pointer='/incident', include_future_action=True)
        return incident

    @classmethod
    def from_mapping(cls, episode_id: str, value: Mapping[str, Any]) -> 'RecoveryIncident':
        item = exact_mapping(value, INCIDENT_FIELDS, pointer='/incident')
        created = cls.create(episode_id=episode_id, action_id=item['action_id'], failure_type=item['failure_type'], confidence=item['confidence'], evidence_frame_ids=item['evidence_frame_ids'], step=item['created_step'], status=item['status'], recovery_attempts=item['recovery_attempts'])
        if item['incident_id'] != created.incident_id:
            raise MethodContractError('method_memory_identity_invalid', 'incident_id differs from its identity')
        updated = integer(item['last_updated_step'], pointer='/incident/last_updated_step')
        if updated < created.created_step:
            raise MethodContractError('method_schema_invalid', 'incident update precedes creation', pointer='/incident')
        return replace(created, last_updated_step=updated)

    def to_mapping(self) -> dict[str, Any]:
        return {'incident_id': self.incident_id, 'action_id': self.action_id, 'failure_type': self.failure_type, 'confidence': self.confidence, 'evidence_frame_ids': list(self.evidence_frame_ids), 'created_step': self.created_step, 'last_updated_step': self.last_updated_step, 'status': self.status, 'recovery_attempts': self.recovery_attempts}

@dataclass(frozen=True)
class PendingObligation:
    obligation_id: str
    type: str
    public_roles: tuple[str, ...]
    created_step: int
    source_incident_id_or_null: str | None
    status: str

    @classmethod
    def create(cls, *, episode_id: str, obligation_type: str, public_roles: Sequence[str], created_step: int, source_incident_id_or_null: str | None=None, status: str='open') -> 'PendingObligation':
        roles = string_sequence(list(public_roles), pointer='/obligation/public_roles', non_empty=True, sorted_utf8=True)
        step = integer(created_step, pointer='/obligation/created_step')
        incident_id = optional_string(source_incident_id_or_null, pointer='/obligation/source_incident_id_or_null')
        if obligation_type not in {'observe', 'act'} or status not in {'open', 'resolved'}:
            raise MethodContractError('method_schema_invalid', 'obligation type/status is invalid')
        obligation = cls(obligation_id_for(episode_id, obligation_type, roles, incident_id, step), obligation_type, roles, step, incident_id, status)
        public_json(obligation.to_mapping(), pointer='/obligation', include_future_action=True)
        return obligation

    @classmethod
    def from_mapping(cls, episode_id: str, value: Mapping[str, Any]) -> 'PendingObligation':
        item = exact_mapping(value, OBLIGATION_FIELDS, pointer='/obligation')
        obligation = cls.create(episode_id=episode_id, obligation_type=item['type'], public_roles=item['public_roles'], created_step=item['created_step'], source_incident_id_or_null=item['source_incident_id_or_null'], status=item['status'])
        if item['obligation_id'] != obligation.obligation_id:
            raise MethodContractError('method_memory_identity_invalid', 'obligation_id differs from its identity')
        return obligation

    def to_mapping(self) -> dict[str, Any]:
        return {'obligation_id': self.obligation_id, 'type': self.type, 'public_roles': list(self.public_roles), 'created_step': self.created_step, 'source_incident_id_or_null': self.source_incident_id_or_null, 'status': self.status}

@dataclass(frozen=True)
class MemoryCapacity:
    max_facts: int = 64
    max_incidents: int = 16
    max_obligations: int = 16

    def __post_init__(self) -> None:
        integer(self.max_facts, pointer='/memory/max_facts', minimum=1)
        integer(self.max_incidents, pointer='/memory/max_incidents', minimum=1)
        integer(self.max_obligations, pointer='/memory/max_obligations', minimum=1)

@dataclass(frozen=True)
class EvidenceMemory:
    episode_id: str
    revision: int
    facts: tuple[EvidenceFact, ...]
    incidents: tuple[RecoveryIncident, ...]
    pending_obligations: tuple[PendingObligation, ...]

    @classmethod
    def empty(cls, episode_id: str) -> 'EvidenceMemory':
        return cls(non_empty_string(episode_id, pointer='/memory/episode_id'), 0, (), (), ())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'EvidenceMemory':
        item = exact_mapping(value, MEMORY_FIELDS, pointer='/memory')
        episode_id = non_empty_string(item['episode_id'], pointer='/memory/episode_id')
        revision = integer(item['revision'], pointer='/memory/revision')
        if not all((isinstance(item[name], list) for name in ('facts', 'incidents', 'pending_obligations'))):
            raise MethodContractError('method_schema_invalid', 'memory collections must be lists')
        facts = tuple((EvidenceFact.from_mapping(row) for row in item['facts']))
        incidents = tuple((RecoveryIncident.from_mapping(episode_id, row) for row in item['incidents']))
        obligations = tuple((PendingObligation.from_mapping(episode_id, row) for row in item['pending_obligations']))
        cls._validate_collection(facts, 'fact_id', '/memory/facts')
        cls._validate_collection(incidents, 'incident_id', '/memory/incidents')
        cls._validate_collection(obligations, 'obligation_id', '/memory/pending_obligations')
        incident_ids = {incident.incident_id for incident in incidents}
        dangling = sorted({obligation.source_incident_id_or_null for obligation in obligations if obligation.source_incident_id_or_null is not None and obligation.source_incident_id_or_null not in incident_ids})
        if dangling:
            raise MethodContractError('method_public_reference_invalid', 'obligations reference missing incidents')
        memory = cls(episode_id, revision, facts, incidents, obligations)
        public_json(memory.to_mapping(), pointer='/memory', include_future_action=True)
        return memory

    @staticmethod
    def _validate_collection(values: Sequence[Any], id_field: str, pointer: str) -> None:
        identifiers = [getattr(value, id_field) for value in values]
        if identifiers != sorted(identifiers, key=lambda item: item.encode('utf-8')):
            raise MethodContractError('method_schema_invalid', 'memory collection is not sorted', pointer=pointer)
        if len(set(identifiers)) != len(identifiers):
            raise MethodContractError('method_memory_identity_invalid', 'memory IDs are duplicated', pointer=pointer)

    def to_mapping(self) -> dict[str, Any]:
        return {'episode_id': self.episode_id, 'revision': self.revision, 'facts': [fact.to_mapping() for fact in self.facts], 'incidents': [incident.to_mapping() for incident in self.incidents], 'pending_obligations': [obligation.to_mapping() for obligation in self.pending_obligations]}

    def validated(self) -> 'EvidenceMemory':
        return EvidenceMemory.from_mapping(self.to_mapping())

@dataclass(frozen=True)
class MemoryCommitResult:
    accepted_by_harness: bool
    reason: str
    memory: EvidenceMemory

    def event_payload(self) -> dict[str, Any]:
        return {'accepted_by_harness': self.accepted_by_harness, 'revision': self.memory.revision, 'facts': [fact.to_mapping() for fact in self.memory.facts], 'incidents': [incident.to_mapping() for incident in self.memory.incidents], 'pending_obligations': [obligation.to_mapping() for obligation in self.memory.pending_obligations]}

def _sort_memory(memory: EvidenceMemory, *, facts: Sequence[EvidenceFact] | None=None, incidents: Sequence[RecoveryIncident] | None=None, obligations: Sequence[PendingObligation] | None=None, revision: int | None=None) -> EvidenceMemory:
    return EvidenceMemory(memory.episode_id, memory.revision if revision is None else revision, tuple(sorted(memory.facts if facts is None else facts, key=lambda item: item.fact_id.encode('utf-8'))), tuple(sorted(memory.incidents if incidents is None else incidents, key=lambda item: item.incident_id.encode('utf-8'))), tuple(sorted(memory.pending_obligations if obligations is None else obligations, key=lambda item: item.obligation_id.encode('utf-8'))))

def enforce_capacity(memory: EvidenceMemory, capacity: MemoryCapacity) -> EvidenceMemory:
    facts = list(memory.facts)
    incidents = list(memory.incidents)
    obligations = list(memory.pending_obligations)
    while len(incidents) > capacity.max_incidents:
        candidates = sorted((item for item in incidents if item.status == 'resolved' and (not any((obligation.source_incident_id_or_null == item.incident_id and obligation.status != 'resolved' for obligation in obligations)))), key=lambda item: item.incident_id.encode('utf-8'))
        if not candidates:
            raise MethodContractError('method_memory_capacity_exceeded', 'open incidents exceed runtime capacity')
        removed_incident = candidates[0]
        incidents.remove(removed_incident)
        obligations = [item for item in obligations if item.source_incident_id_or_null != removed_incident.incident_id]
    while len(obligations) > capacity.max_obligations:
        candidates = sorted((item for item in obligations if item.status == 'resolved'), key=lambda item: item.obligation_id.encode('utf-8'))
        if not candidates:
            raise MethodContractError('method_memory_capacity_exceeded', 'open obligations exceed runtime capacity')
        obligations.remove(candidates[0])
    while len(facts) > capacity.max_facts:
        priority = {'stale': 0, 'contradicted': 1, 'current': 2}
        candidates = sorted(facts, key=lambda item: (priority[item.status], item.observed_step if item.status == 'current' else 0, item.fact_id.encode('utf-8')))
        facts.remove(candidates[0])
    return _sort_memory(memory, facts=facts, incidents=incidents, obligations=obligations).validated()

def record_public_tool_failure(memory: EvidenceMemory, *, action_id: str, public_roles: Sequence[str], evidence_frame_ids: Sequence[str], step: int, confidence: float=1.0, failure_type: str='unresolved_visual_effect', capacity: MemoryCapacity=MemoryCapacity()) -> MemoryCommitResult:
    memory = memory.validated()
    action = non_empty_string(action_id, pointer='/action_id')
    roles = string_sequence(list(public_roles), pointer='/public_roles', non_empty=True, sorted_utf8=True)
    frames = string_sequence(list(evidence_frame_ids), pointer='/evidence_frame_ids', non_empty=True)
    current_step = integer(step, pointer='/step')
    score = probability(confidence, pointer='/confidence')
    incident = RecoveryIncident.create(episode_id=memory.episode_id, action_id=action, failure_type=failure_type, confidence=score, evidence_frame_ids=frames, step=current_step)
    incidents = list(memory.incidents)
    obligations = list(memory.pending_obligations)
    changed = False
    existing = next((item for item in incidents if item.incident_id == incident.incident_id), None)
    if existing is None:
        incidents.append(incident)
        changed = True
    else:
        updated = replace(existing, confidence=score, evidence_frame_ids=frames, last_updated_step=current_step, status='open')
        if updated != existing:
            incidents[incidents.index(existing)] = updated
            changed = True
    if not any((item.status == 'open' and item.type == 'act' and (item.public_roles == roles) and (item.source_incident_id_or_null == incident.incident_id) for item in obligations)):
        candidate = PendingObligation.create(episode_id=memory.episode_id, obligation_type='act', public_roles=roles, created_step=current_step, source_incident_id_or_null=incident.incident_id)
        existing_obligation = next((item for item in obligations if item.obligation_id == candidate.obligation_id), None)
        if existing_obligation is None:
            obligations.append(candidate)
        else:
            obligations[obligations.index(existing_obligation)] = replace(existing_obligation, status='open')
        changed = True
    if not changed:
        return MemoryCommitResult(False, 'public_tool_failure_no_effect', memory)
    updated_memory = _sort_memory(memory, incidents=incidents, obligations=obligations, revision=memory.revision + 1)
    return MemoryCommitResult(True, 'public_tool_failure_commit', enforce_capacity(updated_memory, capacity))

def record_visual_recovery_incident(memory: EvidenceMemory, *, action_id: str, canonical_action: str, public_roles: Sequence[str], evidence_frame_ids: Sequence[str], step: int, confidence: float, source_incident_id_or_null: str | None=None, increment_recovery_attempt: bool=False, capacity: MemoryCapacity=MemoryCapacity()) -> MemoryCommitResult:
    memory = memory.validated()
    action = non_empty_string(action_id, pointer='/action_id')
    canonical = public_semantic_name(canonical_action, pointer='/canonical_action')
    if canonical not in CANONICAL_PHYSICAL_ACTIONS:
        raise MethodContractError('method_memory_vocabulary_invalid', 'visual recovery action is outside the physical vocabulary')
    roles = string_sequence(list(public_roles), pointer='/public_roles', non_empty=True, sorted_utf8=True)
    frames = string_sequence(list(evidence_frame_ids), pointer='/evidence_frame_ids', non_empty=True)
    current_step = integer(step, pointer='/step')
    score = probability(confidence, pointer='/confidence')
    failure_type = 'visual_effect_not_achieved/{}'.format(canonical)
    incidents = list(memory.incidents)
    obligations = list(memory.pending_obligations)
    source = optional_string(source_incident_id_or_null, pointer='/source_incident_id_or_null')
    incident = None
    if source is not None:
        incident = next((item for item in incidents if item.incident_id == source and item.status == 'open'), None)
    if incident is None:
        incident = RecoveryIncident.create(episode_id=memory.episode_id, action_id=action, failure_type=failure_type, confidence=score, evidence_frame_ids=frames, step=current_step)
        existing = next((item for item in incidents if item.incident_id == incident.incident_id), None)
        if existing is None:
            incidents.append(incident)
        else:
            incident = existing
    updated_incident = replace(incident, confidence=score, evidence_frame_ids=frames, last_updated_step=current_step, status='open', recovery_attempts=incident.recovery_attempts + 1 if increment_recovery_attempt else incident.recovery_attempts)
    if incident in incidents:
        incidents[incidents.index(incident)] = updated_incident
    else:
        incidents.append(updated_incident)
    if not any((item.status == 'open' and item.type == 'act' and (item.source_incident_id_or_null == updated_incident.incident_id) for item in obligations)):
        candidate = PendingObligation.create(episode_id=memory.episode_id, obligation_type='act', public_roles=roles, created_step=current_step, source_incident_id_or_null=updated_incident.incident_id)
        existing_obligation = next((item for item in obligations if item.obligation_id == candidate.obligation_id), None)
        if existing_obligation is None:
            obligations.append(candidate)
        else:
            obligations[obligations.index(existing_obligation)] = replace(existing_obligation, status='open', public_roles=roles)
    updated = _sort_memory(memory, incidents=incidents, obligations=obligations, revision=memory.revision + 1)
    return MemoryCommitResult(True, 'visual_recovery_incident_commit', enforce_capacity(updated, capacity))

def assume_recovery_continue(memory: EvidenceMemory, *, incident_id: str, step: int, capacity: MemoryCapacity=MemoryCapacity()) -> MemoryCommitResult:
    memory = memory.validated()
    identifier = non_empty_string(incident_id, pointer='/incident_id')
    current_step = integer(step, pointer='/step')
    if not any((item.incident_id == identifier and item.status == 'open' for item in memory.incidents)):
        return MemoryCommitResult(False, 'recovery_assumed_continue_already_closed', memory)
    incidents = tuple((replace(item, status='resolved', last_updated_step=current_step) if item.incident_id == identifier else item for item in memory.incidents))
    obligations = tuple((replace(item, status='resolved') if item.status == 'open' and item.source_incident_id_or_null == identifier else item for item in memory.pending_obligations))
    updated = _sort_memory(memory, incidents=incidents, obligations=obligations, revision=memory.revision + 1)
    return MemoryCommitResult(True, 'recovery_assumed_continue', enforce_capacity(updated, capacity))

def resolve_public_tool_failure(memory: EvidenceMemory, *, incident_id: str, public_roles: Sequence[str], step: int, capacity: MemoryCapacity=MemoryCapacity()) -> MemoryCommitResult:
    memory = memory.validated()
    identifier = non_empty_string(incident_id, pointer='/incident_id')
    roles = string_sequence(list(public_roles), pointer='/public_roles', non_empty=True, sorted_utf8=True)
    current_step = integer(step, pointer='/step')
    incident = next((item for item in memory.incidents if item.incident_id == identifier and item.status == 'open'), None)
    obligations = tuple((item for item in memory.pending_obligations if item.status == 'open' and item.type == 'act' and (item.source_incident_id_or_null == identifier)))
    if incident is None or len(obligations) != 1 or obligations[0].public_roles != roles:
        return MemoryCommitResult(False, 'public_tool_failure_resolution_mismatch', memory)
    updated_memory = _sort_memory(memory, incidents=tuple((replace(item, status='resolved', last_updated_step=current_step) if item.incident_id == identifier else item for item in memory.incidents)), obligations=tuple((replace(item, status='resolved') if item.obligation_id == obligations[0].obligation_id else item for item in memory.pending_obligations)), revision=memory.revision + 1)
    return MemoryCommitResult(True, 'public_tool_failure_resolved_by_readiness_action', enforce_capacity(updated_memory, capacity))

def resolve_public_tool_failure_by_incident(memory: EvidenceMemory, *, incident_id: str, step: int, capacity: MemoryCapacity=MemoryCapacity()) -> MemoryCommitResult:
    memory = memory.validated()
    identifier = non_empty_string(incident_id, pointer='/incident_id')
    current_step = integer(step, pointer='/step')
    incident = next((item for item in memory.incidents if item.incident_id == identifier and item.status == 'open'), None)
    obligations = tuple((item for item in memory.pending_obligations if item.status == 'open' and item.type == 'act' and (item.source_incident_id_or_null == identifier)))
    if incident is None or len(obligations) != 1:
        return MemoryCommitResult(False, 'public_tool_failure_resolution_mismatch', memory)
    updated_memory = _sort_memory(memory, incidents=tuple((replace(item, status='resolved', last_updated_step=current_step) if item.incident_id == identifier else item for item in memory.incidents)), obligations=tuple((replace(item, status='resolved') if item.obligation_id == obligations[0].obligation_id else item for item in memory.pending_obligations)), revision=memory.revision + 1)
    return MemoryCommitResult(True, 'public_tool_failure_resolved_by_verified_recovery', enforce_capacity(updated_memory, capacity))

def commit_effect_bindings(memory: EvidenceMemory, bindings: Sequence[Mapping[str, Any] | EffectBinding], selection: VerifierSelection, *, success_probability: float, effect_probabilities: Mapping[str, Any], action_id: str, evidence_frame_ids: Sequence[str], step: int, capacity: MemoryCapacity=MemoryCapacity(), observation_public_roles: Sequence[str]=()) -> MemoryCommitResult:
    memory = memory.validated()
    checked_bindings = validate_effect_bindings(bindings)
    success = probability(success_probability, pointer='/verifier_probabilities/outcome/success')
    if not isinstance(effect_probabilities, Mapping):
        raise MethodContractError('method_schema_invalid', 'effect probabilities must be an object')
    checked_effects = {non_empty_string(key, pointer='/verifier_probabilities/effects'): probability(value, pointer='/verifier_probabilities/effects/{}'.format(key)) for key, value in effect_probabilities.items()}
    action = non_empty_string(action_id, pointer='/action_id')
    frames = string_sequence(list(evidence_frame_ids), pointer='/evidence_frame_ids', non_empty=True)
    observed_step = integer(step, pointer='/step')
    observation_roles = string_sequence(list(observation_public_roles), pointer='/observation_public_roles', non_empty=True, sorted_utf8=True) if observation_public_roles else ()
    if selection.effect_conflict or selection.selected_outcome != 'success' or selection.selected_observability == 'unobservable':
        return MemoryCommitResult(False, 'outcome_not_committable', memory)
    selected = set(selection.selected_effects)
    if any((binding.required and binding.effect not in selected for binding in checked_bindings)):
        return MemoryCommitResult(False, 'required_effect_missing', memory)
    required_unknown = tuple((binding for binding in checked_bindings if binding.required and binding.effect in selected and (binding.binding_status != 'known')))
    if required_unknown:
        roles = observation_roles
        if not roles:
            raise MethodContractError('method_public_reference_invalid', 'required unknown binding needs public observation roles', pointer='/observation_public_roles')
        obligations = list(memory.pending_obligations)
        if any((obligation.status == 'open' and obligation.type == 'observe' and (obligation.public_roles == roles) for obligation in obligations)):
            return MemoryCommitResult(False, 'required_binding_unknown', memory)
        candidate = PendingObligation.create(episode_id=memory.episode_id, obligation_type='observe', public_roles=roles, created_step=observed_step)
        existing = next((obligation for obligation in obligations if obligation.obligation_id == candidate.obligation_id), None)
        if existing is None:
            obligations.append(candidate)
        else:
            obligations[obligations.index(existing)] = replace(existing, status='open')
        updated = _sort_memory(memory, obligations=obligations, revision=memory.revision + 1)
        return MemoryCommitResult(True, 'required_binding_unknown_observe_commit', enforce_capacity(updated, capacity))
    facts = {fact.fact_id: fact for fact in memory.facts}
    obligations = list(memory.pending_obligations)
    changed = False
    committable_binding_seen = False
    for binding in checked_bindings:
        if binding.effect not in selected or binding.effect == 'no_effect':
            continue
        if binding.binding_status != 'known':
            if binding.required:
                return MemoryCommitResult(False, 'required_binding_unknown', memory)
            continue
        committable_binding_seen = True
        effect_probability = checked_effects.get(binding.effect)
        if effect_probability is None:
            raise MethodContractError('method_verifier_output_invalid', 'selected effect lacks a probability')
        key = binding_fact_key(binding)
        identifier = fact_id_for(**key)
        if binding.effect in {'relation_removed_inside', 'relation_removed_on_top'}:
            existing = facts.get(identifier)
            if existing is not None and existing.status != 'contradicted':
                facts[identifier] = replace(existing, status='contradicted', confidence=min(success, effect_probability), source_frame_ids=frames, source_action_id=action, observed_step=observed_step)
                changed = True
            continue
        fact = EvidenceFact.create(**key, confidence=min(success, effect_probability), source_frame_ids=frames, source_action_id=action, observed_step=observed_step)
        for conflicting_id, conflicting in tuple(facts.items()):
            if conflicting.status == 'current' and conflicting_id != fact.fact_id and facts_are_mutually_exclusive(existing_subject=conflicting.subject_public_role, existing_relation=conflicting.relation_or_property, existing_value=conflicting.object_public_role_or_value, incoming_subject=fact.subject_public_role, incoming_relation=fact.relation_or_property, incoming_value=fact.object_public_role_or_value):
                facts[conflicting_id] = replace(conflicting, status='contradicted', source_frame_ids=frames, source_action_id=action, observed_step=observed_step)
                changed = True
        if facts.get(fact.fact_id) != fact:
            facts[fact.fact_id] = fact
            changed = True
        if binding.effect == 'relocated' and (not any((obligation.status == 'open' and obligation.type == 'observe' and (obligation.public_roles == (binding.subject_public_role,)) for obligation in obligations))):
            candidate = PendingObligation.create(episode_id=memory.episode_id, obligation_type='observe', public_roles=(binding.subject_public_role,), created_step=observed_step)
            existing = next((obligation for obligation in obligations if obligation.obligation_id == candidate.obligation_id), None)
            if existing is None:
                obligations.append(candidate)
            else:
                obligations[obligations.index(existing)] = replace(existing, status='open')
            changed = True
    if committable_binding_seen and observation_roles:
        matching_obligations = sorted((obligation for obligation in obligations if obligation.status == 'open' and obligation.type == 'observe' and (obligation.public_roles == observation_roles) and (obligation.source_incident_id_or_null is None)), key=lambda obligation: (obligation.created_step, obligation.obligation_id.encode('utf-8')))
        if matching_obligations:
            obligation = matching_obligations[0]
            obligations[obligations.index(obligation)] = replace(obligation, status='resolved')
            changed = True
    if not changed:
        return MemoryCommitResult(False, 'proposal_no_effect', memory)
    updated = _sort_memory(memory, facts=tuple(facts.values()), obligations=obligations, revision=memory.revision + 1)
    updated = enforce_capacity(updated, capacity)
    return MemoryCommitResult(True, 'effect_binding_commit', updated)
__all__ = ['CANONICAL_FAILURE_TYPES', 'CANONICAL_PHYSICAL_ACTIONS', 'EvidenceFact', 'EvidenceMemory', 'MemoryCapacity', 'MemoryCommitResult', 'PendingObligation', 'RecoveryIncident', 'commit_effect_bindings', 'enforce_capacity', 'fact_id_for', 'facts_are_mutually_exclusive', 'incident_id_for', 'obligation_id_for', 'record_public_tool_failure', 'record_visual_recovery_incident', 'assume_recovery_continue', 'resolve_public_tool_failure', 'resolve_public_tool_failure_by_incident', 'validate_failure_type']
