from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from scoper.harness.contracts import MethodContractError
from scoper.harness.validation import exact_mapping, integer, non_empty_string, public_scalar, string_sequence
from scoper.harness.evidence_memory import EvidenceFact, EvidenceMemory, PendingObligation, RecoveryIncident
from scoper.harness.binding import instruction_scene_reference_relevant

@dataclass(frozen=True)
class RouterLimits:
    max_facts: int = 6
    max_incidents: int = 2
    max_obligations: int = 3

    def __post_init__(self) -> None:
        integer(self.max_facts, pointer='/router/max_facts', minimum=1)
        integer(self.max_incidents, pointer='/router/max_incidents', minimum=1)
        integer(self.max_obligations, pointer='/router/max_obligations', minimum=1)
        if self.max_facts > 6 or self.max_incidents > 2 or self.max_obligations > 3:
            raise MethodContractError('method_schema_invalid', 'router limits exceed the frozen 6/2/3 maxima')

@dataclass(frozen=True)
class RoutedMemory:
    facts: tuple[EvidenceFact, ...]
    incidents: tuple[RecoveryIncident, ...]
    obligations: tuple[PendingObligation, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {'facts': [item.to_mapping() for item in self.facts], 'incidents': [item.to_mapping() for item in self.incidents], 'pending_obligations': [item.to_mapping() for item in self.obligations]}

def build_active_roles(*, unresolved_incident_roles: Sequence[str]=(), first_pending_obligation_roles: Sequence[str]=(), last_public_transition_target_or_null: str | None=None) -> tuple[str, ...]:
    ordered = []
    for role in tuple(unresolved_incident_roles) + tuple(first_pending_obligation_roles):
        role = non_empty_string(role, pointer='/active_roles')
        if role not in ordered:
            ordered.append(role)
    if last_public_transition_target_or_null is not None:
        target = non_empty_string(last_public_transition_target_or_null, pointer='/last_public_transition_target_or_null')
        if target not in ordered:
            ordered.append(target)
    return tuple(ordered)

def _fact_hits_role(fact: EvidenceFact, active_roles: set[str]) -> bool:
    return fact.subject_public_role in active_roles or (isinstance(fact.object_public_role_or_value, str) and fact.object_public_role_or_value in active_roles)

def route_memory(memory: EvidenceMemory, active_roles: Sequence[str], *, limits: RouterLimits=RouterLimits(), public_instruction: str | None=None) -> RoutedMemory:
    memory = memory.validated()
    roles = set(string_sequence(list(active_roles), pointer='/active_roles'))
    facts = tuple(sorted(memory.facts, key=lambda fact: (0 if _fact_hits_role(fact, roles) else 1, 1 if public_instruction and fact.relation_or_property == 'scene_observation' and (not instruction_scene_reference_relevant(fact.subject_public_role, public_instruction)) else 0, -fact.observed_step, fact.fact_id.encode('utf-8')))[:limits.max_facts])
    incidents = tuple(sorted(memory.incidents, key=lambda incident: (0 if incident.status == 'open' else 1, -incident.created_step, incident.incident_id.encode('utf-8')))[:limits.max_incidents])
    obligations = tuple(sorted(memory.pending_obligations, key=lambda obligation: (0 if obligation.status == 'open' else 1, obligation.created_step, obligation.obligation_id.encode('utf-8')))[:limits.max_obligations])
    return RoutedMemory(facts, incidents, obligations)

def rank_golden_fact_records(values: Sequence[Mapping[str, Any]], active_roles: Sequence[str], *, max_items: int=6) -> tuple[str, ...]:
    integer(max_items, pointer='/max_items', minimum=1)
    roles = set(string_sequence(list(active_roles), pointer='/active_roles'))
    normalized = []
    fields = frozenset({'fact_id', 'observed_step', 'status', 'subject_public_role'})
    for index, value in enumerate(values):
        item = exact_mapping(value, fields, pointer='/facts/{}'.format(index))
        identifier = non_empty_string(item['fact_id'], pointer='/facts/fact_id')
        subject = non_empty_string(item['subject_public_role'], pointer='/facts/subject_public_role')
        public_scalar(subject, pointer='/facts/subject_public_role')
        step = integer(item['observed_step'], pointer='/facts/observed_step')
        if item['status'] not in {'current', 'stale', 'contradicted'}:
            raise MethodContractError('method_schema_invalid', 'golden fact status is invalid')
        normalized.append((identifier, subject, step))
    ranked = sorted(normalized, key=lambda item: (0 if item[1] in roles else 1, -item[2], item[0].encode('utf-8')))
    return tuple((item[0] for item in ranked[:max_items]))
__all__ = ['RoutedMemory', 'RouterLimits', 'build_active_roles', 'rank_golden_fact_records', 'route_memory']
