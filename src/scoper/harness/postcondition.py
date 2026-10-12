from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from scoper.harness.effect_binding import EffectBinding, validate_effect_bindings
from scoper.harness.contracts import EFFECT_VOCABULARY, MethodContractError
from scoper.harness.validation import public_json, public_semantic_name
from scoper.harness.verifier import VerifierSelection
PUBLIC_ACTION_STATE_SCHEMA = 'method-public-action-state.v1'
PUBLIC_POSTCONDITION_EVIDENCE_SCHEMA = 'method-public-postcondition-evidence.v1'
PUBLIC_POSTCONDITION_RESULT_SCHEMA = 'method-public-postcondition-result.v1'
POSTCONDITION_STATUSES = ('failure', 'success', 'unknown')
_BOOLEAN_PROPERTIES = frozenset({'cooled', 'dirty', 'filled', 'heated', 'open', 'saturated', 'sliced', 'toggled_on'})
_PROPERTY_EFFECTS = {'opened': ('open', True), 'closed': ('open', False), 'toggled_on': ('toggled_on', True), 'toggled_off': ('toggled_on', False), 'cleaned': ('dirty', False), 'dirtied': ('dirty', True), 'sliced': ('sliced', True), 'heated': ('heated', True), 'cooled': ('cooled', True), 'filled': ('filled', True), 'saturated': ('saturated', True)}
_RELATION_EFFECTS = {'relation_added_inside': ('inside', True), 'relation_added_on_top': ('on_top', True), 'relation_removed_inside': ('inside', False), 'relation_removed_on_top': ('on_top', False)}

def _sorted_public_refs(values: Sequence[Any], *, pointer: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise MethodContractError('method_schema_invalid', 'public refs must be a list', pointer=pointer)
    refs = tuple(sorted({public_semantic_name(value, pointer=pointer) for value in values}, key=lambda value: value.encode('utf-8')))
    return refs

@dataclass(frozen=True)
class PublicPropertyValue:
    public_ref: str
    property_name: str
    value: bool

    def __post_init__(self) -> None:
        public_semantic_name(self.public_ref, pointer='/property/public_ref')
        name = public_semantic_name(self.property_name, pointer='/property/property_name')
        if name not in _BOOLEAN_PROPERTIES or not isinstance(self.value, bool):
            raise MethodContractError('method_schema_invalid', 'public property is invalid')

    def to_mapping(self) -> dict[str, Any]:
        return {'public_ref': self.public_ref, 'property_name': self.property_name, 'value': self.value}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'PublicPropertyValue':
        if not isinstance(value, Mapping) or set(value) != {'public_ref', 'property_name', 'value'}:
            raise MethodContractError('method_schema_invalid', 'public property fields are invalid')
        return cls(str(value['public_ref']), str(value['property_name']), value['value'])

@dataclass(frozen=True)
class PublicRelationValue:
    subject_public_ref: str
    relation: str
    object_public_ref: str

    def __post_init__(self) -> None:
        public_semantic_name(self.subject_public_ref, pointer='/relation/subject_public_ref')
        relation = public_semantic_name(self.relation, pointer='/relation/relation')
        if relation not in {'inside', 'on_top'}:
            raise MethodContractError('method_schema_invalid', 'public relation is invalid')
        public_semantic_name(self.object_public_ref, pointer='/relation/object_public_ref')

    def to_mapping(self) -> dict[str, str]:
        return {'subject_public_ref': self.subject_public_ref, 'relation': self.relation, 'object_public_ref': self.object_public_ref}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'PublicRelationValue':
        expected = {'subject_public_ref', 'relation', 'object_public_ref'}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise MethodContractError('method_schema_invalid', 'public relation fields are invalid')
        return cls(str(value['subject_public_ref']), str(value['relation']), str(value['object_public_ref']))

@dataclass(frozen=True)
class PublicActionState:
    held_inventory_observed: bool
    held_public_refs: tuple[str, ...]
    observed_public_refs: tuple[str, ...]
    relation_observed_subjects: tuple[str, ...]
    properties: tuple[PublicPropertyValue, ...]
    relations: tuple[PublicRelationValue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.held_inventory_observed, bool):
            raise MethodContractError('method_schema_invalid', 'held inventory flag is invalid')
        for name in ('held_public_refs', 'observed_public_refs', 'relation_observed_subjects'):
            values = getattr(self, name)
            checked = _sorted_public_refs(values, pointer=f'/{name}')
            if values != checked:
                raise MethodContractError('method_schema_invalid', f'{name} is not canonical')
        property_keys = tuple(((item.public_ref.encode('utf-8'), item.property_name.encode('utf-8')) for item in self.properties))
        relation_keys = tuple(((item.subject_public_ref.encode('utf-8'), item.relation.encode('utf-8'), item.object_public_ref.encode('utf-8')) for item in self.relations))
        if property_keys != tuple(sorted(property_keys)) or len(set(property_keys)) != len(property_keys):
            raise MethodContractError('method_schema_invalid', 'public properties are not canonical')
        if relation_keys != tuple(sorted(relation_keys)) or len(set(relation_keys)) != len(relation_keys):
            raise MethodContractError('method_schema_invalid', 'public relations are not canonical')
        public_json(self.to_mapping(), pointer='/public_action_state', include_future_action=True)

    def to_mapping(self) -> dict[str, Any]:
        return {'schema_version': PUBLIC_ACTION_STATE_SCHEMA, 'held_inventory_observed': self.held_inventory_observed, 'held_public_refs': list(self.held_public_refs), 'observed_public_refs': list(self.observed_public_refs), 'relation_observed_subjects': list(self.relation_observed_subjects), 'properties': [item.to_mapping() for item in self.properties], 'relations': [item.to_mapping() for item in self.relations]}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'PublicActionState':
        expected = {'schema_version', 'held_inventory_observed', 'held_public_refs', 'observed_public_refs', 'relation_observed_subjects', 'properties', 'relations'}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise MethodContractError('method_schema_invalid', 'public action state fields are invalid')
        if value['schema_version'] != PUBLIC_ACTION_STATE_SCHEMA:
            raise MethodContractError('method_schema_invalid', 'public action state schema is invalid')
        properties = value['properties']
        relations = value['relations']
        if not isinstance(properties, list) or not isinstance(relations, list):
            raise MethodContractError('method_schema_invalid', 'public action state values are invalid')
        return cls(value['held_inventory_observed'], _sorted_public_refs(value['held_public_refs'], pointer='/held_public_refs'), _sorted_public_refs(value['observed_public_refs'], pointer='/observed_public_refs'), _sorted_public_refs(value['relation_observed_subjects'], pointer='/relation_observed_subjects'), tuple(sorted((PublicPropertyValue.from_mapping(item) for item in properties), key=lambda item: (item.public_ref.encode('utf-8'), item.property_name.encode('utf-8')))), tuple(sorted((PublicRelationValue.from_mapping(item) for item in relations), key=lambda item: (item.subject_public_ref.encode('utf-8'), item.relation.encode('utf-8'), item.object_public_ref.encode('utf-8')))))

@dataclass(frozen=True)
class PublicPostconditionEvidence:
    pre: PublicActionState
    post: PublicActionState

    def to_mapping(self) -> dict[str, Any]:
        return {'schema_version': PUBLIC_POSTCONDITION_EVIDENCE_SCHEMA, 'pre': self.pre.to_mapping(), 'post': self.post.to_mapping()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'PublicPostconditionEvidence':
        if not isinstance(value, Mapping) or set(value) != {'schema_version', 'pre', 'post'}:
            raise MethodContractError('method_schema_invalid', 'postcondition evidence fields are invalid')
        if value['schema_version'] != PUBLIC_POSTCONDITION_EVIDENCE_SCHEMA:
            raise MethodContractError('method_schema_invalid', 'postcondition evidence schema is invalid')
        return cls(PublicActionState.from_mapping(value['pre']), PublicActionState.from_mapping(value['post']))

@dataclass(frozen=True)
class PublicPostconditionResult:
    status: str
    reason: str
    selected_effects: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in POSTCONDITION_STATUSES or not isinstance(self.reason, str) or (not self.reason):
            raise MethodContractError('method_schema_invalid', 'postcondition result is invalid')
        if self.selected_effects != tuple((effect for effect in EFFECT_VOCABULARY if effect in set(self.selected_effects))):
            raise MethodContractError('method_schema_invalid', 'postcondition effects are not canonical')

    @property
    def conclusive(self) -> bool:
        return self.status != 'unknown'

    def public_record(self) -> dict[str, Any]:
        return {'schema_version': PUBLIC_POSTCONDITION_RESULT_SCHEMA, 'status': self.status, 'reason': self.reason, 'selected_effects': list(self.selected_effects)}

def _property_status(state: PublicActionState, public_ref: str, name: str, expected: bool) -> str:
    values = {item.value for item in state.properties if item.public_ref == public_ref and item.property_name == name}
    if expected in values:
        return 'success'
    if values:
        return 'failure'
    return 'unknown'

def _binding_status(binding: EffectBinding, evidence: PublicPostconditionEvidence) -> str:
    if binding.binding_status != 'known' or binding.subject_public_role is None:
        return 'unknown'
    effect = binding.effect
    subject = binding.subject_public_role
    if effect == 'picked_up':
        if not evidence.post.held_inventory_observed:
            return 'unknown'
        return 'success' if subject in evidence.post.held_public_refs else 'failure'
    if effect == 'released':
        if not evidence.post.held_inventory_observed:
            return 'unknown'
        return 'failure' if subject in evidence.post.held_public_refs else 'success'
    if effect in _PROPERTY_EFFECTS:
        name, expected = _PROPERTY_EFFECTS[effect]
        return _property_status(evidence.post, subject, name, expected)
    if effect in _RELATION_EFFECTS:
        relation, expected_present = _RELATION_EFFECTS[effect]
        target = binding.object_public_role_or_value
        if not isinstance(target, str):
            return 'unknown'
        present = any((item.subject_public_ref == subject and item.relation == relation and (item.object_public_ref == target) for item in evidence.post.relations))
        if present == expected_present:
            if present or subject in evidence.post.relation_observed_subjects:
                return 'success'
        if subject in evidence.post.relation_observed_subjects:
            return 'failure'
        return 'unknown'
    return 'unknown'

def evaluate_public_postcondition(bindings: Sequence[Mapping[str, Any] | EffectBinding], evidence: Mapping[str, Any] | PublicPostconditionEvidence) -> PublicPostconditionResult:
    checked = validate_effect_bindings(bindings)
    public_evidence = evidence if isinstance(evidence, PublicPostconditionEvidence) else PublicPostconditionEvidence.from_mapping(evidence)
    required = tuple((binding for binding in checked if binding.required))
    if not required:
        return PublicPostconditionResult('unknown', 'no_required_public_effect', ())
    statuses = tuple((_binding_status(binding, public_evidence) for binding in required))
    selected_effects = tuple((effect for effect in EFFECT_VOCABULARY if effect in {binding.effect for binding in required}))
    if 'failure' in statuses:
        return PublicPostconditionResult('failure', 'public_postcondition_failed', ())
    if all((status == 'success' for status in statuses)):
        return PublicPostconditionResult('success', 'public_postcondition_satisfied', selected_effects)
    return PublicPostconditionResult('unknown', 'public_postcondition_unobserved', ())

def merge_public_and_visual_verification(public_result: PublicPostconditionResult, visual_selection: VerifierSelection) -> VerifierSelection:
    if public_result.status == 'success':
        return VerifierSelection('success', public_result.selected_effects, 'observable', False)
    if public_result.status == 'failure':
        return VerifierSelection('failure', (), 'observable', False)
    return visual_selection
__all__ = ['POSTCONDITION_STATUSES', 'PUBLIC_ACTION_STATE_SCHEMA', 'PUBLIC_POSTCONDITION_EVIDENCE_SCHEMA', 'PUBLIC_POSTCONDITION_RESULT_SCHEMA', 'PublicActionState', 'PublicPostconditionEvidence', 'PublicPostconditionResult', 'PublicPropertyValue', 'PublicRelationValue', 'evaluate_public_postcondition', 'merge_public_and_visual_verification']
