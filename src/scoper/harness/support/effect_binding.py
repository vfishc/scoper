from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from scoper.benchmark.common.canonical import canonical_json_bytes
from scoper.harness.support.method_contract import EFFECT_VOCABULARY, MethodContractError
from scoper.harness.support.method_validation import exact_mapping, non_empty_string, public_json, public_scalar, public_semantic_name
EFFECT_BINDING_FIELDS = frozenset({'effect', 'subject_public_role', 'relation_or_property', 'object_public_role_or_value', 'required', 'binding_status'})
BINDING_STATUSES = ('known', 'unknown', 'not_applicable')
_PUBLIC_ROLE_VALUE = object()
_EFFECT_FACT_SEMANTICS = {'picked_up': ('held_by_agent', True), 'released': ('held_by_agent', False), 'relation_added_inside': ('inside', _PUBLIC_ROLE_VALUE), 'relation_added_on_top': ('on_top', _PUBLIC_ROLE_VALUE), 'relation_removed_inside': ('inside', _PUBLIC_ROLE_VALUE), 'relation_removed_on_top': ('on_top', _PUBLIC_ROLE_VALUE), 'opened': ('open', True), 'closed': ('open', False), 'toggled_on': ('toggled_on', True), 'toggled_off': ('toggled_on', False), 'cleaned': ('dirty', False), 'dirtied': ('dirty', True), 'sliced': ('sliced', True), 'heated': ('heated', True), 'cooled': ('cooled', True), 'filled': ('filled', True), 'saturated': ('saturated', True), 'relocated': ('location_changed', True)}
CANONICAL_FACT_RELATIONS = frozenset((relation for relation, _ in _EFFECT_FACT_SEMANTICS.values()))
SCENE_OBSERVATION_RELATION = 'scene_observation'

def validate_fact_semantics(relation_or_property: Any, object_public_role_or_value: Any, *, pointer: str, allow_scene_observation: bool=False) -> tuple[str, Any]:
    relation = public_semantic_name(relation_or_property, pointer='{}/relation_or_property'.format(pointer))
    if relation == SCENE_OBSERVATION_RELATION and allow_scene_observation:
        value = public_scalar(object_public_role_or_value, pointer='{}/object_public_role_or_value'.format(pointer))
        if not isinstance(value, str) or not value.strip() or len(value) > 160:
            raise MethodContractError('method_memory_vocabulary_invalid', 'scene observation value must be concise non-empty text', pointer='{}/object_public_role_or_value'.format(pointer))
        return (relation, value)
    if relation not in CANONICAL_FACT_RELATIONS:
        raise MethodContractError('method_memory_vocabulary_invalid', 'fact relation/property is outside the frozen vocabulary', pointer='{}/relation_or_property'.format(pointer))
    value = public_scalar(object_public_role_or_value, pointer='{}/object_public_role_or_value'.format(pointer))
    if relation in {'inside', 'on_top'}:
        public_semantic_name(value, pointer='{}/object_public_role_or_value'.format(pointer))
    elif relation in {'held_by_agent', 'open', 'toggled_on', 'dirty'}:
        if not isinstance(value, bool):
            raise MethodContractError('method_memory_vocabulary_invalid', 'fact property requires a boolean value', pointer='{}/object_public_role_or_value'.format(pointer))
    elif value is not True:
        raise MethodContractError('method_memory_vocabulary_invalid', 'fact property requires canonical true', pointer='{}/object_public_role_or_value'.format(pointer))
    return (relation, value)

def _validate_binding_semantics(binding: 'EffectBinding') -> None:
    if binding.effect == 'no_effect':
        if binding.binding_status != 'not_applicable' or binding.required or any((value is not None for value in (binding.subject_public_role, binding.relation_or_property, binding.object_public_role_or_value))):
            raise MethodContractError('method_effect_binding_invalid', 'no_effect requires a non-required not_applicable null binding')
        return
    if binding.binding_status != 'known':
        if binding.binding_status == 'not_applicable' and binding.required:
            raise MethodContractError('method_effect_binding_invalid', 'a required effect cannot be not_applicable')
        if any((value is not None for value in (binding.subject_public_role, binding.relation_or_property, binding.object_public_role_or_value))):
            raise MethodContractError('method_effect_binding_invalid', 'unknown/not_applicable bindings must use null fact fields')
        return
    subject = public_semantic_name(binding.subject_public_role, pointer='/effect_binding/subject_public_role')
    relation = non_empty_string(binding.relation_or_property, pointer='/effect_binding/relation_or_property')
    expected_relation, expected_value = _EFFECT_FACT_SEMANTICS[binding.effect]
    if relation != expected_relation:
        raise MethodContractError('method_effect_binding_invalid', 'effect relation/property differs from the canonical table', pointer='/effect_binding/relation_or_property')
    value = public_scalar(binding.object_public_role_or_value, pointer='/effect_binding/object_public_role_or_value')
    if expected_value is _PUBLIC_ROLE_VALUE:
        public_semantic_name(value, pointer='/effect_binding/object_public_role_or_value')
    elif type(value) is not type(expected_value) or value != expected_value:
        raise MethodContractError('method_effect_binding_invalid', 'effect object/value differs from the canonical table', pointer='/effect_binding/object_public_role_or_value')
    public_json(subject, pointer='/effect_binding/subject_public_role')

@dataclass(frozen=True)
class EffectBinding:
    effect: str
    subject_public_role: str | None
    relation_or_property: str | None
    object_public_role_or_value: Any
    required: bool
    binding_status: str

    def __post_init__(self) -> None:
        if self.effect not in EFFECT_VOCABULARY:
            raise MethodContractError('method_effect_binding_invalid', 'effect is outside the frozen vocabulary')
        if not isinstance(self.required, bool):
            raise MethodContractError('method_schema_invalid', 'required must be boolean', pointer='/effect_binding/required')
        if self.binding_status not in BINDING_STATUSES:
            raise MethodContractError('method_effect_binding_invalid', 'binding_status is invalid', pointer='/effect_binding/binding_status')
        _validate_binding_semantics(self)
        public_json(self.to_mapping(), pointer='/effect_binding', include_future_action=True)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'EffectBinding':
        item = exact_mapping(value, EFFECT_BINDING_FIELDS, pointer='/effect_binding')
        effect = non_empty_string(item['effect'], pointer='/effect_binding/effect')
        if effect not in EFFECT_VOCABULARY:
            raise MethodContractError('method_effect_binding_invalid', 'effect is outside the frozen vocabulary')
        if not isinstance(item['required'], bool):
            raise MethodContractError('method_schema_invalid', 'required must be boolean', pointer='/effect_binding/required')
        return cls(effect, item['subject_public_role'], item['relation_or_property'], item['object_public_role_or_value'], item['required'], item['binding_status'])

    def to_mapping(self) -> dict[str, Any]:
        return {'effect': self.effect, 'subject_public_role': self.subject_public_role, 'relation_or_property': self.relation_or_property, 'object_public_role_or_value': self.object_public_role_or_value, 'required': self.required, 'binding_status': self.binding_status}

def validate_effect_bindings(values: Sequence[Mapping[str, Any] | EffectBinding]) -> tuple[EffectBinding, ...]:
    bindings = tuple((EffectBinding.from_mapping(value.to_mapping()) if callable(getattr(value, 'to_mapping', None)) else EffectBinding.from_mapping(value) for value in values))
    keys = [(binding.effect.encode('utf-8'), canonical_json_bytes(binding.to_mapping())) for binding in bindings]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise MethodContractError('method_effect_binding_invalid', 'effect bindings must be unique and canonical sorted')
    return bindings

def binding_fact_key(binding: EffectBinding) -> dict[str, Any]:
    if binding.binding_status != 'known':
        raise MethodContractError('method_effect_binding_invalid', 'only known bindings have a fact key')
    return {'subject_public_role': binding.subject_public_role, 'relation_or_property': binding.relation_or_property, 'object_public_role_or_value': binding.object_public_role_or_value}

def rebind_effect_binding_target(values: Sequence[Mapping[str, Any] | EffectBinding], *, original_public_target: str, canonical_public_target: str) -> tuple[EffectBinding, ...]:
    original = public_semantic_name(original_public_target, pointer='/effect_binding_rebind/original_public_target')
    canonical = public_semantic_name(canonical_public_target, pointer='/effect_binding_rebind/canonical_public_target')
    rebound = []
    for raw in values:
        binding = EffectBinding.from_mapping(raw.to_mapping()) if callable(getattr(raw, 'to_mapping', None)) else EffectBinding.from_mapping(raw)
        rebound.append(EffectBinding(binding.effect, canonical if binding.subject_public_role == original else binding.subject_public_role, binding.relation_or_property, canonical if binding.object_public_role_or_value == original else binding.object_public_role_or_value, binding.required, binding.binding_status))
    rebound.sort(key=lambda item: (item.effect.encode('utf-8'), canonical_json_bytes(item.to_mapping())))
    return validate_effect_bindings(rebound)
__all__ = ['BINDING_STATUSES', 'CANONICAL_FACT_RELATIONS', 'SCENE_OBSERVATION_RELATION', 'EFFECT_BINDING_FIELDS', 'EffectBinding', 'binding_fact_key', 'rebind_effect_binding_target', 'validate_fact_semantics', 'validate_effect_bindings']
