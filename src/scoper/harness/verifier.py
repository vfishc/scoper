from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from scoper.harness.contracts import EFFECT_VOCABULARY, MethodContractError
from scoper.harness.validation import exact_mapping, finite_number, non_empty_string, probability, public_json, sha256, string_sequence
OUTCOME_ORDER = ('success', 'failure', 'ambiguous')
OBSERVABILITY_ORDER = ('observable', 'partially_observable', 'unobservable')
IMAGE_METADATA_FIELDS = frozenset({'media_sha256', 'mime_type', 'width', 'height'})
VERIFICATION_ACTION_FIELDS = frozenset({'canonical_action', 'target_public_ref_or_null', 'intended_relation_or_null', 'required_effects'})
VISUAL_VERIFICATION_SUMMARY_FIELDS = frozenset({'tool_name', 'arguments', 'selected_outcome', 'selected_effects', 'selected_observability'})
ORACLE_EFFECT_STATUS_TO_OUTCOME = {'achieved': 'success', 'not_achieved': 'failure', 'partial': 'ambiguous', 'ambiguous': 'ambiguous'}
ORACLE_EFFECT_OBSERVABILITY = {'visible': 'observable', 'occluded': 'partially_observable', 'visually_ambiguous': 'partially_observable', 'mixed': 'partially_observable', 'out_of_view': 'unobservable'}
ACTION_REQUIRED_EFFECTS = {'pick_up': ('picked_up',), 'put_down': ('released',), 'open': ('opened',), 'close': ('closed',), 'toggle_on': ('toggled_on',), 'toggle_off': ('toggled_off',), 'clean': ('cleaned',), 'slice': ('sliced',), 'fill': ('filled',), 'saturate': ('saturated',), 'heat': ('heated',), 'cool': ('cooled',)}
VERIFIER_OUTPUT_FIELDS = frozenset({'schema_version', 'verifier_call_id', 'outcome_probabilities', 'effect_probabilities', 'observability_probabilities', 'selected_outcome', 'selected_effects', 'selected_observability', 'outcome_temperature', 'checkpoint_sha256', 'evidence_frame_ids'})

class VerifierClient(Protocol):
    checkpoint_sha256: str

    def verify(self, request: Mapping[str, Any], *, image_payloads: Mapping[str, bytes]) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...

@dataclass(frozen=True)
class VerifierSelection:
    selected_outcome: str
    selected_effects: tuple[str, ...]
    selected_observability: str
    effect_conflict: bool

    def __post_init__(self) -> None:
        if self.selected_outcome not in OUTCOME_ORDER:
            raise MethodContractError('method_verifier_output_invalid', 'selected outcome is invalid')
        effects = string_sequence(list(self.selected_effects), pointer='/selected_effects')
        if any((effect not in EFFECT_VOCABULARY for effect in effects)):
            raise MethodContractError('method_verifier_output_invalid', 'selected effect is invalid')
        expected_order = tuple((effect for effect in EFFECT_VOCABULARY if effect in effects))
        if effects != expected_order:
            raise MethodContractError('method_verifier_output_invalid', 'selected effects are not in vocabulary order')
        if self.selected_observability not in OBSERVABILITY_ORDER:
            raise MethodContractError('method_verifier_output_invalid', 'selected observability is invalid')
        if not isinstance(self.effect_conflict, bool):
            raise MethodContractError('method_schema_invalid', 'effect_conflict must be boolean')
        expected_conflict = 'no_effect' in effects and len(effects) > 1
        if self.effect_conflict != expected_conflict:
            raise MethodContractError('method_verifier_output_invalid', 'effect_conflict differs from selected effects')
        if self.effect_conflict and self.selected_outcome != 'ambiguous' or (self.selected_outcome == 'success' and 'no_effect' in effects):
            raise MethodContractError('method_verifier_output_invalid', 'selected outcome bypasses verifier downgrade rules')

    def to_mapping(self) -> dict[str, Any]:
        return {'selected_outcome': self.selected_outcome, 'selected_effects': list(self.selected_effects), 'selected_observability': self.selected_observability, 'effect_conflict': self.effect_conflict}

def _probability_mapping(value: Any, *, keys: Sequence[str] | None, pointer: str, require_sum: bool) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise MethodContractError('method_schema_invalid', 'probabilities must be an object', pointer=pointer)
    if keys is not None and set(value) != set(keys):
        raise MethodContractError('method_schema_invalid', 'probability keys differ', pointer=pointer)
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise MethodContractError('method_schema_invalid', 'probability key is invalid', pointer=pointer)
        result[key] = probability(item, pointer='{}/{}'.format(pointer, key))
    if require_sum and abs(sum(result.values()) - 1.0) > 1e-05:
        raise MethodContractError('method_verifier_output_invalid', 'probabilities must sum to one', pointer=pointer)
    return result

def select_verifier_result(probabilities: Mapping[str, Any], thresholds: Mapping[str, Any], *, required_effects: Sequence[str]=()) -> VerifierSelection:
    probabilities = exact_mapping(probabilities, frozenset({'outcome', 'effects', 'observability'}), pointer='/probabilities')
    thresholds = exact_mapping(thresholds, frozenset({'commit_threshold', 'failure_threshold', 'effect_thresholds'}), pointer='/thresholds')
    outcome = _probability_mapping(probabilities['outcome'], keys=OUTCOME_ORDER, pointer='/probabilities/outcome', require_sum=True)
    observability = _probability_mapping(probabilities['observability'], keys=OBSERVABILITY_ORDER, pointer='/probabilities/observability', require_sum=True)
    effects = _probability_mapping(probabilities['effects'], keys=EFFECT_VOCABULARY, pointer='/probabilities/effects', require_sum=False)
    effect_thresholds = _probability_mapping(thresholds['effect_thresholds'], keys=EFFECT_VOCABULARY, pointer='/thresholds/effect_thresholds', require_sum=False)
    commit_threshold = probability(thresholds['commit_threshold'], pointer='/thresholds/commit_threshold')
    failure_threshold = probability(thresholds['failure_threshold'], pointer='/thresholds/failure_threshold')
    required = string_sequence(list(required_effects), pointer='/required_effects')
    if any((effect not in effects for effect in required)):
        raise MethodContractError('method_effect_binding_invalid', 'required effect lacks a probability')
    if outcome['success'] >= commit_threshold:
        selected_outcome = 'success'
    elif outcome['failure'] >= failure_threshold:
        selected_outcome = 'failure'
    else:
        selected_outcome = 'ambiguous'
    selected_effects = tuple((effect for effect in EFFECT_VOCABULARY if effect in effects and effects[effect] >= effect_thresholds[effect]))
    selected_observability = max(OBSERVABILITY_ORDER, key=lambda name: (observability[name], -OBSERVABILITY_ORDER.index(name)))
    effect_conflict = 'no_effect' in selected_effects and len(selected_effects) > 1
    if effect_conflict or (selected_outcome == 'success' and (not set(required).issubset(selected_effects))) or (selected_outcome == 'success' and 'no_effect' in selected_effects) or (selected_observability == 'unobservable'):
        selected_outcome = 'ambiguous'
    return VerifierSelection(selected_outcome, selected_effects, selected_observability, effect_conflict)

def _image_metadata(value: Any, *, pointer: str) -> dict[str, Any]:
    item = exact_mapping(value, IMAGE_METADATA_FIELDS, pointer=pointer)
    result = {'media_sha256': sha256(item['media_sha256'], pointer='{}/media_sha256'.format(pointer)), 'mime_type': item['mime_type'], 'width': item['width'], 'height': item['height']}
    if result['mime_type'] not in {'image/png', 'image/jpeg'}:
        raise MethodContractError('method_media_invalid', 'verifier image mime type is invalid', pointer=pointer)
    for field in ('width', 'height'):
        if isinstance(result[field], bool) or not isinstance(result[field], int) or result[field] < 1:
            raise MethodContractError('method_media_invalid', 'verifier image shape is invalid', pointer=pointer)
    return result

def required_effects_for_action(canonical_action: str, intended_relation_or_null: str | None=None) -> tuple[str, ...]:
    action = non_empty_string(canonical_action, pointer='/action/canonical_action')
    try:
        effects = list(ACTION_REQUIRED_EFFECTS[action])
    except KeyError as exc:
        raise MethodContractError('method_effect_binding_invalid', 'canonical action has no visual effect contract', pointer='/action/canonical_action') from exc
    relation = intended_relation_or_null
    if relation is not None:
        relation = non_empty_string(relation, pointer='/action/intended_relation_or_null')
    if action == 'put_down' and relation is not None:
        relation_effect = {'inside': 'relation_added_inside', 'on_top': 'relation_added_on_top'}.get(relation)
        if relation_effect is None:
            raise MethodContractError('method_effect_binding_invalid', 'put_down relation must be inside or on_top', pointer='/action/intended_relation_or_null')
        effects.append(relation_effect)
    elif action != 'put_down' and relation is not None:
        raise MethodContractError('method_effect_binding_invalid', 'only put_down accepts an intended relation', pointer='/action/intended_relation_or_null')
    return tuple((effect for effect in EFFECT_VOCABULARY if effect in effects))

def _binding_required_effects(effect_bindings: Sequence[Any]) -> tuple[str, ...]:
    effects: list[str] = []
    for index, binding in enumerate(effect_bindings):
        if isinstance(binding, Mapping):
            effect = binding.get('effect')
            required = binding.get('required')
        else:
            effect = getattr(binding, 'effect', None)
            required = getattr(binding, 'required', None)
        if not isinstance(required, bool) or not isinstance(effect, str):
            raise MethodContractError('method_effect_binding_invalid', 'EffectBinding lacks a valid effect or required flag', pointer=f'/effect_bindings/{index}')
        if required:
            if effect not in EFFECT_VOCABULARY or effect == 'no_effect':
                raise MethodContractError('method_effect_binding_invalid', 'required EffectBinding is outside the visual effect vocabulary', pointer=f'/effect_bindings/{index}/effect')
            effects.append(effect)
    if len(effects) != len(set(effects)):
        raise MethodContractError('method_effect_binding_invalid', 'required EffectBindings contain duplicate effects', pointer='/effect_bindings')
    return tuple((effect for effect in EFFECT_VOCABULARY if effect in effects))

def build_verification_action(*, canonical_action: str, target_public_ref_or_null: str | None, intended_relation_or_null: str | None, effect_bindings: Sequence[Any] | None=None) -> dict[str, Any]:
    target = target_public_ref_or_null
    if target is not None:
        from scoper.harness.validation import public_semantic_name
        target = public_semantic_name(target, pointer='/action/target_public_ref_or_null')
    required = required_effects_for_action(canonical_action, intended_relation_or_null)
    if effect_bindings is not None:
        bound = _binding_required_effects(effect_bindings)
        if bound != required:
            raise MethodContractError('method_effect_binding_invalid', 'runtime EffectBinding differs from the Core visual effect contract', pointer='/effect_bindings')
    action = {'canonical_action': canonical_action, 'target_public_ref_or_null': target, 'intended_relation_or_null': intended_relation_or_null, 'required_effects': list(required)}
    exact_mapping(action, VERIFICATION_ACTION_FIELDS, pointer='/action')
    public_json(action, pointer='/action', include_future_action=True)
    return action

def validate_verification_action(action: Mapping[str, Any]) -> dict[str, Any]:
    item = exact_mapping(action, VERIFICATION_ACTION_FIELDS, pointer='/action')
    expected = build_verification_action(canonical_action=item['canonical_action'], target_public_ref_or_null=item['target_public_ref_or_null'], intended_relation_or_null=item['intended_relation_or_null'])
    effects = string_sequence(item['required_effects'], pointer='/action/required_effects', sorted_utf8=False)
    ordered = tuple((effect for effect in EFFECT_VOCABULARY if effect in effects))
    if tuple(effects) != ordered or list(effects) != expected['required_effects']:
        raise MethodContractError('method_effect_binding_invalid', 'action required_effects differ from the Core visual effect contract', pointer='/action/required_effects')
    return expected

def build_visual_verification_summary(*, action: Mapping[str, Any], tool_name: str, arguments: Mapping[str, Any], selected_outcome: str, selected_effects: Sequence[str], selected_observability: str) -> dict[str, Any]:
    action = validate_verification_action(action)
    name = non_empty_string(tool_name, pointer='/visual_verification/tool_name')
    if not isinstance(arguments, Mapping):
        raise MethodContractError('method_schema_invalid', 'visual verification arguments must be an object', pointer='/visual_verification/arguments')
    public_arguments = dict(arguments)
    target = public_arguments.get('target')
    if target != action['target_public_ref_or_null']:
        raise MethodContractError('method_event_correlation_invalid', 'visual verification target differs from the public ToolCall', pointer='/visual_verification/arguments/target')
    if selected_outcome not in OUTCOME_ORDER:
        raise MethodContractError('method_schema_invalid', 'visual verification outcome is invalid', pointer='/visual_verification/selected_outcome')
    effects = string_sequence(list(selected_effects), pointer='/visual_verification/selected_effects')
    ordered_effects = tuple((effect for effect in EFFECT_VOCABULARY if effect in effects))
    if effects != ordered_effects or any((effect not in EFFECT_VOCABULARY for effect in effects)):
        raise MethodContractError('method_schema_invalid', 'visual verification effects are outside canonical order or vocabulary', pointer='/visual_verification/selected_effects')
    if selected_outcome == 'success' and (not set(action['required_effects']).issubset(effects)):
        raise MethodContractError('method_effect_binding_invalid', 'successful visual verification omits a required effect', pointer='/visual_verification/selected_effects')
    if selected_observability not in OBSERVABILITY_ORDER:
        raise MethodContractError('method_schema_invalid', 'visual verification observability is invalid', pointer='/visual_verification/selected_observability')
    VerifierSelection(selected_outcome=selected_outcome, selected_effects=effects, selected_observability=selected_observability, effect_conflict='no_effect' in effects and len(effects) > 1)
    summary = {'tool_name': name, 'arguments': public_arguments, 'selected_outcome': selected_outcome, 'selected_effects': list(effects), 'selected_observability': selected_observability}
    exact_mapping(summary, VISUAL_VERIFICATION_SUMMARY_FIELDS, pointer='/visual_verification')
    public_json(summary, pointer='/visual_verification', include_future_action=True)
    return summary

def build_oracle_visual_verification_summary(*, action: Mapping[str, Any], tool_name: str, arguments: Mapping[str, Any], intended_effect_status: str, effect_observability: str) -> dict[str, Any]:
    action = validate_verification_action(action)
    try:
        outcome = ORACLE_EFFECT_STATUS_TO_OUTCOME[intended_effect_status]
    except (KeyError, TypeError) as exc:
        raise MethodContractError('method_schema_invalid', 'oracle effect status cannot be mapped to the Harness verifier contract', pointer='/oracle_supervision/intended_effect_status') from exc
    try:
        observability = ORACLE_EFFECT_OBSERVABILITY[effect_observability]
    except (KeyError, TypeError) as exc:
        raise MethodContractError('method_schema_invalid', 'oracle effect observability cannot be mapped to the Harness verifier contract', pointer='/oracle_supervision/effect_observability') from exc
    effects = action['required_effects'] if outcome == 'success' else ['no_effect'] if outcome == 'failure' else []
    summary = build_visual_verification_summary(action=action, tool_name=tool_name, arguments=arguments, selected_outcome=outcome, selected_effects=effects, selected_observability=observability)
    return validate_visual_verification_summary(summary, action=action)

def validate_visual_verification_summary(value: Mapping[str, Any], *, action: Mapping[str, Any]) -> dict[str, Any]:
    item = exact_mapping(value, VISUAL_VERIFICATION_SUMMARY_FIELDS, pointer='/visual_verification')
    summary = build_visual_verification_summary(action=action, tool_name=item['tool_name'], arguments=item['arguments'], selected_outcome=item['selected_outcome'], selected_effects=item['selected_effects'], selected_observability=item['selected_observability'])
    public_relation = summary['arguments'].get('relation')
    if public_relation != action['intended_relation_or_null']:
        if public_relation is not None or action['intended_relation_or_null'] is not None:
            raise MethodContractError('method_event_correlation_invalid', 'visual verification relation differs from the public ToolCall', pointer='/visual_verification/arguments/relation')
    return summary

def build_verifier_request(*, verifier_call_id: str, action: Mapping[str, Any], evidence_frame_ids: Sequence[str], pre_rgb: Mapping[str, Any], post_rgb: Mapping[str, Any] | None=None, immediate_post_rgb: Mapping[str, Any] | None=None, confirmation_rgb: Mapping[str, Any] | None=None) -> dict[str, Any]:
    action = validate_verification_action(action)
    frames = string_sequence(list(evidence_frame_ids), pointer='/evidence_frame_ids', non_empty=True)
    if immediate_post_rgb is not None or confirmation_rgb is not None:
        if len(frames) != 3 or immediate_post_rgb is None or confirmation_rgb is None:
            raise MethodContractError('required_frame_missing', 'v12 verifier requires exactly three ordered frames')
        request = {'schema_version': 'transition-verifier-request.v3', 'verifier_call_id': non_empty_string(verifier_call_id, pointer='/verifier_call_id'), 'action': action, 'evidence_frame_ids': list(frames), 'images': {'pre_rgb': _image_metadata(pre_rgb, pointer='/images/pre_rgb'), 'immediate_post_rgb': _image_metadata(immediate_post_rgb, pointer='/images/immediate_post_rgb'), 'confirmation_rgb': _image_metadata(confirmation_rgb, pointer='/images/confirmation_rgb')}}
    else:
        if len(frames) != 2 or post_rgb is None:
            raise MethodContractError('required_frame_missing', 'legacy verifier requires exactly two ordered frames')
        request = {'schema_version': 'transition-verifier-request.v2', 'verifier_call_id': non_empty_string(verifier_call_id, pointer='/verifier_call_id'), 'action': action, 'evidence_frame_ids': list(frames), 'images': {'pre_rgb': _image_metadata(pre_rgb, pointer='/images/pre_rgb'), 'post_rgb': _image_metadata(post_rgb, pointer='/images/post_rgb')}}
    public_json(request, pointer='/verifier_request', include_future_action=True)
    return request

def validate_verifier_output(value: Mapping[str, Any], thresholds: Mapping[str, Any], *, required_effects: Sequence[str]=()) -> VerifierSelection:
    output = exact_mapping(value, VERIFIER_OUTPUT_FIELDS, pointer='/verifier_output')
    if output['schema_version'] not in {'transition-verifier-io.v1', 'transition-verifier-io.v2'}:
        raise MethodContractError('method_verifier_output_invalid', 'verifier output schema is invalid')
    non_empty_string(output['verifier_call_id'], pointer='/verifier_output/verifier_call_id')
    temperature = finite_number(output['outcome_temperature'], pointer='/verifier_output/outcome_temperature', minimum=0)
    if temperature <= 0:
        raise MethodContractError('method_verifier_output_invalid', 'outcome temperature must be positive')
    sha256(output['checkpoint_sha256'], pointer='/verifier_output/checkpoint_sha256')
    frames = string_sequence(output['evidence_frame_ids'], pointer='/verifier_output/evidence_frame_ids', non_empty=True)
    expected_frame_count = 3 if output['schema_version'] == 'transition-verifier-io.v2' else 2
    if len(frames) != expected_frame_count:
        raise MethodContractError('required_frame_missing', 'verifier output evidence count differs from its schema version')
    selection = select_verifier_result({'outcome': output['outcome_probabilities'], 'effects': output['effect_probabilities'], 'observability': output['observability_probabilities']}, thresholds, required_effects=required_effects)
    selected_effects = string_sequence(output['selected_effects'], pointer='/verifier_output/selected_effects')
    if output['selected_outcome'] != selection.selected_outcome or selected_effects != selection.selected_effects or output['selected_observability'] != selection.selected_observability:
        raise MethodContractError('method_verifier_output_invalid', 'selected verifier values differ from calibrated rules')
    public_json(output, pointer='/verifier_output', include_future_action=True)
    return selection

def gate_online_visual_selection(selection: VerifierSelection) -> VerifierSelection:
    if selection.selected_outcome == 'success' and selection.selected_observability == 'unobservable':
        return VerifierSelection(selected_outcome='ambiguous', selected_effects=selection.selected_effects, selected_observability=selection.selected_observability, effect_conflict=selection.effect_conflict)
    return selection
__all__ = ['ACTION_REQUIRED_EFFECTS', 'VERIFICATION_ACTION_FIELDS', 'VISUAL_VERIFICATION_SUMMARY_FIELDS', 'ORACLE_EFFECT_OBSERVABILITY', 'ORACLE_EFFECT_STATUS_TO_OUTCOME', 'VerifierClient', 'OBSERVABILITY_ORDER', 'OUTCOME_ORDER', 'VerifierSelection', 'build_verification_action', 'build_oracle_visual_verification_summary', 'build_visual_verification_summary', 'build_verifier_request', 'gate_online_visual_selection', 'required_effects_for_action', 'select_verifier_result', 'validate_verification_action', 'validate_visual_verification_summary', 'validate_verifier_output']
