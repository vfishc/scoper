from __future__ import annotations
import copy
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Protocol, Sequence
from scoper.harness.canonical import canonical_json_bytes, sha256_json
HARNESS_SPEC_SCHEMA = 'embodied-method-harness-spec.v1'
HARNESS_DEFINITION_SCHEMA = 'embodied-method-harness-definition.v1'
DEFAULT_HARNESS_ID = 'sim_only_core_text_hardqa'
LEGACY_HARNESS_VERSION = '2'
HARNESS_V3_VERSION = '3'
HARNESS_V4_VERSION = '4'
HARNESS_V5_VERSION = '5'
HARNESS_V6_VERSION = '6'
HARNESS_V7_VERSION = '7'
HARNESS_V8_VERSION = '8'
HARNESS_V9_VERSION = '9'
HARNESS_V10_VERSION = '10'
HARNESS_V11_VERSION = '11'
HARNESS_V12_VERSION = '12'
HARNESS_V13_VERSION = '13'
HARNESS_V14_VERSION = '14'
HARNESS_V15_VERSION = '15'
HARNESS_V16_VERSION = '16'
HARNESS_V17_VERSION = '17'
HARNESS_V18_VERSION = '18'
HARNESS_V19_VERSION = '19'
HARNESS_V20_VERSION = '20'
HARNESS_V21_VERSION = 'scoper-1'
DEFAULT_HARNESS_VERSION = HARNESS_V12_VERSION
_V13_PREPARATION_TOOL_NAMES = frozenset({'look_for', 'navigate_to', 'scan'})
_BOUNDED_RECOVERY_HARNESS_VERSIONS = frozenset({HARNESS_V13_VERSION, HARNESS_V14_VERSION, HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION, HARNESS_V15_VERSION, HARNESS_V16_VERSION, HARNESS_V17_VERSION})

def public_transition_error_type(transition: Mapping[str, Any] | None) -> str | None:
    if not isinstance(transition, Mapping) or transition.get('success') is True:
        return None
    feedback = transition.get('env_feedback')
    if not isinstance(feedback, Mapping):
        return None
    error_type = feedback.get('error_type')
    return error_type if isinstance(error_type, str) and error_type else None
TRAINING_COMPONENT_SAMPLE_KINDS = (('memory', ('memory_reconcile', 'memory_update')), ('policy', ('recovery_decision', 'step_decision')), ('verifier', ('verifier_transition',)))

class HarnessCapabilityError(ValueError):
    pass

class PolicyModelClient(Protocol):

    def act(self, request: Any, **kwargs: Any) -> Any:
        ...

    def close(self) -> None:
        ...

class VerifierModelClient(Protocol):

    def verify(self, request: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...

class MemoryModelClient(Protocol):

    def propose(self, request: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...

@dataclass(frozen=True)
class HarnessSpec:
    schema_version: str
    harness_id: str
    harness_version: str
    capabilities: tuple[str, ...]
    evidence_requirements: tuple[str, ...]
    hardqa_roles: tuple[str, ...]
    hardqa_output_mode: str
    source_kinds: tuple[str, ...]
    canonical_sample_kinds: tuple[str, ...]
    effect_vocabulary_version: str
    memory_vocabulary_version: str
    tool_schema_version: str
    prompt_contract_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != HARNESS_SPEC_SCHEMA:
            raise ValueError('unsupported HarnessSpec schema')
        for name in ('harness_id', 'harness_version', 'hardqa_output_mode', 'effect_vocabulary_version', 'memory_vocabulary_version', 'tool_schema_version', 'prompt_contract_sha256'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f'HarnessSpec.{name} must be non-empty text')
        if len(self.prompt_contract_sha256) != 64 or any((character not in '0123456789abcdef' for character in self.prompt_contract_sha256)):
            raise ValueError('HarnessSpec.prompt_contract_sha256 must be sha256')
        for name in ('capabilities', 'evidence_requirements', 'hardqa_roles', 'source_kinds', 'canonical_sample_kinds'):
            values = getattr(self, name)
            if not values or values != tuple(sorted(set(values), key=str.encode)):
                raise ValueError(f'HarnessSpec.{name} must be a non-empty sorted unique tuple')

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ('capabilities', 'evidence_requirements', 'hardqa_roles', 'source_kinds', 'canonical_sample_kinds'):
            value[name] = list(value[name])
        return value

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'HarnessSpec':
        expected = {'schema_version', 'harness_id', 'harness_version', 'capabilities', 'evidence_requirements', 'hardqa_roles', 'hardqa_output_mode', 'source_kinds', 'canonical_sample_kinds', 'effect_vocabulary_version', 'memory_vocabulary_version', 'tool_schema_version', 'prompt_contract_sha256'}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError('HarnessSpec fields are not exact')
        kwargs = dict(value)
        for name in ('capabilities', 'evidence_requirements', 'hardqa_roles', 'source_kinds', 'canonical_sample_kinds'):
            raw = kwargs[name]
            if not isinstance(raw, list) or any((not isinstance(item, str) for item in raw)):
                raise ValueError(f'HarnessSpec.{name} must be a string list')
            kwargs[name] = tuple(raw)
        return cls(**kwargs)

def _default_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import hardqa_prompt_contract_descriptor
    return hardqa_prompt_contract_descriptor()

def _legacy_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import LEGACY_HARDQA_PROMPT_CONTRACT_IDENTITY
    return LEGACY_HARDQA_PROMPT_CONTRACT_IDENTITY

def _v3_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V3_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V3_CONTRACT_IDENTITY

def _v4_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V4_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V4_CONTRACT_IDENTITY

def _v5_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V5_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V5_CONTRACT_IDENTITY

def _v9_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V9_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V9_CONTRACT_IDENTITY

def _v10_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V10_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V10_CONTRACT_IDENTITY

def _v11_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V11_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V11_CONTRACT_IDENTITY

def _v13_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V13_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V13_CONTRACT_IDENTITY

def _v14_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V14_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V14_CONTRACT_IDENTITY

def _v15_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V15_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V15_CONTRACT_IDENTITY

def _v16_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V16_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V16_CONTRACT_IDENTITY

def _v17_prompt_descriptor() -> Mapping[str, Any]:
    from scoper.harness.prompt_contracts import HARDQA_PROMPT_V17_CONTRACT_IDENTITY
    return HARDQA_PROMPT_V17_CONTRACT_IDENTITY
HARNESS_V9_SPEC = HarnessSpec(schema_version=HARNESS_SPEC_SCHEMA, harness_id=DEFAULT_HARNESS_ID, harness_version=HARNESS_V9_VERSION, capabilities=('memory', 'policy', 'recovery', 'stream_evidence', 'verifier'), evidence_requirements=('current_rgb', 'post_action_rgb', 'pre_action_rgb', 'stream_window'), hardqa_roles=('memory_reconcile', 'memory_update', 'policy', 'verifier'), hardqa_output_mode='plain_text', source_kinds=('memory_reconcile', 'memory_route', 'memory_update', 'step_decision', 'stream_decision', 'verification_recovery'), canonical_sample_kinds=('memory_reconcile', 'memory_route', 'memory_update', 'recovery_decision', 'step_decision', 'verifier_transition'), effect_vocabulary_version='method-effects.v1', memory_vocabulary_version='method-memory-delta.v2', tool_schema_version='tools.v1', prompt_contract_sha256=sha256_json(_v9_prompt_descriptor()))
HARNESS_V9_SPEC_SHA256 = HARNESS_V9_SPEC.sha256
HARNESS_V10_SPEC = replace(HARNESS_V9_SPEC, harness_version=HARNESS_V10_VERSION, capabilities=tuple(sorted({*HARNESS_V9_SPEC.capabilities, 'visual_only_physical_verification'}, key=str.encode)), prompt_contract_sha256=sha256_json(_v10_prompt_descriptor()))
HARNESS_V10_SPEC_SHA256 = HARNESS_V10_SPEC.sha256
HARNESS_V11_SPEC = replace(HARNESS_V10_SPEC, harness_version=HARNESS_V11_VERSION, prompt_contract_sha256=sha256_json(_v11_prompt_descriptor()))
HARNESS_V11_SPEC_SHA256 = HARNESS_V11_SPEC.sha256
DEFAULT_HARNESS_SPEC = replace(HARNESS_V11_SPEC, harness_version=DEFAULT_HARNESS_VERSION, capabilities=tuple(sorted({*HARNESS_V11_SPEC.capabilities, 'logical_evidence_1fps', 'three_frame_visual_verification', 'verifier_authoritative_memory_updates'}, key=str.encode)), evidence_requirements=('confirmation_rgb', 'current_rgb', 'immediate_post_action_rgb', 'pre_action_rgb', 'stream_window'), prompt_contract_sha256=sha256_json(_default_prompt_descriptor()))
DEFAULT_HARNESS_SPEC_SHA256 = DEFAULT_HARNESS_SPEC.sha256
HARNESS_V12_SPEC = DEFAULT_HARNESS_SPEC
HARNESS_V12_SPEC_SHA256 = HARNESS_V12_SPEC.sha256
HARNESS_V13_SPEC = replace(HARNESS_V12_SPEC, harness_version=HARNESS_V13_VERSION, capabilities=tuple(sorted({*HARNESS_V12_SPEC.capabilities, 'bounded_original_action_retry'}, key=str.encode)), prompt_contract_sha256=sha256_json(_v13_prompt_descriptor()))
HARNESS_V13_SPEC_SHA256 = HARNESS_V13_SPEC.sha256
HARNESS_V14_SPEC = replace(HARNESS_V13_SPEC, harness_version=HARNESS_V14_VERSION, capabilities=tuple(sorted({*HARNESS_V13_SPEC.capabilities, 'passive_visual_discovery', 'scene_observation_memory'}, key=str.encode)), hardqa_roles=tuple(sorted({*HARNESS_V13_SPEC.hardqa_roles, 'memory_observe'}, key=str.encode)), source_kinds=tuple(sorted({*HARNESS_V13_SPEC.source_kinds, 'memory_observe'}, key=str.encode)), canonical_sample_kinds=tuple(sorted({*HARNESS_V13_SPEC.canonical_sample_kinds, 'memory_observe'}, key=str.encode)), memory_vocabulary_version='method-memory-delta.v3', prompt_contract_sha256=sha256_json(_v14_prompt_descriptor()))
HARNESS_V14_SPEC_SHA256 = HARNESS_V14_SPEC.sha256
HARNESS_V15_SPEC = replace(HARNESS_V14_SPEC, harness_version=HARNESS_V15_VERSION, capabilities=tuple(sorted({*HARNESS_V14_SPEC.capabilities, 'instruction_conditioned_discovery', 'actionable_scene_observation_memory'}, key=str.encode)), prompt_contract_sha256=sha256_json(_v15_prompt_descriptor()))
HARNESS_V15_SPEC_SHA256 = HARNESS_V15_SPEC.sha256
HARNESS_V16_SPEC = replace(HARNESS_V15_SPEC, harness_version=HARNESS_V16_VERSION, capabilities=tuple(sorted({*HARNESS_V15_SPEC.capabilities, 'shared_public_vocabulary'}, key=str.encode)), memory_vocabulary_version='method-memory-delta.v4', prompt_contract_sha256=sha256_json(_v16_prompt_descriptor()))
HARNESS_V16_SPEC_SHA256 = HARNESS_V16_SPEC.sha256
HARNESS_V17_SPEC = replace(HARNESS_V16_SPEC, harness_version=HARNESS_V17_VERSION, memory_vocabulary_version='method-memory-delta.v5', prompt_contract_sha256=sha256_json(_v17_prompt_descriptor()))
HARNESS_V17_SPEC_SHA256 = HARNESS_V17_SPEC.sha256
from scoper.harness.prompt_contracts import hardqa_prompt_contract_descriptor_for_harness_version
HARNESS_V18_SPEC = replace(HARNESS_V14_SPEC, harness_version=HARNESS_V18_VERSION, capabilities=tuple(sorted({*HARNESS_V14_SPEC.capabilities, 'shared_public_vocabulary', 'bounded_concrete_observation_memory'})), memory_vocabulary_version='method-memory-delta.v6', prompt_contract_sha256=sha256_json(hardqa_prompt_contract_descriptor_for_harness_version('18')))
HARNESS_V18_SPEC_SHA256 = HARNESS_V18_SPEC.sha256
HARNESS_V19_SPEC = replace(HARNESS_V18_SPEC, harness_version=HARNESS_V19_VERSION, prompt_contract_sha256=sha256_json(hardqa_prompt_contract_descriptor_for_harness_version('19')))
HARNESS_V19_SPEC_SHA256 = HARNESS_V19_SPEC.sha256
HARNESS_V20_SPEC = replace(HARNESS_V19_SPEC, harness_version=HARNESS_V20_VERSION, capabilities=tuple((c for c in HARNESS_V19_SPEC.capabilities if c != 'three_frame_visual_verification')), evidence_requirements=tuple((c for c in HARNESS_V19_SPEC.evidence_requirements if c != 'confirmation_rgb')), prompt_contract_sha256=sha256_json(hardqa_prompt_contract_descriptor_for_harness_version('20')))
HARNESS_V20_SPEC_SHA256 = HARNESS_V20_SPEC.sha256
HARNESS_V21_SPEC = replace(HARNESS_V20_SPEC, harness_version=HARNESS_V21_VERSION, capabilities=tuple(sorted({*HARNESS_V20_SPEC.capabilities, 'dynamic_evidence_agenda', 'separate_uncertain_planning', 'public_instruction_updates'})), prompt_contract_sha256=sha256_json(hardqa_prompt_contract_descriptor_for_harness_version('scoper-1')))
HARNESS_V21_SPEC_SHA256 = HARNESS_V21_SPEC.sha256
HARNESS_V3_SPEC = replace(HARNESS_V9_SPEC, harness_version=HARNESS_V3_VERSION, prompt_contract_sha256=sha256_json(_v3_prompt_descriptor()), memory_vocabulary_version='method-memory-delta.v1')
HARNESS_V3_SPEC_SHA256 = HARNESS_V3_SPEC.sha256
HARNESS_V4_SPEC = replace(HARNESS_V9_SPEC, harness_version=HARNESS_V4_VERSION, prompt_contract_sha256=sha256_json(_v4_prompt_descriptor()), memory_vocabulary_version='method-memory-delta.v1')
HARNESS_V4_SPEC_SHA256 = HARNESS_V4_SPEC.sha256
HARNESS_V5_SPEC = replace(HARNESS_V9_SPEC, harness_version=HARNESS_V5_VERSION, prompt_contract_sha256=sha256_json(_v5_prompt_descriptor()))
HARNESS_V5_SPEC_SHA256 = HARNESS_V5_SPEC.sha256
HARNESS_V6_SPEC = replace(HARNESS_V9_SPEC, harness_version=HARNESS_V6_VERSION)
HARNESS_V6_SPEC_SHA256 = HARNESS_V6_SPEC.sha256
HARNESS_V7_SPEC = replace(HARNESS_V9_SPEC, harness_version=HARNESS_V7_VERSION)
HARNESS_V7_SPEC_SHA256 = HARNESS_V7_SPEC.sha256
HARNESS_V8_SPEC = replace(HARNESS_V9_SPEC, harness_version=HARNESS_V8_VERSION)
HARNESS_V8_SPEC_SHA256 = HARNESS_V8_SPEC.sha256
LEGACY_HARNESS_SPEC = replace(HARNESS_V9_SPEC, harness_version=LEGACY_HARNESS_VERSION, prompt_contract_sha256=sha256_json(_legacy_prompt_descriptor()), memory_vocabulary_version='method-memory-delta.v1')
LEGACY_HARNESS_SPEC_SHA256 = LEGACY_HARNESS_SPEC.sha256

def assert_harness_spec(value: Mapping[str, Any], expected_sha256: str | None=None) -> HarnessSpec:
    spec = HarnessSpec.from_dict(value)
    if expected_sha256 is not None and spec.sha256 != expected_sha256:
        raise ValueError('HarnessSpec sha256 mismatch')
    return spec

@dataclass(frozen=True)
class HarnessDefinition:
    schema_version: str
    spec: HarnessSpec
    implementation_id: str
    component_sample_kinds: tuple[tuple[str, tuple[str, ...]], ...]
    stream_window_size: int = 16

    def __post_init__(self) -> None:
        if self.schema_version != HARNESS_DEFINITION_SCHEMA:
            raise ValueError('unsupported HarnessDefinition schema')
        if not isinstance(self.implementation_id, str) or not self.implementation_id:
            raise ValueError('HarnessDefinition.implementation_id must be non-empty text')
        if isinstance(self.stream_window_size, bool) or self.stream_window_size < 1:
            raise ValueError('HarnessDefinition.stream_window_size must be positive')
        components = tuple((name for name, _ in self.component_sample_kinds))
        if components != tuple(sorted(set(components), key=str.encode)):
            raise ValueError('HarnessDefinition components must be sorted and unique')
        declared = set(self.spec.canonical_sample_kinds)
        for component, kinds in self.component_sample_kinds:
            if not kinds or kinds != tuple(sorted(set(kinds), key=str.encode)):
                raise ValueError(f'HarnessDefinition sample kinds for {component} are invalid')
            if not set(kinds).issubset(declared):
                raise ValueError(f'HarnessDefinition sample kinds for {component} are undeclared')

    @property
    def harness_spec_sha256(self) -> str:
        return self.spec.sha256

    @property
    def version(self) -> str:
        return self.spec.harness_version

    @property
    def legacy_simulator_postconditions_enabled(self) -> bool:
        return self.version in {HARNESS_V7_VERSION, HARNESS_V8_VERSION, HARNESS_V9_VERSION}

    @property
    def visual_only_state_enabled(self) -> bool:
        return 'visual_only_physical_verification' in self.spec.capabilities

    @property
    def verifier_authoritative_memory_updates(self) -> bool:
        return 'verifier_authoritative_memory_updates' in self.spec.capabilities

    @property
    def passive_observe_enabled(self) -> bool:
        return 'passive_visual_discovery' in self.spec.capabilities

    def project_completed_tool_record(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        from scoper.harness.contracts import MethodContractError
        from scoper.harness.validation import public_json
        if not isinstance(value, Mapping):
            raise MethodContractError('method_schema_invalid', 'tool record must be a mapping')
        tool_name = value.get('tool_name')
        arguments = value.get('arguments')
        if not isinstance(tool_name, str) or not tool_name or (not isinstance(arguments, Mapping)):
            raise MethodContractError('method_schema_invalid', 'tool record is incomplete')
        output: dict[str, Any] = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments)), 'completed': True}
        step = value.get('high_level_step')
        if isinstance(step, int) and (not isinstance(step, bool)) and (step >= 0):
            output['high_level_step'] = step
        if self.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS and isinstance(value.get('binding'), Mapping):
            public_binding_fields = ('status', 'reason', 'source', 'match_kind', 'canonical_public_ref', 'navigation_class')
            output['binding'] = {field: copy.deepcopy(value['binding'][field]) for field in public_binding_fields if field in value['binding']}
        if self.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS and isinstance(value.get('preparation_result'), Mapping):
            from scoper.harness.recovery import _validate_preparation_result
            output['preparation_result'] = _validate_preparation_result(value['preparation_result'])
        if self.version in {HARNESS_V15_VERSION, HARNESS_V16_VERSION} and value.get('success') is False:
            feedback = value.get('env_feedback')
            error_type = feedback.get('error_type') if isinstance(feedback, Mapping) else None
            if isinstance(error_type, str) and error_type:
                output['env_feedback'] = {'error_type': error_type}
        public_json(output, pointer='/completed_tool_record', include_future_action=False)
        return output

    def project_training_source_input(self, source_kind: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if source_kind not in self.spec.source_kinds or not isinstance(value, Mapping):
            raise ValueError('training source kind or input is invalid')
        output = copy.deepcopy(dict(value))
        if not self.visual_only_state_enabled:
            return output
        if source_kind == 'step_decision':
            history = output.get('tool_history')
            if not isinstance(history, list):
                raise ValueError('visual-only policy source requires ordered tool history')
            output['tool_history'] = [dict(self.project_completed_tool_record(item)) for item in history]
            transition = output.get('last_public_transition')
            output['last_public_transition'] = None if transition is None else dict(self.project_completed_tool_record(transition))
            output['prompt_context'] = {}
            output['public_observation'] = {}
        elif source_kind == 'stream_decision':
            output['public_progress'] = None
        elif source_kind == 'verification_recovery':
            for key in ('last_public_transition', 'public_feedback', 'public_memory'):
                output.pop(key, None)
        elif source_kind == 'memory_update':
            if self.version in {HARNESS_V11_VERSION, DEFAULT_HARNESS_VERSION, HARNESS_V13_VERSION, HARNESS_V14_VERSION, HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION, HARNESS_V15_VERSION, HARNESS_V16_VERSION}:
                from scoper.harness.validation import exact_mapping, public_json
                from scoper.harness.verifier import VISUAL_VERIFICATION_SUMMARY_FIELDS
                transition = output.get('public_transition')
                exact_mapping(transition, VISUAL_VERIFICATION_SUMMARY_FIELDS, pointer='/memory_update/public_transition')
                public_json(transition, pointer='/memory_update/public_transition', include_future_action=True)
                bindings = output.get('public_effect_bindings')
                if not isinstance(bindings, list):
                    raise ValueError('v11 memory source requires public effect bindings')
                public_json(bindings, pointer='/memory_update/public_effect_bindings', include_future_action=False)
                allowed_aliases: list[str] = []
                tool_call = output.get('attempted_public_tool_call')
                if isinstance(tool_call, Mapping):
                    arguments = tool_call.get('arguments')
                    if isinstance(arguments, Mapping):
                        target = arguments.get('target')
                        if isinstance(target, str) and target:
                            allowed_aliases.append(target)
                for binding in bindings:
                    if not isinstance(binding, Mapping):
                        continue
                    for key in ('subject_public_role', 'object_public_role_or_value'):
                        alias = binding.get(key)
                        if isinstance(alias, str) and alias and (alias not in allowed_aliases):
                            allowed_aliases.append(alias)
                output['public_entity_candidates'] = sorted(set(allowed_aliases), key=lambda item: item.encode('utf-8'))
            else:
                output['public_effect_bindings'] = []
                output['public_entity_candidates'] = []
                output['public_transition'] = None
        elif source_kind in {'memory_reconcile', 'memory_route'}:
            output['public_observation'] = {}
        return output

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': self.schema_version, 'harness_spec': self.spec.to_dict(), 'harness_spec_sha256': self.harness_spec_sha256, 'implementation_id': self.implementation_id, 'component_sample_kinds': {component: list(kinds) for component, kinds in self.component_sample_kinds}, 'stream_window_size': self.stream_window_size}

    def preflight_capture_capabilities(self, backend: str, capabilities: Mapping[str, bool] | Sequence[str]) -> Mapping[str, Any]:
        if not isinstance(backend, str) or not backend:
            raise HarnessCapabilityError('backend must be non-empty text')
        available = {name for name, enabled in capabilities.items() if enabled is True} if isinstance(capabilities, Mapping) else set(capabilities)
        missing = tuple((requirement for requirement in self.spec.evidence_requirements if requirement not in available))
        if missing:
            raise HarnessCapabilityError(f'{backend} cannot capture Harness {self.version}; missing evidence: {missing!r}')
        return {'schema_version': 'harness-capture-preflight.v1', 'backend': backend, 'harness_spec_sha256': self.harness_spec_sha256, 'evidence_requirements': list(self.spec.evidence_requirements), 'status': 'passed'}

    def required_training_cells(self, backends: Sequence[str]) -> tuple[tuple[str, str], ...]:
        domains = tuple(sorted(set(backends), key=str.encode))
        if not domains or any((not isinstance(item, str) or not item for item in domains)):
            raise ValueError('training backends must be non-empty unique text')
        kinds = {kind for _, component_kinds in self.component_sample_kinds for kind in component_kinds}
        return tuple(((backend, kind) for backend in domains for kind in sorted(kinds, key=str.encode)))

    def validate_training_coverage(self, samples: Sequence[Mapping[str, Any]], *, backends: Sequence[str]) -> Mapping[str, Any]:
        required = self.required_training_cells(backends)
        counts = {cell: 0 for cell in required}
        for sample in samples:
            domain = sample.get('domain')
            kind = sample.get('sample_kind')
            cell = (domain, kind)
            if cell in counts:
                counts[cell] += 1
        missing = tuple((cell for cell in required if counts[cell] == 0))
        if missing:
            raise HarnessCapabilityError(f'joint canonical dataset lacks Harness training cells: {missing!r}')
        return {'schema_version': 'harness-training-coverage.v1', 'harness_spec_sha256': self.harness_spec_sha256, 'counts': [{'domain': domain, 'sample_kind': kind, 'count': counts[domain, kind]} for domain, kind in required], 'status': 'passed'}

    def create_session(self, *, memory_capacity: Any, recovery_budgets: Any, stream_enabled_tools: Sequence[str], event_sink: Any | None=None, router_limits: Any | None=None) -> 'HarnessSession':
        session_type = HarnessSession
        if self.version == 'scoper-1':
            from scoper.harness.session import SCOPERSession
            session_type = SCOPERSession
        return session_type(self, memory_capacity=memory_capacity, recovery_budgets=recovery_budgets, stream_enabled_tools=stream_enabled_tools, event_sink=event_sink, router_limits=router_limits)

class HarnessSession:

    def __init__(self, definition: HarnessDefinition, *, memory_capacity: Any, recovery_budgets: Any, stream_enabled_tools: Sequence[str], event_sink: Any | None, router_limits: Any | None) -> None:
        from scoper.harness.events import NullMethodEventSink
        from scoper.harness.memory_router import RouterLimits
        from scoper.harness.recovery import RecoveryLedger
        from scoper.harness.stream_evidence import StreamEvidenceBuffer
        self.definition = definition
        self.memory_capacity = memory_capacity
        self.recovery_budgets = recovery_budgets
        self.router_limits = router_limits or RouterLimits()
        self.event_sink = event_sink or NullMethodEventSink()
        self.stream_enabled_tools = tuple(stream_enabled_tools)
        self.stream_buffer = StreamEvidenceBuffer(stream_enabled_tools=self.stream_enabled_tools, max_refs=definition.stream_window_size, selection_mode='uniform_v2' if definition.passive_observe_enabled else 'legacy_tail')
        self._episode_id: str | None = None
        self._task_instruction = ''
        self._memory: Any | None = None
        self._recovery_ledger = RecoveryLedger()
        self._last_successful_navigation_roles: tuple[str, ...] = ()
        self._pending_public_retries: dict[str, dict[str, Any]] = {}
        self._recovery_contexts: dict[str, Any] = {}
        self._pending_recovery_actions: dict[str, dict[str, Any]] = {}
        self._placement_recoveries: dict[str, Any] = {}
        self._last_recovery_notice: dict[str, Any] | None = None
        self._assumed_completed_actions: list[dict[str, Any]] = []
        self._exhausted_preparation_fingerprints: set[bytes] = set()
        self._private_binding_receipts: dict[str, tuple[str, ...]] = {}
        self._private_recovery_binding_receipts: dict[str, tuple[str, ...]] = {}
        self._private_processed_binding_role_set: set[str] = set()

    @property
    def active(self) -> bool:
        return self._episode_id is not None

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    @property
    def task_instruction(self) -> str:
        return self._task_instruction

    @property
    def memory(self) -> Any | None:
        return self._memory

    @memory.setter
    def memory(self, value: Any | None) -> None:
        if value is not None:
            validated = value.validated()
            if self._episode_id is not None and validated.episode_id != self._episode_id:
                raise ValueError('HarnessSession memory episode differs')
            value = validated
        self._memory = value
        if value is not None:
            open_incident_ids = {incident.incident_id for incident in value.incidents if incident.status == 'open'}
            self._pending_public_retries = {incident_id: retry for incident_id, retry in self._pending_public_retries.items() if incident_id in open_incident_ids}
            self._recovery_contexts = {incident_id: context for incident_id, context in self._recovery_contexts.items() if incident_id in open_incident_ids}

    @property
    def recovery_ledger(self) -> Any:
        return self._recovery_ledger

    @recovery_ledger.setter
    def recovery_ledger(self, value: Any) -> None:
        self._recovery_ledger = value

    def configure_event_sink(self, event_sink: Any) -> None:
        if self.active:
            raise RuntimeError('cannot replace Harness event sink during an episode')
        self.event_sink = event_sink

    def public_recovery_context(self, recovery_plan: Any) -> Mapping[str, Any] | None:
        if self.definition.version not in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            return None
        incident_id = getattr(recovery_plan, 'incident_id_or_null', None)
        context = self._recovery_contexts.get(incident_id)
        if context is not None:
            if incident_id in self._placement_recoveries:
                return {**context.to_mapping(), 'placement_recovery': True, 'required_tool_call': copy.deepcopy(recovery_plan.required_tool_call_or_null)}
            return context.to_mapping()
        if getattr(recovery_plan, 'reason', None) == 'recovery_assumed_continue':
            return copy.deepcopy(self._last_recovery_notice)
        return None

    def public_assumed_completed_actions(self) -> tuple[Mapping[str, Any], ...]:
        if self.definition.version not in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            return ()
        return tuple((copy.deepcopy(item) for item in self._assumed_completed_actions))

    def public_assumed_effects(self) -> tuple[Mapping[str, Any], ...]:
        if self.definition.version not in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            return ()
        effects: list[dict[str, Any]] = []
        for action in self._assumed_completed_actions:
            for effect in action.get('assumed_effects', ()):
                effects.append(copy.deepcopy(dict(effect)))
        for incident_id, placement in self._placement_recoveries.items():
            if placement.pickup_assumed and placement.phase in {'navigate_destination', 'retry'}:
                effects.append({'effect': 'picked_up', 'subject_public_role': placement.object_ref, 'relation_or_property': 'held_by_agent', 'object_public_role_or_value': True, 'certainty': 'uncertain', 'source_incident_id': incident_id, 'source_frame_ids': []})
        return tuple(effects)

    def public_effective_facts(self) -> tuple[Mapping[str, Any], ...]:
        verified = []
        if self._memory is not None:
            verified = [{**fact.to_mapping(), 'certainty': 'verified'} for fact in self._memory.facts if fact.status == 'current' and fact.relation_or_property != 'scene_observation']
        assumed = list(self.public_assumed_effects())
        assumed_keys = {(item['subject_public_role'], item['relation_or_property']) for item in assumed}
        verified = [item for item in verified if (item.get('subject_public_role'), item.get('relation_or_property')) not in assumed_keys]
        return tuple((*verified, *assumed))

    def public_scene_observations(self) -> tuple[Mapping[str, Any], ...]:
        if self.definition.version not in {HARNESS_V15_VERSION, HARNESS_V16_VERSION, HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION} or self._memory is None:
            return ()
        return tuple((fact.to_mapping() for fact in self._memory.facts if fact.status == 'current' and fact.relation_or_property == 'scene_observation'))

    def public_unresolved_held_placement(self) -> bool:
        if self._memory is None:
            return False
        return any((fact.status == 'current' and fact.relation_or_property == 'held_by_agent' and (fact.object_public_role_or_value is True) for fact in self._memory.facts))

    def record_private_binding_receipt(self, action_id: str, roles: Sequence[str]) -> None:
        if self.definition.version not in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            return
        if not self.active or not isinstance(action_id, str) or (not action_id):
            raise ValueError('private Binding receipt requires an active action')
        normalized = tuple(sorted({str(role) for role in roles if isinstance(role, str) and role}, key=lambda item: item.encode('utf-8')))
        if normalized:
            self._private_binding_receipts[action_id] = normalized

    def private_processed_binding_roles(self) -> tuple[str, ...]:
        if self.definition.version not in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            return ()
        return tuple(sorted(self._private_processed_binding_role_set, key=str.encode))

    def _commit_private_binding_receipt(self, action_id: str) -> None:
        self._private_processed_binding_role_set.update(self._private_binding_receipts.get(action_id, ()))

    def _reconcile_assumed_effects(self, effect_bindings: Sequence[Any], *, selected_effects: Sequence[str]) -> None:
        if self.definition.version not in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            return
        selected = set(selected_effects)
        verified_keys = {(binding.subject_public_role, binding.relation_or_property) for binding in effect_bindings if binding.effect in selected and binding.binding_status == 'known'}
        if not verified_keys:
            return
        retained = []
        for action in self._assumed_completed_actions:
            effects = [effect for effect in action.get('assumed_effects', ()) if (effect.get('subject_public_role'), effect.get('relation_or_property')) not in verified_keys]
            if effects:
                retained.append({**action, 'assumed_effects': effects})
        self._assumed_completed_actions = retained

    @staticmethod
    def _canonical_public_tool_call(value: Mapping[str, Any]) -> dict[str, Any]:
        from scoper.harness.validation import exact_mapping, public_json
        call = exact_mapping(value, frozenset({'tool_name', 'arguments'}), pointer='/failed_tool_call')
        if not isinstance(call['arguments'], Mapping):
            raise ValueError('failed ToolCall arguments must be an object')
        output = {'tool_name': str(call['tool_name']), 'arguments': copy.deepcopy(dict(call['arguments']))}
        public_json(output, pointer='/failed_tool_call', include_future_action=False)
        return output

    @classmethod
    def _preparation_fingerprint(cls, value: Mapping[str, Any]) -> bytes | None:
        call = cls._canonical_public_tool_call(value)
        if call['tool_name'] not in _V13_PREPARATION_TOOL_NAMES:
            return None
        return canonical_json_bytes(call)

    def _record_exhausted_preparation(self, failed_tool_call: Mapping[str, Any]) -> None:
        fingerprint = self._preparation_fingerprint(failed_tool_call)
        if fingerprint is not None:
            self._exhausted_preparation_fingerprints.add(fingerprint)

    def _record_successful_preparation(self, tool_call: Mapping[str, Any], preparation_result: Mapping[str, Any]) -> None:
        if preparation_result['status'] not in {'found', 'navigation_ready', 'scan_completed'}:
            return
        fingerprint = self._preparation_fingerprint(tool_call)
        if fingerprint is not None:
            self._exhausted_preparation_fingerprints.discard(fingerprint)

    def _register_recovery_context(self, *, incident_id: str, failed_tool_call: Mapping[str, Any], failure_kind: str, canonical_action: str, required_effects: Sequence[str], effect_bindings: Sequence[Any]=(), evidence_frame_ids: Sequence[str], phase: str, binding_failure_reason_or_null: str | None=None, uniquely_resolved_public_target: bool=False, public_failure_reason_or_null: str | None=None) -> Any:
        from scoper.harness.recovery import RecoveryContext
        previous = self._recovery_contexts.get(incident_id)
        context = RecoveryContext(incident_id=incident_id, failed_tool_call=self._canonical_public_tool_call(failed_tool_call), failure_kind=failure_kind, canonical_action=canonical_action, required_effects=tuple(sorted(set(required_effects), key=lambda item: item.encode('utf-8'))), retry_index=previous.retry_index if previous is not None else 0, max_retries=2, phase=phase, evidence_frame_ids=tuple(evidence_frame_ids), preparation_attempts=previous.preparation_attempts if previous is not None else 0, recovery_round=previous.recovery_round if previous is not None else 0, dispatched_attempts=previous.dispatched_attempts if previous is not None else 0, last_preparation_result_or_null=previous.last_preparation_result_or_null if previous is not None else None, effect_bindings=tuple((copy.deepcopy(binding.to_mapping() if callable(getattr(binding, 'to_mapping', None)) else dict(binding)) for binding in effect_bindings)), binding_failure_reason_or_null=binding_failure_reason_or_null if binding_failure_reason_or_null is not None else previous.binding_failure_reason_or_null if previous is not None else None, uniquely_resolved_public_target=uniquely_resolved_public_target or bool(previous is not None and previous.uniquely_resolved_public_target), public_failure_reason_or_null=public_failure_reason_or_null, open_prerequisite_attempted=bool(previous is not None and previous.open_prerequisite_attempted))
        self._recovery_contexts[incident_id] = context
        self._last_recovery_notice = None
        return context

    def commit_visual_verification(self, verification: Any, *, action_id: str, evidence_frame_ids: Sequence[str], step: int, recovery_plan: Any, failed_tool_call: Mapping[str, Any] | None=None, public_failure_reason_or_null: str | None=None) -> Any | None:
        if not self.definition.verifier_authoritative_memory_updates:
            return None
        from scoper.harness.controller import VerificationCommitContext, apply_verification_lifecycle
        from scoper.harness.evidence_memory import commit_effect_bindings, resolve_public_tool_failure_by_incident
        if not isinstance(verification, VerificationCommitContext):
            raise TypeError('visual verification context is invalid')
        if self._memory is None:
            raise RuntimeError('Harness episode is not active')
        if public_failure_reason_or_null is not None and (not isinstance(public_failure_reason_or_null, str)):
            raise ValueError('public failure reason must be text or null')
        if self.definition.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            self._exhausted_preparation_fingerprints.clear()
        frames = tuple(evidence_frame_ids)
        if not frames:
            raise ValueError('visual verification requires evidence frames')
        closed_container_failure = bool(self.definition.version in {HARNESS_V15_VERSION, HARNESS_V16_VERSION} and public_failure_reason_or_null == 'container_closed' and (verification.canonical_action == 'put_down') and isinstance(failed_tool_call, Mapping) and (failed_tool_call.get('tool_name') == 'put_down'))
        outcome = 'failure' if closed_container_failure else verification.effective_outcome
        placement_id = getattr(recovery_plan, 'incident_id_or_null', None)
        placement = self._placement_recoveries.get(placement_id)
        placement_pickup = placement is not None and verification.canonical_action == 'pick_up'
        if placement_pickup and outcome != 'success':
            if self.definition.version == 'scoper-1' and outcome == 'failure':
                placement.phase = 'exhausted'
                placement.pickup_assumed = False
                return None
            if public_failure_reason_or_null is not None:
                placement.phase = 'exhausted'
                placement.pickup_assumed = False
                return None
            placement.pickup_assumed = True
            placement.phase = 'navigate_destination'
            return None
        source = 'public_tool_failure' if closed_container_failure else 'visual_verifier_projection'
        if outcome == 'success':
            commit = commit_effect_bindings(self._memory, verification.effect_bindings, verification.selection, success_probability=verification.success_probability, effect_probabilities=dict(verification.effect_probabilities), action_id=action_id, evidence_frame_ids=frames, step=step, capacity=self.memory_capacity, observation_public_roles=verification.public_roles)
            self.memory = commit.memory
            if commit.accepted_by_harness:
                self._commit_private_binding_receipt(action_id)
                self._reconcile_assumed_effects(verification.effect_bindings, selected_effects=verification.selected_effects)
            if commit.accepted_by_harness:
                self.event_sink.emit_memory_update(action_id, {**commit.event_payload(), 'reason': commit.reason, 'source': source, 'verification_outcome': outcome, 'verification_effects': list(verification.selected_effects)})
            incident_id = getattr(recovery_plan, 'incident_id_or_null', None)
            if not isinstance(incident_id, str) or not incident_id or (not commit.accepted_by_harness):
                return commit
            if placement_pickup:
                placement.phase = 'navigate_destination'
                return commit
            if getattr(recovery_plan, 'reason', None) == 'open_closed_container_before_retry':
                context = self._recovery_contexts.get(incident_id)
                if context is not None:
                    self._recovery_contexts[incident_id] = replace(context, phase='retry', public_failure_reason_or_null=None, open_prerequisite_attempted=True)
                return commit
            resolution = resolve_public_tool_failure_by_incident(self._memory, incident_id=incident_id, step=step, capacity=self.memory_capacity)
            self.memory = resolution.memory
            if resolution.accepted_by_harness:
                self.event_sink.emit_memory_update(action_id, {**resolution.event_payload(), 'reason': resolution.reason, 'source': source, 'verification_outcome': outcome, 'verification_effects': list(verification.selected_effects)})
                if self.definition.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
                    self._recovery_contexts.pop(incident_id, None)
                    self._last_recovery_notice = None
            return resolution
        confidence = 1.0 if closed_container_failure else verification.failure_probability if outcome == 'failure' else max(0.0, 1.0 - verification.success_probability - verification.failure_probability)
        if self.definition.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            from scoper.harness.evidence_memory import record_visual_recovery_incident
            if failed_tool_call is None:
                raise ValueError('v13 visual recovery requires the original public ToolCall')
            source_incident_id = getattr(recovery_plan, 'incident_id_or_null', None)
            if getattr(recovery_plan, 'reason', None) == 'open_closed_container_before_retry' and isinstance(source_incident_id, str):
                context = self._recovery_contexts.get(source_incident_id)
                if context is not None:
                    self._recovery_contexts[source_incident_id] = replace(context, phase='retry', public_failure_reason_or_null=None, open_prerequisite_attempted=True)
                return None
            result = record_visual_recovery_incident(self._memory, action_id=action_id, canonical_action=verification.canonical_action, public_roles=verification.public_roles, evidence_frame_ids=frames, step=step, confidence=confidence, source_incident_id_or_null=source_incident_id if isinstance(source_incident_id, str) else None, increment_recovery_attempt=isinstance(source_incident_id, str) and bool(source_incident_id), capacity=self.memory_capacity)
            self.memory = result.memory
            incident = next((item for item in self._memory.incidents if item.status == 'open' and (item.incident_id == source_incident_id or item.action_id == action_id)), None)
            if incident is None:
                raise RuntimeError('v13 visual recovery incident was not created')
            required_effects = tuple((binding.effect for binding in verification.effect_bindings if binding.required))
            previous = self._recovery_contexts.get(incident.incident_id)
            open_prerequisite = bool(closed_container_failure and (not (previous is not None and previous.open_prerequisite_attempted)))
            self._register_recovery_context(incident_id=incident.incident_id, failed_tool_call=failed_tool_call, failure_kind=outcome, canonical_action=verification.canonical_action, required_effects=required_effects, effect_bindings=verification.effect_bindings, evidence_frame_ids=frames, phase='open_prerequisite' if open_prerequisite else 'retry' if bool(getattr(recovery_plan, 'reason', None) == 'reverify_after_look') else 'look', public_failure_reason_or_null=public_failure_reason_or_null)
            if self.definition.version in {HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION} and verification.canonical_action == 'put_down':
                from scoper.harness.placement_recovery import PlacementRecovery, invalidate_old_held_fact, placement_object_ref
                placement = self._placement_recoveries.get(incident.incident_id)
                if placement is None:
                    object_ref = placement_object_ref(self._memory, verification.effect_bindings)
                    placement = PlacementRecovery(object_ref, step)
                    self._placement_recoveries[incident.incident_id] = placement
                placement.failed_step = step
                placement.phase = 'inspect'
                self.memory = invalidate_old_held_fact(self._memory, placement.object_ref, step)
                result = replace(result, memory=self._memory)
            receipt = self._private_binding_receipts.get(action_id, ())
            if receipt:
                self._private_recovery_binding_receipts[incident.incident_id] = receipt
            context = self._recovery_contexts[incident.incident_id]
            if isinstance(source_incident_id, str) and source_incident_id:
                self._recovery_contexts[incident.incident_id] = replace(context, recovery_round=min(context.max_retries, context.recovery_round + int(getattr(recovery_plan, 'reason', None) != 'reverify_after_look')), phase='exhausted' if context.retry_index >= context.max_retries else context.phase)
            if result.accepted_by_harness:
                self.event_sink.emit_memory_update(action_id, {**result.event_payload(), 'reason': result.reason, 'source': source, 'verification_outcome': outcome, 'verification_effects': list(verification.selected_effects)})
            return result
        lifecycle = apply_verification_lifecycle(self._memory, selected_outcome=outcome, canonical_action=verification.canonical_action, confidence=confidence, action_id=action_id, evidence_frame_ids=frames, public_roles=verification.public_roles, step=step, capacity=self.memory_capacity)
        self.memory = lifecycle.memory
        accepted = lifecycle.accepted_mutation()
        if accepted is not None:
            self.event_sink.emit_memory_update(action_id, {**accepted, 'reason': lifecycle.reason, 'source': source, 'verification_outcome': outcome, 'verification_effects': list(verification.selected_effects)})
        return lifecycle

    def record_public_tool_failure(self, *, action_id: str, public_roles: Sequence[str], step: int, confidence: float=1.0, public_failure_reason: str | None=None, failed_tool_call: Mapping[str, Any] | None=None, uniquely_resolved_public_target: bool=False) -> Any:
        from scoper.harness.evidence_memory import record_public_tool_failure
        from scoper.harness.contracts import MethodContractError
        if self.definition.visual_only_state_enabled and self.definition.version not in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            raise MethodContractError('method_visual_evidence_required', 'visual-only Harness cannot create memory from executor or Binding failure')
        if self._memory is None:
            raise RuntimeError('Harness episode is not active')
        evidence_frame_ids = self.event_sink.evidence_frame_ids(action_id)
        failure_type = 'public_interaction_not_ready' if self.definition.version in {HARNESS_V5_VERSION, HARNESS_V6_VERSION, HARNESS_V7_VERSION, HARNESS_V8_VERSION, HARNESS_V9_VERSION, HARNESS_V10_VERSION, HARNESS_V11_VERSION, DEFAULT_HARNESS_VERSION, HARNESS_V13_VERSION, HARNESS_V14_VERSION, HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION, HARNESS_V15_VERSION, HARNESS_V16_VERSION} and public_failure_reason in {'binding_failed', 'motion_failed', 'navigation_failed', 'navigation_not_ready', 'not_found', 'not_interaction_ready', 'navigation_context_required', 'recent_focus_not_interaction_ready', 'successful_navigation_focus_required', 'target_not_interaction_ready'} else 'unresolved_visual_effect'
        result = record_public_tool_failure(self._memory, action_id=action_id, public_roles=public_roles, evidence_frame_ids=evidence_frame_ids, step=step, confidence=confidence, failure_type=failure_type, capacity=self.memory_capacity)
        self._memory = result.memory
        if self.definition.version in {HARNESS_V6_VERSION, HARNESS_V7_VERSION, HARNESS_V8_VERSION, HARNESS_V9_VERSION, HARNESS_V10_VERSION, HARNESS_V11_VERSION, DEFAULT_HARNESS_VERSION, HARNESS_V13_VERSION, HARNESS_V14_VERSION, HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION, HARNESS_V15_VERSION, HARNESS_V16_VERSION} and (failure_type == 'public_interaction_not_ready' or self.definition.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS) and (failed_tool_call is not None):
            from scoper.harness.validation import exact_mapping, public_json
            call = exact_mapping(failed_tool_call, frozenset({'tool_name', 'arguments'}), pointer='/failed_tool_call')
            if not isinstance(call['arguments'], Mapping):
                raise ValueError('failed ToolCall arguments must be an object')
            canonical_call = {'tool_name': str(call['tool_name']), 'arguments': copy.deepcopy(dict(call['arguments']))}
            public_json(canonical_call, pointer='/failed_tool_call', include_future_action=False)
            incident = next((item for item in result.memory.incidents if item.action_id == action_id and item.failure_type == failure_type and (item.status == 'open')), None)
            if incident is not None:
                self._pending_public_retries[incident.incident_id] = {'tool_call': canonical_call, 'public_roles': tuple(public_roles), 'phase': 'navigation_required'}
                if self.definition.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
                    from scoper.harness.verifier import required_effects_for_action
                    try:
                        required_effects = required_effects_for_action(canonical_call['tool_name'])
                    except Exception:
                        required_effects = ()
                    self._register_recovery_context(incident_id=incident.incident_id, failed_tool_call=canonical_call, failure_kind='admission', canonical_action=canonical_call['tool_name'], required_effects=required_effects, effect_bindings=(), evidence_frame_ids=evidence_frame_ids, phase='navigate' if public_failure_reason == 'navigation_context_required' else 'look', binding_failure_reason_or_null=public_failure_reason, uniquely_resolved_public_target=bool(uniquely_resolved_public_target))
        if result.accepted_by_harness:
            self.event_sink.emit_memory_update(action_id, {**result.event_payload(), 'reason': result.reason})
        return result

    def record_public_preparation_failure(self, *, action_id: str, public_roles: Sequence[str], step: int, failed_tool_call: Mapping[str, Any], preparation_result: Mapping[str, Any] | None, uniquely_resolved_public_target: bool=False) -> Any | None:
        if self.definition.version not in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            return None
        if preparation_result is None:
            return None
        from scoper.harness.recovery import _validate_preparation_result
        normalized = _validate_preparation_result(preparation_result)
        canonical_call = self._canonical_public_tool_call(failed_tool_call)
        self._record_successful_preparation(canonical_call, normalized)
        if normalized['status'] not in {'not_found', 'motion_failed', 'binding_failed', 'navigation_not_ready', 'navigation_failed'}:
            return None
        if not public_roles:
            return None
        fingerprint = self._preparation_fingerprint(canonical_call)
        if fingerprint in self._exhausted_preparation_fingerprints:
            return None
        result = self.record_public_tool_failure(action_id=action_id, public_roles=public_roles, step=step, public_failure_reason=normalized['status'], failed_tool_call=canonical_call, uniquely_resolved_public_target=uniquely_resolved_public_target)
        incident = next((item for item in self._memory.incidents if item.action_id == action_id and item.status == 'open'), None)
        if incident is not None and incident.incident_id in self._recovery_contexts:
            self._recovery_contexts[incident.incident_id] = replace(self._recovery_contexts[incident.incident_id], failure_kind='failure')
        return result

    def record_public_recovery_success(self, *, action_id: str, tool_name: str, public_roles: Sequence[str], binding_status: str | None, environment_success: bool, recovery_plan: Any, step: int) -> Any | None:
        if self.definition.visual_only_state_enabled:
            return None
        if tool_name in {'navigate_to', 'navigate_to_room'} and binding_status == 'success' and (environment_success is True) and public_roles:
            self._last_successful_navigation_roles = tuple(public_roles)
        if self.definition.version in {HARNESS_V6_VERSION, HARNESS_V7_VERSION, HARNESS_V8_VERSION, HARNESS_V9_VERSION, HARNESS_V10_VERSION, HARNESS_V11_VERSION, DEFAULT_HARNESS_VERSION, HARNESS_V13_VERSION, HARNESS_V14_VERSION, HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION, HARNESS_V15_VERSION, HARNESS_V16_VERSION} and getattr(recovery_plan, 'mode', None) == 'recovery' and (getattr(recovery_plan, 'reason', None) == 'interaction_readiness_requires_navigation') and self._last_successful_navigation_roles:
            retry = self._pending_public_retries.get(recovery_plan.incident_id_or_null)
            target_values = {value for value in (retry or {}).get('tool_call', {}).get('arguments', {}).values() if isinstance(value, str) and value}
            if retry is not None and target_values.intersection(self._last_successful_navigation_roles):
                retry['phase'] = 'retry_ready'
            return None
        if self.definition.version not in {HARNESS_V4_VERSION, HARNESS_V5_VERSION} or getattr(recovery_plan, 'mode', None) != 'recovery' or (not self._last_successful_navigation_roles):
            return None
        from scoper.harness.evidence_memory import resolve_public_tool_failure
        if self._memory is None:
            raise RuntimeError('Harness episode is not active')
        result = resolve_public_tool_failure(self._memory, incident_id=recovery_plan.incident_id_or_null, public_roles=self._last_successful_navigation_roles, step=step, capacity=self.memory_capacity)
        self._memory = result.memory
        if result.accepted_by_harness:
            self.event_sink.emit_memory_update(action_id, {**result.event_payload(), 'reason': result.reason})
        return result

    def plan_recovery(self, memory: Any, ledger: Any, public_tool_specs: Sequence[Mapping[str, Any]], *, budgets: Any, observation_tool_names: Sequence[str]) -> Any:
        from scoper.harness.recovery import RecoveryPlan, plan_recovery, plan_recovery_v13
        if self.definition.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            from scoper.harness.evidence_memory import assume_recovery_continue
            current = self._memory if self._memory is not None else memory
            if self.definition.version in {HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION}:
                open_incidents = sorted((item for item in current.incidents if item.status == 'open'), key=lambda item: (item.created_step, item.incident_id))
                if open_incidents:
                    incident_id = open_incidents[0].incident_id
                    placement = self._placement_recoveries.get(incident_id)
                    if placement is not None:
                        return placement.plan(current, self._recovery_contexts[incident_id], public_tool_specs)
            exhausted = next((incident for incident in current.incidents if incident.status == 'open' and incident.incident_id in self._recovery_contexts and (self._recovery_contexts[incident.incident_id].retry_index >= self._recovery_contexts[incident.incident_id].max_retries or self._recovery_contexts[incident.incident_id].phase == 'exhausted')), None)
            if exhausted is not None:
                context = self._recovery_contexts[exhausted.incident_id]
                notice = context.to_mapping()
                notice['phase'] = 'complete'
                notice['resolution'] = 'recovery_assumed_continue'
                action_was_dispatched = context.dispatched_attempts > 0 or context.failure_kind in {'failure', 'ambiguous'}
                physical_action_was_dispatched = action_was_dispatched and context.failed_tool_call['tool_name'] not in _V13_PREPARATION_TOOL_NAMES
                if context.failed_tool_call['tool_name'] in _V13_PREPARATION_TOOL_NAMES:
                    self._record_exhausted_preparation(context.failed_tool_call)
                if physical_action_was_dispatched:
                    assumed = {'tool_call': copy.deepcopy(dict(context.failed_tool_call)), 'required_effects': list(context.required_effects), 'source_incident_id': context.incident_id, 'evidence_frame_ids': list(context.evidence_frame_ids), 'retry_count': context.retry_index, 'verification_status': 'uncertain', 'assumed_effects': self._assumed_effects_for_context(context)}
                    if not any((item['tool_call'] == assumed['tool_call'] for item in self._assumed_completed_actions)):
                        self._assumed_completed_actions.append(assumed)
                    self._private_processed_binding_role_set.update(self._private_recovery_binding_receipts.get(context.incident_id, ()))
                result = assume_recovery_continue(current, incident_id=exhausted.incident_id, step=exhausted.last_updated_step + 1, capacity=self.memory_capacity)
                self.memory = result.memory
                self._last_recovery_notice = notice
                if result.accepted_by_harness:
                    self.event_sink.emit_memory_update(exhausted.action_id, {**result.event_payload(), 'reason': result.reason, 'source': 'bounded_recovery', 'recovery_status': 'recovery_assumed_continue'})
                tools = tuple(sorted((str(spec['name']) for spec in public_tool_specs), key=lambda item: item.encode('utf-8')))
                return RecoveryPlan('step', 'recovery_assumed_continue', None, None, tools)
            return plan_recovery_v13(current, ledger, public_tool_specs, contexts=self._recovery_contexts, preparation_tool_names=_V13_PREPARATION_TOOL_NAMES, max_retries=2)
        use_v4 = self.definition.version in {HARNESS_V4_VERSION, HARNESS_V5_VERSION, HARNESS_V6_VERSION, HARNESS_V7_VERSION, HARNESS_V8_VERSION, HARNESS_V9_VERSION, HARNESS_V10_VERSION, HARNESS_V11_VERSION, DEFAULT_HARNESS_VERSION}
        plan = plan_recovery(memory, ledger, public_tool_specs, budgets=budgets, observation_tool_names=observation_tool_names, suppress_repeated_incident_observation=use_v4, prioritize_oldest_incident=use_v4)
        if self.definition.version in {HARNESS_V6_VERSION, HARNESS_V7_VERSION, HARNESS_V8_VERSION, HARNESS_V9_VERSION, HARNESS_V10_VERSION, HARNESS_V11_VERSION, DEFAULT_HARNESS_VERSION} and plan.mode == 'recovery':
            retry = self._pending_public_retries.get(plan.incident_id_or_null)
            if retry is not None and retry.get('phase') == 'retry_ready':
                required_call = copy.deepcopy(retry['tool_call'])
                return replace(plan, reason='retry_original_action_after_navigation', allowed_tool_names=(required_call['tool_name'],), required_tool_call_or_null=required_call)
        return plan

    def policy_tool_specs(self, public_tool_specs: Sequence[Mapping[str, Any]], recovery_plan: Any) -> tuple[Mapping[str, Any], ...]:
        allowed = set(recovery_plan.allowed_tool_names)
        required_call = getattr(recovery_plan, 'required_tool_call_or_null', None)
        output = []
        for raw in public_tool_specs:
            if raw.get('name') not in allowed:
                continue
            spec = copy.deepcopy(dict(raw))
            if self.definition.version in {HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION}:
                schema = spec.get('parameters', spec.get('arguments_schema', {}))
                target = schema.get('properties', {}).get('target')
                if isinstance(target, dict):
                    target['description'] = 'A concrete public vocabulary object name or visually grounded description; never a simulator ID or internal role handle. navigate_to may use an unseen object named in the instruction; interactions require current visual readiness.'
            if required_call is not None and spec.get('name') == required_call.get('tool_name'):
                schema_name = 'parameters' if 'parameters' in spec else 'arguments_schema'
                schema = spec[schema_name]
                properties = schema.get('properties')
                if not isinstance(properties, Mapping):
                    raise ValueError('public tool schema lacks argument properties')
                properties = copy.deepcopy(dict(properties))
                for name, value in required_call['arguments'].items():
                    if name not in properties:
                        raise ValueError('required retry argument is absent from public schema')
                    properties[name] = {**dict(properties[name]), 'enum': [copy.deepcopy(value)]}
                schema['properties'] = properties
                spec[schema_name] = schema
            output.append(spec)
        result = tuple(sorted(output, key=lambda item: item['name'].encode('utf-8')))
        if {item['name'] for item in result} != allowed:
            raise ValueError('recovery plan tools differ from public tool specs')
        return result

    def validate_public_action_preconditions(self, tool_call: Mapping[str, Any], *, public_observation: Mapping[str, Any] | None=None, task_binding_spec: Mapping[str, Any] | None=None, benevolent_binding: bool=False) -> None:
        if self.definition.version not in {HARNESS_V6_VERSION, HARNESS_V7_VERSION, HARNESS_V8_VERSION, HARNESS_V9_VERSION, HARNESS_V10_VERSION, HARNESS_V11_VERSION, DEFAULT_HARNESS_VERSION, HARNESS_V13_VERSION, HARNESS_V14_VERSION, HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION, HARNESS_V15_VERSION, HARNESS_V16_VERSION}:
            return
        from scoper.harness.contracts import MethodContractError
        from scoper.harness.validation import exact_mapping, non_empty_string
        call = exact_mapping(tool_call, frozenset({'tool_name', 'arguments'}), pointer='/tool_call')
        tool_name = non_empty_string(call['tool_name'], pointer='/tool_call/tool_name')
        arguments = call['arguments']
        if not isinstance(arguments, Mapping):
            raise MethodContractError('method_schema_invalid', 'ToolCall arguments must be an object')
        if self._memory is None:
            raise RuntimeError('Harness episode is not active')
        public_call = {'tool_name': tool_name, 'arguments': copy.deepcopy(dict(arguments))}
        completed_equivalent = False
        if self.definition.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            from scoper.harness.binding import TaskBindingSpec, equivalent_unique_task_witness_calls, public_task_witness_role_count
            parsed_binding_spec = TaskBindingSpec.from_dict(task_binding_spec) if isinstance(task_binding_spec, Mapping) else None
            target = public_call['arguments'].get('target')
            duplicate_task_witness = bool(benevolent_binding and parsed_binding_spec is not None and isinstance(target, str) and (public_task_witness_role_count(parsed_binding_spec, target) > 1))
            completed_equivalent = any((not duplicate_task_witness and item.get('tool_call') == public_call or (parsed_binding_spec is not None and equivalent_unique_task_witness_calls(parsed_binding_spec, item.get('tool_call') or {}, public_call)) for item in self._assumed_completed_actions))
        if completed_equivalent:
            raise MethodContractError('method_action_already_satisfied', 'the ToolCall is already assumed complete for its unique public task witness')
        effective_facts = self.public_effective_facts()
        held = tuple((fact['subject_public_role'] for fact in effective_facts if fact.get('relation_or_property') == 'held_by_agent' and fact.get('object_public_role_or_value') is True))
        observation = public_observation if isinstance(public_observation, Mapping) else {}
        observed_held = () if self.definition.visual_only_state_enabled else tuple((value for field in ('held_object_handles', 'held_object_types') for value in observation.get(field, ()) or () if isinstance(value, str) and value))
        held_count = len(set(held)) if held else len(set(observed_held))
        target = arguments.get('target')
        if self.definition.version in {HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION} and tool_name == 'put_down' and (held_count == 0):
            placements = [placement for incident_id, placement in self._placement_recoveries.items() if incident_id in self._recovery_contexts and self._recovery_contexts[incident_id].failed_tool_call == public_call and (placement.phase == 'retry') and (placement.held_state(self._memory) == 'held')]
            if len(placements) == 1:
                held_count = 1
        if tool_name == 'put_down' and held_count != 1:
            raise MethodContractError('method_public_precondition_failed', 'put_down requires exactly one publicly verified held object')
        if tool_name == 'pick_up' and held_count:
            raise MethodContractError('method_public_precondition_failed', 'pick_up is forbidden while a public object is already held')
        if isinstance(target, str) and tool_name in {'open', 'close'}:
            expected = tool_name == 'open'
            if any((fact.get('subject_public_role') == target and fact.get('relation_or_property') == 'open' and (fact.get('object_public_role_or_value') is expected) for fact in effective_facts)):
                raise MethodContractError('method_action_already_satisfied', f'{tool_name} postcondition is already publicly verified')

    def evaluate_public_postcondition(self, bindings: Sequence[Any], evidence: Mapping[str, Any], visual_selection: Any) -> tuple[Any, Any]:
        from scoper.harness.postcondition import PublicPostconditionResult, evaluate_public_postcondition, merge_public_and_visual_verification
        if self.definition.version not in {HARNESS_V7_VERSION, HARNESS_V8_VERSION, HARNESS_V9_VERSION}:
            return (PublicPostconditionResult('unknown', 'public_postconditions_not_enabled', ()), visual_selection)
        result = evaluate_public_postcondition(bindings, evidence)
        return (result, merge_public_and_visual_verification(result, visual_selection))

    def commit_public_postcondition(self, result: Any, bindings: Sequence[Any], *, canonical_action: str, public_roles: Sequence[str], action_id: str, evidence_frame_ids: Sequence[str], step: int, recovery_plan: Any, failed_tool_call: Mapping[str, Any] | None) -> Any | None:
        if self.definition.version not in {HARNESS_V7_VERSION, HARNESS_V8_VERSION, HARNESS_V9_VERSION}:
            return None
        from scoper.harness.controller import apply_verification_lifecycle
        from scoper.harness.evidence_memory import commit_effect_bindings, resolve_public_tool_failure, resolve_public_tool_failure_by_incident
        from scoper.harness.validation import exact_mapping, public_json
        from scoper.harness.postcondition import PublicPostconditionResult
        from scoper.harness.verifier import VerifierSelection
        if not isinstance(result, PublicPostconditionResult) or not result.conclusive:
            return None
        if self._memory is None:
            raise RuntimeError('Harness episode is not active')
        roles = tuple(sorted(set(public_roles), key=lambda item: item.encode('utf-8')))
        if not roles:
            raise ValueError('conclusive public postcondition requires public roles')
        if result.status == 'success':
            selection = VerifierSelection('success', result.selected_effects, 'observable', False)
            commit = commit_effect_bindings(self._memory, bindings, selection, success_probability=1.0, effect_probabilities={effect: 1.0 for effect in result.selected_effects}, action_id=action_id, evidence_frame_ids=evidence_frame_ids, step=step, capacity=self.memory_capacity, observation_public_roles=roles)
            self._memory = commit.memory
            if commit.accepted_by_harness:
                self.event_sink.emit_memory_update(action_id, {**commit.event_payload(), 'reason': commit.reason, 'source': 'public_postcondition'})
            incident_id = getattr(recovery_plan, 'incident_id_or_null', None)
            if isinstance(incident_id, str) and incident_id:
                retry = self._pending_public_retries.get(incident_id)
                resolution_roles = tuple((retry or {}).get('public_roles') or roles)
                resolution = resolve_public_tool_failure_by_incident(self._memory, incident_id=incident_id, step=step, capacity=self.memory_capacity) if self.definition.version == HARNESS_V9_VERSION else resolve_public_tool_failure(self._memory, incident_id=incident_id, public_roles=resolution_roles, step=step, capacity=self.memory_capacity)
                self._memory = resolution.memory
                if resolution.accepted_by_harness:
                    self.event_sink.emit_memory_update(action_id, {**resolution.event_payload(), 'reason': resolution.reason, 'source': 'public_postcondition'})
            return commit
        lifecycle = apply_verification_lifecycle(self._memory, selected_outcome='failure', canonical_action=canonical_action, confidence=1.0, action_id=action_id, evidence_frame_ids=evidence_frame_ids, public_roles=roles, step=step, capacity=self.memory_capacity)
        self._memory = lifecycle.memory
        if lifecycle.accepted_by_harness:
            self.event_sink.emit_memory_update(action_id, {'accepted_by_harness': True, 'revision': lifecycle.memory.revision, 'reason': lifecycle.reason, 'source': 'public_postcondition'})
        if failed_tool_call is not None and canonical_action in {'pick_up', 'open', 'close', 'toggle_on', 'toggle_off'}:
            call = exact_mapping(failed_tool_call, frozenset({'tool_name', 'arguments'}), pointer='/failed_tool_call')
            if not isinstance(call['arguments'], Mapping):
                raise ValueError('failed ToolCall arguments must be an object')
            canonical_call = {'tool_name': str(call['tool_name']), 'arguments': copy.deepcopy(dict(call['arguments']))}
            public_json(canonical_call, pointer='/failed_tool_call', include_future_action=False)
            incident = next((item for item in self._memory.incidents if item.action_id == action_id and item.status == 'open'), None)
            if incident is not None:
                self._pending_public_retries[incident.incident_id] = {'tool_call': canonical_call, 'public_roles': roles, 'phase': 'retry_ready'}
        return lifecycle

    def preserve_memory_on_model_abstention(self, mode: str) -> bool:
        return self.definition.version in {HARNESS_V8_VERSION, HARNESS_V9_VERSION, HARNESS_V10_VERSION, HARNESS_V11_VERSION, DEFAULT_HARNESS_VERSION, HARNESS_V13_VERSION, HARNESS_V14_VERSION, HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION, HARNESS_V15_VERSION, HARNESS_V16_VERSION} and (mode == 'reconcile' or (mode == 'observe' and self.definition.version in {HARNESS_V14_VERSION, HARNESS_V18_VERSION, HARNESS_V19_VERSION, HARNESS_V20_VERSION, HARNESS_V21_VERSION, HARNESS_V15_VERSION, HARNESS_V16_VERSION}))

    def record_recovery_action(self, ledger: Any, plan: Any, tool_call: Mapping[str, Any], *, action_id: str, public_tool_specs: Sequence[Mapping[str, Any]], allowed_public_entity_refs: Sequence[str], public_precondition_satisfied: bool, budgets: Any, public_entity_argument_names: Sequence[str]=('object', 'receptacle', 'room', 'source', 'target')) -> Any:
        from scoper.harness.recovery import record_recovery_action, record_recovery_action_v13
        if self.definition.version in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            updated = record_recovery_action_v13(ledger, plan, tool_call, action_id=action_id, public_tool_specs=public_tool_specs, allowed_public_entity_refs=allowed_public_entity_refs, public_precondition_satisfied=public_precondition_satisfied)
            incident_id = getattr(plan, 'incident_id_or_null', None)
            if isinstance(incident_id, str) and incident_id:
                self._pending_recovery_actions[action_id] = {'incident_id': incident_id, 'is_original_retry': plan.reason in {'retry_original_action', 'final_locked_original_attempt', 'placement_retry'}, 'placement_phase': plan.reason.removeprefix('placement_') if plan.reason.startswith('placement_') else None, 'is_open_prerequisite': plan.reason == 'open_closed_container_before_retry', 'tool_call': self._canonical_public_tool_call(tool_call)}
            return updated
        observation_tools = tuple(sorted(set(self.stream_enabled_tools) & {'look_for', 'scan'}, key=lambda item: item.encode('utf-8')))
        return record_recovery_action(ledger, plan, tool_call, action_id=action_id, public_tool_specs=public_tool_specs, allowed_public_entity_refs=allowed_public_entity_refs, public_precondition_satisfied=public_precondition_satisfied, budgets=budgets, public_entity_argument_names=public_entity_argument_names, observation_tool_names=observation_tools)

    def finalize_recovery_action(self, *, action_id: str, admitted_to_environment: bool, preparation_completed: bool | None=None, preparation_result: Mapping[str, Any] | None=None) -> Any:
        if self.definition.version not in _BOUNDED_RECOVERY_HARNESS_VERSIONS:
            return self._recovery_ledger
        pending = self._pending_recovery_actions.pop(action_id, None)
        if pending is None:
            return self._recovery_ledger
        incident_id = pending['incident_id']
        context = self._recovery_contexts.get(incident_id)
        if context is None:
            return self._recovery_ledger
        from scoper.harness.recovery import RecoveryLedger, _validate_preparation_result
        attempts = dict(self._recovery_ledger.incident_attempts)
        total = self._recovery_ledger.total_recovery_actions
        normalized = _validate_preparation_result(preparation_result) if preparation_result is not None else None
        if normalized is not None:
            self._record_successful_preparation(pending['tool_call'], normalized)
        placement = self._placement_recoveries.get(incident_id)
        if placement is not None and pending.get('placement_phase'):
            placement.finalize(pending['placement_phase'], admitted_to_environment, normalized)
            total += int(admitted_to_environment)
            if pending['is_original_retry'] and admitted_to_environment:
                attempts[incident_id] = attempts.get(incident_id, 0) + 1
                context = replace(context, retry_index=min(context.max_retries, context.retry_index + 1), dispatched_attempts=min(context.max_retries, context.dispatched_attempts + 1))
        elif pending['is_open_prerequisite']:
            total += int(admitted_to_environment)
            context = replace(context, phase='retry', preparation_attempts=context.preparation_attempts + 1, public_failure_reason_or_null=None, open_prerequisite_attempted=True)
        elif pending['is_original_retry']:
            if admitted_to_environment:
                attempts[incident_id] = attempts.get(incident_id, 0) + 1
                total += 1
                context = replace(context, retry_index=min(context.max_retries, context.retry_index + 1), dispatched_attempts=min(context.max_retries, context.dispatched_attempts + 1), phase='retry')
            else:
                next_round = context.recovery_round + 1
                context = replace(context, failure_kind='admission', phase='exhausted' if context.phase == 'final_retry' else 'final_retry' if next_round >= context.max_retries and context.uniquely_resolved_public_target else 'exhausted' if next_round >= context.max_retries else 'look', recovery_round=next_round)
        else:
            total += int(admitted_to_environment)
            completed = bool(preparation_completed) if normalized is None else True
            status = normalized.get('status') if normalized is not None else None
            ready = bool(normalized.get('interaction_ready')) if normalized is not None else False
            tool_name = pending['tool_call']['tool_name']
            if not admitted_to_environment:
                next_phase = 'navigate' if tool_name == 'look_for' else 'look'
                next_round = context.recovery_round + 1
            elif normalized is None:
                next_phase = 'retry' if completed else 'look'
                next_round = context.recovery_round + int(not completed)
            elif tool_name == 'look_for':
                next_phase = 'retry' if status == 'found' and ready else 'navigate'
                next_round = context.recovery_round
            elif tool_name == 'navigate_to':
                next_phase = 'retry' if status == 'navigation_ready' and ready else 'look'
                next_round = context.recovery_round + int(next_phase == 'look')
            else:
                next_phase = 'look'
                next_round = context.recovery_round + 1
            if next_round >= context.max_retries and next_phase != 'retry':
                next_phase = 'final_retry' if context.dispatched_attempts < context.max_retries and context.uniquely_resolved_public_target else 'exhausted'
            context = replace(context, phase=next_phase, preparation_attempts=context.preparation_attempts + 1, recovery_round=next_round, last_preparation_result_or_null=normalized)
        self._recovery_contexts[incident_id] = context
        self._recovery_ledger = RecoveryLedger(total_recovery_actions=total, incident_attempts=tuple(sorted(attempts.items(), key=lambda item: item[0].encode('utf-8'))), obligation_observe_actions=self._recovery_ledger.obligation_observe_actions, recorded_actions=self._recovery_ledger.recorded_actions, incident_observation_actions=self._recovery_ledger.incident_observation_actions)
        return self._recovery_ledger

    @staticmethod
    def _assumed_effects_for_context(context: Any) -> list[dict[str, Any]]:
        effects = []
        required = set(context.required_effects)
        for binding in context.effect_bindings:
            if binding.get('binding_status') != 'known' or binding.get('effect') not in required:
                continue
            effects.append({'effect': binding['effect'], 'subject_public_role': binding['subject_public_role'], 'relation_or_property': binding['relation_or_property'], 'object_public_role_or_value': copy.deepcopy(binding['object_public_role_or_value']), 'certainty': 'uncertain', 'source_incident_id': context.incident_id, 'source_frame_ids': list(context.evidence_frame_ids)})
        return effects

    def start(self, episode_id: str, *, instruction: str='') -> None:
        from scoper.harness.evidence_memory import EvidenceMemory
        from scoper.harness.recovery import RecoveryLedger
        if self.active:
            raise RuntimeError('Harness episode is already active')
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('Harness episode ID must be non-empty text')
        self._episode_id = episode_id
        self._task_instruction = str(instruction or '')
        self._memory = EvidenceMemory.empty(episode_id)
        self._recovery_ledger = RecoveryLedger()
        self._last_successful_navigation_roles = ()
        self._pending_public_retries = {}
        self._recovery_contexts = {}
        self._placement_recoveries = {}
        self._pending_recovery_actions = {}
        self._last_recovery_notice = None
        self._assumed_completed_actions = []
        self._exhausted_preparation_fingerprints = set()
        self._private_binding_receipts = {}
        self._private_recovery_binding_receipts = {}
        self._private_processed_binding_role_set = set()
        self.stream_buffer.reset_episode()

    def finish(self, *, assert_closed: bool=True) -> None:
        from scoper.harness.contracts import MethodContractError
        lifecycle_error: BaseException | None = None
        try:
            if self.stream_buffer.active:
                raise MethodContractError('method_runtime_lifecycle_invalid', 'Harness episode ended during stream collection')
            self.stream_buffer.assert_ready_for_policy_decision()
            check = getattr(self.event_sink, 'assert_closed', None)
            if assert_closed and callable(check):
                check()
        except BaseException as exc:
            lifecycle_error = exc
        try:
            reset = getattr(self.event_sink, 'reset_episode', None)
            if callable(reset):
                reset()
        except BaseException as exc:
            lifecycle_error = lifecycle_error or exc
        self.stream_buffer.reset_episode()
        self._episode_id = None
        self._task_instruction = ''
        self._memory = None
        from scoper.harness.recovery import RecoveryLedger
        self._recovery_ledger = RecoveryLedger()
        self._last_successful_navigation_roles = ()
        self._pending_public_retries = {}
        self._recovery_contexts = {}
        self._placement_recoveries = {}
        self._pending_recovery_actions = {}
        self._last_recovery_notice = None
        self._assumed_completed_actions = []
        self._exhausted_preparation_fingerprints = set()
        self._private_binding_receipts = {}
        self._private_recovery_binding_receipts = {}
        self._private_processed_binding_role_set = set()
        if lifecycle_error is not None:
            raise lifecycle_error

    def abort(self) -> None:
        try:
            self.finish(assert_closed=False)
        except Exception:
            self._episode_id = None
            self._task_instruction = ''
            self._memory = None
            self._recovery_contexts = {}
            self._placement_recoveries = {}
            self._pending_recovery_actions = {}
            self._last_recovery_notice = None
            self._assumed_completed_actions = []
            self._exhausted_preparation_fingerprints = set()
            self._private_binding_receipts = {}
            self._private_recovery_binding_receipts = {}
            self._private_processed_binding_role_set = set()
            self.stream_buffer.reset_episode()

    def provenance(self) -> Mapping[str, Any]:
        return {'harness_id': self.definition.spec.harness_id, 'harness_version': self.definition.version, 'harness_spec_sha256': self.definition.harness_spec_sha256, 'harness_implementation_id': self.definition.implementation_id}
_HARNESS_REGISTRY: dict[tuple[str, str], HarnessDefinition] = {}
_HARNESS_BY_SPEC_SHA256: dict[str, HarnessDefinition] = {}

def register_harness_definition(definition: HarnessDefinition) -> None:
    key = (definition.spec.harness_id, definition.spec.harness_version)
    existing = _HARNESS_REGISTRY.get(key)
    if existing is not None and existing != definition:
        raise ValueError(f'HarnessDefinition {key!r} is already registered differently')
    _HARNESS_REGISTRY[key] = definition
    _HARNESS_BY_SPEC_SHA256[definition.harness_spec_sha256] = definition

def get_harness_definition(harness_id: str=DEFAULT_HARNESS_ID, harness_version: str=DEFAULT_HARNESS_VERSION) -> HarnessDefinition:
    try:
        return _HARNESS_REGISTRY[harness_id, harness_version]
    except KeyError as exc:
        raise ValueError(f'unknown HarnessDefinition {(harness_id, harness_version)!r}') from exc

def get_harness_definition_by_sha256(harness_spec_sha256: str) -> HarnessDefinition:
    definition = _HARNESS_BY_SPEC_SHA256.get(harness_spec_sha256)
    if definition is None:
        raise ValueError(f'unknown or ambiguous HarnessSpec sha256 {harness_spec_sha256!r}')
    return definition
DEFAULT_HARNESS_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=DEFAULT_HARNESS_SPEC, implementation_id='scoper.harness.harness-v12', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V12_DEFINITION = DEFAULT_HARNESS_DEFINITION
HARNESS_V13_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V13_SPEC, implementation_id='scoper.harness.harness-v13', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V14_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V14_SPEC, implementation_id='scoper.harness.harness-v14', component_sample_kinds=(('memory', ('memory_observe', 'memory_reconcile', 'memory_update')), ('policy', ('recovery_decision', 'step_decision')), ('verifier', ('verifier_transition',))), stream_window_size=16)
HARNESS_V15_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V15_SPEC, implementation_id='scoper.harness.harness-v15', component_sample_kinds=HARNESS_V14_DEFINITION.component_sample_kinds, stream_window_size=16)
HARNESS_V16_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V16_SPEC, implementation_id='scoper.harness.harness-v16', component_sample_kinds=HARNESS_V15_DEFINITION.component_sample_kinds, stream_window_size=16)
HARNESS_V17_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V17_SPEC, implementation_id='scoper.harness.harness-v17', component_sample_kinds=HARNESS_V16_DEFINITION.component_sample_kinds, stream_window_size=8)
register_harness_definition(HARNESS_V17_DEFINITION)
HARNESS_V18_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V18_SPEC, implementation_id='scoper.harness.harness-v18', component_sample_kinds=HARNESS_V14_DEFINITION.component_sample_kinds, stream_window_size=8)
register_harness_definition(HARNESS_V18_DEFINITION)
HARNESS_V19_DEFINITION = replace(HARNESS_V18_DEFINITION, spec=HARNESS_V19_SPEC, implementation_id='scoper.harness.harness-v19')
register_harness_definition(HARNESS_V19_DEFINITION)
HARNESS_V20_DEFINITION = replace(HARNESS_V19_DEFINITION, spec=HARNESS_V20_SPEC, implementation_id='scoper.harness.harness-v20')
register_harness_definition(HARNESS_V20_DEFINITION)
HARNESS_V21_DEFINITION = replace(HARNESS_V20_DEFINITION, spec=HARNESS_V21_SPEC, implementation_id='scoper.harness.scoper.harness-v21')
register_harness_definition(HARNESS_V21_DEFINITION)
HARNESS_V11_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V11_SPEC, implementation_id='scoper.harness.harness-v11', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V10_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V10_SPEC, implementation_id='scoper.harness.harness-v10', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V9_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V9_SPEC, implementation_id='scoper.harness.harness-v9', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V8_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V8_SPEC, implementation_id='scoper.harness.harness-v8', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V7_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V7_SPEC, implementation_id='scoper.harness.harness-v7', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V6_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V6_SPEC, implementation_id='scoper.harness.harness-v6', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V4_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V4_SPEC, implementation_id='scoper.harness.harness-v4', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V5_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V5_SPEC, implementation_id='scoper.harness.harness-v5', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
HARNESS_V3_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=HARNESS_V3_SPEC, implementation_id='scoper.harness.harness-v3', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
LEGACY_HARNESS_DEFINITION = HarnessDefinition(schema_version=HARNESS_DEFINITION_SCHEMA, spec=LEGACY_HARNESS_SPEC, implementation_id='scoper.harness.harness-v2', component_sample_kinds=TRAINING_COMPONENT_SAMPLE_KINDS, stream_window_size=16)
register_harness_definition(LEGACY_HARNESS_DEFINITION)
register_harness_definition(HARNESS_V3_DEFINITION)
register_harness_definition(HARNESS_V4_DEFINITION)
register_harness_definition(HARNESS_V5_DEFINITION)
register_harness_definition(HARNESS_V6_DEFINITION)
register_harness_definition(HARNESS_V7_DEFINITION)
register_harness_definition(HARNESS_V8_DEFINITION)
register_harness_definition(HARNESS_V9_DEFINITION)
register_harness_definition(HARNESS_V10_DEFINITION)
register_harness_definition(HARNESS_V11_DEFINITION)
register_harness_definition(DEFAULT_HARNESS_DEFINITION)
register_harness_definition(HARNESS_V13_DEFINITION)
register_harness_definition(HARNESS_V14_DEFINITION)
register_harness_definition(HARNESS_V15_DEFINITION)
register_harness_definition(HARNESS_V16_DEFINITION)
__all__ = ['DEFAULT_HARNESS_ID', 'DEFAULT_HARNESS_DEFINITION', 'DEFAULT_HARNESS_SPEC', 'DEFAULT_HARNESS_SPEC_SHA256', 'DEFAULT_HARNESS_VERSION', 'HARNESS_V3_DEFINITION', 'HARNESS_V3_SPEC', 'HARNESS_V3_SPEC_SHA256', 'HARNESS_V3_VERSION', 'HARNESS_V4_DEFINITION', 'HARNESS_V4_SPEC', 'HARNESS_V4_SPEC_SHA256', 'HARNESS_V4_VERSION', 'HARNESS_V5_DEFINITION', 'HARNESS_V5_SPEC', 'HARNESS_V5_SPEC_SHA256', 'HARNESS_V5_VERSION', 'HARNESS_V6_DEFINITION', 'HARNESS_V6_SPEC', 'HARNESS_V6_SPEC_SHA256', 'HARNESS_V6_VERSION', 'HARNESS_V7_DEFINITION', 'HARNESS_V7_SPEC', 'HARNESS_V7_SPEC_SHA256', 'HARNESS_V7_VERSION', 'HARNESS_V8_DEFINITION', 'HARNESS_V8_SPEC', 'HARNESS_V8_SPEC_SHA256', 'HARNESS_V8_VERSION', 'HARNESS_V9_DEFINITION', 'HARNESS_V9_SPEC', 'HARNESS_V9_SPEC_SHA256', 'HARNESS_V9_VERSION', 'HARNESS_V10_DEFINITION', 'HARNESS_V10_SPEC', 'HARNESS_V10_SPEC_SHA256', 'HARNESS_V10_VERSION', 'HARNESS_V11_DEFINITION', 'HARNESS_V11_SPEC', 'HARNESS_V11_SPEC_SHA256', 'HARNESS_V11_VERSION', 'HARNESS_V12_DEFINITION', 'HARNESS_V12_SPEC', 'HARNESS_V12_SPEC_SHA256', 'HARNESS_V12_VERSION', 'HARNESS_V13_DEFINITION', 'HARNESS_V13_SPEC', 'HARNESS_V13_SPEC_SHA256', 'HARNESS_V13_VERSION', 'HARNESS_V14_DEFINITION', 'HARNESS_V14_SPEC', 'HARNESS_V14_SPEC_SHA256', 'HARNESS_V14_VERSION', 'HARNESS_V15_DEFINITION', 'HARNESS_V15_SPEC', 'HARNESS_V15_SPEC_SHA256', 'HARNESS_V15_VERSION', 'HARNESS_V16_DEFINITION', 'HARNESS_V17_DEFINITION', 'HARNESS_V16_SPEC', 'HARNESS_V17_SPEC', 'HARNESS_V16_SPEC_SHA256', 'HARNESS_V17_SPEC_SHA256', 'HARNESS_V16_VERSION', 'HARNESS_V17_VERSION', 'LEGACY_HARNESS_DEFINITION', 'LEGACY_HARNESS_SPEC', 'LEGACY_HARNESS_SPEC_SHA256', 'LEGACY_HARNESS_VERSION', 'HARNESS_SPEC_SCHEMA', 'HARNESS_DEFINITION_SCHEMA', 'HarnessCapabilityError', 'HarnessDefinition', 'HarnessSession', 'HarnessSpec', 'public_transition_error_type', 'MemoryModelClient', 'PolicyModelClient', 'TRAINING_COMPONENT_SAMPLE_KINDS', 'VerifierModelClient', 'assert_harness_spec', 'get_harness_definition', 'get_harness_definition_by_sha256', 'register_harness_definition']
