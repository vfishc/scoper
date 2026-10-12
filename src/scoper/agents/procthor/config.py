from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple
import yaml
from scoper.benchmark.common import SHARED_VOCABULARY_VERSION
from scoper.benchmark.common.vlm_observation import validate_standard_vlm_exposure
ALLOWED_ADAPTER_TYPES = {'openai', 'gemini', 'anthropic', 'hf_local'}
ALLOWED_OBSERVATION_MODES = {'single_frame', 'multi_frame', 'streaming'}
ALLOWED_IMAGE_FORMATS = {'png', 'jpeg'}
ALLOWED_METADATA_MODES = {'standard', 'privileged'}
ALLOWED_EVAL_MODES = {'agent', 'standard_vlm'}

class VLMConfigError(ValueError):
    pass

@dataclass
class AdapterConfig:
    type: str
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0
    api_key_env: str = ''
    base_url: str = ''
    extra: dict = field(default_factory=dict)

@dataclass
class ObservationConfig:
    mode: str = 'single_frame'
    num_frames: int = 1
    max_history_turns: int = 5
    image_format: str = 'png'
    max_resolution: Tuple[int, int] = (1024, 1024)
    jpeg_quality: int = 90

@dataclass
class MetadataExposureConfig:
    mode: str = 'standard'
    expose_visible_catalog: bool = True
    expose_raw_object_ids: bool = False
    expose_states: bool = False
    expose_bbox: bool = False
    expose_segmentation: bool = False
    log_privileged_metadata: bool = True

@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_cap: float = 30.0
    timeout_seconds: float = 120.0

@dataclass
class PromptExposureConfig:
    tool_specs: bool = True
    tool_contract: bool = True
    visible_catalog: bool = True
    target_handles: bool = True
    room_handles: bool = True
    agent_pose: bool = True
    current_room: bool = True
    distance: bool = True
    states: bool = False
    bbox: bool = False
    raw_object_ids: bool = False
    held_objects: bool = True
    human_messages: bool = True
    visible_issues: bool = True
    scheduler_notices: bool = True
    tool_history: bool = True
    tool_feedback: bool = True
    goal_progress: bool = True
    state_delta: bool = True
    structured_memory: bool = True
    conversation_history: bool = True
    image_history: bool = False

@dataclass
class ObjectBindingConfig:
    allow_role_handles: bool = True
    allow_visible_aliases: bool = True
    allow_memory: bool = True
    allow_description_match: bool = True
    allow_raw_object_ids: bool = False
    allow_task_text_candidates: bool = False
    ambiguous_score_margin: float = 0.05
    benevolent_task_progress: bool = False
    vocabulary_version: str | None = None

@dataclass
class VLMConfig:
    adapter: AdapterConfig
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    metadata: MetadataExposureConfig = field(default_factory=MetadataExposureConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    prompt_template: str = 'minimal'
    eval_mode: str = 'agent'
    exposure: PromptExposureConfig = field(default_factory=PromptExposureConfig)
    binding: ObjectBindingConfig = field(default_factory=ObjectBindingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'VLMConfig':
        config_path = Path(path)
        data = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
        if not isinstance(data, Mapping):
            raise VLMConfigError('VLM config must be a mapping.')
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> 'VLMConfig':
        if 'adapter' not in data:
            raise VLMConfigError("VLM config requires an 'adapter' block.")
        adapter = _adapter_config(data['adapter'])
        observation = _observation_config(data['observation'] if 'observation' in data else {})
        metadata = _metadata_config(data['metadata'] if 'metadata' in data else {})
        retry = _retry_config(data['retry'] if 'retry' in data else {})
        eval_mode = str(data.get('eval_mode') or 'agent').strip().lower()
        prompt_value = data['prompt_template'] if 'prompt_template' in data else 'minimal'
        prompt_template = str(prompt_value or '').strip()
        if not prompt_template:
            raise VLMConfigError('prompt_template must be non-empty.')
        exposure = _exposure_config(data.get('exposure', {}), eval_mode)
        binding = _binding_config(data.get('binding', {}), eval_mode)
        cfg = cls(adapter=adapter, observation=observation, metadata=metadata, retry=retry, prompt_template=prompt_template, eval_mode=eval_mode, exposure=exposure, binding=binding)
        validate_config(cfg)
        return cfg

def _adapter_config(data: Any) -> AdapterConfig:
    if not isinstance(data, Mapping):
        raise VLMConfigError('adapter must be a mapping.')
    adapter_type = str(data.get('type') or '').strip().lower()
    model = str(data.get('model') or '').strip()
    if not adapter_type:
        raise VLMConfigError('adapter.type is required.')
    if not model:
        raise VLMConfigError('adapter.model is required.')
    extra = data['extra'] if 'extra' in data else {}
    if not isinstance(extra, Mapping):
        raise VLMConfigError('adapter.extra must be a mapping when provided.')
    return AdapterConfig(type=adapter_type, model=model, max_tokens=int(data.get('max_tokens', 1024)), temperature=float(data.get('temperature', 0.0)), api_key_env=str(data.get('api_key_env') or ''), base_url=str(data.get('base_url') or ''), extra=dict(extra))

def _observation_config(data: Any) -> ObservationConfig:
    if not isinstance(data, Mapping):
        raise VLMConfigError('observation must be a mapping when provided.')
    max_resolution = _resolution_tuple(data.get('max_resolution', (1024, 1024)))
    return ObservationConfig(mode=str(data.get('mode') or 'single_frame').strip().lower(), num_frames=int(data.get('num_frames', 1)), max_history_turns=int(data.get('max_history_turns', 5)), image_format=str(data.get('image_format') or 'png').strip().lower(), max_resolution=max_resolution, jpeg_quality=int(data.get('jpeg_quality', 90)))

def _retry_config(data: Any) -> RetryConfig:
    if not isinstance(data, Mapping):
        raise VLMConfigError('retry must be a mapping when provided.')
    return RetryConfig(max_retries=int(data.get('max_retries', 3)), backoff_base=float(data.get('backoff_base', 2.0)), backoff_cap=float(data.get('backoff_cap', 30.0)), timeout_seconds=float(data.get('timeout_seconds', 120.0)))

def _metadata_config(data: Any) -> MetadataExposureConfig:
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise VLMConfigError('metadata must be a mapping when provided.')
    return MetadataExposureConfig(mode=str(data.get('mode') or 'standard').strip().lower(), expose_visible_catalog=bool(data.get('expose_visible_catalog', True)), expose_raw_object_ids=bool(data.get('expose_raw_object_ids', False)), expose_states=bool(data.get('expose_states', False)), expose_bbox=bool(data.get('expose_bbox', False)), expose_segmentation=bool(data.get('expose_segmentation', False)), log_privileged_metadata=bool(data.get('log_privileged_metadata', True)))

def _exposure_config(data: Any, eval_mode: str) -> PromptExposureConfig:
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise VLMConfigError('exposure must be a mapping when provided.')
    if eval_mode == 'standard_vlm':
        defaults = {'tool_specs': True, 'tool_contract': True, 'visible_catalog': False, 'target_handles': False, 'room_handles': False, 'agent_pose': False, 'current_room': False, 'distance': False, 'states': False, 'bbox': False, 'raw_object_ids': False, 'held_objects': False, 'human_messages': True, 'visible_issues': False, 'scheduler_notices': False, 'tool_history': True, 'tool_feedback': True, 'goal_progress': False, 'state_delta': False, 'structured_memory': False, 'conversation_history': False, 'image_history': False}
    else:
        defaults = {'tool_specs': True, 'tool_contract': True, 'visible_catalog': True, 'target_handles': False, 'room_handles': True, 'agent_pose': True, 'current_room': True, 'distance': True, 'states': False, 'bbox': False, 'raw_object_ids': False, 'held_objects': True, 'human_messages': True, 'visible_issues': True, 'scheduler_notices': True, 'tool_history': True, 'tool_feedback': True, 'goal_progress': False, 'state_delta': False, 'structured_memory': True, 'conversation_history': True, 'image_history': False}
    unknown = set(data.keys()) - set(defaults.keys())
    if unknown:
        raise VLMConfigError('Unknown exposure option(s): {}.'.format(', '.join(sorted((str(key) for key in unknown)))))
    defaults.update({str(key): bool(value) for key, value in data.items()})
    return PromptExposureConfig(**defaults)

def _binding_config(data: Any, eval_mode: str) -> ObjectBindingConfig:
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise VLMConfigError('binding must be a mapping when provided.')
    if eval_mode == 'standard_vlm':
        defaults: dict[str, Any] = {'allow_role_handles': False, 'allow_visible_aliases': True, 'allow_memory': True, 'allow_description_match': True, 'allow_raw_object_ids': False, 'allow_task_text_candidates': False, 'ambiguous_score_margin': 0.05, 'benevolent_task_progress': False, 'vocabulary_version': None}
    else:
        defaults = {'allow_role_handles': False, 'allow_visible_aliases': True, 'allow_memory': True, 'allow_description_match': True, 'allow_raw_object_ids': False, 'allow_task_text_candidates': False, 'ambiguous_score_margin': 0.05, 'benevolent_task_progress': False, 'vocabulary_version': None}
    unknown = set(data.keys()) - set(defaults.keys())
    if unknown:
        raise VLMConfigError('Unknown binding option(s): {}.'.format(', '.join(sorted((str(key) for key in unknown)))))
    defaults.update(dict(data))
    vocabulary_version = defaults['vocabulary_version']
    if vocabulary_version is not None:
        if not isinstance(vocabulary_version, str) or vocabulary_version != SHARED_VOCABULARY_VERSION:
            raise VLMConfigError('binding.vocabulary_version must be the supported shared vocabulary version.')
    return ObjectBindingConfig(allow_role_handles=bool(defaults['allow_role_handles']), allow_visible_aliases=bool(defaults['allow_visible_aliases']), allow_memory=bool(defaults['allow_memory']), allow_description_match=bool(defaults['allow_description_match']), allow_raw_object_ids=bool(defaults['allow_raw_object_ids']), allow_task_text_candidates=bool(defaults['allow_task_text_candidates']), ambiguous_score_margin=float(defaults['ambiguous_score_margin']), benevolent_task_progress=bool(defaults['benevolent_task_progress']), vocabulary_version=vocabulary_version)

def _resolution_tuple(value: Any) -> Tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise VLMConfigError('observation.max_resolution must be a two-item sequence.')
    if len(value) != 2:
        raise VLMConfigError('observation.max_resolution must contain width and height.')
    width, height = (int(value[0]), int(value[1]))
    return (width, height)

def validate_config(config: VLMConfig) -> None:
    if config.eval_mode == 'standard_vlm':
        try:
            validate_standard_vlm_exposure(config.exposure)
        except ValueError as exc:
            raise VLMConfigError(str(exc)) from exc
    if config.eval_mode not in ALLOWED_EVAL_MODES:
        raise VLMConfigError("Unknown eval_mode '{}'. Allowed: {}.".format(config.eval_mode, ', '.join(sorted(ALLOWED_EVAL_MODES))))
    if config.adapter.type not in ALLOWED_ADAPTER_TYPES:
        raise VLMConfigError("Unknown adapter.type '{}'. Allowed: {}.".format(config.adapter.type, ', '.join(sorted(ALLOWED_ADAPTER_TYPES))))
    if config.adapter.max_tokens <= 0:
        raise VLMConfigError('adapter.max_tokens must be positive.')
    if config.adapter.temperature < 0:
        raise VLMConfigError('adapter.temperature must be non-negative.')
    obs = config.observation
    if obs.mode not in ALLOWED_OBSERVATION_MODES:
        raise VLMConfigError("Unknown observation.mode '{}'. Allowed: {}.".format(obs.mode, ', '.join(sorted(ALLOWED_OBSERVATION_MODES))))
    if obs.num_frames <= 0:
        raise VLMConfigError('observation.num_frames must be positive.')
    if obs.max_history_turns < 0:
        raise VLMConfigError('observation.max_history_turns must be non-negative.')
    if obs.image_format not in ALLOWED_IMAGE_FORMATS:
        raise VLMConfigError("observation.image_format must be 'png' or 'jpeg'.")
    if obs.max_resolution[0] <= 0 or obs.max_resolution[1] <= 0:
        raise VLMConfigError('observation.max_resolution values must be positive.')
    if not 1 <= obs.jpeg_quality <= 100:
        raise VLMConfigError('observation.jpeg_quality must be in [1, 100].')
    if config.metadata.mode not in ALLOWED_METADATA_MODES:
        raise VLMConfigError("metadata.mode must be 'standard' or 'privileged'.")
    retry = config.retry
    if retry.max_retries < 0:
        raise VLMConfigError('retry.max_retries must be non-negative.')
    if retry.backoff_base <= 0 or retry.backoff_cap < 0 or retry.timeout_seconds <= 0:
        raise VLMConfigError('retry timing values must be positive.')
    if config.binding.ambiguous_score_margin < 0:
        raise VLMConfigError('binding.ambiguous_score_margin must be non-negative.')
__all__ = ['AdapterConfig', 'MetadataExposureConfig', 'ObjectBindingConfig', 'ObservationConfig', 'PromptExposureConfig', 'RetryConfig', 'VLMConfig', 'VLMConfigError', 'validate_config']
