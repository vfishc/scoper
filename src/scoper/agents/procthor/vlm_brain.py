from __future__ import annotations
import json
import copy
import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
import numpy as np
from scoper.benchmark.common.vlm_observation import ScanObservationWindow
from scoper.benchmark.procthor.brains.base import BaseBrain, BrainInput, ToolCall
from scoper.agents.procthor.adapters import create_adapter
from scoper.agents.procthor.config import VLMConfig
from scoper.agents.procthor.frame_buffer import FrameBuffer
from scoper.agents.procthor.image_encoder import ImageEncoder
from scoper.agents.procthor.message import Message
from scoper.agents.procthor.prompt_builder import PromptBuilder, text_only_message
from scoper.agents.procthor.response_parser import ResponseParser
from scoper.benchmark.procthor.export.model_io import ModelCallContext
DIRECT_MULTISTEP_V2_REASK = 'Soft review before stop: check every clause in the full instruction. Requested actions alone do not prove success. Return the best next ToolCall, or stop again if current visual evidence supports completion.'

class VLMBrainInfrastructureError(RuntimeError):
    pass

class VLMBrain(BaseBrain):

    def __init__(self, config: VLMConfig, adapter: Optional[Any]=None, *, close_adapter_on_episode_end: bool=True, model_io_sink: Optional[Callable[[Mapping[str, Any]], None]]=None, model_image_sink: Optional[Callable[..., str]]=None) -> None:
        self.config = config
        self.image_encoder = ImageEncoder(max_resolution=config.observation.max_resolution, format=config.observation.image_format, jpeg_quality=config.observation.jpeg_quality)
        self.adapter = adapter or create_adapter(config, self.image_encoder)
        self.close_adapter_on_episode_end = bool(close_adapter_on_episode_end)
        self.frame_buffer = FrameBuffer(mode=config.observation.mode, num_frames=config.observation.num_frames)
        self.scan_window = ScanObservationWindow()
        self.prompt_builder = PromptBuilder(config.prompt_template, exposure=config.exposure, eval_mode=config.eval_mode, direct_multistep_v2=bool(config.adapter.extra.get('direct_multistep_v2', False)))
        self.direct_multistep_v2 = self.prompt_builder.direct_multistep_v2
        self._direct_multistep_stop_reask_used = False
        self.response_parser = ResponseParser()
        self.conversation_history: list[Message] = []
        self.last_raw_response: str = ''
        self.last_messages: list[Message] = []
        self.frame_capture_dir: Optional[Path] = None
        self.frame_capture_manifest: Optional[Path] = None
        self.frame_capture_task_id: Optional[str] = None
        self.frame_capture_call_index = 0
        self.model_io_sink = model_io_sink
        self.model_image_sink = model_image_sink
        self.model_call_context: Optional[ModelCallContext] = None

    def configure_model_io(self, *, model_io_sink: Callable[[Mapping[str, Any]], None], model_image_sink: Callable[..., str]) -> None:
        if self.frame_capture_dir is not None:
            raise VLMBrainInfrastructureError('legacy frame capture and bundle model sinks are mutually exclusive')
        self.model_io_sink = model_io_sink
        self.model_image_sink = model_image_sink

    def set_model_call_context(self, context: ModelCallContext | None) -> None:
        self.model_call_context = context

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        from scoper.benchmark.common.binding_observation_ledger import enable_binding_delivery
        self._binding_environment = env
        self.direct_decision_records = []
        enable_binding_delivery(env, task_spec)
        self.frame_buffer.reset()
        self.scan_window.reset()
        self.conversation_history = []
        self._direct_multistep_stop_reask_used = False
        self.frame_capture_task_id = str(task_spec.get('id') or '')
        self.frame_capture_call_index = 0
        prompt_context = env.get_prompt_context()
        self.prompt_builder.set_task_context(task_spec, prompt_context.get('tool_specs') or [], prompt_context=prompt_context)
        tool_contract = prompt_context.get('tool_contract') or {}
        self.response_parser.set_allowed_tools(tool_contract.get('allowed_model_tools'))

    def on_scan_rgb(self, rgb: Any) -> None:
        self.scan_window.append(rgb)

    def act(self, brain_input: BrainInput) -> ToolCall:
        rgb = brain_input.observation.get('head_rgb')
        if rgb is None:
            error = VLMBrainInfrastructureError('head RGB is missing from observation')
            self._write_failed_model_call(error, messages=[])
            raise error
        if self.model_io_sink is not None and (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or (rgb.shape[-1] != 3)):
            error = VLMBrainInfrastructureError('bundle model input requires HWC RGB uint8')
            self._write_failed_model_call(error, messages=[])
            raise error
        messages = []
        image_frame_ids: list[str] = []
        started = time.monotonic()
        try:
            self.frame_buffer.update(rgb)
            frames = self.scan_window.consume(rgb, self.frame_buffer.get_frames())
            from scoper.benchmark.common.binding_observation_ledger import deliver_binding_frames
            deliver_binding_frames(getattr(self, '_binding_environment', None), frames, brain_input.observation)
            messages = self.prompt_builder.build(brain_input, frames, self.conversation_history)
            self._capture_vlm_input_frames(brain_input=brain_input, frames=frames, messages=messages)
            serialized_messages, image_frame_ids = self._capture_model_images(messages)
            raw_response = self.adapter.call(messages)
        except Exception as exc:
            self._write_failed_model_call(exc, messages=serialized_messages if 'serialized_messages' in locals() else [], image_frame_ids=image_frame_ids, latency_seconds=time.monotonic() - started)
            raise VLMBrainInfrastructureError('VLM adapter failed: {}'.format(type(exc).__name__)) from exc
        self.last_messages = list(messages)
        self.last_raw_response = str(raw_response)
        tool_call = self.response_parser.parse(raw_response)
        from scoper.benchmark.common.direct_response import record_direct_decision
        record_direct_decision(self, brain_input.high_level_step, raw_response, tool_call, self.response_parser.last_parse_error)
        parse_error = self.response_parser.last_parse_error is not None
        history_messages = messages
        if self.direct_multistep_v2 and (not parse_error) and (tool_call.tool_name == 'stop') and (not self._direct_multistep_stop_reask_used):
            self._direct_multistep_stop_reask_used = True
            first_raw, first_call = (raw_response, tool_call)
            messages = [*messages, {'role': 'assistant', 'content': [{'type': 'text', 'text': str(first_raw)}]}, {'role': 'user', 'content': [{'type': 'text', 'text': DIRECT_MULTISTEP_V2_REASK}]}]
            serialized_messages = [*serialized_messages, {'role': 'assistant', 'content': [{'type': 'text', 'text': str(first_raw)}]}, {'role': 'user', 'content': [{'type': 'text', 'text': DIRECT_MULTISTEP_V2_REASK}]}]
            try:
                candidate_raw = self.adapter.call(messages)
                candidate_call = self.response_parser.parse(candidate_raw)
            except Exception:
                candidate_raw, candidate_call = (first_raw, first_call)
            if self.response_parser.last_parse_error is None:
                raw_response, tool_call = (candidate_raw, candidate_call)
            else:
                raw_response, tool_call = (first_raw, first_call)
                self.response_parser.last_parse_error = None
            parse_error = False
            self.last_messages = list(messages)
            self.last_raw_response = str(raw_response)
        parse_failure = None
        if parse_error:
            metadata = getattr(self.adapter, 'last_call_metadata', None)
            prompt_tokens = getattr(metadata, 'prompt_tokens', None)
            completion_tokens = getattr(metadata, 'completion_tokens', None)
            token_usage = None
            if prompt_tokens is not None or completion_tokens is not None:
                token_usage = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': int(prompt_tokens or 0) + int(completion_tokens or 0)}
            parse_failure = VLMBrainInfrastructureError('model response parsing failed')
            parse_failure.evaluation_failure_record = {'model_call_count': 1, 'backend_metrics': {'model_response_parsing_failed': True}, 'token_usage': token_usage, 'failure': {'type': type(parse_failure).__name__, 'code': 'model_response_parsing_failed', 'reason': str(parse_failure), 'raw_response': str(raw_response)}}
        self._write_model_call(messages=serialized_messages, image_frame_ids=image_frame_ids, raw_response=str(raw_response), tool_call=None if parse_error else tool_call, latency_seconds=time.monotonic() - started, error=parse_failure)
        if parse_failure is not None:
            raise parse_failure
        if self.config.exposure.conversation_history:
            self._update_history(history_messages, str(raw_response))
        return tool_call

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any]) -> None:
        if not self.close_adapter_on_episode_end:
            return
        self.close()

    def close(self) -> None:
        close = getattr(self.adapter, 'close', None)
        if callable(close):
            close()

    def _update_history(self, sent: list[Message], raw_response: str) -> None:
        if not sent:
            return
        user_message = sent[-1]
        if self.config.exposure.image_history or self.config.observation.mode == 'streaming':
            self.conversation_history.append(user_message)
        else:
            self.conversation_history.append(text_only_message(user_message))
        self.conversation_history.append({'role': 'assistant', 'content': [{'type': 'text', 'text': raw_response}]})
        self._truncate_history()

    def _truncate_history(self) -> None:
        max_turns = int(self.config.observation.max_history_turns)
        if max_turns <= 0:
            self.conversation_history = []
            return
        max_messages = max_turns * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def set_frame_capture(self, output_dir: Optional[str | Path]) -> None:
        if output_dir is not None and (self.model_io_sink is not None or self.model_image_sink is not None):
            raise VLMBrainInfrastructureError('legacy frame capture and bundle model sinks are mutually exclusive')
        if output_dir is None:
            self.frame_capture_dir = None
            self.frame_capture_manifest = None
            return
        self.frame_capture_dir = Path(output_dir)
        self.frame_capture_dir.mkdir(parents=True, exist_ok=True)
        self.frame_capture_manifest = self.frame_capture_dir / 'manifest.jsonl'
        self.frame_capture_call_index = 0

    def _capture_vlm_input_frames(self, *, brain_input: BrainInput, frames: list[Any], messages: list[Message]) -> None:
        if self.frame_capture_dir is None:
            return
        step = int(brain_input.high_level_step)
        self.frame_capture_call_index += 1
        saved = []
        suffix = 'jpg' if self.image_encoder.format == 'jpeg' else 'png'
        for index, frame in enumerate(frames, start=1):
            filename = 'call_{:04d}_step_{:03d}_frame_{:02d}.{}'.format(self.frame_capture_call_index, step, index, suffix)
            path = self.frame_capture_dir / filename
            image = self.image_encoder.to_pil(frame)
            if self.image_encoder.format == 'jpeg':
                image.save(path, format='JPEG', quality=self.image_encoder.jpeg_quality)
            else:
                image.save(path, format='PNG')
            saved.append(str(path))
        if self.frame_capture_manifest is None:
            return
        record = {'task_id': self.frame_capture_task_id, 'high_level_step': step, 'call_index': self.frame_capture_call_index, 'image_paths': saved, 'prompt_text': _message_text(messages)}
        with self.frame_capture_manifest.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')

    def _capture_model_images(self, messages: list[Message]):
        if self.model_io_sink is None and self.model_image_sink is None:
            return (copy.deepcopy(messages), [])
        context = self._require_model_context()
        if self.model_image_sink is None:
            raise VLMBrainInfrastructureError('bundle model image sink is missing')
        serialized = []
        image_ids = []
        image_index = 0
        for message in messages:
            blocks = []
            for block in message.get('content', []):
                if block.get('type') == 'text':
                    blocks.append({'type': 'text', 'text': str(block.get('text') or '')})
                elif block.get('type') == 'image':
                    image_index += 1
                    encoded, mime_type = self.image_encoder.encode_bytes(block['rgb'])
                    frame_id = self.model_image_sink(context=context, image_index=image_index, encoded=encoded, mime_type=mime_type)
                    image_ids.append(frame_id)
                    blocks.append({'type': 'image', 'frame_id': frame_id, 'mime_type': mime_type, 'sha256': hashlib.sha256(encoded).hexdigest()})
            serialized.append({'role': message.get('role'), 'content': blocks})
        return (serialized, image_ids)

    def _write_model_call(self, *, messages, image_frame_ids, raw_response, tool_call, latency_seconds, error) -> None:
        if self.model_io_sink is None:
            return
        context = self._require_model_context()
        metadata = getattr(self.adapter, 'last_call_metadata', None)
        prompt_tokens = getattr(metadata, 'prompt_tokens', None)
        completion_tokens = getattr(metadata, 'completion_tokens', None)
        total_tokens = prompt_tokens + completion_tokens if prompt_tokens is not None and completion_tokens is not None else None
        parsed = None if tool_call is None else {'tool_name': tool_call.tool_name, 'arguments': copy.deepcopy(dict(tool_call.arguments)), 'reasoning': tool_call.reasoning}
        record = {'model_call_id': context.model_call_id, 'episode_id': context.episode_id, 'call_type': context.call_type, 'decision_id': context.decision_id, 'action_id': context.action_id, 'stream_obs_id': context.stream_obs_id, 'verification_id': context.verification_id, 'provider': self.config.adapter.type, 'adapter_type': self.config.adapter.type, 'model': Path(self.config.adapter.model).name if self.config.adapter.type == 'hf_local' else self.config.adapter.model, 'checkpoint_hash': None, 'adapter_version': '1', 'prompt_template_id': self.config.prompt_template, 'prompt_template_hash': hashlib.sha256(self.config.prompt_template.encode('utf-8')).hexdigest(), 'request': {'messages': copy.deepcopy(self._semantic_request(messages)['messages']), 'image_frame_ids': list(image_frame_ids), 'generation_config': copy.deepcopy(self._semantic_request(messages)['generation_config']), 'exposure_profile': self.config.eval_mode}, 'response': {'raw_text': raw_response, 'parsed_output': parsed, 'parse_status': 'ok' if parsed is not None else 'error' if error else 'not_returned', 'parse_error': None if error is None else type(error).__name__, 'finish_reason': getattr(metadata, 'finish_reason', None)}, 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens, 'source': 'provider' if metadata is not None else 'unavailable'}, 'attempts': copy.deepcopy(getattr(metadata, 'attempts', []) or []), 'latency_seconds': max(0.0, float(latency_seconds)), 'error': None if error is None else {'type': type(error).__name__, 'message': 'model call failed'}}
        self.model_io_sink(record)

    def _semantic_request(self, messages):
        builder = getattr(self.adapter, 'semantic_request', None)
        if callable(builder):
            value = builder(messages)
            if isinstance(value, Mapping):
                return value
        return {'messages': copy.deepcopy(messages), 'generation_config': {'max_tokens': self.config.adapter.max_tokens, 'temperature': self.config.adapter.temperature}}

    def _write_failed_model_call(self, error, *, messages, image_frame_ids=(), latency_seconds=0.0):
        if self.model_io_sink is None:
            return
        self._write_model_call(messages=messages, image_frame_ids=image_frame_ids, raw_response=None, tool_call=None, latency_seconds=latency_seconds, error=error)

    def _require_model_context(self) -> ModelCallContext:
        if self.model_call_context is None:
            raise VLMBrainInfrastructureError('model call context is missing')
        return self.model_call_context

def _message_text(messages: list[Message]) -> list[dict]:
    out = []
    for message in messages:
        text_parts = []
        image_count = 0
        for block in message.get('content', []):
            if block.get('type') == 'text':
                text_parts.append(str(block.get('text') or ''))
            elif block.get('type') == 'image':
                image_count += 1
        out.append({'role': message.get('role'), 'text': '\n'.join(text_parts), 'image_count': image_count})
    return out
__all__ = ['VLMBrain', 'VLMBrainInfrastructureError']
