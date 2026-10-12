from __future__ import annotations
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
import numpy as np
from scoper.benchmark.common.vlm_observation import ScanObservationWindow
from scoper.benchmark.behavior.brains.base import BaseBrain, BrainInput, ToolCall
from scoper.agents.behavior.adapters import create_adapter
from scoper.agents.behavior.config import VLMConfig
from scoper.agents.behavior.frame_buffer import FrameBuffer
from scoper.agents.behavior.image_encoder import ImageEncoder
from scoper.agents.behavior.message import Message
from scoper.agents.behavior.prompt_builder import PromptBuilder, text_only_message
from scoper.agents.behavior.response_parser import ResponseParser
from scoper.agents.behavior.streaming import StreamDecision, StreamFrameObservation, StreamResponseParser
from scoper.benchmark.behavior.json_utils import _json_safe
from scoper.benchmark.behavior.export.ids import sha256_bytes, sha256_json
DIRECT_MULTISTEP_V2_REASK = 'Soft review before stop: check every clause in the full instruction. Requested actions alone do not prove success. Return the best next ToolCall, or stop again if current visual evidence supports completion.'

@dataclass(frozen=True)
class ModelCallContext:
    episode_id: str
    model_call_id: str
    call_type: str
    decision_id: str | None
    action_id: str | None
    stream_obs_id: str | None
    verification_id: str | None

class VLMBrain(BaseBrain):

    def __init__(self, config: VLMConfig, adapter: Optional[Any]=None, io_log_path: str | Path | None=None, image_log_dir: str | Path | None=None, require_valid_visual_input: bool=True, raise_adapter_errors: bool=False, close_adapter_on_episode_end: bool=True, model_io_sink: Callable[[Mapping[str, Any]], None] | None=None, exact_image_sink: Callable[[bytes, str, Mapping[str, Any]], Mapping[str, Any]] | None=None) -> None:
        self.config = config
        self.require_valid_visual_input = bool(require_valid_visual_input)
        self.raise_adapter_errors = bool(raise_adapter_errors)
        self.close_adapter_on_episode_end = bool(close_adapter_on_episode_end)
        self.image_encoder = ImageEncoder(max_resolution=config.observation.max_resolution, format=config.observation.image_format, jpeg_quality=config.observation.jpeg_quality)
        self.adapter = adapter or create_adapter(config, self.image_encoder)
        frame_window = config.observation.stream.recent_window_frames if config.observation.mode == 'streaming' else config.observation.num_frames
        self.frame_buffer = FrameBuffer(mode=config.observation.mode, num_frames=frame_window)
        self.scan_window = ScanObservationWindow()
        self.prompt_builder = PromptBuilder(config.prompt_template, exposure=config.exposure, eval_mode=config.eval_mode, direct_multistep_v2=bool(config.adapter.extra.get('direct_multistep_v2', False)))
        self.direct_multistep_v2 = self.prompt_builder.direct_multistep_v2
        self._direct_multistep_stop_reask_used = False
        self.response_parser = ResponseParser()
        self.stream_response_parser = StreamResponseParser()
        self.conversation_history: list[Message] = []
        self.last_raw_response: str = ''
        self.last_messages: list[Message] = []
        self.model_io_sink = model_io_sink
        self.exact_image_sink = exact_image_sink
        self._model_call_context: ModelCallContext | None = None
        self.io_log_path: Path | None = None
        self._io_log_fh = None
        self.image_log_dir = Path(image_log_dir) if image_log_dir is not None else None
        if self.model_io_sink is not None and io_log_path is not None:
            raise ValueError('bundle model_io_sink cannot be combined with a legacy io_log_path')
        if self.exact_image_sink is not None and image_log_dir is not None:
            raise ValueError('bundle exact_image_sink cannot be combined with legacy image capture')
        self._image_log_counter = 0
        if self.image_log_dir is not None:
            self.image_log_dir.mkdir(parents=True, exist_ok=True)
        if io_log_path is not None:
            self.set_io_log_path(io_log_path)

    def set_io_log_path(self, path: str | Path | None) -> None:
        if self.model_io_sink is not None and path is not None:
            raise ValueError('bundle model_io_sink cannot be combined with a legacy io_log_path')
        if self._io_log_fh is not None:
            self._io_log_fh.close()
            self._io_log_fh = None
        self.io_log_path = Path(path) if path is not None else None
        if self.io_log_path is not None:
            self.io_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._io_log_fh = self.io_log_path.open('w', encoding='utf-8')

    def set_model_call_context(self, context: ModelCallContext | None) -> None:
        if context is not None and (not isinstance(context, ModelCallContext)):
            raise TypeError('model call context must be ModelCallContext or None')
        self._model_call_context = context

    def on_episode_start(self, env: Any, task_spec: Mapping[str, Any]) -> None:
        from scoper.benchmark.common.binding_observation_ledger import enable_binding_delivery
        self._binding_environment = env
        self.direct_decision_records = []
        enable_binding_delivery(env, task_spec)
        self.frame_buffer.reset()
        self.scan_window.reset()
        self.conversation_history = []
        self._direct_multistep_stop_reask_used = False
        prompt_context = env.get_prompt_context()
        self.prompt_builder.set_task_context(task_spec, prompt_context.get('tool_specs') or [], prompt_context=prompt_context)
        tool_contract = prompt_context.get('tool_contract') or {}
        self.response_parser.set_allowed_tools(tool_contract.get('allowed_model_tools'))
        self.stream_response_parser.set_allowed_tools(tool_contract.get('allowed_model_tools'))

    def on_scan_rgb(self, rgb: Any) -> None:
        self.scan_window.append(rgb)

    def act(self, brain_input: BrainInput) -> ToolCall:
        rgb = brain_input.observation.get('head_rgb')
        if rgb is None:
            raise RuntimeError("VLMBrain requires observation['head_rgb']; configure a renderer or use the fake VLM smoke path.")
        messages: list[Message] = []
        visual_quality = _visual_input_quality(rgb)
        visual_context = _visual_context_snapshot(brain_input.observation)
        try:
            self.frame_buffer.update(rgb)
            frames = self.scan_window.consume(rgb, self.frame_buffer.get_frames())
            from scoper.benchmark.common.binding_observation_ledger import deliver_binding_frames
            deliver_binding_frames(getattr(self, '_binding_environment', None), frames, brain_input.observation)
            messages = self.prompt_builder.build(brain_input, frames, self.conversation_history)
            if self.require_valid_visual_input and (not bool(visual_quality.get('usable', False))):
                if self.model_io_sink is not None or self.config.eval_mode == 'standard_vlm':
                    raise RuntimeError('invalid_visual_input: {}'.format(visual_quality.get('reason') or 'unusable_rgb'))
                logged_messages = self._messages_for_log(messages, call_type='act', call_index=brain_input.high_level_step)
                error_reason = 'invalid_visual_input: {}'.format(visual_quality.get('reason') or 'unusable_rgb')
                self._write_vlm_io({'call_type': 'act', 'task_id': brain_input.observation.get('task_id'), 'high_level_step': brain_input.high_level_step, 'messages': logged_messages, 'image_paths': _image_paths_from_logged_messages(logged_messages), 'raw_response': '', 'parsed_tool_call': {'tool_name': 'stop', 'arguments': {}, 'reasoning': error_reason}, 'eval_mode': self.config.eval_mode, 'prompt_exposure': self.prompt_builder.exposure_snapshot(), 'visual_input_quality': visual_quality, 'visual_context': visual_context, 'latency_seconds': None, 'adapter_metadata': _adapter_metadata(self.adapter), 'error': {'type': 'invalid_visual_input', 'message': error_reason}})
                return ToolCall('stop', {}, reasoning=error_reason)
            messages = self._prepare_exact_messages(messages)
            started = time.time()
            raw_response = self.adapter.call(messages)
            latency = time.time() - started
        except Exception as exc:
            error_reason = 'adapter_error: {}: {}'.format(type(exc).__name__, exc)
            logged_messages = self._messages_for_log(messages, call_type='act', call_index=brain_input.high_level_step) if messages else []
            error_record = {'call_type': 'act', 'task_id': brain_input.observation.get('task_id'), 'high_level_step': brain_input.high_level_step, 'messages': logged_messages, 'image_paths': _image_paths_from_logged_messages(logged_messages), 'raw_response': '', 'parsed_tool_call': {'tool_name': 'stop', 'arguments': {}, 'reasoning': error_reason}, 'eval_mode': self.config.eval_mode, 'prompt_exposure': self.prompt_builder.exposure_snapshot(), 'visual_input_quality': visual_quality, 'visual_context': visual_context, 'latency_seconds': None, 'adapter_metadata': _adapter_metadata(self.adapter), 'error': {'type': type(exc).__name__, 'message': str(exc)}}
            self._write_vlm_io(error_record)
            if self.model_io_sink is not None or self.raise_adapter_errors or self.config.eval_mode == 'standard_vlm':
                raise
            return ToolCall('stop', {}, reasoning=error_reason)
        self.last_messages = list(messages)
        self.last_raw_response = str(raw_response)
        tool_call = self.response_parser.parse(raw_response)
        from scoper.benchmark.common.direct_response import record_direct_decision
        record_direct_decision(self, brain_input.high_level_step, raw_response, tool_call, self.response_parser.last_parse_error)
        if self.config.eval_mode == 'standard_vlm' and self.response_parser.last_parse_error is not None:
            from scoper.benchmark.common.direct_response import DirectResponseParseError
            self._write_vlm_io({'call_type': 'act', 'task_id': brain_input.observation.get('task_id'), 'high_level_step': brain_input.high_level_step, 'raw_response': str(raw_response), 'parsed_tool_call': None, 'error': {'type': 'model_response_parsing_failed', 'message': 'No executable final JSON tool call.'}})
            raise DirectResponseParseError(raw_response)
        history_messages = messages
        if self.direct_multistep_v2 and tool_call.tool_name == 'stop' and (not tool_call.reasoning.startswith('parse_error:')) and (not self._direct_multistep_stop_reask_used):
            self._direct_multistep_stop_reask_used = True
            first_raw, first_call = (raw_response, tool_call)
            messages = [*messages, {'role': 'assistant', 'content': [{'type': 'text', 'text': str(first_raw)}]}, {'role': 'user', 'content': [{'type': 'text', 'text': DIRECT_MULTISTEP_V2_REASK}]}]
            try:
                candidate_raw = self.adapter.call(messages)
                candidate_call = self.response_parser.parse(candidate_raw)
            except Exception:
                candidate_raw, candidate_call = (first_raw, first_call)
            if not candidate_call.reasoning.startswith('parse_error:'):
                raw_response, tool_call = (candidate_raw, candidate_call)
            else:
                raw_response, tool_call = (first_raw, first_call)
            self.last_messages = list(messages)
            self.last_raw_response = str(raw_response)
        logged_messages = self._messages_for_log(messages, call_type='act', call_index=brain_input.high_level_step)
        self._write_vlm_io({'call_type': 'act', 'task_id': brain_input.observation.get('task_id'), 'high_level_step': brain_input.high_level_step, 'messages': logged_messages, 'image_paths': _image_paths_from_logged_messages(logged_messages), 'raw_response': str(raw_response), 'parsed_tool_call': {'tool_name': tool_call.tool_name, 'arguments': dict(tool_call.arguments), 'reasoning': tool_call.reasoning}, 'eval_mode': self.config.eval_mode, 'prompt_exposure': self.prompt_builder.exposure_snapshot(), 'visual_input_quality': visual_quality, 'visual_context': visual_context, 'latency_seconds': latency, 'adapter_metadata': _adapter_metadata(self.adapter)})
        self._update_history(history_messages, str(raw_response))
        return tool_call

    def on_stream_start(self, context: Mapping[str, Any]) -> None:
        task_spec = context.get('task_spec')
        prompt_context = context.get('prompt_context')
        if not isinstance(task_spec, Mapping):
            return None
        if not isinstance(prompt_context, Mapping):
            prompt_context = context
        self.prompt_builder.set_task_context(task_spec, context.get('tool_specs') or prompt_context.get('tool_specs') or [], prompt_context=prompt_context)
        tool_contract = context.get('tool_contract') or prompt_context.get('tool_contract') or {}
        self.response_parser.set_allowed_tools(tool_contract.get('allowed_model_tools'))
        self.stream_response_parser.set_allowed_tools(tool_contract.get('allowed_model_tools'))
        return None

    def act_stream_frame(self, frame: StreamFrameObservation) -> StreamDecision:
        if self.model_io_sink is not None:
            frame = _public_stream_frame(frame)
        messages: list[Message] = []
        visual_quality: dict[str, Any] | None = None
        visual_context = _stream_visual_context_snapshot(frame)
        try:
            if frame.head_rgb is not None:
                visual_quality = _visual_input_quality(frame.head_rgb)
                self.frame_buffer.update(frame.head_rgb)
            stream_user = self.prompt_builder.build_stream_message(frame)
            messages = [self.prompt_builder.build_system_message(), *list(self.conversation_history), stream_user]
            if frame.head_rgb is None:
                frames = []
            else:
                frames = self.frame_buffer.get_frames()
            if frames and len(frames) > 1:
                stream_user = self._stream_message_with_recent_frames(stream_user, frames)
                messages[-1] = stream_user
            if self.require_valid_visual_input and visual_quality is not None and (not bool(visual_quality.get('usable', False))):
                if self.model_io_sink is not None:
                    raise RuntimeError('invalid_visual_input: {}'.format(visual_quality.get('reason') or 'unusable_rgb'))
                logged_messages = self._messages_for_log(messages, call_type='stream_frame', call_index=frame.frame_index)
                error_reason = 'invalid_visual_input: {}'.format(visual_quality.get('reason') or 'unusable_rgb')
                self._write_vlm_io({'call_type': 'stream_frame', 'task_id': frame.task_id, 'high_level_step': frame.high_level_step, 'stream_id': frame.stream_id, 'frame_index': frame.frame_index, 'timestamp_s': frame.timestamp_s, 'source': frame.source, 'is_keyframe': frame.is_keyframe, 'keyframe_reason': frame.keyframe_reason, 'active_tool_name': frame.active_tool_name, 'active_tool_args': dict(frame.active_tool_args), 'messages': logged_messages, 'image_paths': _image_paths_from_logged_messages(logged_messages), 'raw_response': '', 'stream_decision': 'continue', 'interrupt_tool_call': None, 'eval_mode': self.config.eval_mode, 'prompt_exposure': self.prompt_builder.exposure_snapshot(), 'visual_input_quality': visual_quality, 'visual_context': visual_context, 'latency_seconds': None, 'adapter_metadata': _adapter_metadata(self.adapter), 'error': {'type': 'invalid_visual_input', 'message': error_reason}})
                return StreamDecision('continue', reasoning=error_reason, parse_error='invalid_visual_input')
            messages = self._prepare_exact_messages(messages)
            started = time.time()
            raw_response = self.adapter.call(messages)
            latency = time.time() - started
        except Exception as exc:
            error_reason = 'adapter_error: {}: {}'.format(type(exc).__name__, exc)
            logged_messages = self._messages_for_log(messages, call_type='stream_frame', call_index=frame.frame_index) if messages else []
            error_record = {'call_type': 'stream_frame', 'task_id': frame.task_id, 'high_level_step': frame.high_level_step, 'stream_id': frame.stream_id, 'frame_index': frame.frame_index, 'timestamp_s': frame.timestamp_s, 'source': frame.source, 'is_keyframe': frame.is_keyframe, 'keyframe_reason': frame.keyframe_reason, 'active_tool_name': frame.active_tool_name, 'active_tool_args': dict(frame.active_tool_args), 'messages': logged_messages, 'image_paths': _image_paths_from_logged_messages(logged_messages), 'raw_response': '', 'stream_decision': 'continue', 'interrupt_tool_call': None, 'eval_mode': self.config.eval_mode, 'prompt_exposure': self.prompt_builder.exposure_snapshot(), 'visual_input_quality': visual_quality, 'visual_context': visual_context, 'latency_seconds': None, 'adapter_metadata': _adapter_metadata(self.adapter), 'error': {'type': type(exc).__name__, 'message': str(exc)}}
            self._write_vlm_io(error_record)
            if self.model_io_sink is not None:
                raise
            return StreamDecision('continue', reasoning=error_reason, parse_error='stream_adapter_error')
        self.last_messages = list(messages)
        self.last_raw_response = str(raw_response)
        decision = self.stream_response_parser.parse(raw_response)
        logged_messages = self._messages_for_log(messages, call_type='stream_frame', call_index=frame.frame_index)
        self._write_vlm_io({'call_type': 'stream_frame', 'task_id': frame.task_id, 'high_level_step': frame.high_level_step, 'stream_id': frame.stream_id, 'frame_index': frame.frame_index, 'timestamp_s': frame.timestamp_s, 'source': frame.source, 'is_keyframe': frame.is_keyframe, 'keyframe_reason': frame.keyframe_reason, 'active_tool_name': frame.active_tool_name, 'active_tool_args': dict(frame.active_tool_args), 'messages': logged_messages, 'image_paths': _image_paths_from_logged_messages(logged_messages), 'raw_response': str(raw_response), 'stream_decision': decision.stream_decision, 'interrupt_tool_call': {'tool_name': decision.tool_call.tool_name, 'arguments': dict(decision.tool_call.arguments), 'reasoning': decision.tool_call.reasoning} if decision.tool_call is not None else None, 'eval_mode': self.config.eval_mode, 'prompt_exposure': self.prompt_builder.exposure_snapshot(), 'visual_input_quality': visual_quality, 'visual_context': visual_context, 'latency_seconds': latency, 'adapter_metadata': _adapter_metadata(self.adapter)})
        self._update_history(messages, str(raw_response))
        return decision

    def on_stream_end(self, summary: Mapping[str, Any]) -> None:
        if self._io_log_fh is not None:
            self._io_log_fh.close()
            self._io_log_fh = None
        return None

    def on_episode_end(self, env: Any, final_goal: Mapping[str, Any]) -> None:
        if self.close_adapter_on_episode_end:
            self.close()
            return
        if self._io_log_fh is not None:
            self._io_log_fh.close()
            self._io_log_fh = None

    def close(self) -> None:
        close = getattr(self.adapter, 'close', None)
        if callable(close):
            close()
        if self._io_log_fh is not None:
            self._io_log_fh.close()
            self._io_log_fh = None

    def _update_history(self, sent: list[Message], raw_response: str) -> None:
        if not sent:
            return
        if not self.config.exposure.conversation_history:
            self.conversation_history = []
            return
        user_message = sent[-1]
        if self.config.exposure.image_history:
            self.conversation_history.append(user_message)
        else:
            self.conversation_history.append(text_only_message(user_message, keep_images=False))
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

    def _stream_message_with_recent_frames(self, message: Message, frames: list[Any]) -> Message:
        content = [block for block in message.get('content', []) if block.get('type') != 'image']
        for index, frame in enumerate(frames, start=1):
            content.append({'type': 'text', 'text': 'Recent stream frame {}/{}.'.format(index, len(frames))})
            content.append({'type': 'image', 'rgb': frame})
        return {'role': 'user', 'content': content}

    def _write_vlm_io(self, record: Mapping[str, Any]) -> None:
        if self.model_io_sink is not None:
            if self._model_call_context is None:
                raise RuntimeError('model_io_sink requires an active ModelCallContext')
            self.model_io_sink(self._model_io_record(record, self._model_call_context))
            return
        if self._io_log_fh is None:
            return
        self._io_log_fh.write(json.dumps(_json_safe(dict(record)), ensure_ascii=False, sort_keys=True))
        self._io_log_fh.write('\n')
        self._io_log_fh.flush()

    def _prepare_exact_messages(self, messages: list[Message]) -> list[Message]:
        if self.exact_image_sink is None:
            return messages
        if self._model_call_context is None:
            raise RuntimeError('exact_image_sink requires an active ModelCallContext')
        prepared: list[Message] = []
        image_ordinal = 0
        image_count = sum((block.get('type') == 'image' for message in messages for block in message.get('content', [])))
        for message_index, message in enumerate(messages):
            content = []
            for block_index, block in enumerate(message.get('content', [])):
                if block.get('type') != 'image':
                    content.append(dict(block))
                    continue
                image_ordinal += 1
                encoded, mime_type = self.image_encoder.encode(block['rgb'])
                metadata = {'context': asdict(self._model_call_context), 'message_index': message_index, 'block_index': block_index, 'image_ordinal': image_ordinal, 'image_count': image_count}
                saved = self.exact_image_sink(encoded, mime_type, metadata)
                if not isinstance(saved, Mapping):
                    raise RuntimeError('exact_image_sink must return frame metadata')
                frame_id = saved.get('frame_id')
                source_frame_id = saved.get('source_frame_id')
                if not isinstance(frame_id, str) or not frame_id:
                    raise RuntimeError('exact_image_sink did not return frame_id')
                prepared_block = dict(block)
                prepared_block.update({'_encoded_bytes': encoded, '_mime_type': mime_type, '_model_frame': {'type': 'image', 'frame_id': frame_id, 'mime_type': mime_type, 'sha256': sha256_bytes(encoded), 'source_frame_id': source_frame_id}})
                content.append(prepared_block)
            prepared.append({'role': message['role'], 'content': content})
        return prepared

    def _model_io_record(self, record: Mapping[str, Any], context: ModelCallContext) -> Mapping[str, Any]:
        adapter_metadata = record.get('adapter_metadata')
        if not isinstance(adapter_metadata, Mapping):
            adapter_metadata = {}
        prompt_tokens = adapter_metadata.get('prompt_tokens')
        completion_tokens = adapter_metadata.get('completion_tokens')
        total_tokens = prompt_tokens + completion_tokens if isinstance(prompt_tokens, int) and (not isinstance(prompt_tokens, bool)) and isinstance(completion_tokens, int) and (not isinstance(completion_tokens, bool)) else None
        logged_messages = record.get('messages') if isinstance(record.get('messages'), list) else []
        image_frame_ids = []
        for message in logged_messages:
            if not isinstance(message, Mapping):
                continue
            for block in message.get('content') or []:
                if isinstance(block, Mapping) and block.get('type') == 'image' and block.get('frame_id'):
                    image_frame_ids.append(block['frame_id'])
        raw_response = record.get('raw_response')
        error = record.get('error') if isinstance(record.get('error'), Mapping) else None
        parsed = record.get('parsed_tool_call')
        if parsed is None:
            parsed = {'stream_decision': record.get('stream_decision'), 'interrupt_tool_call': record.get('interrupt_tool_call')}
        parse_status = 'error' if error else 'not_returned' if raw_response in {None, ''} else 'ok'
        generation_config = {'max_tokens': self.config.adapter.max_tokens, 'temperature': self.config.adapter.temperature, 'extra': _redact_value(self.config.adapter.extra)}
        model_record = {'schema_version': 'model-io.v1', **asdict(context), 'provider': self.config.adapter.type, 'adapter_type': self.config.adapter.type, 'model': Path(self.config.adapter.model).name if self.config.adapter.type == 'hf_local' else self.config.adapter.model, 'checkpoint_hash': None, 'adapter_version': type(self.adapter).__module__ + '.' + type(self.adapter).__name__, 'prompt_template_id': self.config.prompt_template, 'prompt_template_hash': sha256_json({'prompt_template_id': self.config.prompt_template}), 'request': {'messages': logged_messages, 'image_frame_ids': image_frame_ids, 'generation_config': generation_config, 'exposure_profile': 'privileged_oracle' if self.config.eval_mode == 'standard_vlm' and self.config.exposure.raw_object_ids else 'structured_agent'}, 'response': {'raw_text': raw_response if isinstance(raw_response, str) else None, 'parsed_output': parsed, 'parse_status': parse_status, 'parse_error': _redact_text(str(error.get('message'))) if error and error.get('message') else None, 'finish_reason': adapter_metadata.get('finish_reason')}, 'usage': {'prompt_tokens': prompt_tokens if isinstance(prompt_tokens, int) and (not isinstance(prompt_tokens, bool)) else None, 'completion_tokens': completion_tokens if isinstance(completion_tokens, int) and (not isinstance(completion_tokens, bool)) else None, 'total_tokens': total_tokens, 'source': 'provider' if prompt_tokens is not None or completion_tokens is not None else 'unavailable'}, 'attempts': _redact_value(adapter_metadata.get('attempts') or []), 'latency_seconds': record.get('latency_seconds'), 'error': _redact_value(error)}
        return _redact_value(model_record)

    def _messages_for_log(self, messages: list[Message], *, call_type: str, call_index: int) -> list[dict[str, Any]]:
        return _messages_for_log(messages, image_log_dir=self.image_log_dir, image_encoder=self.image_encoder, allocate_image_path=lambda message_index, block_index: self._allocate_image_log_path(call_type=call_type, call_index=call_index, message_index=message_index, block_index=block_index))

    def _allocate_image_log_path(self, *, call_type: str, call_index: int, message_index: int, block_index: int) -> Path:
        assert self.image_log_dir is not None
        self._image_log_counter += 1
        suffix = 'jpg' if self.config.observation.image_format == 'jpeg' else 'png'
        return self.image_log_dir / '{:06d}_{}_{}_m{}_b{}.{}'.format(self._image_log_counter, _safe_path_token(call_type), int(call_index), int(message_index), int(block_index), suffix)

def _messages_for_log(messages: list[Message], *, image_log_dir: Path | None=None, image_encoder: ImageEncoder | None=None, allocate_image_path: Any | None=None) -> list[dict[str, Any]]:
    logged: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        content = []
        for block_index, block in enumerate(message.get('content', [])):
            if block.get('type') == 'text':
                content.append({'type': 'text', 'text': str(block.get('text', ''))})
            elif block.get('type') == 'image':
                model_frame = block.get('_model_frame')
                if isinstance(model_frame, Mapping):
                    content.append({'type': 'image', 'frame_id': model_frame.get('frame_id'), 'mime_type': model_frame.get('mime_type'), 'sha256': model_frame.get('sha256')})
                    continue
                rgb = block.get('rgb')
                shape = getattr(rgb, 'shape', None)
                dtype = getattr(rgb, 'dtype', None)
                image_record = {'type': 'image', 'omitted': True, 'shape': list(shape) if shape is not None else None, 'dtype': str(dtype) if dtype is not None else None}
                if image_log_dir is not None and image_encoder is not None and (allocate_image_path is not None):
                    try:
                        image_path = allocate_image_path(message_index, block_index)
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        image = image_encoder.to_pil(rgb)
                        if str(image_path).lower().endswith(('.jpg', '.jpeg')):
                            image.save(image_path, format='JPEG', quality=image_encoder.jpeg_quality)
                        else:
                            image.save(image_path, format='PNG')
                        image_record['path'] = str(image_path)
                        image_record['omitted'] = False
                    except Exception as exc:
                        image_record['save_error'] = '{}: {}'.format(type(exc).__name__, exc)
                content.append(image_record)
        logged.append({'role': message.get('role'), 'content': content})
    return logged

def _image_paths_from_logged_messages(messages: list[Mapping[str, Any]]) -> list[str]:
    paths: list[str] = []
    for message in messages:
        for block in message.get('content', []) or []:
            if isinstance(block, Mapping) and block.get('type') == 'image' and block.get('path'):
                paths.append(str(block.get('path')))
    return paths

def _safe_path_token(value: Any) -> str:
    chars = []
    for ch in str(value):
        if ch.isalnum() or ch in {'-', '_'}:
            chars.append(ch)
        else:
            chars.append('_')
    return ''.join(chars).strip('_') or 'call'

def _adapter_metadata(adapter: Any) -> dict[str, Any]:
    metadata = getattr(adapter, 'last_call_metadata', None)
    if metadata is None:
        return {}
    return {'elapsed_seconds': getattr(metadata, 'elapsed_seconds', None), 'prompt_tokens': getattr(metadata, 'prompt_tokens', None), 'completion_tokens': getattr(metadata, 'completion_tokens', None), 'finish_reason': getattr(metadata, 'finish_reason', None), 'attempts': list(getattr(metadata, 'attempts', ()) or ())}

def _redact_text(value: str) -> str:
    result = value
    lowered = result.lower()
    for marker in ('api_key=', 'api-key=', 'authorization:', 'bearer ', 'token='):
        start = lowered.find(marker)
        while start >= 0:
            end = result.find(' ', start + len(marker))
            if end < 0:
                end = len(result)
            result = result[:start + len(marker)] + '[REDACTED]' + result[end:]
            lowered = result.lower()
            start = lowered.find(marker, start + len(marker) + len('[REDACTED]'))
    return re.sub('([?&](?:x-amz-signature|signature|sig|token|key)=)[^&\\s]+', '\\1[REDACTED]', result, flags=re.IGNORECASE)

def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in {'api_key', 'apikey', 'authorization', 'headers', 'access_token', 'refresh_token', 'secret', 'password', 'credential', 'credentials'} or normalized.endswith(('_api_key', '_access_token', '_password')):
                continue
            result[str(key)] = _redact_value(item)
        return result
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value

def _visual_context_snapshot(observation: Mapping[str, Any]) -> dict[str, Any]:
    visible_catalog = observation.get('visible_object_catalog') or []
    categories: set[str] = set()
    if isinstance(visible_catalog, list):
        for item in visible_catalog:
            if not isinstance(item, Mapping):
                continue
            category = item.get('objectType') or item.get('category')
            if category:
                categories.add(str(category))
    visible_ids = observation.get('visible_object_ids') or []
    discoverability = observation.get('visual_discoverability') or {}
    if not isinstance(discoverability, Mapping):
        discoverability = {}
    return {'agent_pose': observation.get('agent_pose'), 'observation_pose': observation.get('observation_pose'), 'visibility_mode': observation.get('visibility_mode'), 'visible_object_count': len(visible_ids) if isinstance(visible_ids, list) else None, 'visible_object_categories': sorted(categories), 'visual_discoverability': {key: discoverability.get(key) for key in ('mode', 'visibility_source', 'rendered_segmentation_available', 'rendered_visibility_status', 'symbolic_fov_fallback', 'renderer_configured') if key in discoverability}}

def _stream_visual_context_snapshot(frame: StreamFrameObservation) -> dict[str, Any]:
    metadata = frame.privileged_metadata if isinstance(frame.privileged_metadata, Mapping) else {}
    if metadata:
        return _visual_context_snapshot(metadata)
    return {'visibility_mode': None, 'visible_object_count': len(frame.visible_object_catalog or []), 'visible_object_categories': sorted({str(item.get('objectType') or item.get('category')) for item in frame.visible_object_catalog or [] if isinstance(item, Mapping) and (item.get('objectType') or item.get('category'))}), 'visual_discoverability': {}}

def _public_stream_frame(frame: StreamFrameObservation) -> StreamFrameObservation:
    allowed_prompt_keys = {'tool_progress', 'tool_result', 'tool_target_handles', 'room_target_handles', 'human_messages', 'visible_issues', 'metadata_mode'}
    prompt_metadata = {key: _public_runtime_value(value) for key, value in frame.prompt_metadata.items() if key in allowed_prompt_keys}
    return replace(frame, prompt_metadata=prompt_metadata, privileged_metadata={})

def _public_runtime_value(value: Any) -> Any:
    private_keys = {'object_id', 'objectid', 'object_ids', 'visible_object_ids', 'target_object_ids', 'states', 'goal_result', 'state_delta', 'scheduler_truth'}
    if isinstance(value, Mapping):
        return {str(key): _public_runtime_value(item) for key, item in value.items() if str(key).casefold() not in private_keys and (not str(key).casefold().startswith(('private_', 'privileged')))}
    if isinstance(value, (list, tuple)):
        return [_public_runtime_value(item) for item in value]
    return value

def _visual_input_quality(rgb: Any) -> dict[str, Any]:
    try:
        array = np.asarray(rgb)
    except Exception as exc:
        return {'usable': False, 'reason': 'array_conversion_failed', 'error': repr(exc)}
    payload: dict[str, Any] = {'shape': list(getattr(array, 'shape', ())), 'dtype': str(getattr(array, 'dtype', ''))}
    if array.ndim != 3 or array.shape[-1] < 3:
        payload.update({'usable': False, 'reason': 'expected_hwc_rgb'})
        return payload
    height, width = (int(array.shape[0]), int(array.shape[1]))
    if height < 4 or width < 4:
        payload.update({'usable': False, 'reason': 'image_too_small'})
        return payload
    rgb3 = array[:, :, :3].astype(np.float32, copy=False)
    if not np.all(np.isfinite(rgb3)):
        payload.update({'usable': False, 'reason': 'non_finite_pixels'})
        return payload
    luminance = 0.2126 * rgb3[:, :, 0] + 0.7152 * rgb3[:, :, 1] + 0.0722 * rgb3[:, :, 2]
    dynamic_range = float(np.max(rgb3) - np.min(rgb3))
    luminance_std = float(np.std(luminance))
    gradient = float(np.mean(np.abs(np.diff(luminance, axis=0)))) + float(np.mean(np.abs(np.diff(luminance, axis=1))))
    payload.update({'mean': float(np.mean(rgb3)), 'std': float(np.std(rgb3)), 'min': float(np.min(rgb3)), 'max': float(np.max(rgb3)), 'dynamic_range': dynamic_range, 'luminance_std': luminance_std, 'gradient': gradient})
    if dynamic_range < 2.0 and luminance_std < 1.0:
        payload.update({'usable': False, 'reason': 'blank_or_constant_rgb'})
        return payload
    if float(np.mean(rgb3)) <= 1.0 and luminance_std < 2.0:
        payload.update({'usable': False, 'reason': 'near_black_rgb'})
        return payload
    if float(np.mean(rgb3)) >= 254.0 and luminance_std < 2.0:
        payload.update({'usable': False, 'reason': 'near_white_rgb'})
        return payload
    payload.update({'usable': True, 'reason': 'ok'})
    if luminance_std < 3.0 and gradient < 1.0:
        payload['warning'] = 'low_information_rgb'
    return payload
__all__ = ['ModelCallContext', 'VLMBrain']
