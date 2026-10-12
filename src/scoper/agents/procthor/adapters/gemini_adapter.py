from __future__ import annotations
import base64
import os
from typing import Mapping, Sequence
from scoper.agents.procthor.adapters.base import BaseAdapter, CallMetadata, NonRetryableVLMError, redact_semantic_request
from scoper.agents.procthor.config import AdapterConfig, RetryConfig
from scoper.agents.procthor.image_encoder import ImageEncoder
from scoper.agents.procthor.message import Message

class GeminiAdapter(BaseAdapter):

    def __init__(self, config: AdapterConfig, retry: RetryConfig, image_encoder: ImageEncoder) -> None:
        super().__init__(config, retry, image_encoder)
        try:
            from google import genai
        except ImportError as exc:
            raise NonRetryableVLMError("Missing optional dependency 'google-genai'. Install scoper-supplement[api].") from exc
        api_key = os.environ.get(config.api_key_env, '')
        if not api_key:
            raise NonRetryableVLMError("Gemini API key env var '{}' is not set.".format(config.api_key_env))
        self.client = genai.Client(api_key=api_key)

    def _call_once(self, messages: Sequence[Message]) -> str:
        try:
            from google.genai import types
        except ImportError as exc:
            raise NonRetryableVLMError("Missing optional dependency 'google-genai'. Install scoper-supplement[api].") from exc
        system_text, contents = self._convert_messages(messages, types)
        config_kwargs = {'system_instruction': system_text or None, 'max_output_tokens': self.config.max_tokens, 'temperature': self.config.temperature}
        thinking_config = self._thinking_config(types)
        if thinking_config is not None:
            config_kwargs['thinking_config'] = thinking_config
        generation_params = self.config.extra.get('generation_params') or {}
        if isinstance(generation_params, dict):
            config_kwargs.update(generation_params)
        cfg = types.GenerateContentConfig(**config_kwargs)
        response = self.client.models.generate_content(model=self.config.model, contents=contents, config=cfg)
        usage = getattr(response, 'usage_metadata', None)
        self.last_call_metadata = CallMetadata(elapsed_seconds=0.0, prompt_tokens=getattr(usage, 'prompt_token_count', None), completion_tokens=getattr(usage, 'candidates_token_count', None), finish_reason=_gemini_finish_reason(response))
        return _gemini_response_text(response)

    def _thinking_config(self, types):
        extra = self.config.extra
        if 'thinking_budget' not in extra and 'include_thoughts' not in extra:
            return None
        thinking_cls = getattr(types, 'ThinkingConfig', None)
        if thinking_cls is None:
            return None
        kwargs = {}
        if 'thinking_budget' in extra:
            kwargs['thinking_budget'] = int(extra['thinking_budget'])
        if 'include_thoughts' in extra:
            kwargs['include_thoughts'] = bool(extra['include_thoughts'])
        try:
            return thinking_cls(**kwargs)
        except TypeError:
            return None

    def _convert_messages(self, messages: Sequence[Message], types):
        system_text = ''
        contents = []
        for message in messages:
            if message['role'] == 'system':
                system_text += self._render_text(message)
                continue
            parts = []
            for block in message.get('content', []):
                if block.get('type') == 'text':
                    parts.append(types.Part.from_text(text=str(block.get('text', ''))))
                elif block.get('type') == 'image':
                    b64, mime = self.image_encoder.to_base64(block['rgb'])
                    parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime))
            contents.append(types.Content(role='model' if message['role'] == 'assistant' else 'user', parts=parts))
        return (system_text, contents)

    def _render_text(self, message: Message) -> str:
        return '\n'.join((str(block.get('text', '')) for block in message.get('content', []) if block.get('type') == 'text'))

    def semantic_request(self, messages):
        system_text = ''
        contents = []
        for message in messages:
            if message.get('role') == 'system':
                system_text += '\n'.join((str(block.get('text', '')) for block in message.get('content', []) if block.get('type') == 'text'))
                continue
            contents.append({'role': 'model' if message.get('role') == 'assistant' else 'user', 'parts': [{'type': 'text', 'text': str(block.get('text', ''))} if block.get('type') == 'text' else {'type': 'image', 'frame_id': block['frame_id'], 'mime_type': block['mime_type'], 'sha256': block['sha256']} for block in message.get('content', [])]})
        config = {'system_instruction': system_text or None, 'max_output_tokens': self.config.max_tokens, 'temperature': self.config.temperature}
        if 'thinking_budget' in self.config.extra or 'include_thoughts' in self.config.extra:
            config['thinking_config'] = {key: self.config.extra[key] for key in ('thinking_budget', 'include_thoughts') if key in self.config.extra}
        generation_params = self.config.extra.get('generation_params') or {}
        if isinstance(generation_params, Mapping):
            config.update(dict(generation_params))
        return {'messages': contents, 'generation_config': redact_semantic_request(config)}

def _gemini_finish_reason(response) -> str | None:
    candidates = getattr(response, 'candidates', None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], 'finish_reason', None)
    return str(reason) if reason is not None else None

def _gemini_response_text(response) -> str:
    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass
    candidates = getattr(response, 'candidates', None) or []
    chunks = []
    for candidate in candidates:
        content = getattr(candidate, 'content', None)
        parts = getattr(content, 'parts', None) or []
        for part in parts:
            if getattr(part, 'thought', False):
                continue
            part_text = getattr(part, 'text', None)
            if part_text:
                chunks.append(part_text)
    return ''.join(chunks)
__all__ = ['GeminiAdapter']
