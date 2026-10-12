from __future__ import annotations
import os
from typing import Mapping, Sequence
from scoper.agents.procthor.adapters.base import BaseAdapter, CallMetadata, NonRetryableVLMError, redact_semantic_request
from scoper.agents.procthor.config import AdapterConfig, RetryConfig
from scoper.agents.procthor.image_encoder import ImageEncoder
from scoper.agents.procthor.message import Message

class AnthropicAdapter(BaseAdapter):

    def __init__(self, config: AdapterConfig, retry: RetryConfig, image_encoder: ImageEncoder) -> None:
        super().__init__(config, retry, image_encoder)
        try:
            import anthropic
        except ImportError as exc:
            raise NonRetryableVLMError("Missing optional dependency 'anthropic'. Install scoper-supplement[api].") from exc
        api_key = os.environ.get(config.api_key_env, '')
        if not api_key:
            raise NonRetryableVLMError("Anthropic API key env var '{}' is not set.".format(config.api_key_env))
        self.client = anthropic.Anthropic(api_key=api_key, timeout=retry.timeout_seconds)

    def _call_once(self, messages: Sequence[Message]) -> str:
        system_text, api_messages = self._convert_messages(messages)
        kwargs = {'model': self.config.model, 'system': system_text, 'messages': api_messages, 'max_tokens': self.config.max_tokens, 'temperature': self.config.temperature}
        request_params = self.config.extra.get('request_params') or {}
        if isinstance(request_params, dict):
            kwargs.update(request_params)
        if 'thinking' in kwargs and isinstance(kwargs['thinking'], dict):
            if kwargs['thinking'].get('type') == 'enabled':
                kwargs['temperature'] = 1.0
        response = self.client.messages.create(**kwargs)
        usage = getattr(response, 'usage', None)
        self.last_call_metadata = CallMetadata(elapsed_seconds=0.0, prompt_tokens=getattr(usage, 'input_tokens', None), completion_tokens=getattr(usage, 'output_tokens', None), finish_reason=getattr(response, 'stop_reason', None))
        for block in getattr(response, 'content', []) or []:
            if getattr(block, 'type', None) == 'text':
                return getattr(block, 'text', '') or ''
        return ''

    def _convert_messages(self, messages: Sequence[Message]):
        system_text = ''
        api_messages = []
        for message in messages:
            if message['role'] == 'system':
                system_text += self._render_text(message)
                continue
            content = []
            for block in message.get('content', []):
                if block.get('type') == 'text':
                    content.append({'type': 'text', 'text': str(block.get('text', ''))})
                elif block.get('type') == 'image':
                    b64, mime = self.image_encoder.to_base64(block['rgb'])
                    content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': b64}})
            api_messages.append({'role': message['role'], 'content': content})
        return (system_text, api_messages)

    def _render_text(self, message: Message) -> str:
        return '\n'.join((str(block.get('text', '')) for block in message.get('content', []) if block.get('type') == 'text'))

    def semantic_request(self, messages):
        system_text = ''
        api_messages = []
        for message in messages:
            if message.get('role') == 'system':
                system_text += '\n'.join((str(block.get('text', '')) for block in message.get('content', []) if block.get('type') == 'text'))
                continue
            api_messages.append({'role': message.get('role'), 'content': [{'type': 'text', 'text': str(block.get('text', ''))} if block.get('type') == 'text' else {'type': 'image', 'source': {'type': 'frame_ref', 'frame_id': block['frame_id'], 'media_type': block['mime_type'], 'sha256': block['sha256']}} for block in message.get('content', [])]})
        config = {'system': system_text, 'max_tokens': self.config.max_tokens, 'temperature': self.config.temperature}
        request_params = self.config.extra.get('request_params') or {}
        if isinstance(request_params, Mapping):
            config.update(dict(request_params))
        if isinstance(config.get('thinking'), Mapping) and config['thinking'].get('type') == 'enabled':
            config['temperature'] = 1.0
        return {'messages': api_messages, 'generation_config': redact_semantic_request(config)}
__all__ = ['AnthropicAdapter']
