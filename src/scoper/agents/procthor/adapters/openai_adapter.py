from __future__ import annotations
import os
from typing import Any, Dict, Mapping, Sequence
from scoper.agents.procthor.adapters.base import BaseAdapter, CallMetadata, NonRetryableVLMError, redact_semantic_request
from scoper.agents.procthor.config import AdapterConfig, RetryConfig
from scoper.agents.procthor.image_encoder import ImageEncoder
from scoper.agents.procthor.message import Message

def _is_reasoning_model(model: str) -> bool:
    name = model.lower().replace('_', '-')
    if name.startswith(('o1', 'o3', 'o4')):
        return True
    if name.startswith('gpt-5') and (not name.startswith('gpt-5-chat')):
        return True
    return False

def _uses_completion_tokens(model: str) -> bool:
    name = model.lower().replace('_', '-')
    return name.startswith(('o1', 'o3', 'o4', 'gpt-5'))

class OpenAIAdapter(BaseAdapter):

    def __init__(self, config: AdapterConfig, retry: RetryConfig, image_encoder: ImageEncoder) -> None:
        super().__init__(config, retry, image_encoder)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise NonRetryableVLMError("Missing optional dependency 'openai'. Install scoper-supplement[api].") from exc
        extra = config.extra
        if 'reasoning_model' in extra:
            self._reasoning = bool(extra['reasoning_model'])
        else:
            self._reasoning = _is_reasoning_model(config.model)
        if 'use_max_completion_tokens' in extra:
            self._completion_tokens = bool(extra['use_max_completion_tokens'])
        else:
            self._completion_tokens = self._reasoning or _uses_completion_tokens(config.model)
        self._system_role = 'developer' if self._reasoning else 'system'
        self.client = OpenAI(max_retries=0, api_key=os.environ.get(config.api_key_env, 'EMPTY'), base_url=config.base_url or None, timeout=retry.timeout_seconds)
        self.transport_retry_budget_seconds = float(config.extra.get('transport_retry_budget_seconds', 240))

    def _call_once(self, messages: Sequence[Message]) -> str:
        budget = getattr(self, '_transport_retry_budget', None)
        client = self.client.with_options(timeout=min(self.retry.timeout_seconds, budget.remaining()), max_retries=0) if budget is not None else self.client
        response = client.chat.completions.create(**self._build_request_kwargs(messages))
        choice = response.choices[0]
        usage = getattr(response, 'usage', None)
        self.last_call_metadata = CallMetadata(elapsed_seconds=0.0, prompt_tokens=getattr(usage, 'prompt_tokens', None), completion_tokens=getattr(usage, 'completion_tokens', None), finish_reason=getattr(choice, 'finish_reason', None))
        return choice.message.content or ''

    def _build_request_kwargs(self, messages: Sequence[Message]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {'model': self.config.model, 'messages': [self._convert_message(message) for message in messages]}
        token_param = 'max_completion_tokens' if self._completion_tokens else 'max_tokens'
        kwargs[token_param] = self.config.max_tokens
        if not self._reasoning or self.config.extra.get('send_temperature'):
            kwargs['temperature'] = self.config.temperature
        request_params = self.config.extra.get('request_params') or {}
        if isinstance(request_params, Mapping):
            kwargs.update(dict(request_params))
        return kwargs

    def _convert_message(self, message: Message) -> Mapping[str, Any]:
        role = message['role']
        if role == 'system':
            role = self._system_role
        content = []
        for block in message.get('content', []):
            if block.get('type') == 'text':
                content.append({'type': 'text', 'text': str(block.get('text', ''))})
            elif block.get('type') == 'image':
                b64, mime = self.image_encoder.to_base64(block['rgb'])
                content.append({'type': 'image_url', 'image_url': {'url': 'data:{};base64,{}'.format(mime, b64), 'detail': self.config.extra.get('image_detail', 'high')}})
        if all((item.get('type') == 'text' for item in content)):
            text = '\n'.join((str(item.get('text', '')) for item in content))
            return {'role': role, 'content': text}
        return {'role': role, 'content': content}

    def semantic_request(self, messages):
        converted = []
        for message in messages:
            role = self._system_role if message.get('role') == 'system' else message.get('role')
            content = []
            for block in message.get('content', []):
                if block.get('type') == 'text':
                    content.append({'type': 'text', 'text': str(block.get('text', ''))})
                elif block.get('type') == 'image':
                    content.append({'type': 'image_url', 'image_url': {'frame_id': block['frame_id'], 'mime_type': block['mime_type'], 'sha256': block['sha256'], 'detail': self.config.extra.get('image_detail', 'high')}})
            converted.append({'role': role, 'content': '\n'.join((item['text'] for item in content)) if all((item.get('type') == 'text' for item in content)) else content})
        kwargs = {'max_completion_tokens' if self._completion_tokens else 'max_tokens': self.config.max_tokens}
        if not self._reasoning or self.config.extra.get('send_temperature'):
            kwargs['temperature'] = self.config.temperature
        request_params = self.config.extra.get('request_params') or {}
        if isinstance(request_params, Mapping):
            kwargs.update(dict(request_params))
        return {'messages': converted, 'generation_config': redact_semantic_request(kwargs)}
__all__ = ['OpenAIAdapter']
