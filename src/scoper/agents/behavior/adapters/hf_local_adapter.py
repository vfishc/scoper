from __future__ import annotations
from typing import Sequence
from scoper.agents.behavior.adapters.base import BaseAdapter, CallMetadata, NonRetryableVLMError
from scoper.agents.behavior.config import AdapterConfig, RetryConfig
from scoper.agents.behavior.image_encoder import ImageEncoder
from scoper.agents.behavior.message import Message

class HFLocalAdapter(BaseAdapter):

    def __init__(self, config: AdapterConfig, retry: RetryConfig, image_encoder: ImageEncoder) -> None:
        super().__init__(config, retry, image_encoder)
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise NonRetryableVLMError("Missing optional dependencies 'torch' and 'transformers'. Install scoper-supplement[local].") from exc
        self.torch = torch
        dtype_name = str(config.extra.get('torch_dtype', 'bfloat16'))
        self.dtype = getattr(torch, dtype_name)
        processor_compat = config.extra.get('processor_compat')
        if processor_compat == 'molmo2_transformers5':
            from scoper.benchmark.common.hf_processor_compat import load_molmo2_processor
            self.processor = load_molmo2_processor(config.model, trust_remote_code=bool(config.extra.get('trust_remote_code', True)))
        elif processor_compat:
            raise NonRetryableVLMError('Unsupported processor_compat: ' + str(processor_compat))
        else:
            self.processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=bool(config.extra.get('trust_remote_code', True)))
        chat_template_kwargs = config.extra.get('chat_template_kwargs') or {}
        if not isinstance(chat_template_kwargs, dict):
            raise NonRetryableVLMError('adapter.extra.chat_template_kwargs must be a mapping when provided.')
        self.chat_template_kwargs = dict(chat_template_kwargs)
        self.merge_system_into_first_user = bool(config.extra.get('merge_system_into_first_user', False))
        kwargs = {'torch_dtype': self.dtype, 'device_map': config.extra.get('device_map', 'auto'), 'trust_remote_code': bool(config.extra.get('trust_remote_code', True))}
        attn_impl = config.extra.get('attn_implementation')
        if attn_impl:
            kwargs['attn_implementation'] = attn_impl
        model_class = AutoModelForImageTextToText
        if processor_compat == 'molmo2_transformers5':
            from scoper.benchmark.common.hf_processor_compat import prepare_molmo2_model
            model_class = prepare_molmo2_model(config.model)
        self.model = model_class.from_pretrained(config.model, **kwargs).eval()

    def _call_once(self, messages: Sequence[Message]) -> str:
        from scoper.benchmark.common.hf_processor_compat import prepare_hf_messages
        hf_messages = prepare_hf_messages([self._convert_message(message) for message in messages], merge_system=getattr(self, 'merge_system_into_first_user', False), error_type=NonRetryableVLMError)
        inputs = self.processor.apply_chat_template(hf_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt', **self.chat_template_kwargs)
        inputs = self._move_inputs_to_model(inputs)
        generate_kwargs = {'max_new_tokens': self.config.max_tokens, 'do_sample': self.config.temperature > 0}
        if self.config.temperature > 0:
            generate_kwargs['temperature'] = self.config.temperature
        generation_params = self.config.extra.get('generation_params') or {}
        if isinstance(generation_params, dict):
            generate_kwargs.update(generation_params)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)
        prompt_len = inputs['input_ids'].shape[1]
        new_tokens = output[0, prompt_len:]
        self.last_call_metadata = CallMetadata(elapsed_seconds=0.0, prompt_tokens=int(prompt_len), completion_tokens=int(new_tokens.shape[0]), finish_reason=None)
        return self.processor.decode(new_tokens, skip_special_tokens=True)

    def close(self) -> None:
        try:
            del self.model
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def _convert_message(self, message: Message) -> dict:
        content = []
        for block in message.get('content', []):
            if block.get('type') == 'text':
                content.append({'type': 'text', 'text': str(block.get('text', ''))})
            elif block.get('type') == 'image':
                content.append({'type': 'image', 'image': self.image_encoder.block_to_pil(block)})
        return {'role': message['role'], 'content': content}

    def _move_inputs_to_model(self, inputs):
        device = getattr(self.model, 'device', None)
        if device is None:
            return inputs
        return {key: value.to(device) if hasattr(value, 'to') else value for key, value in inputs.items()}
__all__ = ['HFLocalAdapter']
