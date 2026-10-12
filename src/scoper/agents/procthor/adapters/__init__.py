from __future__ import annotations
from scoper.agents.procthor.config import VLMConfig
from scoper.agents.procthor.image_encoder import ImageEncoder

def create_adapter(config: VLMConfig, encoder: ImageEncoder):
    adapter_type = config.adapter.type.lower()
    if adapter_type == 'openai':
        from scoper.agents.procthor.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(config.adapter, config.retry, encoder)
    if adapter_type == 'gemini':
        from scoper.agents.procthor.adapters.gemini_adapter import GeminiAdapter
        return GeminiAdapter(config.adapter, config.retry, encoder)
    if adapter_type == 'anthropic':
        from scoper.agents.procthor.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(config.adapter, config.retry, encoder)
    if adapter_type == 'hf_local':
        from scoper.agents.procthor.adapters.hf_local_adapter import HFLocalAdapter
        return HFLocalAdapter(config.adapter, config.retry, encoder)
    raise ValueError("Unknown adapter type '{}'.".format(adapter_type))
__all__ = ['create_adapter']
