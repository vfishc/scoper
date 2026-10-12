from __future__ import annotations
import importlib.util
import os
from typing import List
from scoper.agents.behavior.config import VLMConfig

class VLMRuntimeError(RuntimeError):
    pass

def missing_runtime_requirements(config: VLMConfig) -> List[str]:
    adapter = config.adapter
    adapter_type = adapter.type.lower()
    missing: List[str] = []
    if adapter_type == 'openai':
        _require_module('openai', 'pip install scoper-supplement[api]', missing)
        if adapter.api_key_env and (not adapter.base_url) and (not os.environ.get(adapter.api_key_env)):
            missing.append('environment variable {} is required for OpenAI API access'.format(adapter.api_key_env))
    elif adapter_type == 'gemini':
        _require_module('google.genai', 'pip install scoper-supplement[api]', missing)
        if not adapter.api_key_env:
            missing.append('adapter.api_key_env must be set for Gemini')
        elif not os.environ.get(adapter.api_key_env):
            missing.append('environment variable {} is required for Gemini API access'.format(adapter.api_key_env))
    elif adapter_type == 'anthropic':
        _require_module('anthropic', 'pip install scoper-supplement[api]', missing)
        if not adapter.api_key_env:
            missing.append('adapter.api_key_env must be set for Anthropic')
        elif not os.environ.get(adapter.api_key_env):
            missing.append('environment variable {} is required for Anthropic API access'.format(adapter.api_key_env))
    elif adapter_type == 'hf_local':
        _require_module('torch', 'pip install scoper-supplement[local]', missing)
        _require_module('transformers', 'pip install scoper-supplement[local]', missing)
        _require_module('accelerate', 'pip install scoper-supplement[local]', missing)
    return missing

def assert_runtime_requirements(config: VLMConfig) -> None:
    missing = missing_runtime_requirements(config)
    if missing:
        raise VLMRuntimeError('VLM runtime requirements are missing:\n- {}'.format('\n- '.join(missing)))

def _require_module(module_name: str, install_hint: str, missing: List[str]) -> None:
    if not _module_available(module_name):
        missing.append("Python module '{}' is missing ({})".format(module_name, install_hint))

def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False
__all__ = ['VLMRuntimeError', 'assert_runtime_requirements', 'missing_runtime_requirements']
