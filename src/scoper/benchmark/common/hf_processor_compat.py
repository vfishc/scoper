import json
from pathlib import Path

def prepare_hf_messages(messages, *, merge_system=False, error_type=ValueError):
    prepared = [{'role': m['role'], 'content': list(m.get('content') or [])} for m in messages]
    if not merge_system:
        return prepared
    systems = []
    while prepared and prepared[0]['role'] == 'system':
        systems.append(prepared.pop(0))
    if not systems:
        return prepared
    if not prepared or prepared[0]['role'] != 'user':
        raise error_type('merge_system_into_first_user requires a leading user message.')
    text = '\n\n'.join((str(b.get('text', '')) for m in systems for b in m['content'] if b.get('type') == 'text'))
    prepared[0]['content'] = [{'type': 'text', 'text': text + '\n\n'}, *prepared[0]['content']]
    return prepared

def legacy_default_rope(config, device=None, seq_len=None):
    import torch
    base = config.rope_theta
    dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    dim = int(dim * getattr(config, 'partial_rotary_factor', 1.0))
    return (1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim), 1.0)

def prepare_molmo2_model(model):
    import sys
    import functools
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    original = get_class_from_dynamic_module('modeling_molmo2.Molmo2ForConditionalGeneration', str(Path(model).resolve(strict=True)), local_files_only=True)
    module = sys.modules[original.__module__]
    if 'default' not in module.ROPE_INIT_FUNCTIONS:
        module.ROPE_INIT_FUNCTIONS = {**module.ROPE_INIT_FUNCTIONS, 'default': legacy_default_rope}
    if not getattr(module, '_molmo2_tf5_masks_compat', False):

        def rename_embeddings(function):

            @functools.wraps(function)
            def call(*args, **kwargs):
                if 'input_embeds' in kwargs:
                    kwargs['inputs_embeds'] = kwargs.pop('input_embeds')
                return function(*args, **kwargs)
            return call
        module.create_causal_mask = rename_embeddings(module.create_causal_mask)
        module.create_masks_for_generate = rename_embeddings(module.create_masks_for_generate)
        module._molmo2_tf5_masks_compat = True

    class CompatibleMolmo2Model(original):

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model = super().from_pretrained(*args, **kwargs)
            for item in model.modules():
                if isinstance(item, module.Molmo2RotaryEmbedding):
                    inv_freq, scaling = item.rope_init_fn(item.config, device=item.inv_freq.device)
                    item.register_buffer('inv_freq', inv_freq, persistent=False)
                    item.original_inv_freq = inv_freq
                    item.attention_scaling = scaling
            return model

        @functools.wraps(original.prepare_inputs_for_generation)
        def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
            if kwargs.get('cache_position') is None:
                import torch
                cache = kwargs.get('past_key_values')
                start = cache.get_seq_length() if cache is not None else 0
                length = kwargs.get('next_sequence_length')
                if length is None:
                    length = input_ids.shape[-1]
                kwargs['cache_position'] = torch.arange(start, start + length, device=input_ids.device)
            return super().prepare_inputs_for_generation(input_ids, *args, **kwargs)
    CompatibleMolmo2Model.register_for_auto_class('AutoModelForImageTextToText')
    return CompatibleMolmo2Model

def load_molmo2_processor(model, *, trust_remote_code):
    if not trust_remote_code:
        raise ValueError('Molmo2 processor compatibility requires trusted local checkpoint code')
    root = Path(model).resolve(strict=True)
    config = json.loads((root / 'processor_config.json').read_text())
    entry = (config.get('auto_map') or {}).get('AutoProcessor')
    if entry != 'processing_molmo2.Molmo2Processor':
        raise ValueError('Unexpected processor for Molmo2 compatibility')
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.processing_utils import ProcessorMixin
    original = get_class_from_dynamic_module(entry, str(root), local_files_only=True)

    class LegacyOptionalAttributes(ProcessorMixin):

        def __init__(self, *args, **kwargs):
            for key in self.optional_attributes:
                if key != 'chat_template' and key in kwargs:
                    setattr(self, key, kwargs.pop(key))
            super().__init__(*args, **kwargs)

    class CompatibleMolmo2Processor(original, LegacyOptionalAttributes):
        pass
    return CompatibleMolmo2Processor.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
