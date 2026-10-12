from collections.abc import Mapping

def sync(env, prompt_context):
    messages = prompt_context.get('human_messages') or ()
    public = tuple(({'type': 'insert', 'instruction': m['instruction']} for m in messages if isinstance(m, Mapping) and m.get('type') == 'insert' and isinstance(m.get('instruction'), str)))
    env._public_binding_human_messages = public
