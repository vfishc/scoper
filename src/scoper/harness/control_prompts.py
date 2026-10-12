import copy
from scoper.harness.compatibility import MEMORY_SYSTEM_V14_BYTES, POLICY_SYSTEM_V14_BYTES, QWEN35_MEMORY_V2_PROMPT_SUFFIX_BYTES, QWEN35_POLICY_D4_GUIDANCE_SUFFIX_BYTES, VERIFIER_SYSTEM_V13_BYTES
from scoper.harness.vocabulary import PUBLIC_VOCABULARY_PROMPT_BYTES, resolve_public_family
POLICY_SYSTEM_BYTES = POLICY_SYSTEM_V14_BYTES.replace(b'{"arguments":{"num_views":4},"tool_name":"scan"}', b'{"arguments":{},"tool_name":"scan"}') + b'Before every interaction action (pick_up, put_down, open, close, toggle_on, toggle_off, clean, slice, fill, saturate, heat, or cool), first use navigate_to the concrete target and wait for navigation to reach it; a scan or look_for does not satisfy this interaction-range requirement. A put_down is legal only while the intended object is visibly held. If put_down fails or the object is no longer held, stop retrying put_down: reacquire the object with navigate_to then pick_up, verify that it is held, and only then retry put_down. Never stop while this recovery obligation is open.\nScan accepts an empty arguments object and supplies at most eight views. Use concrete vocabulary names; plate stays plate, never dishware. Remembered visibility is historical, not current interaction readiness. Missing held evidence means unknown, not proof of an empty hand. Recovery exhaustion is not visual success.\n' + PUBLIC_VOCABULARY_PROMPT_BYTES
MEMORY_SYSTEM_BYTES = MEMORY_SYSTEM_V14_BYTES.replace(b'Return at most eight facts.', b'Return at most four facts.').replace(b'Copy public references exactly from supplied candidates.', b'In update, reconcile and route, copy public references exactly from supplied candidates.').replace(b'Use only supplied public candidates.', b'In update, reconcile and route, use only supplied public candidates.').replace(b'Copy public_entity_ref values, public_roles items, relevant_memory_ids items, and per_fact_status fact_id values exactly from the corresponding candidate arrays in the request; never invent or paraphrase a candidate.', b'In update, reconcile and route, copy public_entity_ref values, public_roles items, relevant_memory_ids items, and per_fact_status fact_id values exactly from the corresponding candidate arrays in the request; never invent or paraphrase a candidate.').replace(b'a category-level public_entity_ref', b'a concrete public-vocabulary object name as public_entity_ref') + b"For observe, public_entity_ref must name a concrete visible object from the public vocabulary. The global vocabulary does not assert scene presence. Keep plate as plate and mug as mug; do not promote them to dishware or drinking vessel. Keep drawer as drawer, not storage container. Prioritize task-relevant objects on the floor, possibly dropped objects, visibly open drawers, cabinets or fridges, and faucets with visible flowing water. An open container is not necessarily abnormal; follow the instruction. Also retain the task's visible objects and destination containers. Describe a visible location, relation or state in at most 160 characters. Do not infer water flow from faucet presence. Background furniture presence must not displace actionable task observations. Avoid duplicate facts. Observation never proves action success or current interaction readiness. Observe uses the vocabulary and RGB, not the update-mode candidate whitelist.\nEmit compact JSON without indentation or line breaks. The memory response has a 512-token output budget; prefer fewer facts to an incomplete JSON object.\n" + PUBLIC_VOCABULARY_PROMPT_BYTES
VERIFIER_SYSTEM_BYTES = VERIFIER_SYSTEM_V13_BYTES
D6_POLICY_SYSTEM_BYTES = POLICY_SYSTEM_BYTES + QWEN35_POLICY_D4_GUIDANCE_SUFFIX_BYTES
D6_MEMORY_SYSTEM_BYTES = MEMORY_SYSTEM_BYTES + QWEN35_MEMORY_V2_PROMPT_SUFFIX_BYTES

def component_generation_budget(harness_version, component, configured_budget):
    if str(harness_version) in {'18', '19', '20'} and component in {'memory', 'memory_proposer', 'u_omega'}:
        return 512
    return configured_budget

def d6_policy_adapter_descriptor(harness_version='18'):
    if str(harness_version) in {'19', '20'}:
        from scoper.harness import recovery_prompts
        if str(harness_version) == '20':
            from scoper.harness import visual_prompts as recovery_prompts
        from scoper.harness.canonical import sha256_bytes
        descriptor = dict(d6_policy_adapter_descriptor('18'))
        descriptor.update(schema_version='qwen35-policy-prompt-adapter.d6.v2', adapter_id='qwen3.5-policy-d6.v2', base_prompt_sha256=sha256_bytes(recovery_prompts.POLICY_SYSTEM_BYTES), effective_prompt_sha256=sha256_bytes(recovery_prompts.D6_POLICY_SYSTEM_BYTES), suffix_sha256=sha256_bytes(recovery_prompts.D6_POLICY_SYSTEM_BYTES[len(recovery_prompts.POLICY_SYSTEM_BYTES):]))
        return descriptor
    if str(harness_version) not in {'18', '19', '20'}:
        raise ValueError('D6 requires Harness v18 or newer compatible version')
    from scoper.harness.canonical import sha256_bytes
    from scoper.harness.prompt_contracts import qwen35_policy_semantic_retry_v12_prompt_adapter_descriptor
    descriptor = dict(qwen35_policy_semantic_retry_v12_prompt_adapter_descriptor('14'))
    descriptor.update(schema_version='qwen35-policy-prompt-adapter.d6.v1', adapter_id='qwen3.5-policy-d6.v1', base_prompt_sha256=sha256_bytes(POLICY_SYSTEM_BYTES), effective_prompt_sha256=sha256_bytes(D6_POLICY_SYSTEM_BYTES))
    return descriptor

def bound_observation_proposal(proposal):
    result = copy.deepcopy(dict(proposal))
    if result.get('mode') != 'observe':
        return result
    mutations = []
    seen = set()
    raw_mutations = result.get('fact_mutations', [])
    if not isinstance(raw_mutations, list):
        return result
    for mutation in raw_mutations:
        if not isinstance(mutation, dict):
            continue
        family = resolve_public_family(mutation.get('public_entity_ref'))
        description = mutation.get('object_public_role_or_value')
        if family is None or not isinstance(description, str) or (not 0 < len(description) <= 160):
            continue
        if mutation.get('relation_or_property') != 'scene_observation':
            continue
        reference = ' '.join(mutation['public_entity_ref'].strip().lower().split())
        if reference in {'drinking vessel', 'drinking_vessel'}:
            continue
        mutation['public_entity_ref'] = reference
        key = (reference, description.casefold().strip())
        if key not in seen:
            seen.add(key)
            mutations.append(mutation)
    mutations.sort(key=lambda item: not any((cue in item['object_public_role_or_value'].casefold() for cue in ('floor', 'open', 'flow', 'running', 'dropped'))))
    result['fact_mutations'] = mutations[:4]
    return result
