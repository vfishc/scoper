from __future__ import annotations
import copy
import re
from typing import Any, Mapping, Sequence
from scoper.harness.canonical import canonical_json_bytes, sha256_bytes, sha256_json, strict_json_loads, validate_json_value
from scoper.harness.exploration_guard import qwen_policy_verified_held_refs
from scoper.harness.vocabulary import PUBLIC_VOCABULARY_PROMPT_BYTES, SHARED_VOCABULARY_SHA256, SHARED_VOCABULARY_VERSION
LEGACY_HARDQA_PROMPT_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v3'
HARDQA_PROMPT_V3_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v4'
HARDQA_PROMPT_V4_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v5'
HARDQA_PROMPT_V5_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v6'
HARDQA_PROMPT_V9_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v7'
HARDQA_PROMPT_V10_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v8'
HARDQA_PROMPT_V11_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v9'
HARDQA_PROMPT_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v10'
HARDQA_PROMPT_V13_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v12'
HARDQA_PROMPT_V14_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v13'
HARDQA_PROMPT_V15_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v14'
HARDQA_PROMPT_V16_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v15'
HARDQA_PROMPT_V17_CONTRACT_VERSION = 'method-text-hardqa-prompt-contract.v16'
HARDQA_PROMPT_ENVELOPE_VERSION = 'method-text-hardqa-prompt.v2'
HARDQA_MESSAGE_LAYOUT = 'system-images-user-json-assistant.v1'
HARDQA_PROMPT_BUILDER = 'hardqa-prompt-builder.v3'
QWEN35_POLICY_PROMPT_ADAPTER_ID = 'qwen3.5-policy-format.v1'
QWEN35_POLICY_PROMPT_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v1'
QWEN35_POLICY_PROMPT_SUFFIX_BYTES = b'Do not use code fences. Emit compact JSON. The first output character must be { and the last output character must be }.\n'
QWEN35_POLICY_GUIDANCE_ADAPTER_ID = 'qwen3.5-policy-guidance.v2'
QWEN35_POLICY_GUIDANCE_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v2'
QWEN35_POLICY_CONTEXT_FOOTER_SCHEMA_VERSION = 'qwen3.5-policy-context-footer.v1'
QWEN35_POLICY_CONTEXT_FOOTER_PREFIX_BYTES = b'QWEN_DECISION_CHECK='
QWEN35_POLICY_CONTEXT_FOOTER_SERIALIZER_IDENTITY_BYTES = b'canonical-policy-context-newline-qwen-decision-check-canonical-json.v1'
QWEN35_POLICY_GUIDANCE_SUFFIX_BYTES = b'Do not use code fences. Emit compact JSON. The first output character must be { and the last output character must be }.\nIf a new decision is requested after stop, the episode is still active; do not repeat stop until public evidence supports every instruction clause.\nDo not stop while evidence_memory contains an open incident or pending obligation.\nFollow the public recovery field exactly: perform its required preparation or saved ToolCall before returning to the task.\nAfter not_interaction_ready or binding_failed, use an allowed observation or navigation action before manipulating again.\nDo not repeat the same failed ToolCall unless the current image, recovery phase, or public evidence changed.\n'
QWEN35_POLICY_D4_GUIDANCE_SUFFIX_BYTES = QWEN35_POLICY_GUIDANCE_SUFFIX_BYTES + b'For put_down, target the destination receptacle or surface named by the instruction, never the object being carried.\n'
QWEN35_MEMORY_PROMPT_ADAPTER_ID = 'qwen3.5-memory-mode-guidance.v1'
QWEN35_MEMORY_PROMPT_ADAPTER_SCHEMA_VERSION = 'qwen35-memory-prompt-adapter.v1'
QWEN35_MEMORY_PROMPT_SUFFIX_BYTES = b'For update, relevant_memory_ids and per_fact_status must be [].\nFor reconcile, relevant_memory_ids must be [] and per_fact_status must contain exactly one entry for every candidate fact ID.\nFor route, fact_mutations, incident_mutations, obligation_mutations, and per_fact_status must be [].\nUse only fields allowed by the requested mode.\n'
QWEN35_MEMORY_V2_PROMPT_ADAPTER_ID = 'qwen3.5-memory-mode-guidance.v2'
QWEN35_MEMORY_V2_PROMPT_ADAPTER_SCHEMA_VERSION = 'qwen35-memory-prompt-adapter.v2'
QWEN35_MEMORY_V2_PROMPT_SUFFIX_BYTES = QWEN35_MEMORY_PROMPT_SUFFIX_BYTES + b'For observe, use schema_version method-text-hardqa.v2; only scene_observation fact_mutations are allowed, all other collections must be [], and a canonical empty abstain result is valid.\n'
INSTRUCTION_GROUNDED_MEMORY_PROFILE_ID = 'instruction-grounded-memory.v1'
QWEN35_MEMORY_V3_PROMPT_ADAPTER_ID = 'qwen3.5-memory-mode-guidance.v3'
QWEN35_MEMORY_V3_PROMPT_ADAPTER_SCHEMA_VERSION = 'qwen35-memory-prompt-adapter.v3'
QWEN35_MEMORY_V3_PROMPT_SUFFIX_BYTES = QWEN35_MEMORY_V2_PROMPT_SUFFIX_BYTES + b'For observe, use public_instruction as the preferred entity vocabulary. When the instruction uses a broad task category, report that category instead of guessing a subtype: plate, bowl, cup, mug, or dish become dishware; an open drawer, cabinet, cupboard, fridge, or refrigerator becomes storage container; faucet, tap, water source, or running water becomes faucet. Include a visually supported task-relevant fact whenever one is present. Other visually supported facts may remain, but do not let them replace a task-relevant finding. Scene observations remain advisory.\n'
QWEN35_MEMORY_V4_PROMPT_ADAPTER_ID = 'qwen3.5-memory-mode-guidance.v4'
QWEN35_MEMORY_V4_PROMPT_ADAPTER_SCHEMA_VERSION = 'qwen35-memory-prompt-adapter.v4'
QWEN35_MEMORY_V4_PROMPT_SUFFIX_BYTES = QWEN35_MEMORY_V2_PROMPT_SUFFIX_BYTES + b'For update, reconcile, and route, use schema_version method-text-hardqa.v1; never use the prompt envelope version method-text-hardqa-prompt.v2. For observe, preserve each visually identifiable concrete public entity name. Broad instruction categories guide attention but must not replace concrete tool targets. Re-check every current RGB frame for each conditional state named by public_instruction before repeating prior memory. Return at most four observe facts, prioritizing instruction-relevant states over generic object presence. Visible flowing water should be reported as faucet appears to be running; visibly ajar storage should use its concrete name, such as drawer or cabinet, and appears open. Emit a canonical empty result only when the current RGB does not support a task-relevant finding. When the instruction requires placement, exploratory scan or look_for should report a visible concrete destination such as a shelf, drawer, or cabinet only as present in the scene or with a visible location cue; do not infer whether it is open, closed, or ready. Only after observation_tool_call_or_null is navigate_to for that destination or its storage query may current RGB support appears open or appears closed. If neither state is clear, keep only the presence or location observation. Do not report an unrelated surface as the destination. An instruction explicitly asking to inspect open storage may still report visibly open storage as its safety issue. For a targeted look_for, inspect the queried entity and named condition first; if unsupported, abstain instead of filling the fact budget with unrelated presence facts. For running-water checks, a visible light-blue or cyan vertical stream beneath a tap is flowing water: report faucet appears to be running, not merely sink present.\n'
QWEN35_MEMORY_V5_PROMPT_ADAPTER_ID = 'qwen3.5-memory-mode-guidance.v5'
QWEN35_MEMORY_V5_PROMPT_ADAPTER_SCHEMA_VERSION = 'qwen35-memory-prompt-adapter.v5'
QWEN35_MEMORY_V5_PROMPT_SUFFIX_BYTES = QWEN35_MEMORY_V4_PROMPT_SUFFIX_BYTES + b'Exclude wall, floor, picture, picture frame, and door unless directly queried. Prefer on-floor, on-top, inside, outside-storage, open, closed, running, or off when visibly supported. Return at most four task-relevant facts.\n'
QWEN35_POLICY_EXPERIMENT_ADAPTER_ID = 'qwen3.5-policy-public-failure-guidance.v1'
QWEN35_POLICY_EXPERIMENT_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v3'
QWEN35_POLICY_EXPERIMENT_CONTEXT_FOOTER_SCHEMA_VERSION = 'qwen3.5-policy-context-footer.v2'
QWEN35_POLICY_EXPERIMENT_PROFILES = ('a_stop_rejection', 'b_placement_target', 'c_binding_disambiguation')
QWEN35_POLICY_EXPERIMENT_CONTEXT_FOOTER_SERIALIZER_IDENTITY_BYTES = b'canonical-policy-context-newline-qwen-decision-check-public-failure-v2;profiles=a_stop_rejection,b_placement_target,c_binding_disambiguation;public-fields=tool-call,stop-missing-tools,binding-admission.v1'
QWEN35_POLICY_SEMANTIC_RETRY_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v1'
QWEN35_POLICY_SEMANTIC_RETRY_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v4'
QWEN35_POLICY_SEMANTIC_RETRY_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v1'
QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS = 2
QWEN35_POLICY_SEMANTIC_RETRY_DETECTOR_IDENTITY_BYTES = b'qwen35-public-semantic-retry-detector.v1;rules=unique-held-put-down-target,singleton-recovery-enum,rejected-stop;public-fields=instruction,allowed-tool-specs,public-observation,effective-facts,projected-transition-feedback.v1'
QWEN35_POLICY_SEMANTIC_RETRY_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v1;fields=attempt,guidance,held-public-refs,instruction,max-attempts,rejected-reason,rejected-tool-call,required-tool-call-or-null,schema-version,stop-rejected-missing-tools.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V2_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v2'
QWEN35_POLICY_SEMANTIC_RETRY_V2_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v5'
QWEN35_POLICY_SEMANTIC_RETRY_V2_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v2'
QWEN35_POLICY_SEMANTIC_RETRY_V2_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_DETECTOR_IDENTITY_BYTES + b';combined-rule=missing-put-down-plus-unique-held-target.v2'
QWEN35_POLICY_SEMANTIC_RETRY_V2_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v2;fields=attempt,guidance,held-public-refs,instruction,max-attempts,rejected-reason,rejected-tool-call,required-tool-call-or-null,schema-version,stop-rejected-missing-tools;combined-guidance=missing-put-down-plus-unique-held-target.v2'
QWEN35_POLICY_SEMANTIC_RETRY_V3_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v3'
QWEN35_POLICY_SEMANTIC_RETRY_V3_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v6'
QWEN35_POLICY_SEMANTIC_RETRY_V3_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v3'
QWEN35_POLICY_SEMANTIC_RETRY_V3_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V2_DETECTOR_IDENTITY_BYTES + b';rule=put-down-requires-unique-verified-effective-fact-held.v3'
QWEN35_POLICY_SEMANTIC_RETRY_V3_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v3;fields=attempt,guidance,held-public-refs,instruction,max-attempts,rejected-reason,rejected-tool-call,required-tool-call-or-null,schema-version,stop-rejected-missing-tools;held-source=unique-verified-effective-facts-only.v1;second-violation=fail-open-to-harness.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V4_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v4'
QWEN35_POLICY_SEMANTIC_RETRY_V4_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v7'
QWEN35_POLICY_SEMANTIC_RETRY_V4_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v4'
QWEN35_POLICY_SEMANTIC_RETRY_V4_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V3_DETECTOR_IDENTITY_BYTES + b';same-public-held-detector-acquisition-first-reask.v4'
QWEN35_POLICY_SEMANTIC_RETRY_V4_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v4;fields=attempt,guidance,held-public-refs,instruction,max-attempts,rejected-reason,rejected-tool-call,required-tool-call-or-null,schema-version,stop-rejected-missing-tools;held-source=unique-verified-effective-facts-only.v1;no-held-guidance=soft-acquisition-first-no-put-down-or-stop.v1;second-violation=fail-open-to-harness.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V5_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v5'
QWEN35_POLICY_SEMANTIC_RETRY_V5_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v8'
QWEN35_POLICY_SEMANTIC_RETRY_V5_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v5'
QWEN35_POLICY_SEMANTIC_RETRY_V5_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V4_DETECTOR_IDENTITY_BYTES + b';same-public-held-detector-explicit-alternatives.v5'
QWEN35_POLICY_SEMANTIC_RETRY_V5_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v5;fields=allowed-alternative-tool-names,attempt,guidance,held-public-refs,instruction,max-attempts,rejected-reason,rejected-tool-call,required-tool-call-or-null,schema-version,stop-rejected-missing-tools;alternatives=public-allowed-minus-put-down-stop.v1;second-violation=fail-open-to-harness.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V6_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v6'
QWEN35_POLICY_SEMANTIC_RETRY_V6_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v9'
QWEN35_POLICY_SEMANTIC_RETRY_V6_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v6'
QWEN35_POLICY_SEMANTIC_RETRY_V6_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V5_DETECTOR_IDENTITY_BYTES + b';fresh-public-context-reask.v6'
QWEN35_POLICY_SEMANTIC_RETRY_V6_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v6;payload=v5-plus-fresh-reask;retry-context=public-allowed-specs-filtered-to-explicit-alternatives.v1;assistant-history=omitted;second-violation=fail-open-to-harness.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V6_PREFIX_BYTES = b'QWEN_SEMANTIC_RETRY='
QWEN35_POLICY_SEMANTIC_RETRY_V7_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v7'
QWEN35_POLICY_SEMANTIC_RETRY_V7_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v10'
QWEN35_POLICY_SEMANTIC_RETRY_V7_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v7'
QWEN35_POLICY_ACTION_PLAYBOOK_SCHEMA_VERSION = 'qwen3.5-policy-action-playbook.v1'
QWEN35_POLICY_ACTION_PLAYBOOK_PREFIX_BYTES = b'QWEN_ACTION_PLAYBOOK='
QWEN35_POLICY_ACTION_PLAYBOOK_SERIALIZER_IDENTITY_BYTES = b'canonical-qwen-action-playbook.v1;public-fields=mode,allowed-tools,effective-held-facts,assumed-completed-tool-calls,projected-binding-failure;phases=recover,observe,navigate-before-retry,place,acquire,continue;no-instruction-target-parser,no-hidden-state,no-action-rewrite.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V7_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V6_DETECTOR_IDENTITY_BYTES + b';assumed-pick-up-is-placement-planning-evidence.v7'
QWEN35_POLICY_SEMANTIC_RETRY_V7_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v7;payload=v6-with-assumed-completion-reconciliation;initial-and-retry-context=public-action-playbook.v1;assistant-history=omitted;second-violation=fail-open-to-harness.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V8_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v8'
QWEN35_POLICY_SEMANTIC_RETRY_V8_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v11'
QWEN35_POLICY_SEMANTIC_RETRY_V8_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v8'
QWEN35_POLICY_SEMANTIC_RETRY_V8_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V7_DETECTOR_IDENTITY_BYTES + b';harness-locked-recovery-bypass.v8'
QWEN35_POLICY_SEMANTIC_RETRY_V8_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v8;payload=v7-outside-recovery;harness-locked-recovery=no-reask;guard-composition=preserve-current-public-allowed-spec-intersection.v1;one-semantic-reask-total-per-decision.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V9_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v9'
QWEN35_POLICY_SEMANTIC_RETRY_V9_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v12'
QWEN35_POLICY_SEMANTIC_RETRY_V9_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v9'
QWEN35_POLICY_ACTION_PLAYBOOK_V2_SCHEMA_VERSION = 'qwen3.5-policy-action-playbook.v2'
QWEN35_POLICY_ACTION_PLAYBOOK_V2_SERIALIZER_IDENTITY_BYTES = QWEN35_POLICY_ACTION_PLAYBOOK_SERIALIZER_IDENTITY_BYTES + b';verified-release-after-chronological-pick-up-put-down=continue.v2'
QWEN35_POLICY_SEMANTIC_RETRY_V9_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V8_DETECTOR_IDENTITY_BYTES + b';verified-placement-candidate-retry.v9'
QWEN35_POLICY_SEMANTIC_RETRY_V9_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v9;payload=v8-plus-verified-placement-retry;public-state=chronological-tool-history-plus-verified-held-false;no-goal-state,no-simulator-state,no-action-rewrite.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V10_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v10'
QWEN35_POLICY_SEMANTIC_RETRY_V10_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v13'
QWEN35_POLICY_SEMANTIC_RETRY_V10_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v10'
QWEN35_POLICY_ACTION_PLAYBOOK_V3_SCHEMA_VERSION = 'qwen3.5-policy-action-playbook.activation-compatible.v3'
QWEN35_POLICY_ACTION_PLAYBOOK_V3_SERIALIZER_IDENTITY_BYTES = QWEN35_POLICY_ACTION_PLAYBOOK_V2_SERIALIZER_IDENTITY_BYTES + b';inactive-output-is-byte-identical-to-action-playbook.v1;verified-placement-only-activates-v2.v3'
QWEN35_POLICY_SEMANTIC_RETRY_V10_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V9_DETECTOR_IDENTITY_BYTES + b';inactive-detector-is-byte-identical-to-v8.v10'
QWEN35_POLICY_SEMANTIC_RETRY_V10_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v10;inactive-initial-and-retry-bytes=v8;verified-placement-only=playbook-v2-and-retry-v10;no-goal-state,no-simulator-state,no-action-rewrite.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V11_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v11'
QWEN35_POLICY_SEMANTIC_RETRY_V11_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v14'
QWEN35_POLICY_SEMANTIC_RETRY_V11_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v11'
QWEN35_POLICY_ACTION_HINT_SCHEMA_VERSION = 'qwen3.5-policy-action-hint.d5a.v1'
QWEN35_POLICY_ACTION_HINT_PREFIX_BYTES = b'QWEN_ACTION_HINT='
QWEN35_POLICY_ACTION_HINT_SERIALIZER_IDENTITY_BYTES = b'canonical-qwen-action-hint.d5a.v1;public-inputs=mode,allowed-tool-names,effective-facts,assumed-completed-actions,projected-binding-failure,tool-history;priority=not-ready,verified-placement,verified-held,no-held-acquisition;advisory-only,no-target-inference,no-hidden-state,no-goal-progress.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V11_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V2_DETECTOR_IDENTITY_BYTES + b';recovery-bypass;verified-placement-repeat-only.d5a.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V11_SERIALIZER_IDENTITY_BYTES = b'canonical-json-qwen35-policy-semantic-retry-message.v11;base=historical-semantic-retry-v2;recovery=no-reask;verified-placement-repeat=reask-with-public-alternatives;initial-context=historical-d2-footer-plus-compact-action-hint.d5a.v1'
QWEN35_POLICY_SEMANTIC_RETRY_V12_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v12'
QWEN35_POLICY_SEMANTIC_RETRY_V12_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v15'
QWEN35_POLICY_SEMANTIC_RETRY_V12_MESSAGE_SCHEMA_VERSION = QWEN35_POLICY_SEMANTIC_RETRY_V11_MESSAGE_SCHEMA_VERSION
QWEN35_POLICY_SEMANTIC_RETRY_V12_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V11_DETECTOR_IDENTITY_BYTES + b';harness-v14-scene-observations-remain-advisory.v12'
QWEN35_POLICY_SEMANTIC_RETRY_V12_SERIALIZER_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V11_SERIALIZER_IDENTITY_BYTES + b';base-prompt=harness-v14;behavior=d5a.v12'
QWEN35_POLICY_SEMANTIC_RETRY_V13_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v13'
QWEN35_POLICY_SEMANTIC_RETRY_V13_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v16'
QWEN35_POLICY_SEMANTIC_RETRY_V13_MESSAGE_SCHEMA_VERSION = 'qwen3.5-policy-semantic-retry-message.v12'
QWEN35_POLICY_SEMANTIC_RETRY_V13_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V12_DETECTOR_IDENTITY_BYTES + b';public-required-argument-presence.v13'
QWEN35_POLICY_SEMANTIC_RETRY_V13_SERIALIZER_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V12_SERIALIZER_IDENTITY_BYTES + b';missing-required-arguments=one-public-schema-reask.v13'
QWEN35_POLICY_SEMANTIC_RETRY_V14_ADAPTER_ID = 'qwen3.5-policy-semantic-retry.v14'
QWEN35_POLICY_SEMANTIC_RETRY_V14_ADAPTER_SCHEMA_VERSION = 'qwen35-policy-prompt-adapter.v17'
QWEN35_POLICY_SEMANTIC_RETRY_V14_MESSAGE_SCHEMA_VERSION = QWEN35_POLICY_SEMANTIC_RETRY_V13_MESSAGE_SCHEMA_VERSION
QWEN35_POLICY_SEMANTIC_RETRY_V14_DETECTOR_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V13_DETECTOR_IDENTITY_BYTES + b';harness-v15-actionable-discovery-target-semantics;post-navigation-storage-state;bounded-correction-consumption;actionable-discovery-diversion;instruction-room-entry.v16'
QWEN35_POLICY_SEMANTIC_RETRY_V14_SERIALIZER_IDENTITY_BYTES = QWEN35_POLICY_SEMANTIC_RETRY_V13_SERIALIZER_IDENTITY_BYTES + b';base-prompt=harness-v15;behavior=d5a.v14;scene-target-reask.v2'
_QWEN35_POLICY_EXPERIMENT_SUFFIXES = {'a_stop_rejection': QWEN35_POLICY_GUIDANCE_SUFFIX_BYTES + b'When the public decision check reports stop_rejected_missing_tools, the stop request was rejected and the episode is active. Do not repeat stop; choose an allowed missing tool and use it to make task progress.\n', 'b_placement_target': QWEN35_POLICY_GUIDANCE_SUFFIX_BYTES + b'When the public decision check reports stop_rejected_missing_tools, the stop request was rejected and the episode is active. Do not repeat stop; choose an allowed missing tool and use it to make task progress.\nFor put_down, target the destination receptacle or surface named by the instruction, never the object being carried.\n', 'c_binding_disambiguation': QWEN35_POLICY_GUIDANCE_SUFFIX_BYTES + b'When the public decision check reports stop_rejected_missing_tools, the stop request was rejected and the episode is active. Do not repeat stop; choose an allowed missing tool and use it to make task progress.\nFor put_down, target the destination receptacle or surface named by the instruction, never the object being carried.\nAfter a public Binding failure with reason multiple_public_matches, do not repeat the same generic target. Use a more specific public reference supported by the instruction or current image, or observe if none is available.\n'}
_STOP_MISSING_TOOLS_RE = re.compile('\\Astandard_vlm stop requires prior requested action\\(s\\): ([a-z][a-z0-9_]*(?:, [a-z][a-z0-9_]*)*)\\Z')
_STOP_HELD_PLACEMENT_RE = re.compile('(?i)stop rejected: a publicly held object still needs placement before stopping\\.')
POLICY_DECISION_CHECK_ADAPTER_ID = 'public-instruction-decision-check.v1'
POLICY_DECISION_CHECK_ADAPTER_SCHEMA_VERSION = 'method-policy-context-footer-adapter.v1'
POLICY_DECISION_CHECK_PREFIX_BYTES = b'DECISION_CHECK='
POLICY_DECISION_CHECK_BYTES = b'Before returning a manipulation ToolCall, re-read every instruction clause. If that manipulation has any conditional or ordered prerequisite without public evidence, observe or navigate to its named target and complete the prerequisite first. Never assume an if-condition is false without evidence. Do not repeat a manipulation after public evidence shows its intended effect.'
POLICY_DECISION_CHECK_SERIALIZER_IDENTITY_BYTES = b'canonical-policy-context-newline-decision-check-static-bytes.v1'
_FROZEN_V11_PROMPT_EXPOSURE_HASHES = {'memory_proposer': '3eb3e7e5cb7785eca7b1828966e1de9abfe4b00eaf9fc2d52ae7b4bbc4445cd5', 'policy': '7de1cabb01b3abbacf2bbf65529fbb9481c53c0125860ad4d28995d1019abe06', 'verifier': '15dee5b8e192a98c1d63bf886efc3906d74d00ad057e79e9fd8b4ebaf82fea09'}
POLICY_SYSTEM_V2_BYTES = b'You are a closed-loop embodied tool policy.\nUse only the current image and the public JSON context.\nTreat evidence-memory confidence and unresolved incidents as uncertain evidence.\nUse natural public references from the instruction, visual observation, public history, or evidence memory; never emit internal role names or simulator object IDs.\nAn unseen target named by the instruction may be used only with navigate_to; manipulation requires a currently public interaction-ready reference.\nSelect only a tool and arguments allowed by allowed_tool_specs.\nReturn exactly one JSON object with keys arguments and tool_name, and no other text.\nCanonical format example (choose an allowed tool and arguments from the request): {"arguments":{"target":"container"},"tool_name":"navigate_to"}\n'
POLICY_SYSTEM_V3_BYTES = b'You are a closed-loop embodied tool policy.\nUse only the current image and the public JSON context.\nTreat evidence-memory confidence and unresolved incidents as uncertain evidence.\nUse natural public references from the instruction, visual observation, public history, or evidence memory; never emit internal role names or simulator object IDs.\nAn unseen target named by the instruction may be used only with navigate_to; manipulation requires a currently public interaction-ready reference.\nRead mode before choosing an action. allowed_tool_specs is the complete and exclusive tool set for this decision; never emit a tool that is absent from it, even if the instruction, history, or an example suggests that tool.\nIn observe mode, pause the original task and choose only an available observation tool such as scan or look_for. In recovery mode, address the unresolved incident using only the tools currently listed. If the desired next task tool is absent, choose a listed tool instead.\nReturn exactly one JSON object with keys arguments and tool_name, and no other text.\nObserve-mode canonical format example when scan is listed: {"arguments":{"num_views":4},"tool_name":"scan"}\n'
POLICY_SYSTEM_V4_BYTES = b'You are a closed-loop embodied tool policy.\nUse only the current image and the public JSON context.\nThe instruction is the complete overall task. Before each action, privately derive a short ordered plan for its clauses, then choose only the next useful tool; revise that plan when observation or execution changes the public evidence. Keep pursuing every clause across decisions. tool_history and last_public_transition are the authoritative public record of your prior actions; evidence_memory is your current public, evidence-based task status. Never expose hidden reasoning, and never use or infer private goal progress.\nTreat evidence-memory confidence and unresolved incidents as uncertain evidence. Do not repeat a completed action when public history or current memory already supports its effect. Request stop only when public history and memory support completion of the whole instruction.\nUse natural public references from the instruction, visual observation, public history, or evidence memory; never emit internal role names or simulator object IDs.\nAn unseen target named by the instruction may be used only with navigate_to; manipulation requires a currently public interaction-ready reference. If the last Binding failed with not_interaction_ready, navigate to its canonical public reference before retrying the interaction.\nRead mode before choosing an action. allowed_tool_specs is the complete and exclusive tool set for this decision; never emit a tool that is absent from it, even if the instruction, history, or an example suggests that tool.\nIn observe mode, pause the original task and choose only an available observation tool. In recovery mode, inspect the linked pending obligation and last public failure. At most one observation action is useful for the same unresolved incident; after observation tools disappear from allowed_tool_specs, choose a listed navigation or interaction action instead.\nReturn exactly one JSON object with keys arguments and tool_name, and no other text.\nObserve-mode canonical format example when scan is listed: {"arguments":{"num_views":4},"tool_name":"scan"}\n'
POLICY_SYSTEM_V5_BYTES = POLICY_SYSTEM_V4_BYTES.replace(b'Use only the current image and the public JSON context.\n', b'Use only the current image and the public JSON context. If the instruction names one or more rooms, first navigate_to one named room; do not call scan or stop before entering a named room, and inspect every named room before stopping. Scan views may be deliberately tilted downward: inspect every view carefully, including the lower corners and floor, where small objects can be easy to miss. In each room, first look for task-relevant interactive or pickable objects such as bowl, plate, cup, mug, dish, box, food, book, or toy; then record their visible relation or state (on the floor, on a countertop, inside or outside storage, open, closed, or running). Prefer these concrete objects over large background furniture. Do not spend attention on wall, floor, picture, or door as entities unless directly relevant.\n').replace(b'Observe-mode canonical format example when scan is listed: {"arguments":{"num_views":4},"tool_name":"scan"}', b'scan accepts no arguments; its canonical call is exactly {}. Observe-mode canonical format example when scan is listed: {"arguments":{},"tool_name":"scan"}').replace(b'tool_history and last_public_transition are the authoritative public record of your prior actions; ', b'tool_history is chronological, oldest first and newest last; last_public_transition is the newest completed public action. Together they are the authoritative public record of your prior actions; ').replace(b'In recovery mode, inspect the linked pending obligation and last public failure. ', b'In recovery mode, inspect the linked pending obligation and last public failure. If failure_type is public_interaction_not_ready, use navigate_to on the linked public role; do not scan or retry manipulation. ').replace(b'An unseen target named by the instruction may be used only with navigate_to; manipulation requires a currently public interaction-ready reference. ', b'An unseen target named by the instruction may be used only with navigate_to; manipulation requires a currently visible public interaction-ready reference in the current image. Before pick_up, put_down, open, close, toggle_on, toggle_off, slice, clean, fill, or saturate, verify both that tool_history contains a successful navigation to the relevant concrete target and that the current image still shows that target at interaction range; scan and look_for do not count as navigation. If either check fails, choose navigate_to first. ')
POLICY_SYSTEM_V9_BYTES = POLICY_SYSTEM_V5_BYTES.replace(b'Treat evidence-memory confidence and unresolved incidents as uncertain evidence. Do not repeat a completed action when public history or current memory already supports its effect. Request stop only when public history and memory support completion of the whole instruction.\n', b'Treat evidence-memory confidence and unresolved incidents as uncertain evidence. Evidence memory is the public task-progress ledger: compare its current verified facts and pending obligations with every clause of the full instruction before advancing or stopping. An attempted or environment-accepted interaction is not completed until public post-action evidence confirms its required effect. After pick_up, the target must be recorded as held_by_agent=true. After put_down, the previously held object must be held_by_agent=false and, when required by the instruction, recorded inside or on_top of the target. After open or close, the target must be recorded with open=true or open=false respectively. Apply the same verified-state rule to toggle and other physical state changes. If the required fact is absent, ambiguous, contradicted, or linked to an open incident, treat that task step as incomplete and follow recovery instead of advancing. Do not repeat a completed action when public history or current memory already supports its effect. Request stop only when verified public facts support completion of the whole instruction.\n').replace(b'In recovery mode, inspect the linked pending obligation and last public failure. If failure_type is public_interaction_not_ready, use navigate_to on the linked public role; do not scan or retry manipulation. ', b'In recovery mode, inspect the linked pending obligation and last public failure. If failure_type is public_interaction_not_ready, first use navigate_to on the linked public role; do not scan or retry manipulation before navigation succeeds. After that recovery navigation, retry the original failed interaction before advancing the task. When an allowed tool argument has a one-value enum, copy that value exactly because it is the public retry obligation. ')
POLICY_SYSTEM_V10_BYTES = POLICY_SYSTEM_V9_BYTES.replace(b'tool_history is chronological, oldest first and newest last; last_public_transition is the newest completed public action. Together they are the authoritative public record of your prior actions; ', b'tool_history is chronological, oldest first and newest last; every row records only a requested tool whose execution lifecycle reached action-end. last_public_transition is the newest such completed request. Neither completed nor action-end means that the tool succeeded. Evidence memory, updated by visual model calls, is the only task-state ledger; ').replace(b'An attempted or environment-accepted interaction is not completed until public post-action evidence confirms its required effect. ', b'A requested interaction is not successful until the visual verifier and visual memory update support its required effect from RGB evidence. The environment never supplies action success, Binding success, inventory, object state, relation state, interaction readiness, or goal progress to this Harness. ').replace(b'Before pick_up, put_down, open, close, toggle_on, toggle_off, slice, clean, fill, or saturate, verify both that tool_history contains a successful navigation to the relevant concrete target and that the current image still shows that target at interaction range; ', b'Before pick_up, put_down, open, close, toggle_on, toggle_off, slice, clean, fill, or saturate, verify both that tool_history contains a completed navigation request to the relevant concrete target and that the current image visually shows that target at interaction range; ').replace(b'If failure_type is public_interaction_not_ready, first use navigate_to on the linked public role; do not scan or retry manipulation before navigation succeeds. After that recovery navigation, retry the original failed interaction before advancing the task. When an allowed tool argument has a one-value enum, copy that value exactly because it is the public retry obligation. ', b'Use recovery only for incidents created from visual verifier or visual memory evidence. Judge whether navigation or observation changed the situation from the current image; do not infer success from action-end or hidden execution feedback. ').replace(b'If the last Binding failed with not_interaction_ready, navigate to its canonical public reference before retrying the interaction.\n', b'\n')
POLICY_SYSTEM_BYTES = POLICY_SYSTEM_V10_BYTES
POLICY_SYSTEM_V13_BYTES = POLICY_SYSTEM_V10_BYTES.replace(b'Return exactly one JSON object with keys arguments and tool_name, and no other text.\n', b'When method_state.recovery is present, failed_tool_call is the exact public action being recovered. failure_kind says why it needs recovery. phase=look requires the locked look_for(original target), or the locked scan only when no target exists. phase=navigate requires the locked navigate_to(original target). These preparation actions do not replace the failed action. phase=retry or phase=final_retry requires failed_tool_call with exactly the saved tool_name and arguments. retry_index counts only actual retries admitted for execution; recovery_round also counts failed admission/preparation cycles. last_preparation_result_or_null reports found/not_found/motion_failed/binding_failed or navigation readiness without exposing private state.\nmethod_state.assumed_completed_actions is the persistent planning ledger for actions that remained visually uncertain after two retries. Treat each listed ToolCall as already completed for planning, continue with the next task step, and do not repeat it using another phrase for the same unique public task witness. These entries are not verified visual facts and must not be copied into evidence memory.\nmethod_state.effective_facts combines visually verified facts with separately marked uncertain assumed effects. Use it for planning and public preconditions; verified facts take precedence over assumptions. Never describe an uncertain fact as a verified training label.\nWhen human_messages contains a pending QA, answer it before any physical or recovery action. Use the only exposed answer tool, copy its locked question_id exactly, and choose value only from answer_set. Base the answer on the current RGB, chronological tool_history, and verified evidence memory; for a question about the next action, do not recommend an action whose requested state is already visibly satisfied. Answering pauses recovery for one decision and does not cancel, advance, or reset the failed action. Insert messages add public instructions to the overall task.\nA placement supported by verified evidence memory, or listed as an exact put_down ToolCall in assumed_completed_actions, is complete for planning. Continue to the next required task step; do not pick up or place that object again unless a later RGB observation provides a clear visual contradiction. Action-end alone still never proves success.\nReturn exactly one JSON object with keys arguments and tool_name, and no other text.\n')
POLICY_SYSTEM_V14_BYTES = POLICY_SYSTEM_V13_BYTES.replace(b'Return exactly one JSON object with keys arguments and tool_name, and no other text.\n', b'Scene-observation facts in evidence_memory are advisory visual discoveries. They may guide later observation, navigation, or manipulation, but they do not prove current visibility, interaction readiness, or task completion.\nReturn exactly one JSON object with keys arguments and tool_name, and no other text.\n')
POLICY_SYSTEM_V15_BYTES = POLICY_SYSTEM_V14_BYTES.replace(b'Return exactly one JSON object with keys arguments and tool_name, and no other text.\n', b'When the instruction is conditional or names a broad category, actively inspect the relevant area before deciding that no action is needed. After navigating to an inspection room, scan once before targeted search unless relevant concrete discoveries already exist. A broad scan is orientation, not proof that an instruction condition is absent: before stop, use targeted look_for for each unresolved named condition, such as faucet for running water and storage container for open storage. Use unresolved instruction conditions as search queries; do not switch to an unrelated visible object. If a task requires moving a discovered object to category-defined storage, use navigate_to with the broad storage query when needed, identify a concrete destination such as a shelf, drawer, or cabinet, and place only into that destination, never onto an unrelated surface. Exploratory discovery of a placement destination establishes only presence or location, not whether it is open or ready. After navigating to that destination, judge the current RGB: if it appears closed, open it once before put_down; otherwise proceed directly to put_down. Treat instruction-relevant scene observations as a discovery queue: subject_public_role is the concrete observed entity name that may be used by look_for, navigate_to, or manipulation, while object_public_role_or_value is only descriptive state/location text and must never be copied as a ToolCall target. Broad categories such as dishware, storage, or safety issue are inspection queries, not manipulation targets, once memory provides a concrete entity. Resolve applicable discovered states before moving on: observed does not mean resolved. A running faucet or open storage remains unresolved until verified effective facts or an exact assumed-completed corrective ToolCall supports toggled_on=false or open=false. After targeted look_for successfully locates an entity whose scene observation states an instruction-relevant abnormal condition, use the matching corrective tool before inspecting another condition. A later verified or assumed-completed correction consumes the earlier abnormal observation; only a newer contradictory visual observation may reactivate it. A discovery remains advisory and never proves visibility, interaction readiness, action success, or completion; re-observe or navigate when needed. Do not let unrelated observations displace an unresolved instruction-relevant discovery.\nReturn exactly one JSON object with keys arguments and tool_name, and no other text.\n')
POLICY_SYSTEM_V15_BYTES = POLICY_SYSTEM_V15_BYTES.replace(b'A later verified or assumed-completed correction consumes the earlier abnormal observation; only a newer contradictory visual observation may reactivate it. ', b'A verified or assumed-completed correction consumes advisory observations of that abnormality; a scene observation alone cannot reactivate the hard requirement or override effective facts. ')
POLICY_SYSTEM_V15_BYTES = POLICY_SYSTEM_V15_BYTES.replace(b'Return exactly one JSON object with keys arguments and tool_name, and no other text.\n', b'When method_state.recovery phase is open_prerequisite, the public put_down failed because its destination was closed. Execute the single required open ToolCall; the Harness will then restore the original put_down retry.\nReturn exactly one JSON object with keys arguments and tool_name, and no other text.\n')
POLICY_SYSTEM_V16_BYTES = POLICY_SYSTEM_V15_BYTES + (b'The public semantic vocabulary below is fixed globally and is not a task-specific target list. Use the most specific visible family or one of its listed aliases for ToolCall references. Broad query terms are for inspection only; unknown visual names may be retained as advisory observations but are never interaction targets. Binding is shared Core infrastructure and remains fail-closed when a family has multiple candidates.\n' + PUBLIC_VOCABULARY_PROMPT_BYTES)
VERIFIER_V9_BASE_SYSTEM_BYTES = b'Compare the BEFORE and AFTER egocentric images using only visual evidence.\nThe attempted action is context and does not imply that the action succeeded.\nRepresent success, failure, ambiguity, visible effects, and observability without guessing hidden state.\n'
VERIFIER_V10_BASE_SYSTEM_BYTES = VERIFIER_V9_BASE_SYSTEM_BYTES + b'You receive no simulator success flag, Binding result, inventory, object state, relation state, interaction readiness, or goal progress. Action-end only means execution stopped. Decide the outcome exclusively from the BEFORE and AFTER RGB images; return ambiguous when the required effect is not visually supported.\n'
MEMORY_V9_BASE_SYSTEM_BYTES = b'Propose structured public evidence-memory changes using only the supplied visual evidence.\nTreat prior memory as uncertain evidence and never infer private simulator state.\nUse only the requested update, reconcile, or route output head.\n'
MEMORY_V10_BASE_SYSTEM_BYTES = MEMORY_V9_BASE_SYSTEM_BYTES + b'Simulator success, Binding results, inventory, object states, relations, interaction readiness, and goal progress are unavailable. Tool completion and attempted-action context are not evidence of success. Mutate memory only when the supplied RGB evidence supports the change; otherwise abstain or mark prior facts unknown.\n'
VERIFIER_INSTRUCTION_BYTES = b'Return exactly one canonical JSON object with exact keys abstain,effects,mode,observability,outcome,schema_version. Use schema_version method-text-hardqa.v1 and mode verifier. Do not output probabilities, scores, markdown, or explanation.\n'
MEMORY_INSTRUCTION_BYTES = b'Return exactly one canonical JSON object with exact keys abstain,fact_mutations,incident_mutations,mode,obligation_mutations,per_fact_status,relevant_memory_ids,schema_version. Use schema_version method-text-hardqa.v1 and the requested mode. Use only supplied public candidates. Copy public_entity_ref values, public_roles items, relevant_memory_ids items, and per_fact_status fact_id values exactly from the corresponding candidate arrays in the request; never invent or paraphrase a candidate. For update or route, if no supplied candidate supports a valid answer, return the canonical abstain shape with the requested mode. For reconcile, cover every candidate_memory_ids item with status unknown when visual evidence does not resolve it. Do not output probabilities, scores, markdown, or explanation.\n'
VERIFIER_ABSTAIN_EXAMPLE_BYTES = b'{"abstain":true,"effects":[],"mode":"verifier","observability":"unobservable","outcome":"ambiguous","schema_version":"method-text-hardqa.v1"}'
MEMORY_ABSTAIN_EXAMPLE_BYTES = b'{"abstain":true,"fact_mutations":[],"incident_mutations":[],"mode":"update","obligation_mutations":[],"per_fact_status":[],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v1"}'
VERIFIER_V9_SYSTEM_BYTES = VERIFIER_V9_BASE_SYSTEM_BYTES + VERIFIER_INSTRUCTION_BYTES + b'Canonical format example: ' + VERIFIER_ABSTAIN_EXAMPLE_BYTES + b'\n'
MEMORY_V9_SYSTEM_BYTES = MEMORY_V9_BASE_SYSTEM_BYTES + MEMORY_INSTRUCTION_BYTES + b'Canonical format example: ' + MEMORY_ABSTAIN_EXAMPLE_BYTES + b'\n'
VERIFIER_V10_SYSTEM_BYTES = VERIFIER_V10_BASE_SYSTEM_BYTES + VERIFIER_INSTRUCTION_BYTES + b'Canonical format example: ' + VERIFIER_ABSTAIN_EXAMPLE_BYTES + b'\n'
MEMORY_V10_SYSTEM_BYTES = MEMORY_V10_BASE_SYSTEM_BYTES + MEMORY_INSTRUCTION_BYTES + b'Canonical format example: ' + MEMORY_ABSTAIN_EXAMPLE_BYTES + b'\n'
VERIFIER_BASE_SYSTEM_BYTES = VERIFIER_V9_BASE_SYSTEM_BYTES + b'You receive no simulator success flag, Binding result, inventory, object state, relation state, interaction readiness, or goal progress. Action-end only means execution stopped. Decide the outcome exclusively from the ordered RGB evidence; do not infer success from tool completion or hidden state.\nThe public action context names the model-requested target and the visual effects that would count as success. required_effects are expectations, never observations or proof of success.\nFirst locate the same public target in BEFORE and AFTER, then judge the requested effect from the RGB change. For pick_up, success means the target is in the hand or visual-attach region in AFTER, or has clearly moved from its original location into that held region. For put_down, success means the held object leaves the held region and appears in or on the intended target; a visually credible placement may be partially_observable and successful. For open and close, compare the same door, drawer, or container geometry; an AFTER image already clearly in the requested state is sufficient and the change need not be exaggerated. Apply the equivalent visible-state comparison to toggle, clean, slice, fill, saturate, heat, and cool.\nPartial observability does not imply ambiguity. When credible positive visual evidence exists and there is no clear contradiction, prefer success. Use failure only for clear counter-evidence, such as the same visible target remaining in the opposite state. Use ambiguous only when the target is absent, seriously occluded, or the two images genuinely cannot establish the effect.\n'
MEMORY_BASE_SYSTEM_BYTES = MEMORY_V10_BASE_SYSTEM_BYTES + b'The observed public action includes expected visual effects, but expectations are not facts. Use only RGB plus the visual verifier conclusion. When visually supported, map picked_up to held_by_agent=true, released to held_by_agent=false, opened to open=true, closed to open=false, toggled_on or toggled_off to toggled_on=true or false, and placement effects to inside or on_top. Apply the corresponding public fact for clean, slice, fill, saturate, heat, and cool. An ambiguous verifier result does not authorize a verified fact mutation.\nFor a visually verified success, use operation add when the fact is absent from previous_evidence_memory and confirm when the exact fact already exists. Use invalidate only when visual evidence removes or contradicts an existing fact. In update mode per_fact_status must be empty. In reconcile mode cover every candidate_memory_ids fact exactly once and do not invent fact IDs. For a visually verified failure, create one incident using failure_type visual_effect_not_achieved/<canonical_action>; the Harness creates the recovery obligation. Copy every public reference exactly from public_entity_candidates.\n'
VERIFIER_SUCCESS_EXAMPLE_BYTES = b'{"abstain":false,"effects":["picked_up"],"mode":"verifier","observability":"partially_observable","outcome":"success","schema_version":"method-text-hardqa.v1"}'
VERIFIER_FAILURE_EXAMPLE_BYTES = b'{"abstain":false,"effects":["no_effect"],"mode":"verifier","observability":"observable","outcome":"failure","schema_version":"method-text-hardqa.v1"}'
VERIFIER_SYSTEM_BYTES = VERIFIER_BASE_SYSTEM_BYTES + VERIFIER_INSTRUCTION_BYTES + b'Success example: ' + VERIFIER_SUCCESS_EXAMPLE_BYTES + b'\nFailure example: ' + VERIFIER_FAILURE_EXAMPLE_BYTES + b'\nAmbiguous example: ' + VERIFIER_ABSTAIN_EXAMPLE_BYTES + b'\n'
MEMORY_SYSTEM_BYTES = MEMORY_BASE_SYSTEM_BYTES + MEMORY_INSTRUCTION_BYTES + b'Successful pickup update example when mug is a supplied public candidate and the verified effect is picked_up: ' + b'{"abstain":false,"fact_mutations":[{"object_public_role_or_value":true,"operation":"add","public_entity_ref":"mug","relation_or_property":"held_by_agent"}],"incident_mutations":[],"mode":"update","obligation_mutations":[],"per_fact_status":[],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v1"}' + b'\nFailed pickup update example when mug is a supplied public candidate and the verifier found clear failure: ' + b'{"abstain":false,"fact_mutations":[],"incident_mutations":[{"failure_type":"visual_effect_not_achieved/pick_up","operation":"create","public_roles":["mug"]}],"mode":"update","obligation_mutations":[],"per_fact_status":[],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v1"}' + b'\nReconcile example when fact-1 is the only candidate fact and remains visually supported: ' + b'{"abstain":false,"fact_mutations":[],"incident_mutations":[],"mode":"reconcile","obligation_mutations":[],"per_fact_status":[{"fact_id":"fact-1","status":"confirm"}],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v1"}' + b'\nCanonical update abstain example: ' + MEMORY_ABSTAIN_EXAMPLE_BYTES + b'\n'
MEMORY_SYSTEM_V14_SUFFIX_BYTES = b'In observe mode, jointly inspect the ordered public RGB sequence after the information action finishes. Record only task-relevant objects or clearly abnormal visible states. Use schema_version method-text-hardqa.v2. Each fact mutation must use relation_or_property scene_observation, a category-level public_entity_ref, and a concise visually supported text value. Do not infer obligations, destinations, task completion, Binding, coordinates, instance identity, or hidden simulator state. incident_mutations, obligation_mutations, per_fact_status, and relevant_memory_ids must be empty. Return at most eight facts. If nothing relevant is visible, return the canonical empty abstain result.\nObserve discovery example: {"abstain":false,"fact_mutations":[{"object_public_role_or_value":"on the floor","operation":"add","public_entity_ref":"plate","relation_or_property":"scene_observation"}],"incident_mutations":[],"mode":"observe","obligation_mutations":[],"per_fact_status":[],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v2"}\nObserve empty example: {"abstain":true,"fact_mutations":[],"incident_mutations":[],"mode":"observe","obligation_mutations":[],"per_fact_status":[],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v2"}\n'
VERIFIER_V11_SYSTEM_BYTES = VERIFIER_SYSTEM_BYTES
MEMORY_V11_SYSTEM_BYTES = MEMORY_SYSTEM_BYTES
VERIFIER_BASE_SYSTEM_BYTES = VERIFIER_V9_BASE_SYSTEM_BYTES + b'You receive no simulator success flag, Binding result, inventory, object state, relation state, interaction readiness, or goal progress. Action-end only means execution stopped. Decide the outcome exclusively from the ordered RGB evidence; do not infer success from tool completion or hidden state.\nThe public action context names the model-requested target and the visual effects that would count as success. required_effects are expectations, never observations or proof of success.\nThe images object has three named, ordered RGB views and they are not interchangeable: BEFORE (pre_rgb) is the policy view immediately before the tool call; IMMEDIATE_AFTER (immediate_post_rgb) is the first view after environment execution stops, before any confirmation camera motion; CONFIRMATION (confirmation_rgb) is the final deterministic camera view aimed at the action target or held region. Confirmation is selected from the public action and camera adapter only; it is never selected from success, inventory, object state, goal progress, or simulator truth. Judge only from this BEFORE -> IMMEDIATE_AFTER -> CONFIRMATION sequence.\nFor pick_up, locate the target in BEFORE, use IMMEDIATE_AFTER to check whether it left its original location, and use CONFIRMATION to inspect the hand or visual-attach region. A target visibly held there is success. For put_down, use CONFIRMATION to check that the object leaves the held region and appears in or on the requested placement target. For open and close, compare the same target geometry across all three images; a clearly requested final state in CONFIRMATION is sufficient. Apply the equivalent target-centered comparison to toggle, clean, slice, fill, saturate, heat, and cool.\nPartial observability does not imply ambiguity. When credible positive visual evidence exists without clear contradiction, prefer success. Use failure only for clear counter-evidence. Use ambiguous only when all three views genuinely cannot establish the requested effect.\n'
MEMORY_BASE_SYSTEM_BYTES = MEMORY_V10_BASE_SYSTEM_BYTES + b'For an update request, the visual verifier conclusion in public_transition_or_null is the sole authority for action outcome and selected effects. Do not independently rejudge success or failure from the images, tool completion, or hidden state. Deterministically project that already-verified conclusion into the required public fact or incident mutation; only use the images as evidence lineage for the update. For reconcile, inspect current RGB evidence to confirm, contradict, or mark prior facts unknown. Never infer simulator truth.\nThe three image keys have fixed meanings: pre_rgb is before execution, immediate_post_rgb is the first post-execution view before camera confirmation, and confirmation_rgb is the deterministic target/held-region view after camera adjustment. Keep this order when relating a visual fact to the action.\nFor a verifier success, map picked_up to held_by_agent=true, released to held_by_agent=false, opened to open=true, closed to open=false, toggled_on or toggled_off to toggled_on=true or false, and placement effects to inside or on_top. Apply the corresponding public fact for clean, slice, fill, saturate, heat, and cool. For a verifier failure, create one incident using failure_type visual_effect_not_achieved/<canonical_action>. For a verifier ambiguous result, abstain from action facts.\nFor verified success, use operation add when the fact is absent and confirm when the exact fact already exists. Use invalidate only during reconcile when current RGB contradicts an existing fact. Copy public references exactly from supplied candidates.\n'
VERIFIER_SYSTEM_BYTES = VERIFIER_BASE_SYSTEM_BYTES + VERIFIER_INSTRUCTION_BYTES + b'Success example: ' + VERIFIER_SUCCESS_EXAMPLE_BYTES + b'\nFailure example: ' + VERIFIER_FAILURE_EXAMPLE_BYTES + b'\nAmbiguous example: ' + VERIFIER_ABSTAIN_EXAMPLE_BYTES + b'\n'
MEMORY_SYSTEM_BYTES = MEMORY_BASE_SYSTEM_BYTES + MEMORY_INSTRUCTION_BYTES + b'Successful pickup update example when mug is supplied and the verified effect is picked_up: ' + b'{"abstain":false,"fact_mutations":[{"object_public_role_or_value":true,"operation":"add","public_entity_ref":"mug","relation_or_property":"held_by_agent"}],"incident_mutations":[],"mode":"update","obligation_mutations":[],"per_fact_status":[],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v1"}' + b'\nFailed pickup update example when the verifier conclusion is failure: ' + b'{"abstain":false,"fact_mutations":[],"incident_mutations":[{"failure_type":"visual_effect_not_achieved/pick_up","operation":"create","public_roles":["mug"]}],"mode":"update","obligation_mutations":[],"per_fact_status":[],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v1"}' + b'\nReconcile example when fact-1 is the only candidate fact and the confirmation view still supports it: ' + b'{"abstain":false,"fact_mutations":[],"incident_mutations":[],"mode":"reconcile","obligation_mutations":[],"per_fact_status":[{"fact_id":"fact-1","status":"confirm"}],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v1"}' + b'\nCanonical update abstain example: ' + MEMORY_ABSTAIN_EXAMPLE_BYTES + b'\n'
MEMORY_SYSTEM_V14_BYTES = MEMORY_SYSTEM_BYTES + MEMORY_SYSTEM_V14_SUFFIX_BYTES
MEMORY_SYSTEM_V15_BYTES = MEMORY_SYSTEM_BYTES + b'Observe answers must be exactly one raw JSON object. Do not use Markdown code fences, prose, prefixes, suffixes, or multiple JSON values. The first non-whitespace character must be { and the last non-whitespace character must be }. A legal empty answer is the canonical abstain object shown below.\nOutputs for update, reconcile, and route must use schema_version method-text-hardqa.v1. Never output the prompt envelope schema_version method-text-hardqa-prompt.v2. In observe mode, jointly inspect the ordered public RGB sequence after the information action finishes. Use the public instruction to prioritize visible entities and abnormal states that could change the remaining task plan. Preserve the most specific visually identifiable natural public entity name, such as cup, bowl, faucet, drawer, cabinet, or fridge; broad instruction categories such as dishware, storage, or safety issue guide attention but are not actionable entity names. The text value must state only a concise visible state or location cue, such as on the floor, outside storage, appears to be running, appears open, or present in the scene. Use schema_version method-text-hardqa.v2 and relation_or_property scene_observation. Do not infer obligations, destinations, task completion, Binding, coordinates, instance identity, or hidden simulator state. incident_mutations, obligation_mutations, per_fact_status, and relevant_memory_ids must be empty. Return at most four facts, prioritizing instruction-relevant states over generic object presence. Inspect every current frame anew for each conditional state in the public instruction; previous memory is context, not an output template. A visible water stream from a sink is faucet appears to be running, and visibly ajar storage is the concrete drawer or cabinet appears open. For a placement instruction, an exploratory scan or look_for should record each visible concrete destination only as present in the scene or with a visible location cue, separately from the misplaced object; it must not infer open, closed, available, or ready state. Only when observation_tool_call_or_null is navigate_to for that destination or its storage query may current RGB support appears open or appears closed. If neither state is clear, keep only its presence or location observation. For an open-storage safety condition, emit appears open only when the opening is visually supported, otherwise omit that condition. Do not repeat unrelated prior facts instead of a current task-relevant finding. If nothing relevant is visible, return the canonical empty abstain result. For a targeted look_for, inspect the queried entity and named condition first; if unsupported, abstain instead of filling the fact budget with unrelated presence facts. For running-water checks, a visible light-blue or cyan vertical stream beneath a tap is flowing water: report faucet appears to be running, not merely sink present.\nObserve actionable discovery example: {"abstain":false,"fact_mutations":[{"object_public_role_or_value":"on the countertop outside storage","operation":"add","public_entity_ref":"cup","relation_or_property":"scene_observation"},{"object_public_role_or_value":"present in the scene","operation":"add","public_entity_ref":"shelf","relation_or_property":"scene_observation"}],"incident_mutations":[],"mode":"observe","obligation_mutations":[],"per_fact_status":[],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v2"}\nObserve empty example: {"abstain":true,"fact_mutations":[],"incident_mutations":[],"mode":"observe","obligation_mutations":[],"per_fact_status":[],"relevant_memory_ids":[],"schema_version":"method-text-hardqa.v2"}\n'
MEMORY_SYSTEM_V16_BYTES = MEMORY_SYSTEM_V15_BYTES.replace(b'at most four facts', b'at most eight facts') + (b'The public semantic vocabulary below is global, not a task target list. In observe mode preserve the most specific visible family or alias when one is supported by RGB. Unknown visual nouns may remain advisory text, but do not turn them into an interaction target. Return one raw canonical JSON object, at most eight scene observation facts in this call, and rely on the memory capacity to retain at most sixteen scene observations. A canonical empty abstain result is valid.\n' + PUBLIC_VOCABULARY_PROMPT_BYTES)
MEMORY_SYSTEM_V17_BYTES = MEMORY_SYSTEM_V15_BYTES + b'For this memory version, return at most four facts. Prefer a visually supported task-relevant relation or state over generic presence. Exclude wall, floor, picture, picture frame, and door unless one is the direct named interaction target. For placement tasks prefer on the floor, on the countertop, inside the cabinet, or outside storage. For openable or toggleable targets prefer appears open, appears closed, appears to be running, or appears to be off only when RGB supports that state.\n'
VERIFIER_SYSTEM_V13_BYTES = VERIFIER_SYSTEM_BYTES.replace(b'For open and close, compare the same target geometry across all three images; a clearly requested final state in CONFIRMATION is sufficient. ', b'For open and close, compare the same target geometry across all three images. Success requires a clearly requested final state and should normally be supported by a visible geometric change from BEFORE to IMMEDIATE_AFTER or CONFIRMATION. If BEFORE already unambiguously shows the requested state, an equally clear final state may count as idempotent success. A narrow crack, slight displacement, nearly unchanged geometry, or a clearly partial open or close state is not success. When that partial or opposite state is clearly visible, return failure; when the target is cropped, occluded, or too unclear to judge, return ambiguous. ') + b'Open success example: the same door or drawer moves substantially and the interior or opening is clearly exposed in the final view. Partial-open failure example: only a narrow slit appears or the target remains visibly only partly open. Cropped ambiguous example: the relevant door, drawer, or container geometry is outside the frame or too occluded to establish its final state.\n'
HARDQA_VERIFIER_PROJECTED_FIELDS = ('schema_version', 'action', 'images')
HARDQA_MEMORY_PROJECTED_FIELDS = ('schema_version', 'mode', 'base_revision', 'images', 'previous_evidence_memory', 'observed_action_or_null', 'public_transition_or_null', 'public_observation_or_null', 'public_entity_candidates', 'candidate_memory_ids')
HARDQA_MEMORY_V14_PROJECTED_FIELDS = ('schema_version', 'proposal_id', 'mode', 'base_revision', 'evidence_frame_ids', 'images', 'previous_evidence_memory', 'public_instruction', 'public_task_context', 'observation_tool_call_or_null')
HARDQA_IMAGE_ROLE_POLICY_V11 = {'verifier': ['pre_rgb', 'post_rgb'], 'memory': {'update': ['pre_rgb', 'post_rgb'], 'reconcile': 'current_rgb_or_ordered_window_rgb', 'route': 'optional_current_rgb'}}
HARDQA_IMAGE_ROLE_POLICY = {'verifier': ['pre_rgb', 'immediate_post_rgb', 'confirmation_rgb'], 'memory': {'update': ['pre_rgb', 'immediate_post_rgb', 'confirmation_rgb'], 'reconcile': 'current_rgb_or_ordered_window_rgb', 'route': 'optional_current_rgb'}}
HARDQA_IMAGE_ROLE_POLICY_V14 = copy.deepcopy(HARDQA_IMAGE_ROLE_POLICY)
HARDQA_IMAGE_ROLE_POLICY_V14['memory']['observe'] = 'ordered_window_rgb'
HARDQA_IMAGE_ROLE_POLICY_V16 = copy.deepcopy(HARDQA_IMAGE_ROLE_POLICY_V14)
HARDQA_IMAGE_ROLE_POLICY_V17 = copy.deepcopy(HARDQA_IMAGE_ROLE_POLICY_V14)
HARDQA_MEMORY_V16_PROJECTED_FIELDS = HARDQA_MEMORY_V14_PROJECTED_FIELDS

def _image_role_policy(prompt_contract_version: str) -> Mapping[str, Any]:
    return HARDQA_IMAGE_ROLE_POLICY_V14 if prompt_contract_version in {HARDQA_PROMPT_V14_CONTRACT_VERSION, HARDQA_PROMPT_V15_CONTRACT_VERSION, HARDQA_PROMPT_V16_CONTRACT_VERSION, HARDQA_PROMPT_V17_CONTRACT_VERSION} else HARDQA_IMAGE_ROLE_POLICY if prompt_contract_version in {HARDQA_PROMPT_CONTRACT_VERSION, HARDQA_PROMPT_V13_CONTRACT_VERSION, HARDQA_PROMPT_V14_CONTRACT_VERSION, HARDQA_PROMPT_V15_CONTRACT_VERSION, HARDQA_PROMPT_V16_CONTRACT_VERSION, HARDQA_PROMPT_V17_CONTRACT_VERSION} else HARDQA_IMAGE_ROLE_POLICY_V11

def hardqa_system_bytes(kind: str) -> bytes:
    if kind == 'verifier':
        return VERIFIER_SYSTEM_BYTES
    if kind == 'memory':
        return MEMORY_SYSTEM_BYTES
    raise ValueError(f'unsupported Hard-QA prompt kind: {kind!r}')

def hardqa_system_bytes_for_harness_version(kind: str, harness_version: str) -> bytes:
    if str(harness_version) == 'scoper-1':
        from scoper.harness import prompts
        return {'memory': prompts.MEMORY_SYSTEM_BYTES, 'verifier': prompts.VERIFIER_SYSTEM_BYTES}[kind]
    if str(harness_version) == '20':
        from scoper.harness import visual_prompts
        return {'memory': visual_prompts.MEMORY_SYSTEM_BYTES, 'verifier': visual_prompts.VERIFIER_SYSTEM_BYTES}[kind]
    if str(harness_version) == '19':
        from scoper.harness import recovery_prompts
        return {'memory': recovery_prompts.MEMORY_SYSTEM_BYTES, 'verifier': recovery_prompts.VERIFIER_SYSTEM_BYTES}[kind]
    if str(harness_version) == '18':
        from scoper.harness import control_prompts
        return {'memory': control_prompts.MEMORY_SYSTEM_BYTES, 'verifier': control_prompts.VERIFIER_SYSTEM_BYTES}[kind]
    if str(harness_version) in {'13', '14', '15', '16'} and kind == 'verifier':
        return VERIFIER_SYSTEM_V13_BYTES
    if str(harness_version) == '14' and kind == 'memory':
        return MEMORY_SYSTEM_V14_BYTES
    if str(harness_version) == '15' and kind == 'memory':
        return MEMORY_SYSTEM_V15_BYTES
    if str(harness_version) == '16' and kind == 'memory':
        return MEMORY_SYSTEM_V16_BYTES
    if str(harness_version) == '17' and kind == 'memory':
        return MEMORY_SYSTEM_V17_BYTES
    return hardqa_system_bytes(kind)

def _hardqa_system_bytes_for_contract(kind: str, *, verifier_system_bytes: bytes, memory_system_bytes: bytes) -> bytes:
    if kind == 'verifier':
        return verifier_system_bytes
    if kind == 'memory':
        return memory_system_bytes
    raise ValueError(f'unsupported Hard-QA prompt kind: {kind!r}')

def project_hardqa_request(kind: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
    if kind not in {'verifier', 'memory'} or not isinstance(request, Mapping):
        raise ValueError('Hard-QA request kind or value is invalid')
    fields = HARDQA_VERIFIER_PROJECTED_FIELDS if kind == 'verifier' else HARDQA_MEMORY_V14_PROJECTED_FIELDS if request.get('schema_version') == 'memory-proposer-request.v3' else HARDQA_MEMORY_PROJECTED_FIELDS
    missing = set(fields).difference(request)
    if missing:
        raise ValueError(f'Hard-QA request omits public semantic fields: {sorted(missing)!r}')
    return {field: copy.deepcopy(request[field]) for field in fields}

def build_hardqa_envelope(*, kind: str, mode: str, request: Mapping[str, Any], image_roles: Sequence[str]) -> Mapping[str, Any]:
    return {'schema_version': HARDQA_PROMPT_ENVELOPE_VERSION, 'kind': kind, 'mode': mode, 'request': project_hardqa_request(kind, request), 'image_roles': list(image_roles)}

def hardqa_prompt_exposure(component: str, *, history_limit: int | None=None, policy_system_bytes: bytes=POLICY_SYSTEM_BYTES, prompt_contract_version: str=HARDQA_PROMPT_CONTRACT_VERSION, verifier_system_bytes: bytes | None=None, memory_system_bytes: bytes | None=None) -> Mapping[str, Any]:
    if verifier_system_bytes is None:
        verifier_system_bytes = {HARDQA_PROMPT_V13_CONTRACT_VERSION: VERIFIER_SYSTEM_V13_BYTES, HARDQA_PROMPT_CONTRACT_VERSION: VERIFIER_SYSTEM_BYTES, HARDQA_PROMPT_V11_CONTRACT_VERSION: VERIFIER_V11_SYSTEM_BYTES, HARDQA_PROMPT_V10_CONTRACT_VERSION: VERIFIER_V10_SYSTEM_BYTES}.get(prompt_contract_version, VERIFIER_V9_SYSTEM_BYTES)
    if memory_system_bytes is None:
        memory_system_bytes = {HARDQA_PROMPT_V13_CONTRACT_VERSION: MEMORY_SYSTEM_BYTES, HARDQA_PROMPT_CONTRACT_VERSION: MEMORY_SYSTEM_BYTES, HARDQA_PROMPT_V11_CONTRACT_VERSION: MEMORY_V11_SYSTEM_BYTES, HARDQA_PROMPT_V10_CONTRACT_VERSION: MEMORY_V10_SYSTEM_BYTES, HARDQA_PROMPT_V14_CONTRACT_VERSION: MEMORY_SYSTEM_V14_BYTES, HARDQA_PROMPT_V15_CONTRACT_VERSION: MEMORY_SYSTEM_V15_BYTES, HARDQA_PROMPT_V16_CONTRACT_VERSION: MEMORY_SYSTEM_V16_BYTES, HARDQA_PROMPT_V17_CONTRACT_VERSION: MEMORY_SYSTEM_V17_BYTES}.get(prompt_contract_version, MEMORY_V9_SYSTEM_BYTES)
    if component == 'policy':
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit <= 0:
            raise ValueError('policy history_limit must be a positive integer')
        return {'builder': 'policy-input-builder.v1', 'history_limit': history_limit, 'template_sha256': sha256_bytes(policy_system_bytes)}
    kind = {'verifier': 'verifier', 'memory_proposer': 'memory'}.get(component)
    if kind is None:
        raise ValueError(f'unsupported Hard-QA component exposure: {component!r}')
    fields = HARDQA_VERIFIER_PROJECTED_FIELDS if kind == 'verifier' else HARDQA_MEMORY_V14_PROJECTED_FIELDS if prompt_contract_version in {HARDQA_PROMPT_V14_CONTRACT_VERSION, HARDQA_PROMPT_V15_CONTRACT_VERSION, HARDQA_PROMPT_V16_CONTRACT_VERSION, HARDQA_PROMPT_V17_CONTRACT_VERSION} else HARDQA_MEMORY_PROJECTED_FIELDS
    return {'builder': HARDQA_PROMPT_BUILDER, 'prompt_contract_version': prompt_contract_version, 'envelope_schema_version': HARDQA_PROMPT_ENVELOPE_VERSION, 'message_layout': HARDQA_MESSAGE_LAYOUT, 'projected_request_fields': list(fields), 'image_role_policy': copy.deepcopy(_image_role_policy(prompt_contract_version)[kind]), 'template_sha256': sha256_bytes(_hardqa_system_bytes_for_contract(kind, verifier_system_bytes=verifier_system_bytes, memory_system_bytes=memory_system_bytes))}

def hardqa_prompt_exposure_hash(component: str, *, history_limit: int | None=None, policy_system_bytes: bytes=POLICY_SYSTEM_BYTES, prompt_contract_version: str=HARDQA_PROMPT_CONTRACT_VERSION, verifier_system_bytes: bytes | None=None, memory_system_bytes: bytes | None=None) -> str:
    return sha256_json(hardqa_prompt_exposure(component, history_limit=history_limit, policy_system_bytes=policy_system_bytes, prompt_contract_version=prompt_contract_version, verifier_system_bytes=verifier_system_bytes, memory_system_bytes=memory_system_bytes))

def hardqa_prompt_contract_descriptor(*, history_limit: int=16, policy_system_bytes: bytes=POLICY_SYSTEM_BYTES, prompt_contract_version: str=HARDQA_PROMPT_CONTRACT_VERSION, verifier_system_bytes: bytes | None=None, memory_system_bytes: bytes | None=None) -> Mapping[str, Any]:
    descriptor = {'schema_version': prompt_contract_version, 'envelope_schema_version': HARDQA_PROMPT_ENVELOPE_VERSION, 'message_layout': HARDQA_MESSAGE_LAYOUT, 'policy_history_limit': history_limit, 'task_binding_contract': 'task-binding-spec.v1', 'training_observation_contract': 'standard_vlm+streaming_evidence.v1', 'policy_target_reference_contract': 'natural_public_reference.v1', 'projected_request_fields': {'memory': list(HARDQA_MEMORY_V14_PROJECTED_FIELDS if prompt_contract_version in {HARDQA_PROMPT_V14_CONTRACT_VERSION, HARDQA_PROMPT_V15_CONTRACT_VERSION, HARDQA_PROMPT_V16_CONTRACT_VERSION} else HARDQA_MEMORY_PROJECTED_FIELDS), 'verifier': list(HARDQA_VERIFIER_PROJECTED_FIELDS)}, 'image_role_policy': copy.deepcopy(_image_role_policy(prompt_contract_version)), 'prompt_exposure_hashes': copy.deepcopy(_FROZEN_V11_PROMPT_EXPOSURE_HASHES) if prompt_contract_version == HARDQA_PROMPT_V11_CONTRACT_VERSION else {component: hardqa_prompt_exposure_hash(component, history_limit=history_limit, policy_system_bytes=policy_system_bytes, prompt_contract_version=prompt_contract_version, verifier_system_bytes=verifier_system_bytes, memory_system_bytes=memory_system_bytes) for component in ('policy', 'verifier', 'memory_proposer')}}
    if prompt_contract_version == HARDQA_PROMPT_V16_CONTRACT_VERSION:
        descriptor.update({'shared_vocabulary_version': SHARED_VOCABULARY_VERSION, 'shared_vocabulary_sha256': SHARED_VOCABULARY_SHA256})
    return descriptor

def hardqa_policy_system_bytes_for_harness_version(harness_version: str) -> bytes:
    if harness_version == 'scoper-1':
        from scoper.harness.prompts import POLICY_SYSTEM_BYTES
        return POLICY_SYSTEM_BYTES
    if harness_version == '20':
        from scoper.harness.visual_prompts import POLICY_SYSTEM_BYTES
        return POLICY_SYSTEM_BYTES
    if harness_version == '19':
        from scoper.harness.recovery_prompts import POLICY_SYSTEM_BYTES as policy
        return policy
    if harness_version == '18':
        from scoper.harness.control_prompts import POLICY_SYSTEM_BYTES as policy
        return policy
    if harness_version == '14':
        return POLICY_SYSTEM_V14_BYTES
    if harness_version == '15':
        return POLICY_SYSTEM_V15_BYTES
    if harness_version in {'16', '17'}:
        return POLICY_SYSTEM_V16_BYTES
    if harness_version == '13':
        return POLICY_SYSTEM_V13_BYTES
    return POLICY_SYSTEM_BYTES

def qwen35_policy_system_bytes_for_harness_version(harness_version: str) -> bytes:
    return hardqa_policy_system_bytes_for_harness_version(harness_version) + QWEN35_POLICY_PROMPT_SUFFIX_BYTES

def qwen35_policy_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    base = hardqa_policy_system_bytes_for_harness_version(harness_version)
    effective = qwen35_policy_system_bytes_for_harness_version(harness_version)
    return {'schema_version': QWEN35_POLICY_PROMPT_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_PROMPT_ADAPTER_ID, 'component': 'policy', 'base_prompt_sha256': sha256_bytes(base), 'suffix_sha256': sha256_bytes(QWEN35_POLICY_PROMPT_SUFFIX_BYTES), 'effective_prompt_sha256': sha256_bytes(effective)}

def qwen35_policy_guidance_system_bytes_for_harness_version(harness_version: str) -> bytes:
    return hardqa_policy_system_bytes_for_harness_version(harness_version) + QWEN35_POLICY_GUIDANCE_SUFFIX_BYTES

def qwen35_policy_d4_guidance_system_bytes_for_harness_version(harness_version: str) -> bytes:
    if str(harness_version) == 'scoper-1':
        from scoper.harness.prompts import D6_POLICY_SYSTEM_BYTES
        return D6_POLICY_SYSTEM_BYTES
    if str(harness_version) == '20':
        from scoper.harness.visual_prompts import D6_POLICY_SYSTEM_BYTES
        return D6_POLICY_SYSTEM_BYTES
    if str(harness_version) == '19':
        from scoper.harness.recovery_prompts import D6_POLICY_SYSTEM_BYTES
        return D6_POLICY_SYSTEM_BYTES
    if str(harness_version) == '18':
        from scoper.harness.control_prompts import D6_POLICY_SYSTEM_BYTES
        return D6_POLICY_SYSTEM_BYTES
    'Reconstruct the archived D4-C system prompt under its published IDs.'
    return hardqa_policy_system_bytes_for_harness_version(harness_version) + QWEN35_POLICY_D4_GUIDANCE_SUFFIX_BYTES

def qwen35_policy_d5a_system_bytes_for_harness_version(harness_version: str) -> bytes:
    return qwen35_policy_d4_guidance_system_bytes_for_harness_version(harness_version)

def _qwen35_experiment_profile(profile: str) -> str:
    if profile not in QWEN35_POLICY_EXPERIMENT_PROFILES:
        raise ValueError('unsupported Qwen policy experiment profile')
    return profile

def qwen35_policy_experiment_system_bytes_for_harness_version(harness_version: str, profile: str) -> bytes:
    return hardqa_policy_system_bytes_for_harness_version(harness_version) + _QWEN35_POLICY_EXPERIMENT_SUFFIXES[_qwen35_experiment_profile(profile)]

def _public_binding_admission(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('Qwen public Binding admission must be an object or null')
    fields = ('canonical_public_ref', 'match_kind', 'navigation_class', 'reason', 'source', 'status')
    result = {field: copy.deepcopy(value[field]) for field in fields if field in value}
    if not result:
        return None
    if any((item is not None and (not isinstance(item, str)) for item in result.values())):
        raise ValueError('Qwen public Binding admission values must be text or null')
    validate_json_value(result, json_pointer='/qwen_public_binding_admission')
    return result

def qwen35_public_transition_feedback(transition: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if transition is None:
        return None
    if not isinstance(transition, Mapping):
        raise ValueError('Qwen public transition feedback must be an object or null')
    nested = transition.get('qwen_public_transition_feedback')
    if nested is not None:
        if not isinstance(nested, Mapping) or set(nested) != {'binding_admission_or_null', 'stop_rejected_missing_tools', 'tool_call'}:
            raise ValueError('Qwen projected public transition feedback is invalid')
        call = _public_tool_call(nested.get('tool_call'))
        binding = _public_binding_admission(nested.get('binding_admission_or_null'))
        missing = nested.get('stop_rejected_missing_tools')
        if call is None or not isinstance(missing, list) or any((not isinstance(item, str) or re.fullmatch('[a-z][a-z0-9_]*', item) is None for item in missing)) or (len(set(missing)) != len(missing)):
            raise ValueError('Qwen projected public transition feedback differs')
        result = {'binding_admission_or_null': binding, 'stop_rejected_missing_tools': list(missing), 'tool_call': call}
        validate_json_value(result, json_pointer='/qwen_public_transition_feedback')
        return result
    tool_call = _public_tool_call({'tool_name': transition.get('tool_name'), 'arguments': transition.get('arguments', {})})
    if tool_call is None:
        raise ValueError('Qwen public transition feedback lacks its ToolCall')
    binding = _public_binding_admission(transition.get('binding'))
    if binding is None:
        bindings = transition.get('bindings')
        if bindings is not None:
            if not isinstance(bindings, list):
                raise ValueError('Qwen public Binding admissions must be a list')
            projected = [item for item in (_public_binding_admission(raw) for raw in bindings) if item is not None]
            binding = projected[-1] if projected else None
    missing_tools: list[str] = []
    candidates: list[str] = []
    raw_execution = transition.get('raw_execution')
    if isinstance(raw_execution, Mapping):
        for field in ('error_message', 'message'):
            if isinstance(raw_execution.get(field), str):
                candidates.append(raw_execution[field])
    for field in ('feedback', 'execution_error'):
        if isinstance(transition.get(field), str):
            candidates.append(transition[field])
    env_feedback = transition.get('env_feedback')
    if isinstance(env_feedback, str):
        candidates.append(env_feedback)
    elif isinstance(env_feedback, Mapping):
        for field in ('error_message', 'message'):
            if isinstance(env_feedback.get(field), str):
                candidates.append(env_feedback[field])
    for candidate in candidates:
        match = _STOP_MISSING_TOOLS_RE.fullmatch(candidate)
        if match is not None:
            missing_tools = match.group(1).split(', ')
            break
        if _STOP_HELD_PLACEMENT_RE.fullmatch(candidate):
            missing_tools = ['put_down']
            break
    result = {'binding_admission_or_null': binding, 'stop_rejected_missing_tools': missing_tools, 'tool_call': tool_call}
    validate_json_value(result, json_pointer='/qwen_public_transition_feedback')
    return result

def qwen35_projected_public_transition_feedback(feedback: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if feedback is None:
        return None
    return qwen35_public_transition_feedback({'qwen_public_transition_feedback': feedback})

def _open_rows(value: Any, *, id_field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError('Qwen policy context memory rows must be lists')
    output: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError('Qwen policy context memory row must be an object')
        if raw.get('status') != 'open':
            continue
        identifier = raw.get(id_field)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('Qwen policy context memory row lacks its public ID')
        output.append(copy.deepcopy(dict(raw)))
    output.sort(key=lambda item: canonical_json_bytes(item))
    return output

def _open_obligations(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return _open_rows(value, id_field='obligation_id')
    if not isinstance(value, Mapping):
        raise ValueError('Qwen policy pending obligations must be a list or object')
    output = []
    for identifier, raw in value.items():
        if not isinstance(identifier, str) or not identifier or (not isinstance(raw, Mapping)):
            raise ValueError('Qwen policy pending obligation entry is invalid')
        row = copy.deepcopy(dict(raw))
        row.setdefault('obligation_id', identifier)
        row.setdefault('status', 'open')
        if row['obligation_id'] != identifier:
            raise ValueError('Qwen policy pending obligation ID differs')
        if row.get('status') == 'open':
            output.append(row)
    output.sort(key=canonical_json_bytes)
    return output

def _public_tool_call(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('Qwen policy context ToolCall must be an object or null')
    tool_name = value.get('tool_name')
    arguments = value.get('arguments')
    if not isinstance(tool_name, str) or not tool_name or (not isinstance(arguments, Mapping)):
        raise ValueError('Qwen policy context ToolCall is incomplete')
    call = {'arguments': copy.deepcopy(dict(arguments)), 'tool_name': tool_name}
    validate_json_value(call, json_pointer='/qwen_policy_context/tool_call')
    return call

def _history_tool_call(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('Qwen policy history row must be an object or null')
    nested = value.get('tool_call')
    if nested is not None:
        return _public_tool_call(nested)
    tool_name = value.get('tool_name', value.get('name'))
    arguments = value.get('arguments', value.get('public_arguments'))
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError('Qwen policy history row lacks a tool name')
    if arguments is None:
        arguments = {}
    return _public_tool_call({'arguments': arguments, 'tool_name': tool_name})

def _assumed_tool_calls_in_order(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError('Qwen assumed-completed actions must be a list')
    output = []
    for raw in values:
        if not isinstance(raw, Mapping):
            raise ValueError('Qwen assumed-completed action must be an object')
        call = _public_tool_call(raw.get('tool_call'))
        if call is None:
            raise ValueError('Qwen assumed-completed action lacks its ToolCall')
        output.append(call)
    return output

def _assumed_tool_calls(values: Any) -> list[dict[str, Any]]:
    output = _assumed_tool_calls_in_order(values)
    output.sort(key=canonical_json_bytes)
    return output

def _qwen35_assumed_placement_state(policy_context: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    calls = _assumed_tool_calls_in_order(policy_context.get('assumed_completed_actions', []))
    last_pick_up = max((index for index, call in enumerate(calls) if call['tool_name'] == 'pick_up'), default=-1)
    last_put_down = max((index for index, call in enumerate(calls) if call['tool_name'] == 'put_down'), default=-1)
    if last_pick_up > last_put_down:
        return ('assumed_pick_up', calls)
    if last_put_down >= 0 and last_put_down > last_pick_up:
        return ('assumed_put_down', calls)
    return ('none', calls)

def _qwen35_allowed_tool_names(policy_context: Mapping[str, Any]) -> list[str]:
    specs = policy_context.get('allowed_tool_specs')
    if not isinstance(specs, list):
        raise ValueError('Qwen allowed tool specs must be a list')
    names: list[str] = []
    for raw in specs:
        name = raw.get('name') if isinstance(raw, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValueError('Qwen allowed tool spec lacks a name')
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError('Qwen allowed tool names are duplicated')
    return sorted(names, key=lambda item: item.encode('utf-8'))

def qwen35_policy_action_playbook_payload(policy_context: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any]:
    if not isinstance(policy_context, Mapping):
        raise ValueError('Qwen action playbook context must be an object')
    mode = policy_context.get('mode')
    if mode not in {'step', 'recovery', 'observe'}:
        raise ValueError('Qwen action playbook mode is invalid')
    allowed = _qwen35_allowed_tool_names(policy_context)
    verified_held = qwen_policy_verified_held_refs(policy_context)
    assumed_state, assumed_calls = _qwen35_assumed_placement_state(policy_context)
    feedback = qwen35_projected_public_transition_feedback(public_transition_feedback_or_null)
    binding = feedback.get('binding_admission_or_null') if isinstance(feedback, Mapping) else None
    failed_call = feedback.get('tool_call') if isinstance(feedback, Mapping) else None
    not_ready = bool(isinstance(binding, Mapping) and (binding.get('status') == 'not_interaction_ready' or binding.get('reason') in {'not_interaction_ready', 'target_not_interaction_ready', 'recent_focus_not_interaction_ready', 'successful_navigation_focus_required', 'navigation_context_required'}) and isinstance(failed_call, Mapping))
    prohibited_repeat: Mapping[str, Any] | None = None
    if mode == 'recovery':
        phase = 'recover'
        planning_basis = 'harness_recovery'
        guidance = 'Follow method_state.recovery and its locked allowed ToolCall exactly; do not reorder acquisition or placement while recovery is active.'
    elif mode == 'observe':
        phase = 'observe'
        planning_basis = 'public_mode'
        guidance = 'Use one allowed observation ToolCall for the active observation request, then resume the instruction on the next decision.'
    elif not_ready and 'navigate_to' in allowed:
        phase = 'navigate_before_retry'
        planning_basis = 'public_binding_not_ready'
        prohibited_repeat = copy.deepcopy(dict(failed_call))
        guidance = 'The last physical ToolCall was publicly not interaction-ready and was not executed. Use navigate_to with that same public target before retrying it.'
    elif len(verified_held) == 1:
        phase = 'place'
        planning_basis = 'verified_held_fact'
        guidance = 'Plan placement next: navigate_to the instruction destination when needed, then put_down with the destination receptacle or surface as target, never the carried object.'
    elif len(verified_held) == 0 and assumed_state == 'assumed_pick_up':
        phase = 'place'
        planning_basis = 'assumed_completed_pick_up'
        guidance = 'Harness lists pick_up as completed for planning but not visually verified. Do not repeat pick_up. Continue to the instruction destination: navigate_to it when needed, then put_down with the destination as target.'
    elif assumed_state == 'assumed_put_down':
        phase = 'continue'
        planning_basis = 'assumed_completed_put_down'
        guidance = 'Harness lists put_down as completed for planning but not visually verified. Do not repeat acquisition or placement; continue other public instruction obligations or use stop only when the public stop gate permits it.'
    elif len(verified_held) == 0:
        phase = 'acquire'
        planning_basis = 'no_public_held_evidence'
        guidance = 'Plan acquisition before placement: navigate_to the instruction object when needed, then pick_up it. Do not put_down before public held evidence or a Harness assumed-completed pick_up exists.'
    else:
        phase = 'continue'
        planning_basis = 'ambiguous_public_held_evidence'
        guidance = 'Public held evidence is ambiguous. Use an allowed observation or navigation ToolCall to resolve it; do not guess a carried object or destination.'
    payload = {'allowed_tool_names': allowed, 'assumed_completed_tool_calls': assumed_calls, 'guidance': guidance, 'mode': mode, 'phase': phase, 'planning_basis': planning_basis, 'prohibited_repeat_or_null': prohibited_repeat, 'schema_version': QWEN35_POLICY_ACTION_PLAYBOOK_SCHEMA_VERSION, 'verified_held_public_refs': verified_held}
    validate_json_value(payload, json_pointer='/qwen_policy_action_playbook')
    return payload

def qwen35_policy_action_playbook_v2_payload(policy_context: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any]:
    payload = dict(qwen35_policy_action_playbook_payload(policy_context, public_transition_feedback_or_null=public_transition_feedback_or_null))
    verified_placement = _qwen35_latest_verified_placement(policy_context)
    if policy_context.get('mode') == 'step' and verified_placement is not None:
        payload.update({'guidance': 'Public visual evidence confirms release after the latest pick_up then put_down. Do not pick up that same object or repeat its placement. Continue any other instruction clause, or request stop for Harness validation when none remains.', 'phase': 'continue', 'planning_basis': 'verified_completed_placement'})
    payload['schema_version'] = QWEN35_POLICY_ACTION_PLAYBOOK_V2_SCHEMA_VERSION
    validate_json_value(payload, json_pointer='/qwen_policy_action_playbook_v2')
    return payload

def qwen35_policy_action_playbook_v3_user_bytes(canonical_policy_context_bytes: bytes, policy_context: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> bytes:
    if _qwen35_latest_verified_placement(policy_context) is None:
        return qwen35_policy_action_playbook_user_bytes(canonical_policy_context_bytes, policy_context, public_transition_feedback_or_null=public_transition_feedback_or_null)
    if canonical_policy_context_bytes != canonical_json_bytes(policy_context):
        raise ValueError('Qwen action playbook context bytes are not exact canonical JSON')
    return canonical_policy_context_bytes + b'\n' + QWEN35_POLICY_ACTION_PLAYBOOK_PREFIX_BYTES + canonical_json_bytes(qwen35_policy_action_playbook_v2_payload(policy_context, public_transition_feedback_or_null=public_transition_feedback_or_null))

def qwen35_policy_action_playbook_user_bytes(canonical_policy_context_bytes: bytes, policy_context: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> bytes:
    if not isinstance(canonical_policy_context_bytes, bytes) or canonical_policy_context_bytes != canonical_json_bytes(policy_context):
        raise ValueError('Qwen action playbook context bytes are not exact canonical JSON')
    payload = qwen35_policy_action_playbook_payload(policy_context, public_transition_feedback_or_null=public_transition_feedback_or_null)
    return canonical_policy_context_bytes + b'\n' + QWEN35_POLICY_ACTION_PLAYBOOK_PREFIX_BYTES + canonical_json_bytes(payload)

def qwen35_policy_d5a_action_hint_payload(policy_context: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any]:
    if not isinstance(policy_context, Mapping):
        raise ValueError('Qwen D5-A policy context must be an object')
    mode = policy_context.get('mode')
    if mode not in {'step', 'recovery', 'observe'}:
        raise ValueError('Qwen D5-A policy mode is invalid')
    allowed = _qwen35_allowed_tool_names(policy_context)
    held = qwen_policy_verified_held_refs(policy_context)
    placement = _qwen35_latest_verified_placement(policy_context)
    feedback = qwen35_projected_public_transition_feedback(public_transition_feedback_or_null)
    binding = feedback.get('binding_admission_or_null') if feedback else None
    failed_call = feedback.get('tool_call') if feedback else None
    not_ready = bool(isinstance(binding, Mapping) and (binding.get('status') == 'not_interaction_ready' or binding.get('reason') in {'not_interaction_ready', 'target_not_interaction_ready', 'recent_focus_not_interaction_ready', 'successful_navigation_focus_required', 'navigation_context_required'}) and isinstance(failed_call, Mapping))
    if mode == 'recovery':
        phase = 'recovery'
        guidance = 'Follow method_state.recovery and its locked ToolCall exactly.'
    elif mode == 'observe':
        phase = 'observe'
        guidance = 'Use one allowed observation ToolCall for this request.'
    elif not_ready and 'navigate_to' in allowed:
        phase = 'navigate_before_retry'
        guidance = 'The last physical call was publicly not interaction-ready. Navigate to its same public target before retrying it.'
    elif placement is not None:
        phase = 'continue_after_placement'
        guidance = "Public visual evidence supports the latest placement. Do not repeat that object's pick_up or put_down; continue another instruction clause or stop only when the public stop gate permits."
    elif _qwen35_assumed_placement_state(policy_context)[0] == 'assumed_pick_up':
        phase = 'place_after_assumed_pick_up'
        guidance = 'Harness lists pick_up as completed for planning but not visually verified. Do not repeat pick_up; continue to the instruction destination and use it as put_down target.'
    elif _qwen35_assumed_placement_state(policy_context)[0] == 'assumed_put_down':
        phase = 'continue_after_assumed_put_down'
        guidance = 'Harness lists put_down as completed for planning but not visually verified. Do not repeat acquisition or placement; continue another instruction clause or stop only when the public stop gate permits.'
    elif len(held) == 1:
        phase = 'place'
        guidance = "One object is publicly verified held. Continue to the instruction's destination; put_down must target that receptacle or surface."
    elif len(held) == 0:
        phase = 'acquire'
        guidance = 'No object is publicly verified held. Prepare or perform pick_up of the instruction object before put_down.'
    else:
        phase = 'resolve_held_evidence'
        guidance = 'Public held evidence is ambiguous; use an allowed observation or navigation action instead of guessing a placement.'
    payload = {'guidance': guidance, 'phase': phase, 'schema_version': QWEN35_POLICY_ACTION_HINT_SCHEMA_VERSION}
    validate_json_value(payload, json_pointer='/qwen_policy_d5a_action_hint')
    return payload

def qwen35_policy_d5a_user_bytes(canonical_policy_context_bytes: bytes, policy_context: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> bytes:
    base = qwen35_policy_user_bytes(canonical_policy_context_bytes, policy_context)
    return base + b'\n' + QWEN35_POLICY_ACTION_HINT_PREFIX_BYTES + canonical_json_bytes(qwen35_policy_d5a_action_hint_payload(policy_context, public_transition_feedback_or_null=public_transition_feedback_or_null))

def qwen35_policy_context_footer_payload(policy_context: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(policy_context, Mapping):
        raise ValueError('Qwen policy context must be an object')
    mode = policy_context.get('mode')
    if mode not in {'step', 'recovery', 'observe'}:
        raise ValueError('Qwen policy context mode is invalid')
    specs = policy_context.get('allowed_tool_specs')
    if not isinstance(specs, list):
        raise ValueError('Qwen allowed tool specs must be a list')
    allowed_names = []
    for raw in specs:
        name = raw.get('name') if isinstance(raw, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValueError('Qwen allowed tool spec lacks a name')
        allowed_names.append(name)
    if len(set(allowed_names)) != len(allowed_names):
        raise ValueError('Qwen allowed tool names are duplicated')
    allowed_names.sort(key=lambda item: item.encode('utf-8'))
    history = policy_context.get('tool_history')
    if not isinstance(history, list):
        raise ValueError('Qwen policy tool history must be a list')
    calls = []
    for raw in history:
        call = _history_tool_call(raw)
        if call is None:
            raise ValueError('Qwen policy tool history contains null')
        calls.append(call)
    last_call = calls[-1] if calls else None
    repeat_count = 0
    if last_call is not None:
        for call in reversed(calls):
            if call != last_call:
                break
            repeat_count += 1
    memory = policy_context.get('evidence_memory')
    if not isinstance(memory, Mapping):
        raise ValueError('Qwen policy evidence memory is missing')
    assumed = policy_context.get('assumed_completed_actions', [])
    payload = {'allowed_tool_names': allowed_names, 'assumed_completed_tool_calls': _assumed_tool_calls(assumed), 'last_tool_call_or_null': last_call, 'last_tool_call_was_stop': bool(last_call is not None and last_call['tool_name'] == 'stop'), 'mode': mode, 'open_incidents': _open_rows(memory.get('incidents', []), id_field='incident_id'), 'open_obligations': _open_obligations(memory.get('pending_obligations', [])), 'repeated_last_tool_call_count': repeat_count, 'schema_version': QWEN35_POLICY_CONTEXT_FOOTER_SCHEMA_VERSION}
    validate_json_value(payload, json_pointer='/qwen_policy_context_footer')
    return payload

def qwen35_policy_user_bytes(canonical_policy_context_bytes: bytes, policy_context: Mapping[str, Any]) -> bytes:
    if not isinstance(canonical_policy_context_bytes, bytes) or canonical_policy_context_bytes != canonical_json_bytes(policy_context):
        raise ValueError('Qwen policy context bytes are not exact canonical JSON')
    footer = canonical_json_bytes(qwen35_policy_context_footer_payload(policy_context))
    return canonical_policy_context_bytes + b'\n' + QWEN35_POLICY_CONTEXT_FOOTER_PREFIX_BYTES + footer

def qwen35_policy_experiment_context_footer_payload(policy_context: Mapping[str, Any], *, profile: str, public_transition_feedback_or_null: Mapping[str, Any] | None) -> Mapping[str, Any]:
    profile = _qwen35_experiment_profile(profile)
    base = dict(qwen35_policy_context_footer_payload(policy_context))
    base['schema_version'] = QWEN35_POLICY_EXPERIMENT_CONTEXT_FOOTER_SCHEMA_VERSION
    base['experiment_profile'] = profile
    feedback = qwen35_projected_public_transition_feedback(public_transition_feedback_or_null)
    if feedback is not None and profile != 'c_binding_disambiguation':
        feedback = {**feedback, 'binding_admission_or_null': None}
    base['last_public_failure_or_null'] = feedback
    validate_json_value(base, json_pointer='/qwen_policy_experiment_footer')
    return base

def qwen35_policy_experiment_user_bytes(canonical_policy_context_bytes: bytes, policy_context: Mapping[str, Any], *, profile: str, public_transition_feedback_or_null: Mapping[str, Any] | None) -> bytes:
    if not isinstance(canonical_policy_context_bytes, bytes) or canonical_policy_context_bytes != canonical_json_bytes(policy_context):
        raise ValueError('Qwen policy context bytes are not exact canonical JSON')
    footer = qwen35_policy_experiment_context_footer_payload(policy_context, profile=profile, public_transition_feedback_or_null=public_transition_feedback_or_null)
    return canonical_policy_context_bytes + b'\n' + QWEN35_POLICY_CONTEXT_FOOTER_PREFIX_BYTES + canonical_json_bytes(footer)

def qwen35_policy_guidance_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    base = hardqa_policy_system_bytes_for_harness_version(harness_version)
    effective = qwen35_policy_guidance_system_bytes_for_harness_version(harness_version)
    return {'schema_version': QWEN35_POLICY_GUIDANCE_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_GUIDANCE_ADAPTER_ID, 'component': 'policy', 'base_prompt_sha256': sha256_bytes(base), 'suffix_sha256': sha256_bytes(QWEN35_POLICY_GUIDANCE_SUFFIX_BYTES), 'effective_prompt_sha256': sha256_bytes(effective), 'context_footer_schema_version': QWEN35_POLICY_CONTEXT_FOOTER_SCHEMA_VERSION, 'context_footer_serializer_sha256': sha256_bytes(QWEN35_POLICY_CONTEXT_FOOTER_SERIALIZER_IDENTITY_BYTES)}

def qwen35_policy_d4_guidance_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    base = hardqa_policy_system_bytes_for_harness_version(harness_version)
    effective = qwen35_policy_d4_guidance_system_bytes_for_harness_version(harness_version)
    descriptor = dict(qwen35_policy_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'suffix_sha256': sha256_bytes(QWEN35_POLICY_D4_GUIDANCE_SUFFIX_BYTES), 'effective_prompt_sha256': sha256_bytes(effective)})
    return descriptor

def qwen35_policy_experiment_prompt_adapter_descriptor(harness_version: str, profile: str) -> Mapping[str, Any]:
    profile = _qwen35_experiment_profile(profile)
    base = hardqa_policy_system_bytes_for_harness_version(harness_version)
    suffix = _QWEN35_POLICY_EXPERIMENT_SUFFIXES[profile]
    effective = qwen35_policy_experiment_system_bytes_for_harness_version(harness_version, profile)
    return {'schema_version': QWEN35_POLICY_EXPERIMENT_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_EXPERIMENT_ADAPTER_ID, 'component': 'policy', 'experiment_profile': profile, 'base_prompt_sha256': sha256_bytes(base), 'suffix_sha256': sha256_bytes(suffix), 'effective_prompt_sha256': sha256_bytes(effective), 'context_footer_schema_version': QWEN35_POLICY_EXPERIMENT_CONTEXT_FOOTER_SCHEMA_VERSION, 'context_footer_serializer_sha256': sha256_bytes(QWEN35_POLICY_EXPERIMENT_CONTEXT_FOOTER_SERIALIZER_IDENTITY_BYTES)}

def _qwen35_reference_key(value: str) -> str:
    text = re.sub('(?<!^)(?=[A-Z])', ' ', value)
    return ' '.join(re.findall('[a-z0-9]+', text.lower()))

def _qwen35_entity_reference_key(value: str) -> str:
    tokens = _qwen35_reference_key(value).split()
    if tokens and tokens[-1].isdigit():
        tokens = tokens[:-1]
    return ' '.join(tokens)

def _qwen35_latest_verified_placement(policy_context: Mapping[str, Any]) -> Mapping[str, Any] | None:
    history = policy_context.get('tool_history', [])
    facts = policy_context.get('effective_facts', [])
    if not isinstance(history, list) or not isinstance(facts, list):
        raise ValueError('Qwen verified placement public state is invalid')
    calls = [_history_tool_call(item) for item in history]
    if any((item is None for item in calls)):
        raise ValueError('Qwen verified placement history is invalid')
    last_pick_up = max((index for index, call in enumerate(calls) if call['tool_name'] == 'pick_up'), default=-1)
    last_put_down = max((index for index, call in enumerate(calls) if call['tool_name'] == 'put_down'), default=-1)
    if last_pick_up < 0 or last_put_down <= last_pick_up:
        return None
    picked_target = calls[last_pick_up]['arguments'].get('target')
    destination = calls[last_put_down]['arguments'].get('target')
    if not isinstance(picked_target, str) or not isinstance(destination, str):
        return None
    picked_key = _qwen35_entity_reference_key(picked_target)
    released_refs = sorted({str(item.get('subject_public_role')) for item in facts if isinstance(item, Mapping) and item.get('relation_or_property') == 'held_by_agent' and (item.get('object_public_role_or_value') is False) and (item.get('certainty') == 'verified') and (item.get('status', 'current') == 'current') and isinstance(item.get('subject_public_role'), str) and (_qwen35_entity_reference_key(str(item['subject_public_role'])) == picked_key)}, key=lambda item: item.encode('utf-8'))
    if len(released_refs) != 1:
        return None
    return {'destination_public_ref': destination, 'object_public_ref': released_refs[0]}

def _qwen35_public_held_refs(policy_context: Mapping[str, Any], *, public_observation_or_null: Mapping[str, Any] | None=None) -> list[str]:
    candidates: list[str] = []
    if public_observation_or_null is not None:
        if not isinstance(public_observation_or_null, Mapping):
            raise ValueError('Qwen public observation must be an object or null')
        raw = public_observation_or_null.get('held_object_handles', [])
        if not isinstance(raw, list) or any((not isinstance(item, str) or not item for item in raw)):
            raise ValueError('Qwen public held-object handles are invalid')
        candidates.extend(raw)
    observation = policy_context.get('public_observation_without_rgb')
    if isinstance(observation, Mapping):
        raw = observation.get('held_object_handles', [])
        if not isinstance(raw, list) or any((not isinstance(item, str) or not item for item in raw)):
            raise ValueError('Qwen public held-object handles are invalid')
        candidates.extend(raw)
    facts = policy_context.get('effective_facts', [])
    if not isinstance(facts, list) or any((not isinstance(item, Mapping) for item in facts)):
        raise ValueError('Qwen public effective facts are invalid')
    candidates.extend((str(item['subject_public_role']) for item in facts if item.get('relation_or_property') == 'held_by_agent' and item.get('object_public_role_or_value') is True and isinstance(item.get('subject_public_role'), str) and item['subject_public_role']))
    by_normalized: dict[str, set[str]] = {}
    for item in candidates:
        normalized = _qwen35_reference_key(item)
        if normalized:
            by_normalized.setdefault(normalized, set()).add(item)
    return [sorted(values, key=lambda item: item.encode('utf-8'))[0] for _, values in sorted(by_normalized.items(), key=lambda item: item[0].encode('utf-8'))]

def _qwen35_singleton_enum_requirements(policy_context: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    specs = policy_context.get('allowed_tool_specs')
    if not isinstance(specs, list):
        raise ValueError('Qwen allowed tool specs must be a list')
    matching = [item for item in specs if isinstance(item, Mapping) and item.get('name') == tool_name]
    if len(matching) != 1:
        return {}
    schema = matching[0].get('arguments_schema', matching[0].get('parameters'))
    properties = schema.get('properties') if isinstance(schema, Mapping) else None
    if not isinstance(properties, Mapping):
        return {}
    result: dict[str, Any] = {}
    for name, raw in properties.items():
        values = raw.get('enum') if isinstance(raw, Mapping) else None
        if isinstance(name, str) and isinstance(values, list) and (len(values) == 1):
            result[name] = copy.deepcopy(values[0])
    validate_json_value(result, json_pointer='/qwen_required_tool_arguments')
    return result

def _qwen35_stop_missing_tools(public_transition_feedback_or_null: Mapping[str, Any] | None) -> list[str]:
    feedback = qwen35_projected_public_transition_feedback(public_transition_feedback_or_null)
    if feedback is None:
        return []
    missing = feedback['stop_rejected_missing_tools']
    return list(missing)

def qwen35_policy_semantic_retry_reason(policy_context: Mapping[str, Any], tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> str | None:
    call = _public_tool_call(tool_call)
    if call is None:
        raise ValueError('Qwen semantic retry requires a ToolCall')
    held = _qwen35_public_held_refs(policy_context, public_observation_or_null=public_observation_or_null)
    target = call['arguments'].get('target')
    if call['tool_name'] == 'put_down' and len(held) == 1 and isinstance(target, str) and (_qwen35_reference_key(target) == _qwen35_reference_key(held[0])):
        return 'put_down_target_is_held_object'
    if policy_context.get('mode') == 'recovery':
        required = _qwen35_singleton_enum_requirements(policy_context, call['tool_name'])
        if any((call['arguments'].get(name) != value for name, value in required.items())):
            return 'tool_call_differs_from_required_public_enum'
    if call['tool_name'] == 'stop' and _qwen35_stop_missing_tools(public_transition_feedback_or_null):
        return 'stop_repeated_after_missing_tools_rejection'
    return None
_QWEN35_SEMANTIC_RETRY_GUIDANCE = {'put_down_target_is_held_object': 'Return one new canonical ToolCall. put_down target must be the destination receptacle, not a held object. Use an exact public reference from the instruction or public context.', 'tool_call_differs_from_required_public_enum': 'Return one new canonical ToolCall. Follow the single allowed public enum value in allowed_tool_specs exactly.', 'stop_repeated_after_missing_tools_rejection': 'Return one new canonical ToolCall. The previous stop was rejected and the episode is active; choose an allowed missing tool instead of stop.'}

def qwen35_policy_semantic_retry_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    reason = qwen35_policy_semantic_retry_reason(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if reason is None:
        return None
    call = _public_tool_call(rejected_tool_call)
    if call is None:
        raise ValueError('Qwen semantic retry requires a ToolCall')
    instruction = policy_context.get('instruction')
    if not isinstance(instruction, str):
        raise ValueError('Qwen semantic retry instruction is invalid')
    required = _qwen35_singleton_enum_requirements(policy_context, call['tool_name'])
    required_call = {'arguments': required, 'tool_name': call['tool_name']} if required else None
    payload = {'attempt': 2, 'guidance': _QWEN35_SEMANTIC_RETRY_GUIDANCE[reason], 'held_public_refs': _qwen35_public_held_refs(policy_context, public_observation_or_null=public_observation_or_null), 'instruction': instruction, 'max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS, 'rejected_reason': reason, 'rejected_tool_call': call, 'required_tool_call_or_null': required_call, 'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_MESSAGE_SCHEMA_VERSION, 'stop_rejected_missing_tools': _qwen35_stop_missing_tools(public_transition_feedback_or_null)}
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry')
    return payload

def qwen35_policy_semantic_retry_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_ADAPTER_ID, 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v2_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    base = qwen35_policy_semantic_retry_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    payload = dict(base)
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V2_MESSAGE_SCHEMA_VERSION
    missing = payload['stop_rejected_missing_tools']
    held = payload['held_public_refs']
    if payload['rejected_reason'] == 'stop_repeated_after_missing_tools_rejection' and missing == ['put_down'] and (len(held) == 1):
        payload['guidance'] = 'Return one new canonical put_down ToolCall. The previous stop was rejected and put_down is still required. put_down target must be the destination receptacle, not the held object. Use an exact public reference from the instruction or public context.'
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v2')
    return payload

def qwen35_policy_semantic_retry_v2_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v2_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v2_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V2_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V2_ADAPTER_ID, 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V2_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V2_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V2_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v3_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    call = _public_tool_call(rejected_tool_call)
    if call is None:
        raise ValueError('Qwen semantic retry requires a ToolCall')
    verified_held = qwen_policy_verified_held_refs(policy_context)
    if call['tool_name'] == 'put_down' and len(verified_held) != 1:
        instruction = policy_context.get('instruction')
        if not isinstance(instruction, str):
            raise ValueError('Qwen semantic retry instruction is invalid')
        payload = {'attempt': 2, 'guidance': 'The proposed action was not executed. No unique verified public held fact supports put_down. Return one canonical ToolCall that acquires or prepares acquisition of the instruction object before placement.', 'held_public_refs': verified_held, 'instruction': instruction, 'max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS, 'rejected_reason': 'put_down_without_unique_verified_public_held_object', 'rejected_tool_call': call, 'required_tool_call_or_null': None, 'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V3_MESSAGE_SCHEMA_VERSION, 'stop_rejected_missing_tools': _qwen35_stop_missing_tools(public_transition_feedback_or_null)}
        validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v3')
        return payload
    base = qwen35_policy_semantic_retry_v2_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    payload = dict(base)
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V3_MESSAGE_SCHEMA_VERSION
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v3')
    return payload

def qwen35_policy_semantic_retry_v3_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v3_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v3_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_d4_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V3_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V3_ADAPTER_ID, 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V3_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V3_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V3_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v4_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    base = qwen35_policy_semantic_retry_v3_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    payload = dict(base)
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V4_MESSAGE_SCHEMA_VERSION
    if payload.get('rejected_reason') == 'put_down_without_unique_verified_public_held_object':
        payload['guidance'] = 'The proposed put_down was not executed because no unique verified public held fact supports placement. On this retry, do not output put_down or stop. If pick_up is allowed, pick up the instruction object first. Otherwise use one allowed navigation or observation ToolCall only to locate or prepare acquisition of that object.'
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v4')
    return payload

def qwen35_policy_semantic_retry_v4_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v4_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v4_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_d4_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V4_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V4_ADAPTER_ID, 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V4_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V4_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V4_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v5_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    base = qwen35_policy_semantic_retry_v4_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    specs = policy_context.get('allowed_tool_specs')
    if not isinstance(specs, list):
        raise ValueError('Qwen semantic retry allowed tool specs are invalid')
    alternatives = sorted({str(item['name']) for item in specs if isinstance(item, Mapping) and isinstance(item.get('name'), str) and (item['name'] not in {'put_down', 'stop'})}, key=lambda item: item.encode('utf-8'))
    payload = dict(base)
    payload['allowed_alternative_tool_names'] = alternatives
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V5_MESSAGE_SCHEMA_VERSION
    if payload.get('rejected_reason') == 'put_down_without_unique_verified_public_held_object':
        payload['guidance'] = 'The proposed put_down was not executed because no unique verified public held fact supports placement. On this one retry, select one tool_name from allowed_alternative_tool_names. Prefer pick_up for the instruction object when available; otherwise prepare only its acquisition. Do not output put_down or stop.'
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v5')
    return payload

def qwen35_policy_semantic_retry_v5_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v5_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v5_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_d4_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V5_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V5_ADAPTER_ID, 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V5_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V5_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V5_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v6_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    base = qwen35_policy_semantic_retry_v5_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    payload = dict(base)
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V6_MESSAGE_SCHEMA_VERSION
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v6')
    return payload

def qwen35_policy_semantic_retry_v6_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v6_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v6_user_bytes(policy_context: Mapping[str, Any], retry_payload: Mapping[str, Any]) -> bytes:
    alternatives = retry_payload.get('allowed_alternative_tool_names')
    if not isinstance(alternatives, list) or any((not isinstance(item, str) or not item for item in alternatives)):
        raise ValueError('Qwen fresh semantic retry alternatives are invalid')
    allowed = set(alternatives)
    specs = policy_context.get('allowed_tool_specs')
    if not isinstance(specs, list):
        raise ValueError('Qwen fresh semantic retry tool specs are invalid')
    retry_context = copy.deepcopy(dict(policy_context))
    retry_context['allowed_tool_specs'] = [copy.deepcopy(dict(item)) for item in specs if isinstance(item, Mapping) and item.get('name') in allowed]
    if len(retry_context['allowed_tool_specs']) != len(allowed):
        raise ValueError('Qwen fresh semantic retry alternatives differ from specs')
    validate_json_value(retry_context, json_pointer='/qwen_semantic_retry_context')
    validate_json_value(retry_payload, json_pointer='/qwen_semantic_retry_payload')
    return canonical_json_bytes(retry_context) + b'\n' + QWEN35_POLICY_SEMANTIC_RETRY_V6_PREFIX_BYTES + canonical_json_bytes(retry_payload)

def qwen35_policy_semantic_retry_v6_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_d4_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V6_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V6_ADAPTER_ID, 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V6_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V6_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V6_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v7_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    call = _public_tool_call(rejected_tool_call)
    if call is None:
        raise ValueError('Qwen semantic retry requires a ToolCall')
    assumed_state, assumed_calls = _qwen35_assumed_placement_state(policy_context)
    verified_held = qwen_policy_verified_held_refs(policy_context)
    if call['tool_name'] == 'put_down' and len(verified_held) == 0 and (assumed_state == 'assumed_pick_up'):
        assumed_pick_up = next((item for item in reversed(assumed_calls) if item['tool_name'] == 'pick_up'), None)
        proposed_target = call['arguments'].get('target')
        assumed_target = assumed_pick_up['arguments'].get('target') if isinstance(assumed_pick_up, Mapping) else None
        if isinstance(proposed_target, str) and isinstance(assumed_target, str) and (_qwen35_reference_key(proposed_target) == _qwen35_reference_key(assumed_target)):
            instruction = policy_context.get('instruction')
            if not isinstance(instruction, str):
                raise ValueError('Qwen semantic retry instruction is invalid')
            payload = {'attempt': 2, 'guidance': 'The proposed put_down was not executed. Harness lists pick_up as completed for planning, but put_down target must be the instruction destination receptacle or surface, never the object whose pick_up is assumed complete. Return one canonical ToolCall.', 'held_public_refs': [], 'instruction': instruction, 'max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS, 'rejected_reason': 'put_down_target_is_assumed_picked_up_object', 'rejected_tool_call': call, 'required_tool_call_or_null': None, 'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V7_MESSAGE_SCHEMA_VERSION, 'stop_rejected_missing_tools': _qwen35_stop_missing_tools(public_transition_feedback_or_null)}
            validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v7')
            return payload
        return qwen35_policy_semantic_retry_v2_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    base = qwen35_policy_semantic_retry_v6_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    payload = dict(base)
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V7_MESSAGE_SCHEMA_VERSION
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v7')
    return payload

def qwen35_policy_semantic_retry_v7_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v7_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v7_user_bytes(policy_context: Mapping[str, Any], retry_payload: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> bytes:
    retry_context = copy.deepcopy(dict(policy_context))
    alternatives = retry_payload.get('allowed_alternative_tool_names')
    if alternatives is not None:
        if not isinstance(alternatives, list) or any((not isinstance(item, str) or not item for item in alternatives)):
            raise ValueError('Qwen fresh semantic retry alternatives are invalid')
        allowed = set(alternatives)
        specs = policy_context.get('allowed_tool_specs')
        if not isinstance(specs, list):
            raise ValueError('Qwen fresh semantic retry tool specs are invalid')
        retry_context['allowed_tool_specs'] = [copy.deepcopy(dict(item)) for item in specs if isinstance(item, Mapping) and item.get('name') in allowed]
        if len(retry_context['allowed_tool_specs']) != len(allowed):
            raise ValueError('Qwen fresh semantic retry alternatives differ from specs')
    validate_json_value(retry_context, json_pointer='/qwen_semantic_retry_context')
    validate_json_value(retry_payload, json_pointer='/qwen_semantic_retry_payload')
    fresh = canonical_json_bytes(retry_context) + b'\n' + QWEN35_POLICY_SEMANTIC_RETRY_V6_PREFIX_BYTES + canonical_json_bytes(retry_payload)
    playbook = qwen35_policy_action_playbook_payload(retry_context, public_transition_feedback_or_null=public_transition_feedback_or_null)
    return fresh + b'\n' + QWEN35_POLICY_ACTION_PLAYBOOK_PREFIX_BYTES + canonical_json_bytes(playbook)

def qwen35_policy_semantic_retry_v7_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_d4_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V7_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V7_ADAPTER_ID, 'action_playbook_schema_version': QWEN35_POLICY_ACTION_PLAYBOOK_SCHEMA_VERSION, 'action_playbook_serializer_sha256': sha256_bytes(QWEN35_POLICY_ACTION_PLAYBOOK_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V7_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V7_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V7_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v8_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    if policy_context.get('mode') == 'recovery':
        return None
    base = qwen35_policy_semantic_retry_v7_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    payload = dict(base)
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V8_MESSAGE_SCHEMA_VERSION
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v8')
    return payload

def qwen35_policy_semantic_retry_v8_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v8_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v8_user_bytes(policy_context: Mapping[str, Any], retry_payload: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> bytes:
    return qwen35_policy_semantic_retry_v7_user_bytes(policy_context, retry_payload, public_transition_feedback_or_null=public_transition_feedback_or_null)

def qwen35_policy_semantic_retry_v8_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_d4_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V8_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V8_ADAPTER_ID, 'action_playbook_schema_version': QWEN35_POLICY_ACTION_PLAYBOOK_SCHEMA_VERSION, 'action_playbook_serializer_sha256': sha256_bytes(QWEN35_POLICY_ACTION_PLAYBOOK_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V8_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V8_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V8_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v9_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    if policy_context.get('mode') == 'recovery':
        return None
    call = _public_tool_call(rejected_tool_call)
    placement = _qwen35_latest_verified_placement(policy_context)
    if call is None:
        raise ValueError('Qwen semantic retry requires a ToolCall')
    if placement is not None:
        tool_name = call['tool_name']
        target = call['arguments'].get('target')
        repeats_object = tool_name == 'pick_up' and isinstance(target, str) and (_qwen35_entity_reference_key(target) == _qwen35_entity_reference_key(placement['object_public_ref']))
        repeats_destination = tool_name == 'put_down' and isinstance(target, str) and (_qwen35_entity_reference_key(target) == _qwen35_entity_reference_key(placement['destination_public_ref']))
        if repeats_object or repeats_destination:
            allowed = [name for name in _qwen35_allowed_tool_names(policy_context) if name not in {'pick_up', 'put_down'}]
            instruction = policy_context.get('instruction')
            if not isinstance(instruction, str):
                raise ValueError('Qwen semantic retry instruction is invalid')
            payload = {'allowed_alternative_tool_names': allowed, 'attempt': 2, 'guidance': 'The proposed action was not executed. Public visual evidence already confirms release after the latest pick_up and put_down. Do not repeat acquisition or placement of that same object. Choose another instruction action, or stop for Harness validation.', 'held_public_refs': [], 'instruction': instruction, 'max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS, 'rejected_reason': 'repeat_after_verified_completed_placement', 'rejected_tool_call': call, 'required_tool_call_or_null': None, 'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V9_MESSAGE_SCHEMA_VERSION, 'stop_rejected_missing_tools': _qwen35_stop_missing_tools(public_transition_feedback_or_null)}
            validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v9')
            return payload
    base = qwen35_policy_semantic_retry_v8_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    payload = dict(base)
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V9_MESSAGE_SCHEMA_VERSION
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v9')
    return payload

def qwen35_policy_semantic_retry_v9_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v9_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v9_user_bytes(policy_context: Mapping[str, Any], retry_payload: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> bytes:
    retry_context = copy.deepcopy(dict(policy_context))
    alternatives = retry_payload.get('allowed_alternative_tool_names')
    if alternatives is not None:
        allowed = set(alternatives)
        retry_context['allowed_tool_specs'] = [copy.deepcopy(dict(item)) for item in policy_context.get('allowed_tool_specs', []) if isinstance(item, Mapping) and item.get('name') in allowed]
        if len(retry_context['allowed_tool_specs']) != len(allowed):
            raise ValueError('Qwen v9 retry alternatives differ from specs')
    fresh = canonical_json_bytes(retry_context) + b'\n' + QWEN35_POLICY_SEMANTIC_RETRY_V6_PREFIX_BYTES + canonical_json_bytes(retry_payload)
    playbook = qwen35_policy_action_playbook_v2_payload(retry_context, public_transition_feedback_or_null=public_transition_feedback_or_null)
    return fresh + b'\n' + QWEN35_POLICY_ACTION_PLAYBOOK_PREFIX_BYTES + canonical_json_bytes(playbook)

def qwen35_policy_semantic_retry_v9_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_d4_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V9_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V9_ADAPTER_ID, 'action_playbook_schema_version': QWEN35_POLICY_ACTION_PLAYBOOK_V2_SCHEMA_VERSION, 'action_playbook_serializer_sha256': sha256_bytes(QWEN35_POLICY_ACTION_PLAYBOOK_V2_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V9_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V9_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V9_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v10_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    placement_payload = qwen35_policy_semantic_retry_v9_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if placement_payload is not None and placement_payload.get('rejected_reason') == 'repeat_after_verified_completed_placement':
        result = dict(placement_payload)
        result['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V10_MESSAGE_SCHEMA_VERSION
        validate_json_value(result, json_pointer='/qwen_policy_semantic_retry_v10')
        return result
    return qwen35_policy_semantic_retry_v8_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)

def qwen35_policy_semantic_retry_v10_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v10_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v10_user_bytes(policy_context: Mapping[str, Any], retry_payload: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None) -> bytes:
    if retry_payload.get('rejected_reason') != 'repeat_after_verified_completed_placement':
        return qwen35_policy_semantic_retry_v8_user_bytes(policy_context, retry_payload, public_transition_feedback_or_null=public_transition_feedback_or_null)
    retry_context = copy.deepcopy(dict(policy_context))
    alternatives = retry_payload.get('allowed_alternative_tool_names')
    if alternatives is not None:
        allowed = set(alternatives)
        retry_context['allowed_tool_specs'] = [copy.deepcopy(dict(item)) for item in policy_context.get('allowed_tool_specs', []) if isinstance(item, Mapping) and item.get('name') in allowed]
        if len(retry_context['allowed_tool_specs']) != len(allowed):
            raise ValueError('Qwen v10 retry alternatives differ from specs')
    fresh = canonical_json_bytes(retry_context) + b'\n' + QWEN35_POLICY_SEMANTIC_RETRY_V6_PREFIX_BYTES + canonical_json_bytes(retry_payload)
    return fresh + b'\n' + QWEN35_POLICY_ACTION_PLAYBOOK_PREFIX_BYTES + canonical_json_bytes(qwen35_policy_action_playbook_v2_payload(retry_context, public_transition_feedback_or_null=public_transition_feedback_or_null))

def qwen35_policy_semantic_retry_v10_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    descriptor = dict(qwen35_policy_d4_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V10_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V10_ADAPTER_ID, 'action_playbook_schema_version': QWEN35_POLICY_ACTION_PLAYBOOK_V3_SCHEMA_VERSION, 'action_playbook_serializer_sha256': sha256_bytes(QWEN35_POLICY_ACTION_PLAYBOOK_V3_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V10_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V10_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V10_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    return descriptor

def qwen35_policy_semantic_retry_v11_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    if policy_context.get('mode') == 'recovery':
        return None
    call = _public_tool_call(rejected_tool_call)
    if call is None:
        raise ValueError('Qwen semantic retry requires a ToolCall')
    placement = _qwen35_latest_verified_placement(policy_context)
    target = call['arguments'].get('target')
    repeats_placement = bool(placement is not None and (call['tool_name'] == 'pick_up' and isinstance(target, str) and (_qwen35_entity_reference_key(target) == _qwen35_entity_reference_key(placement['object_public_ref'])) or (call['tool_name'] == 'put_down' and isinstance(target, str) and (_qwen35_entity_reference_key(target) == _qwen35_entity_reference_key(placement['destination_public_ref'])))))
    if repeats_placement:
        instruction = policy_context.get('instruction')
        if not isinstance(instruction, str):
            raise ValueError('Qwen semantic retry instruction is invalid')
        payload = {'attempt': 2, 'guidance': "Public visual evidence supports the latest placement. Return one new canonical ToolCall that does not repeat that object's pick_up or put_down; continue another instruction clause, or stop only when the public stop gate permits.", 'held_public_refs': [], 'instruction': instruction, 'max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS, 'rejected_reason': 'repeat_after_verified_completed_placement', 'rejected_tool_call': call, 'required_tool_call_or_null': None, 'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V11_MESSAGE_SCHEMA_VERSION, 'stop_rejected_missing_tools': _qwen35_stop_missing_tools(public_transition_feedback_or_null)}
        validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v11')
        return payload
    base = qwen35_policy_semantic_retry_v2_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    payload = dict(base)
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V11_MESSAGE_SCHEMA_VERSION
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v11')
    return payload

def qwen35_policy_semantic_retry_v11_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v11_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def _qwen35_missing_required_arguments(policy_context: Mapping[str, Any], tool_call: Mapping[str, Any]) -> list[str]:
    call = _public_tool_call(tool_call)
    if call is None:
        raise ValueError('Qwen semantic retry requires a ToolCall')
    specs = policy_context.get('allowed_tool_specs')
    if not isinstance(specs, list):
        raise ValueError('Qwen allowed tool specs must be a list')
    matches = [item for item in specs if isinstance(item, Mapping) and item.get('name') == call['tool_name']]
    if len(matches) != 1:
        return []
    schema = matches[0].get('arguments_schema', matches[0].get('parameters'))
    required = schema.get('required') if isinstance(schema, Mapping) else None
    if not isinstance(required, list) or any((not isinstance(name, str) or not name for name in required)):
        return []
    return sorted((name for name in required if name not in call['arguments']), key=str.encode)

def qwen35_policy_semantic_retry_v13_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    call = _public_tool_call(rejected_tool_call)
    if call is None:
        raise ValueError('Qwen semantic retry requires a ToolCall')
    missing = _qwen35_missing_required_arguments(policy_context, call)
    if missing:
        instruction = policy_context.get('instruction')
        if not isinstance(instruction, str):
            raise ValueError('Qwen semantic retry instruction is invalid')
        payload = {'attempt': 2, 'guidance': 'Return one new canonical ToolCall for the same intended action. Include every field named in missing_required_arguments, using only public references supported by the instruction or public context.', 'held_public_refs': [], 'instruction': instruction, 'max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS, 'missing_required_arguments': missing, 'rejected_reason': 'missing_required_tool_arguments', 'rejected_tool_call': call, 'required_tool_call_or_null': None, 'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V13_MESSAGE_SCHEMA_VERSION, 'stop_rejected_missing_tools': _qwen35_stop_missing_tools(public_transition_feedback_or_null)}
        validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v13')
        return payload
    base = qwen35_policy_semantic_retry_v11_payload(policy_context, call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is None:
        return None
    payload = dict(base)
    payload['schema_version'] = QWEN35_POLICY_SEMANTIC_RETRY_V13_MESSAGE_SCHEMA_VERSION
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v13')
    return payload

def qwen35_policy_semantic_retry_v13_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v13_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def _qwen35_v15_scene_observations(policy_context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    memory = policy_context.get('evidence_memory')
    facts = memory.get('facts') if isinstance(memory, Mapping) else None
    if not isinstance(facts, list):
        return []
    return [fact for fact in facts if isinstance(fact, Mapping) and fact.get('relation_or_property') == 'scene_observation' and (fact.get('status') == 'current') and isinstance(fact.get('subject_public_role'), str) and fact['subject_public_role']]

def _qwen35_scene_description_key(value: str) -> str:
    tokens = _qwen35_reference_key(value).split()
    while tokens and tokens[0] in {'a', 'an', 'at', 'in', 'inside', 'near', 'on', 'the'}:
        tokens.pop(0)
    return ' '.join(tokens)

def _qwen35_v15_discovery_queries(policy_context: Mapping[str, Any], instruction: str) -> list[str]:
    from scoper.harness.binding import instruction_scene_reference_category
    observations = _qwen35_v15_scene_observations(policy_context)
    history = policy_context.get('tool_history')
    history = history if isinstance(history, list) else []
    calls = [call for call in (_history_tool_call(item) for item in history) if call is not None]
    queries = []
    for query in ('faucet', 'storage container', 'dishware', 'storage'):
        category = instruction_scene_reference_category(query, instruction)
        if category is None:
            continue
        observed = any((instruction_scene_reference_category(fact['subject_public_role'], instruction) == category for fact in observations))
        if category == 'faucet':
            observed = any((instruction_scene_reference_category(fact['subject_public_role'], instruction) == category and any((cue in _qwen35_reference_key(fact.get('object_public_role_or_value') or '') for cue in ('running', 'water stream', 'flowing'))) for fact in observations))
        inspected = any((call['tool_name'] in {'clean', 'close', 'fill', 'look_for', 'open', 'put_down', 'toggle_off', 'toggle_on'} and instruction_scene_reference_category(call['arguments'].get('target'), instruction) == category for call in calls))
        if not observed and (not inspected):
            queries.append(query)
    return queries

def _qwen35_v15_instruction_room_target(policy_context: Mapping[str, Any], instruction: str) -> str | None:
    from scoper.harness.room_semantics import CANONICAL_ROOM_TYPES, room_display_label
    normalized = instruction.casefold()
    labels = {room_display_label(room_type) for room_type in CANONICAL_ROOM_TYPES if re.search('\\b{}\\b'.format(re.escape(room_display_label(room_type))), normalized)}
    if len(labels) != 1:
        return None
    target = next(iter(labels))
    public_context = policy_context.get('public_observation_without_rgb')
    scoped_room_ids = public_context.get('task_scope_room_ids') if isinstance(public_context, Mapping) else None
    if isinstance(scoped_room_ids, list):
        scoped_room_ids = [item for item in scoped_room_ids if isinstance(item, str) and item]
        room_catalog = public_context.get('room_catalog', [])
        matching_scope_ids = []
        for room_id in scoped_room_ids:
            row = next((item for item in room_catalog if isinstance(item, Mapping) and item.get('room_id') == room_id), None)
            if row is None or _qwen35_reference_key(str(row.get('room_type') or '')) == _qwen35_reference_key(target):
                matching_scope_ids.append(room_id)
        if len(matching_scope_ids) == 1:
            target = matching_scope_ids[0]
    history = policy_context.get('tool_history')
    history = history if isinstance(history, list) else []
    if any((call is not None and call['tool_name'] == 'navigate_to_room' and (_qwen35_reference_key(call['arguments'].get('room', call['arguments'].get('target'))) == _qwen35_reference_key(target)) for call in (_history_tool_call(item) for item in history))):
        return None
    return target

def _qwen35_v15_category_action_requested(policy_context: Mapping[str, Any], instruction: str, category: str, tool_name: str) -> bool:
    from scoper.harness.binding import instruction_scene_reference_category
    history = policy_context.get('tool_history')
    if not isinstance(history, list):
        return False
    for raw in history:
        call = _history_tool_call(raw)
        if call is None or call['tool_name'] != tool_name:
            continue
        binding = raw.get('binding') if isinstance(raw, Mapping) else None
        refs = (call['arguments'].get('target'), binding.get('canonical_public_ref') if isinstance(binding, Mapping) else None)
        if not any((isinstance(ref, str) and instruction_scene_reference_category(ref, instruction) == category for ref in refs)):
            continue
        return True
    return False

def _qwen35_v15_observation_resolved(policy_context: Mapping[str, Any], instruction: str, category: str, tool_name: str, relation: str, expected_value: Any) -> bool:
    from scoper.harness.binding import instruction_scene_reference_category
    assumed = policy_context.get('assumed_completed_actions', [])
    if isinstance(assumed, list) and any((call['tool_name'] == tool_name and instruction_scene_reference_category(call['arguments'].get('target'), instruction) == category for call in _assumed_tool_calls_in_order(assumed))):
        return True
    facts = policy_context.get('effective_facts')
    if not isinstance(facts, list):
        return False
    for fact in facts:
        if not (isinstance(fact, Mapping) and fact.get('certainty') == 'verified' and (fact.get('status', 'current') == 'current') and (fact.get('relation_or_property') == relation) and (fact.get('object_public_role_or_value') == expected_value)):
            continue
        subject = fact.get('subject_public_role')
        if isinstance(subject, str) and instruction_scene_reference_category(subject, instruction) == category or _qwen35_v15_category_action_requested(policy_context, instruction, category, tool_name):
            return True
    return False

def _qwen35_v15_actionable_observation(policy_context: Mapping[str, Any], instruction: str, allowed_tools: set[Any]) -> tuple[str, str, str] | None:
    from scoper.harness.binding import instruction_scene_reference_category, normalize_reference
    repairs = (('faucet', 'toggle_off', ('running', 'water stream')), ('storage container', 'close', ('open', 'ajar')))
    for category, tool_name, state_cues in repairs:
        if tool_name not in allowed_tools:
            continue
        for fact in _qwen35_v15_scene_observations(policy_context):
            value = normalize_reference(fact.get('object_public_role_or_value'))
            if instruction_scene_reference_category(fact['subject_public_role'], instruction) == category and any((cue in value for cue in state_cues)) and (not _qwen35_v15_observation_resolved(policy_context, instruction, category, tool_name, 'toggled_on' if category == 'faucet' else 'open', False)):
                return (fact['subject_public_role'], tool_name, category)
    return None

def _qwen35_v15_closed_storage_observation(policy_context: Mapping[str, Any], instruction: str) -> tuple[str, str] | None:
    from scoper.harness.binding import instruction_scene_reference_category, normalize_reference
    for fact in _qwen35_v15_scene_observations(policy_context):
        category = instruction_scene_reference_category(fact['subject_public_role'], instruction)
        value = normalize_reference(fact.get('object_public_role_or_value'))
        if category in {'storage', 'storage container'} and any((cue in value for cue in ('closed', 'shut'))) and _qwen35_v15_category_action_requested(policy_context, instruction, category, 'navigate_to') and (not _qwen35_v15_observation_resolved(policy_context, instruction, category, 'open', 'open', True)):
            return (fact['subject_public_role'], category)
    return None

def qwen35_policy_semantic_retry_v14_payload(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> Mapping[str, Any] | None:
    base = qwen35_policy_semantic_retry_v13_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    if base is not None or policy_context.get('mode') == 'recovery':
        return base
    call = _public_tool_call(rejected_tool_call)
    instruction = policy_context.get('instruction')
    target = None if call is None else call['arguments'].get('target')
    if call is None or not isinstance(instruction, str) or (call['tool_name'] not in {'scan', 'stop'} and (not isinstance(target, str))):
        return None
    observations = _qwen35_v15_scene_observations(policy_context)
    history = policy_context.get('tool_history')
    history = history if isinstance(history, list) else []
    target_key = _qwen35_scene_description_key(target or '')
    description_target = any((isinstance(fact.get('object_public_role_or_value'), str) and _qwen35_scene_description_key(fact['object_public_role_or_value']) == target_key for fact in observations))
    from scoper.harness.binding import instruction_scene_reference_category, instruction_scene_reference_relevant
    category = instruction_scene_reference_category(target, instruction)
    concrete_refs = sorted({fact['subject_public_role'] for fact in observations if (instruction_scene_reference_category(fact['subject_public_role'], instruction) == category if category is not None else instruction_scene_reference_relevant(fact['subject_public_role'], instruction)) and _qwen35_entity_reference_key(fact['subject_public_role']) != _qwen35_entity_reference_key(target or '')}, key=str.encode)
    broad_manipulation = bool(category is not None and concrete_refs and (call['tool_name'] in {'clean', 'close', 'cool', 'fill', 'heat', 'open', 'pick_up', 'put_down', 'saturate', 'slice', 'toggle_off', 'toggle_on', 'wash'}))
    allowed_tools = {item.get('name') for item in policy_context.get('allowed_tool_specs', []) if isinstance(item, Mapping)}
    room_target = _qwen35_v15_instruction_room_target(policy_context, instruction) if call['tool_name'] == 'scan' and 'navigate_to_room' in allowed_tools else None
    missing_queries = _qwen35_v15_discovery_queries(policy_context, instruction)
    repeated_scan_query = missing_queries[0] if call['tool_name'] == 'scan' and 'look_for' in allowed_tools and any((history_call is not None and history_call['tool_name'] == 'scan' for history_call in (_history_tool_call(item) for item in history))) and missing_queries else None
    premature_stop = bool(call['tool_name'] == 'stop' and 'look_for' in allowed_tools and missing_queries)
    storage_misplacement = bool(call['tool_name'] == 'put_down' and 'storage' in missing_queries and (instruction_scene_reference_category(target, instruction) != 'storage') and ('navigate_to' in allowed_tools))
    feedback = qwen35_projected_public_transition_feedback(public_transition_feedback_or_null)
    binding = feedback.get('binding_admission_or_null') if isinstance(feedback, Mapping) else None
    bound_ref = binding.get('canonical_public_ref') if isinstance(binding, Mapping) else None
    feedback_call = feedback.get('tool_call') if isinstance(feedback, Mapping) else None
    failed_target = feedback_call.get('arguments', {}).get('target') if isinstance(feedback_call, Mapping) else None
    failed_category = instruction_scene_reference_category(failed_target, instruction)
    actionable = _qwen35_v15_actionable_observation(policy_context, instruction, allowed_tools)
    actionable_stop = actionable if call['tool_name'] == 'stop' else None
    closed_storage = _qwen35_v15_closed_storage_observation(policy_context, instruction) if call['tool_name'] == 'put_down' and 'open' in allowed_tools else None
    repair_after_inspection = bool(actionable is not None and call['tool_name'] != actionable[1] and isinstance(binding, Mapping) and (binding.get('status') == 'success') and isinstance(feedback_call, Mapping) and (feedback_call.get('tool_name') == 'look_for') and (instruction_scene_reference_category(failed_target, instruction) == actionable[2]))
    actionable_diversion = actionable if actionable is not None and (not repair_after_inspection) and (call['tool_name'] not in {actionable[1], 'navigate_to'}) and (not (call['tool_name'] == 'look_for' and instruction_scene_reference_category(target, instruction) == actionable[2])) else None
    binding_fallback = bool(isinstance(binding, Mapping) and binding.get('status') != 'success' and (failed_category in {'storage', 'storage container'}) and isinstance(target, str) and (instruction_scene_reference_category(target, instruction) == failed_category) and ('navigate_to' in allowed_tools))
    focus_mismatch = bool(isinstance(binding, Mapping) and binding.get('status') == 'success' and isinstance(bound_ref, str) and isinstance(target, str) and (call['tool_name'] in {'clean', 'close', 'fill', 'open', 'put_down', 'toggle_off', 'toggle_on'}) and (instruction_scene_reference_category(target, instruction) == instruction_scene_reference_category(bound_ref, instruction)) and (_qwen35_entity_reference_key(target) != _qwen35_entity_reference_key(bound_ref)))
    if not any((description_target, broad_manipulation, room_target is not None, repeated_scan_query is not None, premature_stop, storage_misplacement, repair_after_inspection, actionable_stop is not None, actionable_diversion is not None, closed_storage is not None, binding_fallback, focus_mismatch)):
        return None
    preferred = next((ref for ref in concrete_refs if isinstance(bound_ref, str) and _qwen35_entity_reference_key(ref) == _qwen35_entity_reference_key(bound_ref)), concrete_refs[0] if len(concrete_refs) == 1 else None)
    required_call = None
    if room_target is not None:
        required_call = {'tool_name': 'navigate_to_room', 'arguments': {'room': room_target}}
    elif repair_after_inspection:
        required_call = {'tool_name': actionable[1], 'arguments': {'target': actionable[0]}}
    elif actionable_stop is not None or actionable_diversion is not None:
        pending_actionable = actionable_stop or actionable_diversion
        required_call = {'tool_name': pending_actionable[1] if _qwen35_v15_category_action_requested(policy_context, instruction, pending_actionable[2], 'navigate_to') or 'navigate_to' not in allowed_tools else 'navigate_to', 'arguments': {'target': pending_actionable[0]}}
    elif repeated_scan_query is not None:
        required_call = {'tool_name': 'look_for', 'arguments': {'target': repeated_scan_query}}
    elif closed_storage is not None:
        required_call = {'tool_name': 'open', 'arguments': {'target': target}}
    elif premature_stop:
        required_call = {'tool_name': 'look_for', 'arguments': {'target': missing_queries[0]}}
    elif storage_misplacement:
        required_call = {'tool_name': 'navigate_to', 'arguments': {'target': 'storage'}}
    elif binding_fallback:
        required_call = {'tool_name': 'navigate_to', 'arguments': {'target': failed_category}}
    elif focus_mismatch:
        required_call = {'tool_name': call['tool_name'], 'arguments': {**call['arguments'], 'target': bound_ref}}
    elif broad_manipulation and preferred is not None:
        required_call = {'tool_name': call['tool_name'], 'arguments': {**call['arguments'], 'target': preferred}}
    payload = {'attempt': 2, 'guidance': 'Enter the single room explicitly named by the instruction before the first broad scan.' if room_target is not None else 'An instruction-relevant abnormal scene observation remains unresolved. Navigate to its concrete public entity when needed, apply the matching corrective tool, and require verified effective facts before stopping.' if actionable_stop is not None or actionable_diversion is not None else 'A broad scan was already completed. Inspect the next unresolved instruction condition with targeted look_for instead of repeating scan.' if repeated_scan_query is not None else 'The observed storage destination appears closed. Navigate to it when needed and open it before put_down.' if closed_storage is not None else 'The previous targeted look_for located an entity whose scene observation states an instruction-relevant abnormal condition. Apply the matching corrective tool before inspecting another condition.' if repair_after_inspection else 'Inspect the unresolved instruction condition with the required targeted look_for before stopping.' if premature_stop else "Navigate to the instruction's storage destination before placement; do not place the object on an unrelated surface." if storage_misplacement else 'The concrete scene-observation reference failed public Binding. Navigate with its instruction-level storage query instead.' if binding_fallback else 'Use the concrete public reference from the latest successful Binding instead of a different scene-observation name.' if focus_mismatch else 'A scene_observation object_public_role_or_value is descriptive text, not an entity reference. Choose a concrete destination from the instruction; if it has not been discovered, inspect it with look_for or navigate_to before manipulation.' if description_target else 'Replace the broad manipulation target with a concrete observed entity. The public scene-observation candidates are: {}.'.format(', '.join(concrete_refs)), 'held_public_refs': [], 'instruction': instruction, 'max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS, 'rejected_reason': 'instruction_room_entry_before_scan' if room_target is not None else 'targeted_condition_search_after_broad_scan' if repeated_scan_query is not None else 'actionable_scene_observation_not_repaired' if repair_after_inspection else 'actionable_scene_observation_before_stop' if actionable_stop is not None else 'actionable_scene_observation_before_more_exploration' if actionable_diversion is not None else 'closed_storage_prerequisite_before_put_down' if closed_storage is not None else 'unresolved_instruction_condition_before_stop' if premature_stop else 'placement_target_is_not_instruction_storage' if storage_misplacement else 'scene_observation_reference_binding_failed' if binding_fallback else 'scene_observation_reference_differs_from_bound_focus' if focus_mismatch else 'scene_observation_description_is_not_target' if description_target else 'broad_category_has_concrete_scene_target', 'rejected_tool_call': call, 'required_tool_call_or_null': required_call, 'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V14_MESSAGE_SCHEMA_VERSION, 'stop_rejected_missing_tools': _qwen35_stop_missing_tools(public_transition_feedback_or_null)}
    validate_json_value(payload, json_pointer='/qwen_policy_semantic_retry_v14')
    return payload

def qwen35_policy_semantic_retry_v14_message_bytes(policy_context: Mapping[str, Any], rejected_tool_call: Mapping[str, Any], *, public_transition_feedback_or_null: Mapping[str, Any] | None=None, public_observation_or_null: Mapping[str, Any] | None=None) -> bytes | None:
    payload = qwen35_policy_semantic_retry_v14_payload(policy_context, rejected_tool_call, public_transition_feedback_or_null=public_transition_feedback_or_null, public_observation_or_null=public_observation_or_null)
    return None if payload is None else canonical_json_bytes(payload)

def qwen35_policy_semantic_retry_v11_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    base = hardqa_policy_system_bytes_for_harness_version(harness_version)
    effective = qwen35_policy_d5a_system_bytes_for_harness_version(harness_version)
    descriptor = dict(qwen35_policy_guidance_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V11_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V11_ADAPTER_ID, 'suffix_sha256': sha256_bytes(QWEN35_POLICY_D4_GUIDANCE_SUFFIX_BYTES), 'effective_prompt_sha256': sha256_bytes(effective), 'action_hint_schema_version': QWEN35_POLICY_ACTION_HINT_SCHEMA_VERSION, 'action_hint_serializer_sha256': sha256_bytes(QWEN35_POLICY_ACTION_HINT_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V11_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V11_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V11_SERIALIZER_IDENTITY_BYTES), 'semantic_retry_max_attempts': QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS})
    if descriptor['base_prompt_sha256'] != sha256_bytes(base):
        raise AssertionError('Qwen D5-A Harness base prompt differs')
    return descriptor

def qwen35_memory_system_bytes_for_harness_version(harness_version: str) -> bytes:
    if str(harness_version) == '17':
        return qwen35_memory_v5_system_bytes_for_harness_version(harness_version)
    return hardqa_system_bytes_for_harness_version('memory', harness_version) + QWEN35_MEMORY_PROMPT_SUFFIX_BYTES

def qwen35_memory_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    if str(harness_version) == '17':
        return qwen35_memory_v5_prompt_adapter_descriptor(harness_version)
    base = hardqa_system_bytes_for_harness_version('memory', harness_version)
    effective = qwen35_memory_system_bytes_for_harness_version(harness_version)
    return {'schema_version': QWEN35_MEMORY_PROMPT_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_MEMORY_PROMPT_ADAPTER_ID, 'component': 'memory', 'base_prompt_sha256': sha256_bytes(base), 'suffix_sha256': sha256_bytes(QWEN35_MEMORY_PROMPT_SUFFIX_BYTES), 'effective_prompt_sha256': sha256_bytes(effective)}

def qwen35_memory_v2_system_bytes_for_harness_version(harness_version: str) -> bytes:
    return hardqa_system_bytes_for_harness_version('memory', harness_version) + QWEN35_MEMORY_V2_PROMPT_SUFFIX_BYTES

def qwen35_memory_v2_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    base = hardqa_system_bytes_for_harness_version('memory', harness_version)
    effective = qwen35_memory_v2_system_bytes_for_harness_version(harness_version)
    return {'schema_version': QWEN35_MEMORY_V2_PROMPT_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_MEMORY_V2_PROMPT_ADAPTER_ID, 'component': 'memory', 'base_prompt_sha256': sha256_bytes(base), 'suffix_sha256': sha256_bytes(QWEN35_MEMORY_V2_PROMPT_SUFFIX_BYTES), 'effective_prompt_sha256': sha256_bytes(effective)}

def qwen35_memory_v3_system_bytes_for_harness_version(harness_version: str) -> bytes:
    if str(harness_version) != '14':
        raise ValueError('Qwen instruction-grounded memory requires Harness v14')
    return hardqa_system_bytes_for_harness_version('memory', harness_version) + QWEN35_MEMORY_V3_PROMPT_SUFFIX_BYTES

def qwen35_memory_v3_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    base = hardqa_system_bytes_for_harness_version('memory', harness_version)
    effective = qwen35_memory_v3_system_bytes_for_harness_version(harness_version)
    return {'schema_version': QWEN35_MEMORY_V3_PROMPT_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_MEMORY_V3_PROMPT_ADAPTER_ID, 'component': 'memory', 'base_prompt_sha256': sha256_bytes(base), 'suffix_sha256': sha256_bytes(QWEN35_MEMORY_V3_PROMPT_SUFFIX_BYTES), 'effective_prompt_sha256': sha256_bytes(effective), 'extension_profile': INSTRUCTION_GROUNDED_MEMORY_PROFILE_ID}

def qwen35_memory_v4_system_bytes_for_harness_version(harness_version: str) -> bytes:
    if str(harness_version) != '15':
        raise ValueError('Qwen actionable discovery memory requires Harness v15')
    return hardqa_system_bytes_for_harness_version('memory', harness_version) + QWEN35_MEMORY_V4_PROMPT_SUFFIX_BYTES

def qwen35_memory_v4_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    base = hardqa_system_bytes_for_harness_version('memory', harness_version)
    effective = qwen35_memory_v4_system_bytes_for_harness_version(harness_version)
    return {'schema_version': QWEN35_MEMORY_V4_PROMPT_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_MEMORY_V4_PROMPT_ADAPTER_ID, 'component': 'memory', 'base_prompt_sha256': sha256_bytes(base), 'suffix_sha256': sha256_bytes(QWEN35_MEMORY_V4_PROMPT_SUFFIX_BYTES), 'effective_prompt_sha256': sha256_bytes(effective)}

def qwen35_memory_v5_system_bytes_for_harness_version(harness_version: str) -> bytes:
    return hardqa_system_bytes_for_harness_version('memory', harness_version) + QWEN35_MEMORY_V5_PROMPT_SUFFIX_BYTES

def qwen35_memory_v5_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    base = hardqa_system_bytes_for_harness_version('memory', harness_version)
    effective = qwen35_memory_v5_system_bytes_for_harness_version(harness_version)
    return {'schema_version': QWEN35_MEMORY_V5_PROMPT_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_MEMORY_V5_PROMPT_ADAPTER_ID, 'component': 'memory', 'base_prompt_sha256': sha256_bytes(base), 'suffix_sha256': sha256_bytes(QWEN35_MEMORY_V5_PROMPT_SUFFIX_BYTES), 'effective_prompt_sha256': sha256_bytes(effective)}

def qwen35_policy_semantic_retry_v12_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    if str(harness_version) != '14':
        raise ValueError('Qwen semantic-retry v12 requires Harness v14')
    descriptor = dict(qwen35_policy_semantic_retry_v11_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V12_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V12_ADAPTER_ID, 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V12_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V12_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V12_SERIALIZER_IDENTITY_BYTES)})
    return descriptor

def qwen35_policy_semantic_retry_v13_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    if str(harness_version) != '14':
        raise ValueError('Qwen semantic-retry v13 requires Harness v14')
    descriptor = dict(qwen35_policy_semantic_retry_v12_prompt_adapter_descriptor(harness_version))
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V13_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V13_ADAPTER_ID, 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V13_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V13_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V13_SERIALIZER_IDENTITY_BYTES)})
    return descriptor

def qwen35_policy_semantic_retry_v14_prompt_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    if str(harness_version) != '15':
        raise ValueError('Qwen semantic-retry v14 requires Harness v15')
    descriptor = dict(qwen35_policy_semantic_retry_v13_prompt_adapter_descriptor('14'))
    base = hardqa_policy_system_bytes_for_harness_version(harness_version)
    effective = qwen35_policy_d5a_system_bytes_for_harness_version(harness_version)
    descriptor.update({'schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V14_ADAPTER_SCHEMA_VERSION, 'adapter_id': QWEN35_POLICY_SEMANTIC_RETRY_V14_ADAPTER_ID, 'base_prompt_sha256': sha256_bytes(base), 'effective_prompt_sha256': sha256_bytes(effective), 'semantic_retry_message_schema_version': QWEN35_POLICY_SEMANTIC_RETRY_V14_MESSAGE_SCHEMA_VERSION, 'semantic_retry_detector_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V14_DETECTOR_IDENTITY_BYTES), 'semantic_retry_serializer_sha256': sha256_bytes(QWEN35_POLICY_SEMANTIC_RETRY_V14_SERIALIZER_IDENTITY_BYTES)})
    return descriptor

def policy_decision_check_adapter_descriptor(harness_version: str) -> Mapping[str, Any]:
    base = hardqa_policy_system_bytes_for_harness_version(harness_version)
    return {'schema_version': POLICY_DECISION_CHECK_ADAPTER_SCHEMA_VERSION, 'adapter_id': POLICY_DECISION_CHECK_ADAPTER_ID, 'component': 'policy', 'base_prompt_sha256': sha256_bytes(base), 'footer_prefix_sha256': sha256_bytes(POLICY_DECISION_CHECK_PREFIX_BYTES), 'footer_sha256': sha256_bytes(POLICY_DECISION_CHECK_BYTES), 'serializer_sha256': sha256_bytes(POLICY_DECISION_CHECK_SERIALIZER_IDENTITY_BYTES)}

def policy_decision_check_user_bytes(canonical_policy_context_bytes: bytes) -> bytes:
    if not isinstance(canonical_policy_context_bytes, bytes):
        raise ValueError('policy context bytes must be bytes')
    context = strict_json_loads(canonical_policy_context_bytes, source='canonical policy context')
    if not isinstance(context, Mapping) or canonical_json_bytes(context) != canonical_policy_context_bytes:
        raise ValueError('policy context bytes are not one canonical JSON object')
    return canonical_policy_context_bytes + b'\n' + POLICY_DECISION_CHECK_PREFIX_BYTES + POLICY_DECISION_CHECK_BYTES

def hardqa_prompt_contract_descriptor_for_harness_version(harness_version: str, *, history_limit: int=16) -> Mapping[str, Any]:
    if harness_version == 'scoper-1':
        history_limit = 48
        from scoper.harness import prompts
        descriptor = hardqa_prompt_contract_descriptor(history_limit=history_limit, policy_system_bytes=prompts.POLICY_SYSTEM_BYTES, prompt_contract_version='method-text-hardqa-prompt-contract.v21', verifier_system_bytes=prompts.VERIFIER_SYSTEM_BYTES, memory_system_bytes=prompts.MEMORY_SYSTEM_BYTES)
        descriptor = dict(descriptor)
        descriptor['agenda_system_sha256'] = sha256_bytes(prompts.AGENDA_SYSTEM_BYTES)
        descriptor['answer_system_sha256'] = sha256_bytes(prompts.ANSWER_SYSTEM_BYTES)
        descriptor['context_limits'] = dict(prompts.CONTEXT_LIMITS)
        descriptor['verification_thresholds'] = dict(prompts.VERIFICATION_THRESHOLDS)
        descriptor['recovery_limits'] = dict(prompts.RECOVERY_LIMITS)
        return descriptor
    if harness_version == '20':
        from scoper.harness import visual_prompts
        return hardqa_prompt_contract_descriptor(history_limit=history_limit, policy_system_bytes=visual_prompts.POLICY_SYSTEM_BYTES, prompt_contract_version='method-text-hardqa-prompt-contract.v20', verifier_system_bytes=visual_prompts.VERIFIER_SYSTEM_BYTES, memory_system_bytes=visual_prompts.MEMORY_SYSTEM_BYTES)
    if harness_version == '19':
        from scoper.harness import recovery_prompts
        return hardqa_prompt_contract_descriptor(history_limit=history_limit, policy_system_bytes=recovery_prompts.POLICY_SYSTEM_BYTES, prompt_contract_version='method-text-hardqa-prompt-contract.v19', verifier_system_bytes=recovery_prompts.VERIFIER_SYSTEM_BYTES, memory_system_bytes=recovery_prompts.MEMORY_SYSTEM_BYTES)
    if harness_version == '18':
        from scoper.harness import control_prompts
        return hardqa_prompt_contract_descriptor(history_limit=history_limit, policy_system_bytes=control_prompts.POLICY_SYSTEM_BYTES, prompt_contract_version='method-text-hardqa-prompt-contract.v18', verifier_system_bytes=control_prompts.VERIFIER_SYSTEM_BYTES, memory_system_bytes=control_prompts.MEMORY_SYSTEM_BYTES)
    if harness_version == '16':
        return hardqa_prompt_contract_descriptor(history_limit=history_limit, policy_system_bytes=POLICY_SYSTEM_V16_BYTES, prompt_contract_version=HARDQA_PROMPT_V16_CONTRACT_VERSION, verifier_system_bytes=VERIFIER_SYSTEM_V13_BYTES, memory_system_bytes=MEMORY_SYSTEM_V16_BYTES)
    if harness_version == '17':
        return hardqa_prompt_contract_descriptor(history_limit=history_limit, policy_system_bytes=POLICY_SYSTEM_V16_BYTES, prompt_contract_version=HARDQA_PROMPT_V17_CONTRACT_VERSION, verifier_system_bytes=VERIFIER_SYSTEM_V13_BYTES, memory_system_bytes=MEMORY_SYSTEM_V17_BYTES)
    if harness_version == '15':
        return hardqa_prompt_contract_descriptor(history_limit=history_limit, policy_system_bytes=POLICY_SYSTEM_V15_BYTES, prompt_contract_version=HARDQA_PROMPT_V15_CONTRACT_VERSION, verifier_system_bytes=VERIFIER_SYSTEM_V13_BYTES, memory_system_bytes=MEMORY_SYSTEM_V15_BYTES)
    if harness_version == '14':
        return hardqa_prompt_contract_descriptor(history_limit=history_limit, policy_system_bytes=POLICY_SYSTEM_V14_BYTES, prompt_contract_version=HARDQA_PROMPT_V14_CONTRACT_VERSION, verifier_system_bytes=VERIFIER_SYSTEM_V13_BYTES, memory_system_bytes=MEMORY_SYSTEM_V14_BYTES)
    if harness_version == '13':
        return hardqa_prompt_contract_descriptor(history_limit=history_limit, policy_system_bytes=POLICY_SYSTEM_V13_BYTES, prompt_contract_version=HARDQA_PROMPT_V13_CONTRACT_VERSION, verifier_system_bytes=VERIFIER_SYSTEM_V13_BYTES, memory_system_bytes=MEMORY_SYSTEM_BYTES)
    return hardqa_prompt_contract_descriptor(history_limit=history_limit)
HARDQA_PROMPT_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor()
HARDQA_PROMPT_CONTRACT_SHA256 = sha256_json(HARDQA_PROMPT_CONTRACT_IDENTITY)
HARDQA_PROMPT_V13_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor_for_harness_version('13')
HARDQA_PROMPT_V14_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor_for_harness_version('14')
HARDQA_PROMPT_V15_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor_for_harness_version('15')
HARDQA_PROMPT_V16_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor_for_harness_version('16')
HARDQA_PROMPT_V17_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor_for_harness_version('17')
HARDQA_PROMPT_V11_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor(prompt_contract_version=HARDQA_PROMPT_V11_CONTRACT_VERSION, verifier_system_bytes=VERIFIER_V11_SYSTEM_BYTES, memory_system_bytes=MEMORY_V11_SYSTEM_BYTES)
HARDQA_PROMPT_V10_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor(policy_system_bytes=POLICY_SYSTEM_V10_BYTES, prompt_contract_version=HARDQA_PROMPT_V10_CONTRACT_VERSION, verifier_system_bytes=VERIFIER_V10_SYSTEM_BYTES, memory_system_bytes=MEMORY_V10_SYSTEM_BYTES)
HARDQA_PROMPT_V9_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor(policy_system_bytes=POLICY_SYSTEM_V9_BYTES, prompt_contract_version=HARDQA_PROMPT_V9_CONTRACT_VERSION, verifier_system_bytes=VERIFIER_V9_SYSTEM_BYTES, memory_system_bytes=MEMORY_V9_SYSTEM_BYTES)
LEGACY_HARDQA_PROMPT_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor(policy_system_bytes=POLICY_SYSTEM_V2_BYTES, prompt_contract_version=LEGACY_HARDQA_PROMPT_CONTRACT_VERSION)
HARDQA_PROMPT_V3_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor(policy_system_bytes=POLICY_SYSTEM_V3_BYTES, prompt_contract_version=HARDQA_PROMPT_V3_CONTRACT_VERSION)
HARDQA_PROMPT_V4_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor(policy_system_bytes=POLICY_SYSTEM_V4_BYTES, prompt_contract_version=HARDQA_PROMPT_V4_CONTRACT_VERSION)
HARDQA_PROMPT_V5_CONTRACT_IDENTITY = hardqa_prompt_contract_descriptor(policy_system_bytes=POLICY_SYSTEM_V5_BYTES, prompt_contract_version=HARDQA_PROMPT_V5_CONTRACT_VERSION)
__all__ = ['HARDQA_IMAGE_ROLE_POLICY', 'HARDQA_IMAGE_ROLE_POLICY_V14', 'HARDQA_IMAGE_ROLE_POLICY_V16', 'HARDQA_IMAGE_ROLE_POLICY_V17', 'HARDQA_IMAGE_ROLE_POLICY_V11', 'HARDQA_MEMORY_PROJECTED_FIELDS', 'HARDQA_MEMORY_V14_PROJECTED_FIELDS', 'HARDQA_MEMORY_V16_PROJECTED_FIELDS', 'HARDQA_MESSAGE_LAYOUT', 'HARDQA_PROMPT_BUILDER', 'HARDQA_PROMPT_CONTRACT_IDENTITY', 'HARDQA_PROMPT_CONTRACT_SHA256', 'HARDQA_PROMPT_CONTRACT_VERSION', 'HARDQA_PROMPT_V3_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V3_CONTRACT_VERSION', 'HARDQA_PROMPT_V4_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V4_CONTRACT_VERSION', 'HARDQA_PROMPT_V5_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V5_CONTRACT_VERSION', 'HARDQA_PROMPT_V9_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V9_CONTRACT_VERSION', 'HARDQA_PROMPT_V10_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V10_CONTRACT_VERSION', 'HARDQA_PROMPT_V11_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V11_CONTRACT_VERSION', 'HARDQA_PROMPT_V13_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V13_CONTRACT_VERSION', 'HARDQA_PROMPT_V14_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V14_CONTRACT_VERSION', 'HARDQA_PROMPT_V17_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V17_CONTRACT_VERSION', 'HARDQA_PROMPT_V15_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V15_CONTRACT_VERSION', 'HARDQA_PROMPT_V16_CONTRACT_IDENTITY', 'HARDQA_PROMPT_V16_CONTRACT_VERSION', 'HARDQA_PROMPT_ENVELOPE_VERSION', 'HARDQA_VERIFIER_PROJECTED_FIELDS', 'LEGACY_HARDQA_PROMPT_CONTRACT_IDENTITY', 'LEGACY_HARDQA_PROMPT_CONTRACT_VERSION', 'MEMORY_SYSTEM_BYTES', 'MEMORY_SYSTEM_V14_BYTES', 'MEMORY_SYSTEM_V15_BYTES', 'MEMORY_SYSTEM_V16_BYTES', 'MEMORY_V10_SYSTEM_BYTES', 'MEMORY_V11_SYSTEM_BYTES', 'MEMORY_V9_SYSTEM_BYTES', 'POLICY_SYSTEM_BYTES', 'POLICY_SYSTEM_V13_BYTES', 'POLICY_SYSTEM_V14_BYTES', 'POLICY_SYSTEM_V15_BYTES', 'POLICY_SYSTEM_V16_BYTES', 'POLICY_SYSTEM_V10_BYTES', 'POLICY_SYSTEM_V2_BYTES', 'POLICY_SYSTEM_V3_BYTES', 'POLICY_SYSTEM_V4_BYTES', 'POLICY_SYSTEM_V5_BYTES', 'POLICY_SYSTEM_V9_BYTES', 'POLICY_DECISION_CHECK_ADAPTER_ID', 'POLICY_DECISION_CHECK_ADAPTER_SCHEMA_VERSION', 'POLICY_DECISION_CHECK_BYTES', 'POLICY_DECISION_CHECK_PREFIX_BYTES', 'POLICY_DECISION_CHECK_SERIALIZER_IDENTITY_BYTES', 'QWEN35_POLICY_PROMPT_ADAPTER_ID', 'QWEN35_POLICY_PROMPT_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_PROMPT_SUFFIX_BYTES', 'QWEN35_MEMORY_PROMPT_ADAPTER_ID', 'QWEN35_MEMORY_PROMPT_ADAPTER_SCHEMA_VERSION', 'QWEN35_MEMORY_PROMPT_SUFFIX_BYTES', 'QWEN35_MEMORY_V2_PROMPT_ADAPTER_ID', 'QWEN35_MEMORY_V2_PROMPT_ADAPTER_SCHEMA_VERSION', 'QWEN35_MEMORY_V2_PROMPT_SUFFIX_BYTES', 'INSTRUCTION_GROUNDED_MEMORY_PROFILE_ID', 'QWEN35_MEMORY_V3_PROMPT_ADAPTER_ID', 'QWEN35_MEMORY_V3_PROMPT_ADAPTER_SCHEMA_VERSION', 'QWEN35_MEMORY_V3_PROMPT_SUFFIX_BYTES', 'QWEN35_MEMORY_V4_PROMPT_ADAPTER_ID', 'QWEN35_MEMORY_V4_PROMPT_ADAPTER_SCHEMA_VERSION', 'QWEN35_MEMORY_V4_PROMPT_SUFFIX_BYTES', 'QWEN35_POLICY_CONTEXT_FOOTER_PREFIX_BYTES', 'QWEN35_POLICY_CONTEXT_FOOTER_SCHEMA_VERSION', 'QWEN35_POLICY_CONTEXT_FOOTER_SERIALIZER_IDENTITY_BYTES', 'QWEN35_POLICY_GUIDANCE_ADAPTER_ID', 'QWEN35_POLICY_GUIDANCE_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_GUIDANCE_SUFFIX_BYTES', 'QWEN35_POLICY_D4_GUIDANCE_SUFFIX_BYTES', 'QWEN35_POLICY_EXPERIMENT_ADAPTER_ID', 'QWEN35_POLICY_EXPERIMENT_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_EXPERIMENT_CONTEXT_FOOTER_SCHEMA_VERSION', 'QWEN35_POLICY_EXPERIMENT_CONTEXT_FOOTER_SERIALIZER_IDENTITY_BYTES', 'QWEN35_POLICY_EXPERIMENT_PROFILES', 'QWEN35_POLICY_SEMANTIC_RETRY_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_MAX_ATTEMPTS', 'QWEN35_POLICY_SEMANTIC_RETRY_DETECTOR_IDENTITY_BYTES', 'QWEN35_POLICY_SEMANTIC_RETRY_SERIALIZER_IDENTITY_BYTES', 'QWEN35_POLICY_SEMANTIC_RETRY_V2_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V2_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V2_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V3_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V3_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V3_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V4_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V4_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V4_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V5_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V5_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V5_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V6_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V6_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V6_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V7_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V7_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V7_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V8_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V8_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V8_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V11_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V11_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V11_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V12_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V12_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V12_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V13_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V13_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V13_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V14_ADAPTER_ID', 'QWEN35_POLICY_SEMANTIC_RETRY_V14_ADAPTER_SCHEMA_VERSION', 'QWEN35_POLICY_SEMANTIC_RETRY_V14_MESSAGE_SCHEMA_VERSION', 'QWEN35_POLICY_ACTION_HINT_PREFIX_BYTES', 'QWEN35_POLICY_ACTION_HINT_SCHEMA_VERSION', 'QWEN35_POLICY_ACTION_HINT_SERIALIZER_IDENTITY_BYTES', 'QWEN35_POLICY_ACTION_PLAYBOOK_PREFIX_BYTES', 'QWEN35_POLICY_ACTION_PLAYBOOK_SCHEMA_VERSION', 'QWEN35_POLICY_ACTION_PLAYBOOK_SERIALIZER_IDENTITY_BYTES', 'VERIFIER_SYSTEM_BYTES', 'VERIFIER_SYSTEM_V13_BYTES', 'VERIFIER_V10_SYSTEM_BYTES', 'VERIFIER_V11_SYSTEM_BYTES', 'VERIFIER_V9_SYSTEM_BYTES', 'build_hardqa_envelope', 'hardqa_prompt_contract_descriptor', 'hardqa_prompt_contract_descriptor_for_harness_version', 'hardqa_prompt_exposure', 'hardqa_prompt_exposure_hash', 'hardqa_policy_system_bytes_for_harness_version', 'hardqa_system_bytes', 'hardqa_system_bytes_for_harness_version', 'project_hardqa_request', 'policy_decision_check_adapter_descriptor', 'policy_decision_check_user_bytes', 'qwen35_policy_prompt_adapter_descriptor', 'qwen35_policy_system_bytes_for_harness_version', 'qwen35_memory_prompt_adapter_descriptor', 'qwen35_memory_system_bytes_for_harness_version', 'qwen35_memory_v2_prompt_adapter_descriptor', 'qwen35_memory_v2_system_bytes_for_harness_version', 'qwen35_memory_v3_prompt_adapter_descriptor', 'qwen35_memory_v3_system_bytes_for_harness_version', 'qwen35_memory_v4_prompt_adapter_descriptor', 'qwen35_memory_v4_system_bytes_for_harness_version', 'qwen35_memory_v5_prompt_adapter_descriptor', 'qwen35_memory_v5_system_bytes_for_harness_version', 'QWEN35_MEMORY_V5_PROMPT_ADAPTER_ID', 'QWEN35_MEMORY_V5_PROMPT_ADAPTER_SCHEMA_VERSION', 'QWEN35_MEMORY_V5_PROMPT_SUFFIX_BYTES', 'qwen35_policy_context_footer_payload', 'qwen35_policy_guidance_prompt_adapter_descriptor', 'qwen35_policy_guidance_system_bytes_for_harness_version', 'qwen35_policy_d4_guidance_prompt_adapter_descriptor', 'qwen35_policy_d4_guidance_system_bytes_for_harness_version', 'qwen35_policy_d5a_action_hint_payload', 'qwen35_policy_d5a_system_bytes_for_harness_version', 'qwen35_policy_d5a_user_bytes', 'qwen35_policy_experiment_context_footer_payload', 'qwen35_policy_experiment_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_message_bytes', 'qwen35_policy_semantic_retry_payload', 'qwen35_policy_semantic_retry_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_reason', 'qwen35_policy_semantic_retry_v2_message_bytes', 'qwen35_policy_semantic_retry_v2_payload', 'qwen35_policy_semantic_retry_v2_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v3_message_bytes', 'qwen35_policy_semantic_retry_v3_payload', 'qwen35_policy_semantic_retry_v3_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v4_message_bytes', 'qwen35_policy_semantic_retry_v4_payload', 'qwen35_policy_semantic_retry_v4_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v5_message_bytes', 'qwen35_policy_semantic_retry_v5_payload', 'qwen35_policy_semantic_retry_v5_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v6_message_bytes', 'qwen35_policy_semantic_retry_v6_payload', 'qwen35_policy_semantic_retry_v6_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v6_user_bytes', 'qwen35_policy_semantic_retry_v7_message_bytes', 'qwen35_policy_semantic_retry_v7_payload', 'qwen35_policy_semantic_retry_v7_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v7_user_bytes', 'qwen35_policy_semantic_retry_v8_message_bytes', 'qwen35_policy_semantic_retry_v8_payload', 'qwen35_policy_semantic_retry_v8_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v8_user_bytes', 'qwen35_policy_semantic_retry_v9_message_bytes', 'qwen35_policy_semantic_retry_v9_payload', 'qwen35_policy_semantic_retry_v9_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v9_user_bytes', 'qwen35_policy_semantic_retry_v10_message_bytes', 'qwen35_policy_semantic_retry_v10_payload', 'qwen35_policy_semantic_retry_v10_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v10_user_bytes', 'qwen35_policy_semantic_retry_v11_message_bytes', 'qwen35_policy_semantic_retry_v11_payload', 'qwen35_policy_semantic_retry_v11_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v12_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v13_message_bytes', 'qwen35_policy_semantic_retry_v13_payload', 'qwen35_policy_semantic_retry_v13_prompt_adapter_descriptor', 'qwen35_policy_semantic_retry_v14_message_bytes', 'qwen35_policy_semantic_retry_v14_payload', 'qwen35_policy_semantic_retry_v14_prompt_adapter_descriptor', 'qwen35_policy_action_playbook_v3_user_bytes', 'qwen35_policy_action_playbook_payload', 'qwen35_policy_action_playbook_v2_payload', 'qwen35_policy_action_playbook_user_bytes', 'qwen35_policy_experiment_system_bytes_for_harness_version', 'qwen35_policy_experiment_user_bytes', 'qwen35_projected_public_transition_feedback', 'qwen35_public_transition_feedback', 'qwen35_policy_user_bytes']
