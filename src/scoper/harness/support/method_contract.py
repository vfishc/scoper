from __future__ import annotations
METHOD_SAMPLE_VERSION = 'method-sample.v4'
SAMPLE_KINDS = ('verifier_transition', 'step_decision', 'recovery_decision', 'stream_gate', 'preference', 'memory_update', 'memory_reconcile', 'memory_route')
DOMAINS = ('wild', 'scoper.benchmark.behavior', 'eb_procthor')
OUTCOMES = ('success', 'failure', 'ambiguous', 'unknown')
OBSERVABILITY = ('observable', 'partially_observable', 'unobservable', 'unknown')
EFFECT_VOCABULARY = ('picked_up', 'released', 'relation_added_inside', 'relation_added_on_top', 'relation_removed_inside', 'relation_removed_on_top', 'opened', 'closed', 'toggled_on', 'toggled_off', 'cleaned', 'dirtied', 'sliced', 'heated', 'cooled', 'filled', 'saturated', 'relocated', 'no_effect')
PRIVATE_DENY_KEYS = ('object_id', 'objectId', 'private_id', 'role_map', 'role_bindings', 'goal_predicates', 'predicate_records', 'scheduler_truth', 'state_snapshot', 'normalized_state', 'private_pose', 'oracle_hint', 'repair_plan_template', 'env.core')

class MethodContractError(ValueError):

    def __init__(self, code: str, message: str, *, pointer: str | None=None) -> None:
        self.code = code
        self.pointer = pointer
        detail = f'{code}: {message}'
        if pointer is not None:
            detail = f'{detail} ({pointer})'
        super().__init__(detail)
__all__ = ['DOMAINS', 'EFFECT_VOCABULARY', 'METHOD_SAMPLE_VERSION', 'MethodContractError', 'OBSERVABILITY', 'OUTCOMES', 'PRIVATE_DENY_KEYS', 'SAMPLE_KINDS']
