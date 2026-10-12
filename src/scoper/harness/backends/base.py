from scoper.harness.adapters import CallbackEnvironmentAdapter
from scoper.harness.evaluation import score_episode

class EvaluationAdapter(CallbackEnvironmentAdapter):

    def __init__(self, backend, *, execute, expected_effect, goal_predicates, submit_answer, poll_public_messages, scoring_counts, on_stop=lambda: None):
        self._goals = goal_predicates
        self._counts = scoring_counts
        self._frozen = None

        def stop_and_freeze():
            on_stop()
            self._frozen = tuple(goal_predicates())
        super().__init__(backend, execute=execute, expected_effect=expected_effect, stop_and_freeze_score=stop_and_freeze, submit_answer=submit_answer, poll_public_messages=poll_public_messages)

    def expected_effect(self, call, public_memory):
        effect = super().expected_effect(call, public_memory)
        if call['tool_name'] not in {'scan', 'look_for'} and effect is None:
            raise ValueError('navigation and manipulation require a public expected-effect specification')
        return effect

    def finalize(self):
        goals = self._frozen if self._frozen is not None else tuple(self._goals())
        return score_episode(stopped=self._stopped, goal_satisfied=goals, **self._counts())
