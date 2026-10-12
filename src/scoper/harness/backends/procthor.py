from scoper.harness.backends.base import EvaluationAdapter

class ProcTHORAdapter(EvaluationAdapter):

    def __init__(self, **callbacks):
        super().__init__('procthor', **callbacks)
