from scoper.harness.backends.base import EvaluationAdapter

class BEHAVIORAdapter(EvaluationAdapter):

    def __init__(self, **callbacks):
        super().__init__('behavior', **callbacks)
