from scoper.harness.runtime import ExecutionReceipt

class CallbackEnvironmentAdapter:

    def __init__(self, backend_name, *, execute, expected_effect, stop_and_freeze_score, submit_answer, poll_public_messages):
        if backend_name not in {'procthor', 'behavior'}:
            raise ValueError('backend must be procthor or behavior')
        self.backend_name = backend_name
        self._execute = execute
        self._expected_effect = expected_effect
        self._stop = stop_and_freeze_score
        self._answer = submit_answer
        self._poll = poll_public_messages
        self._stopped = False

    def expected_effect(self, call, public_memory):
        return self._expected_effect(call, public_memory)

    def execute(self, call):
        if self._stopped:
            raise RuntimeError('physical actions are forbidden after stop')
        receipt = self._execute(call)
        if not isinstance(receipt, ExecutionReceipt):
            raise TypeError('execute must explicitly project an ExecutionReceipt')
        return receipt

    def stop(self):
        if not self._stopped:
            self._stop()
            self._stopped = True

    def submit_answer(self, answer):
        return self._answer(answer)

    def poll_public_messages(self):
        return self._poll()
