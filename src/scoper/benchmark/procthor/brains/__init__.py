from scoper.benchmark.procthor.brains.base import BaseBrain, Brain, BrainError, BrainInput, ToolCall
from scoper.benchmark.procthor.brains.oracle import OracleBrain
from scoper.benchmark.procthor.brains.recovery_oracle import RecoveryOracleBrain
from scoper.benchmark.procthor.brains.scripted import ScriptedBrain
__all__ = ['BaseBrain', 'Brain', 'BrainError', 'BrainInput', 'OracleBrain', 'RecoveryOracleBrain', 'ScriptedBrain', 'ToolCall', 'VLMBrain']

def __getattr__(name):
    if name == 'VLMBrain':
        from scoper.agents.procthor import VLMBrain
        return VLMBrain
    raise AttributeError(name)
