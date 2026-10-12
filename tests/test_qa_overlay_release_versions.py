import copy
import json
from pathlib import Path
import pytest
from scoper.benchmark.common.situated_qa import validate_qa_overlay

def fixture():
    root=Path(__file__).resolve().parents[1]/'tests/fixtures/tasks/procthor'
    task=next(json.loads(p.read_text()) for p in root.glob('*.json') if json.loads(p.read_text()).get('qa_overlay'))
    return task,copy.deepcopy(task['qa_overlay'])

@pytest.mark.parametrize('version,count,valid',[('qa_overlay.v1',3,True),('qa_overlay.v1',4,False),('qa_overlay.v2',6,True),('qa_overlay.v2',7,False)])
def test_preserve_v1_and_support_extended_v2(version,count,valid):
    task,overlay=fixture()
    question=overlay['questions'][0]
    overlay={'schema_version':version,'questions':[dict(copy.deepcopy(question),id=f'q{i}') for i in range(count)]}
    if valid:
        assert validate_qa_overlay(overlay,role_names=list(task['roles']))['schema_version']==version
    else:
        with pytest.raises(ValueError):validate_qa_overlay(overlay,role_names=list(task['roles']))
