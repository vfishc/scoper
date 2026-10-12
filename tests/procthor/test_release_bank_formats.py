import json
import pytest
from scoper.benchmark.procthor.online_task_loader import load_task_bank
from scoper.benchmark.procthor.online_task_spec import TaskSpecValidationError

@pytest.mark.parametrize('shape', ['list', 'object', 'jsonl'])
def test_bank_formats_agree(tmp_path, shape):
    tasks=[{'id':'a'}, {'id':'b'}]
    path=tmp_path/('tasks.jsonl' if shape=='jsonl' else 'tasks.json')
    path.write_text('\n'.join(json.dumps(t) for t in tasks) if shape=='jsonl' else json.dumps(tasks if shape=='list' else {'tasks':tasks},indent=2))
    assert load_task_bank(str(path),validate=False)=={t['id']:t for t in tasks}

@pytest.mark.parametrize('value', [{'tasks':{}}, {'tasks':[1]}, {'unrelated':[]}])
def test_reject_malformed_bank(tmp_path,value):
    path=tmp_path/'tasks.json';path.write_text(json.dumps(value))
    with pytest.raises(TaskSpecValidationError):load_task_bank(str(path),validate=False)
