import hashlib
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
TEMPLATE = 'inspect_and_restore_storage_container_001'
CATEGORY = 'observation_conditioned_container_inspection'
DISHWARE = {'Cup', 'Mug', 'Bowl', 'Plate'}
INSTRUCTION = 'Inspect the cabinet. Open it to check its contents. If it contains dishware (cups, mugs, bowls, or plates), close it; otherwise, leave it open.'

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
