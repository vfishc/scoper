import sys
from pathlib import Path
import pytest
import scoper.benchmark.behavior
from scoper.benchmark.behavior.execution_core.api import make_core
from scoper.benchmark.behavior.execution_core.ego_agent import EgoPose
from scoper.benchmark.behavior.execution_core.navigation import NavigationConfig
from scoper.benchmark.behavior.execution_core.occupancy import AABBObject, OccupancyConfig

def build_kitchen_scene():
    return [AABBObject('floors_1', 'floors', (5.0, 3.0, -0.02), (10.0, 6.0, 0.04)), AABBObject('walls_1', 'walls', (-0.1, 3.0, 1.0), (0.2, 6.2, 2.0)), AABBObject('countertop.n.01_1', 'countertop', (3.2, 1.8, 0.5), (1.2, 2.0, 1.0), abilities=('support_surface',)), AABBObject('breakfast_table.n.01_1', 'breakfast_table', (5.3, 4.4, 0.5), (1.2, 1.8, 1.0), abilities=('support_surface',)), AABBObject('bowl.n.01_1', 'bowl', (3.2, 1.8, 1.06), (0.2, 0.2, 0.12), abilities=('fillable', 'pickupable'), states={'filled': {}}), AABBObject('electric_refrigerator.n.01_1', 'fridge', (8.0, 3.0, 1.0), (0.8, 0.8, 2.0), abilities=('openable',), states={'open': False}), AABBObject('stove.n.01_1', 'stove', (7.8, 4.7, 0.55), (1.0, 0.8, 1.1), abilities=('toggleable',), states={'toggled_on': False})]

def build_smoke_task_spec():
    return {'id': 'behavior_stage1_smoke_000001', 'version': '1.0', 'backend': 'behavior', 'category': 'static_simple', 'source': {'type': 'template', 'scene_model': 'Rs_int'}, 'instruction': 'Put the bowl on the breakfast table and open the fridge.', 'roles': {'target_object': {'synset': 'bowl.n.01', 'bddl_inst': 'bowl.n.01_1'}, 'target_receptacle': {'synset': 'breakfast_table.n.01', 'bddl_inst': 'breakfast_table.n.01_1'}, 'fridge': {'synset': 'electric_refrigerator.n.01', 'bddl_inst': 'electric_refrigerator.n.01_1'}}, 'init': [], 'goal': {'type': 'predicates', 'require_model_stop': False, 'bddl_goal': None, 'predicates': [{'type': 'object_on_receptacle', 'args': {'object': 'target_object', 'receptacle': 'target_receptacle'}}, {'type': 'object_open', 'args': {'object': 'fridge'}}]}}

@pytest.fixture
def core():
    return make_core(objects=build_kitchen_scene(), start_pose=EgoPose(1.0, 1.0, yaw=0.0), occupancy_config=OccupancyConfig(resolution=0.1, agent_radius=0.2, safety_margin=0.05, bounds=(-0.5, -0.5, 10.5, 6.5)), navigation_config=NavigationConfig(frame_spacing=0.2, candidate_count=192))

@pytest.fixture
def smoke_task_spec():
    return build_smoke_task_spec()

@pytest.fixture
def role_map(smoke_task_spec):
    return {role: spec['bddl_inst'] for role, spec in smoke_task_spec['roles'].items()}
import json
from scoper.benchmark.behavior.paths import data_dir

def build_stage2_task_spec():
    path = Path(__file__).parent / 'fixtures/task_2.json'
    return json.loads(path.read_text(encoding='utf-8'))['tasks'][0]

@pytest.fixture
def stage2_task_spec():
    return build_stage2_task_spec()

def build_stage3_task_spec():
    path = Path(__file__).parent / 'fixtures/task_3.json'
    return json.loads(path.read_text(encoding='utf-8'))['tasks'][0]

@pytest.fixture
def stage3_task_spec():
    return build_stage3_task_spec()

def build_stage5_tasks():
    path = Path(__file__).parent / 'fixtures/task_5.json'
    return json.loads(path.read_text(encoding='utf-8'))['tasks']

def stage5_task_by_id(task_id_substr: str):
    for t in build_stage5_tasks():
        if task_id_substr in t['id']:
            return t
    raise KeyError(task_id_substr)
