from types import SimpleNamespace
from scoper.benchmark.behavior.brains.base import BrainInput
from scoper.benchmark.behavior.brains.teachers.recover import RecoverTeacher

def _env():
    objects = {'food': SimpleNamespace(abilities=('pickupable',), states={}), 'fridge': SimpleNamespace(abilities=('openable',), states={'open': False}), 'faucet': SimpleNamespace(abilities=('toggleable',), states={'toggled_on': True}), 'dish': SimpleNamespace(abilities=('pickupable',), states={'dirty': True})}
    return SimpleNamespace(role_map={name: name for name in objects}, core=SimpleNamespace(objects=objects, ego_state=SimpleNamespace(held_object_name=None)))

def _next(teacher):
    return teacher.act(BrainInput(observation={}, prompt_context={}, last_transition=None, high_level_step=0))

def test_single_incident_visual_discovery_probes_primary_route_before_recovery():
    teacher = RecoverTeacher()
    teacher.on_episode_start(_env(), {'family_spec': {'recover': {'primary_task': {'object_role': 'food', 'receptacle_role': 'fridge'}, 'incident': {'type': 'faucet_on', 'object_role': 'faucet', 'urgency': 'immediate', 'activation': {'type': 'visual_discovery'}}}}})
    assert (_next(teacher).tool_name, _next(teacher).tool_name) == ('navigate_to', 'scan')

def test_multi_incident_visual_discovery_uses_explicit_nonvisual_work_item():
    teacher = RecoverTeacher()
    teacher.on_episode_start(_env(), {'family_spec': {'recover': {'incidents': [{'type': 'faucet_on', 'object_role': 'faucet', 'urgency': 'high', 'activation': {'type': 'visual_discovery'}}, {'type': 'dirty_dish', 'object_role': 'dish', 'urgency': 'normal'}]}}})
    first = _next(teacher)
    second = _next(teacher)
    assert first.tool_name == 'navigate_to'
    assert first.arguments == {'target': 'dish'}
    assert second.tool_name == 'scan'
