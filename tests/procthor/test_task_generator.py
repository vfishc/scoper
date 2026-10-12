import json
import os
import tempfile
import unittest
from scoper.benchmark.procthor.object_taxonomy import load_object_taxonomy
from scoper.benchmark.procthor.online_task_loader import load_task_bank
from scoper.benchmark.procthor.online_task_spec import validate_task_bank
from scoper.benchmark.procthor.task_generator import generate_candidate_tasks, load_scene_profiles, write_task_bank
from scoper.benchmark.procthor.task_templates import enumerate_role_bindings, load_task_template, load_task_templates
DATA_DIR = str(__import__('scoper.benchmark.procthor.paths', fromlist=['data_dir']).data_dir())
TAXONOMY_PATH = os.path.join(DATA_DIR, 'object_taxonomy.json')
TEMPLATE_DIR = os.path.join(DATA_DIR, 'task_templates')

def _obj(object_id, object_type, **kwargs):
    data = {'object_id': object_id, 'object_type': object_type, 'room_id': 'room_0', 'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'pickupable': False, 'receptacle': False, 'openable': False, 'toggleable': False, 'sliceable': False, 'dirtyable': False, 'breakable': False, 'cookable': False, 'parent_receptacles': [], 'visible_from_initial_pose': True, 'is_open': None, 'is_toggled': None}
    data.update(kwargs)
    return data

def _scene_profile(house_id='synthetic_house'):
    cabinet_id = 'Cabinet|+02.00|+00.90|+00.00'
    objects = [_obj('Apple|+00.00|+00.90|+00.00', 'Apple', pickupable=True, visible_from_initial_pose=True), _obj('Plate|+00.10|+00.90|+00.00', 'Plate', pickupable=True, receptacle=True, dirtyable=True, breakable=True, visible_from_initial_pose=True), _obj('Book|+00.20|+00.90|+00.00', 'Book', pickupable=True, visible_from_initial_pose=False), _obj('Bowl|+00.22|+00.90|+00.10', 'Bowl', pickupable=True, receptacle=True, dirtyable=True, breakable=True, visible_from_initial_pose=False), _obj('SoapBottle|+00.25|+00.90|+00.00', 'SoapBottle', pickupable=True, visible_from_initial_pose=True), _obj('Mug|+00.30|+00.90|+00.00', 'Mug', pickupable=True, visible_from_initial_pose=False, parent_receptacles=[cabinet_id]), _obj(cabinet_id, 'Cabinet', receptacle=True, openable=True, is_open=False), _obj('Fridge|+02.50|+00.90|+00.00', 'Fridge', receptacle=True, openable=True, is_open=False), _obj('Sink|+01.00|+00.90|+00.00', 'Sink', receptacle=True), _obj('SinkBasin|+01.05|+00.90|+00.00', 'SinkBasin', receptacle=True), _obj('CounterTop|+01.50|+00.90|+00.00', 'CounterTop', receptacle=True), _obj('Faucet|+01.10|+00.90|+00.00', 'Faucet', toggleable=True, is_toggled=False, visible_from_initial_pose=False, parent_receptacles=['SinkBasin|+01.05|+00.90|+00.00']), _obj('DeskLamp|+03.00|+00.90|+00.00', 'DeskLamp', toggleable=True, is_toggled=True, visible_from_initial_pose=True), _obj('Drawer|+02.70|+00.90|+00.00', 'Drawer', receptacle=True, openable=True, is_open=False, visible_from_initial_pose=False)]
    return {'house_id': house_id, 'source': 'procthor', 'profile_version': '0.1', 'split': 'train', 'house_path': '/tmp/{}.json'.format(house_id), 'rooms': [{'room_id': 'room_0', 'room_type': 'Kitchen'}], 'objects': objects, 'affordance_index': {'pickupable': [obj['object_id'] for obj in objects if obj.get('pickupable')], 'receptacle': [obj['object_id'] for obj in objects if obj.get('receptacle')], 'openable': [obj['object_id'] for obj in objects if obj.get('openable')], 'toggleable': [obj['object_id'] for obj in objects if obj.get('toggleable')], 'sliceable': [], 'dirtyable': [obj['object_id'] for obj in objects if obj.get('dirtyable')], 'breakable': [obj['object_id'] for obj in objects if obj.get('breakable')], 'cookable': [obj['object_id'] for obj in objects if obj.get('cookable')]}, 'object_type_index': {obj['object_type']: [candidate['object_id'] for candidate in objects if candidate['object_type'] == obj['object_type']] for obj in objects}, 'reachable': {'num_positions': 2, 'positions': [{'x': 0.0, 'y': 0.9, 'z': 0.0}, {'x': 0.25, 'y': 0.9, 'z': 0.0}, {'x': 0.5, 'y': 0.9, 'z': 0.0}, {'x': 0.75, 'y': 0.9, 'z': 0.0}, {'x': 1.0, 'y': 0.9, 'z': 0.0}, {'x': 1.25, 'y': 0.9, 'z': 0.0}, {'x': 1.5, 'y': 0.9, 'z': 0.0}], 'sampled_agent_poses': [{'position': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'rotation': {'y': 90.0}, 'horizon': 0.0}]}}

def _three_step_composite_template():
    return {'template_id': 'test_three_step_composite', 'template_version': '0.1', 'category': 'composite', 'requires': {'target_object': {'object_type_allowlist': ['Plate'], 'must_be_pickupable': True, 'must_be_dirtyable': True, 'same_room_as': 'storage_receptacle', 'different_from': ['sink_basin', 'faucet', 'storage_receptacle']}, 'sink_basin': {'object_type_allowlist': ['SinkBasin'], 'must_be_receptacle': True, 'same_room_as': 'storage_receptacle'}, 'faucet': {'object_type_allowlist': ['Faucet'], 'must_be_toggleable': True, 'parent_of_role_base': 'sink_basin'}, 'storage_receptacle': {'object_type_allowlist': ['Cabinet'], 'must_be_receptacle': True, 'same_room_as': 'sink_basin'}}, 'initial_state_constraints': [{'type': 'object_dirty', 'object_role': 'target_object'}], 'goal_predicates': [{'type': 'object_closed', 'object_role': 'storage_receptacle', 'owner_subtask_id': 'finish', 'include_if': {'role': 'storage_receptacle', 'field': 'openable', 'equals': True}}], 'goal_options': {'require_model_stop': True}, 'instruction_templates': ['Wash, store, then close the {storage_receptacle}.'], 'expected_events': ['three_step_composite'], 'expected_brain_behaviors': ['sequential_composition'], 'family_spec_template': {'family': 'composite', 'scope': {'type': 'single_room', 'label': 'kitchen', 'room_ids_from_role': 'storage_receptacle'}, 'composite': {'ordering': 'sequential', 'subtasks': [{'subtask_id': 'wash', 'category': 'prepare_multi_stage', 'instruction_fragment': 'Wash the plate.', 'family_spec': {'family': 'prepare_multi_stage', 'scope': {'type': 'single_room', 'label': 'kitchen', 'room_ids_from_role': 'storage_receptacle'}, 'prepare': {'target_role': 'target_object', 'final_receptacle_role': 'sink_basin', 'device_roles': ['sink_basin', 'faucet'], 'tool_roles': [], 'stages': [{'type': 'wash', 'receptacle_role': 'sink_basin', 'faucet_role': 'faucet'}]}}, 'goal_predicates': [{'type': 'object_clean', 'object_role': 'target_object'}, {'type': 'object_closed', 'object_role': 'storage_receptacle', 'include_if': {'role': 'storage_receptacle', 'field': 'openable', 'equals': False}}]}, {'subtask_id': 'store', 'category': 'put_away_by_category', 'instruction_fragment': 'Put the plate away.', 'family_spec': {'family': 'put_away_by_category', 'scope': {'type': 'single_room', 'label': 'kitchen', 'room_ids_from_role': 'storage_receptacle'}, 'put_away': {'semantic_category': 'dishware', 'target_roles': ['target_object'], 'storage_roles': ['storage_receptacle'], 'protected_distractor_roles': []}}, 'goal_predicates': [{'type': 'object_in_receptacle', 'object_role': 'target_object', 'receptacle_role': 'storage_receptacle'}]}, {'subtask_id': 'finish', 'category': 'put_away_by_category', 'instruction_fragment': 'Close the storage.', 'family_spec': {'family': 'put_away_by_category', 'scope': {'type': 'single_room', 'label': 'kitchen', 'room_ids_from_role': 'storage_receptacle'}, 'put_away': {'semantic_category': 'dishware', 'target_roles': [], 'storage_roles': ['storage_receptacle'], 'protected_distractor_roles': []}}, 'goal_predicates': [{'type': 'object_closed', 'object_role': 'storage_receptacle', 'include_if': {'role': 'storage_receptacle', 'field': 'openable', 'equals': True}}]}]}}}

class TaskGeneratorTest(unittest.TestCase):

    def test_tidy_policy_template_marks_listed_issue_types_as_conditional(self):
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'tidy_by_policy_001.json'))
        instruction = template['instruction_templates'][0]
        self.assertEqual(template['template_version'], '0.2')
        self.assertIn('issues you actually observe', instruction)
        self.assertIn('Some issue types may not be present', instruction)
        self.assertIn('Do not repeatedly search', instruction)

    def test_enumerate_role_bindings_returns_multiple_static_bindings(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'static_simple.json'))
        bindings, reason = enumerate_role_bindings(template, _scene_profile(), taxonomy)
        self.assertIsNone(reason)
        self.assertGreater(len(bindings), 2)
        for binding in bindings:
            self.assertIn('target_object', binding)
            self.assertIn('target_receptacle', binding)
            self.assertNotEqual(binding['target_object']['object_id'], binding['target_receptacle']['object_id'])

    def test_generate_candidate_tasks_produces_valid_task_bank(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        templates = load_task_templates(TEMPLATE_DIR)
        tasks, summary = generate_candidate_tasks([_scene_profile()], templates, taxonomy, seed=3, max_tasks_per_house=60, max_tasks_per_category_per_house=4, max_tasks_per_template_per_house=2)
        validate_task_bank(tasks)
        self.assertEqual(summary['num_profiles'], 1)
        self.assertEqual(summary['num_templates'], len(templates))
        self.assertEqual(summary['num_tasks'], len(tasks))
        self.assertIn('static_simple', summary['category_counts'])
        self.assertIn('hidden_precondition', summary['category_counts'])
        self.assertIn('mid_execution_discovery', summary['category_counts'])
        self.assertIn('dynamic_state_change', summary['category_counts'])
        self.assertIn('completion_timing', summary['category_counts'])
        self.assertIn('composite', summary['category_counts'])
        self.assertIn('explore_then_act', summary['category_counts'])
        self.assertIn('put_away_visual_judgment', summary['category_counts'])
        self.assertTrue(all((task['id'].startswith('procthor_') for task in tasks)))
        self.assertTrue(all((task['validation']['status'] == 'candidate' for task in tasks)))

    def test_prepare_family_spec_is_attached_and_scope_resolved(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'clean_then_place.json'))
        tasks, _ = generate_candidate_tasks([_scene_profile()], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task['category'], 'prepare_then_place')
        self.assertEqual(task['family_spec']['family'], 'prepare_then_place')
        self.assertEqual(task['family_spec']['scope']['room_ids'], ['room_0'])
        self.assertEqual(task['family_spec']['prepare']['subtemplate'], 'wash_then_store')
        predicate_types = [p['type'] for p in task['goal']['predicates']]
        self.assertIn('object_closed', predicate_types)

    def test_recover_family_spec_is_attached_and_incident_is_initially_hidden(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'recover_open_container_then_simple_place_001.json'))
        tasks, _ = generate_candidate_tasks([_scene_profile()], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task['category'], 'recover_and_store_incident_object')
        self.assertEqual(task['family_spec']['family'], 'recover_and_store_incident_object')
        self.assertEqual(task['family_spec']['scope']['room_ids'], ['room_0'])
        self.assertEqual(task['family_spec']['recover']['incident']['object_role'], 'incident_container')
        self.assertEqual(task['init']['object_state_overrides'], [{'type': 'set_open_state', 'object_role': 'incident_container', 'is_open': True}])

    def test_recover_faucet_and_pick_uses_existing_sink_object(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'recover_faucet_and_pick_001.json'))
        profile = _scene_profile()
        sink_id = 'SinkBasin|+01.05|+00.90|+00.00'
        for obj in profile['objects']:
            if obj['object_id'] == 'Faucet|+01.10|+00.90|+00.00':
                obj['parent_receptacles'] = [sink_id]
            elif obj['object_id'] == 'SoapBottle|+00.25|+00.90|+00.00':
                obj['parent_receptacles'] = [sink_id]
        tasks, summary = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1, max_tasks_per_house=4, max_tasks_per_category_per_house=4)
        self.assertEqual(summary['num_tasks'], 1)
        task = tasks[0]
        self.assertEqual(task['template']['template_id'], 'recover_faucet_and_pick_001')
        self.assertEqual(task['roles']['primary_object']['object_id'], 'SoapBottle|+00.25|+00.90|+00.00')
        self.assertNotIn('spawn_objects', task['init'])
        self.assertEqual(task['family_spec']['recover']['incident']['object_role'], 'incident_faucet')
        self.assertEqual(task['init']['object_state_overrides'], [{'type': 'set_toggle_state', 'object_role': 'incident_faucet', 'is_toggled': True}])
        validate_task_bank(tasks)

    def test_recover_faucet_and_pick_rejects_empty_sink(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'recover_faucet_and_pick_001.json'))
        profile = _scene_profile()
        sink_id = 'SinkBasin|+01.05|+00.90|+00.00'
        for obj in profile['objects']:
            if obj['object_id'] == 'Faucet|+01.10|+00.90|+00.00':
                obj['parent_receptacles'] = [sink_id]
            else:
                obj['parent_receptacles'] = [parent for parent in obj.get('parent_receptacles', []) if parent != sink_id]
        tasks, summary = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1, max_tasks_per_house=4, max_tasks_per_category_per_house=4)
        self.assertEqual(tasks, [])
        self.assertEqual(summary['num_tasks'], 0)
        self.assertEqual(len(summary['rejections']), 1)
        self.assertEqual(summary['rejections'][0]['category'], 'recover_and_store_incident_object')
        self.assertIn('requires a real pickupable object', summary['rejections'][0]['reason'])

    def test_recover_faucet_and_dirty_dish_relocates_to_bathroom_scope(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'recover_faucet_and_dirty_dish.json'))
        profile = _scene_profile()
        profile['rooms'] = [{'room_id': 'room_kitchen', 'room_type': 'Kitchen'}, {'room_id': 'room_bathroom', 'room_type': 'Bathroom'}]
        for obj in profile['objects']:
            obj['room_id'] = 'room_bathroom' if obj['object_type'] in {'CounterTop', 'Faucet', 'Sink', 'SinkBasin'} else 'room_kitchen'
        tasks, summary = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(summary['num_tasks'], 1)
        task = tasks[0]
        self.assertEqual(task['family_spec']['scope']['room_ids'], ['room_bathroom'])
        self.assertEqual(task['family_spec']['scope']['label'], 'bathroom')
        self.assertIn('dirty dish in the bathroom', task['instruction'])
        faucet_incident = task['family_spec']['recover']['incidents'][0]
        self.assertEqual(faucet_incident['activation'], {'expected_visibility': 'navigation_stream_or_scan', 'initially_visible': False, 'type': 'visual_discovery'})
        relocate = [item for item in task['init']['object_pose_overrides'] if item.get('object_role') == 'dirty_plate']
        self.assertEqual(len(relocate), 1)
        self.assertEqual(relocate[0]['mode'], 'same_room_as')

    def test_recover_faucet_then_food_uses_cross_room_whole_house_scope(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'recover_faucet_on_then_food_to_fridge_001.json'))
        profile = _scene_profile()
        profile['rooms'] = [{'room_id': 'room_kitchen', 'room_type': 'Kitchen'}, {'room_id': 'room_bathroom', 'room_type': 'Bathroom'}]
        for obj in profile['objects']:
            obj['room_id'] = 'room_bathroom' if obj['object_type'] in {'CounterTop', 'Faucet', 'Sink', 'SinkBasin'} else 'room_kitchen'
        tasks, summary = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(summary['num_tasks'], 1)
        task = tasks[0]
        self.assertEqual(task['family_spec']['scope'], {'type': 'whole_house', 'label': 'house', 'room_ids': []})
        self.assertIn('keep the house tidy and safe', task['instruction'])
        role_rooms = {role: next((obj['room_id'] for obj in profile['objects'] if obj['object_id'] == binding['object_id'])) for role, binding in task['roles'].items() if role in {'incident_faucet', 'primary_receptacle'}}
        self.assertNotEqual(role_rooms['incident_faucet'], role_rooms['primary_receptacle'])
        relocate = [item for item in task['init']['object_pose_overrides'] if item.get('object_role') == 'primary_object']
        self.assertEqual(len(relocate), 1)
        self.assertEqual(relocate[0]['mode'], 'same_room_as')
        self.assertEqual(relocate[0]['to_receptacle_role'], 'primary_object_relocate_target')

    def test_observation_conditioned_template_materializes_latent_payload(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'observation_conditioned_dirty_dish_inspection_001.json'))
        tasks, _ = generate_candidate_tasks([_scene_profile()], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task['category'], 'observation_conditioned_room_inspection')
        obs = task['family_spec']['observation_conditioned']
        self.assertEqual(obs['rooms'], ['room_0'])
        self.assertEqual(len(obs['latent_payloads']), 1)
        payload = obs['latent_payloads'][0]
        self.assertEqual(payload['issue_type'], 'dirty_dish')
        self.assertEqual(payload['object_role'], 'dirty_dish')
        self.assertEqual(payload['room_id'], 'room_0')
        self.assertEqual(payload['support_roles']['sink'], 'sink_basin')
        self.assertEqual(task['init']['object_state_overrides'], [{'type': 'set_dirty_state', 'object_role': 'dirty_dish', 'is_dirty': True}])

    def test_three_step_composite_template_materializes_subtasks(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = _three_step_composite_template()
        tasks, _ = generate_candidate_tasks([_scene_profile()], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(len(tasks), 1)
        validate_task_bank(tasks)
        task = tasks[0]
        self.assertEqual(task['category'], 'composite')
        subtasks = task['family_spec']['composite']['subtasks']
        self.assertEqual([subtask['subtask_id'] for subtask in subtasks], ['wash', 'store', 'finish'])
        for subtask in subtasks:
            self.assertEqual(subtask['family_spec']['scope']['room_ids'], ['room_0'])
        self.assertEqual(subtasks[0]['goal_predicates'], [{'type': 'object_clean', 'object_role': 'target_object'}])
        self.assertEqual(subtasks[1]['goal_predicates'], [{'type': 'object_in_receptacle', 'object_role': 'target_object', 'receptacle_role': 'storage_receptacle'}])
        owner_ids = [predicate.get('owner_subtask_id') for predicate in task['goal']['predicates']]
        self.assertIn('wash', owner_ids)
        self.assertIn('store', owner_ids)
        self.assertIn('finish', owner_ids)
        self.assertEqual([predicate for predicate in task['goal']['predicates'] if predicate.get('type') == 'object_closed' and predicate.get('object_role') == 'storage_receptacle'], [{'type': 'object_closed', 'object_role': 'storage_receptacle', 'owner_subtask_id': 'finish'}])

    def test_composite_template_merges_child_disturbance_specs(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = _three_step_composite_template()
        subtasks = template['family_spec_template']['composite']['subtasks']
        subtasks[0]['disturbance_spec'] = {'seed': 7, 'disturbances': [{'type': 'dirty_object', 'target_role': 'target_object', 'after_step': 3, 'requires_out_of_view': False, 'once': True}]}
        subtasks[1]['disturbance_spec'] = {'seed': 7, 'failures': [{'tool': 'put_down', 'mode': 'drop_beside', 'target_role': 'storage_receptacle', 'after_step': 0, 'max_count': 1}]}
        tasks, _ = generate_candidate_tasks([_scene_profile()], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        disturbance_spec = tasks[0]['disturbance_spec']
        self.assertEqual(disturbance_spec['seed'], 7)
        self.assertEqual(disturbance_spec['disturbances'][0]['target_role'], 'target_object')
        self.assertEqual(disturbance_spec['failures'][0]['target_role'], 'storage_receptacle')

    def test_augmented_layout_profile_metadata_is_copied_to_task_source(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'static_simple.json'))
        profile = _scene_profile()
        profile.update({'source_json': '/assets/behavior/datasets/generated-scene-variants/scenes/Rs_int/json/Rs_int_variant_0000.json', 'layout_variant_id': 'Rs_int_variant_0000', 'profile_source': 'behavior_scene_augmentation', 'occupancy_map_path': '/assets/behavior/outputs/example/scene_variant_occupancy_occupied.png', 'bev_map_path': '/assets/behavior/datasets/generated-scene-variants/scenes/Rs_int/json/Rs_int_variant_0000_bev.png', 'labels_json_path': '/assets/behavior/datasets/generated-scene-variants/scenes/Rs_int/json/Rs_int_variant_0000_labels.json'})
        tasks, _ = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(len(tasks), 1)
        source = tasks[0]['source']
        self.assertEqual(source['layout_variant_id'], 'Rs_int_variant_0000')
        self.assertEqual(source['profile_source'], 'behavior_scene_augmentation')
        self.assertIn('source_json', source)
        self.assertIn('occupancy_map_path', source)

    def test_conditional_goal_predicate_is_dropped_when_binding_does_not_match(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'clean_then_place.json'))
        profile = _scene_profile()
        for obj in profile['objects']:
            if obj['object_type'] in {'Cabinet', 'Drawer'}:
                obj['object_type'] = 'Shelf'
                obj['openable'] = False
                obj['is_open'] = None
        tasks, _ = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(len(tasks), 1)
        predicate_types = [p['type'] for p in tasks[0]['goal']['predicates']]
        self.assertNotIn('object_closed', predicate_types)

    def test_generated_hidden_precondition_task_has_open_state_override(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'hidden_precondition.json'))
        tasks, _ = generate_candidate_tasks([_scene_profile()], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]['category'], 'hidden_precondition')
        self.assertEqual(tasks[0]['init']['object_state_overrides'], [{'type': 'set_open_state', 'object_role': 'source_container', 'is_open': False}])

    def test_put_away_target_in_goal_receptacle_gets_pose_override(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'put_away_food_to_cold_storage.json'))
        profile = _scene_profile()
        apple = next((obj for obj in profile['objects'] if obj['object_type'] == 'Apple'))
        apple['parent_receptacles'] = ['Fridge|+02.50|+00.90|+00.00']
        apple['visible_from_initial_pose'] = False
        tasks, summary = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1, max_tasks_per_house=4, max_tasks_per_category_per_house=4)
        self.assertEqual(summary['num_tasks'], 1)
        override = tasks[0]['init']['object_pose_overrides'][0]
        self.assertEqual(override['mode'], 'evacuate_from_goal_receptacle')
        self.assertEqual(override['object_role'], 'target_0')
        self.assertEqual(override['to_receptacle_role'], 'target_0_relocate_target')
        self.assertIn('target_0_relocate_target', tasks[0]['roles'])

    def test_recover_dirty_dish_gets_floor_pose_override(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'recover_dirty_dish_then_simple_place_001.json'))
        profile = _scene_profile()
        object_positions = {'Apple|+00.00|+00.90|+00.00': {'x': 0.0, 'y': 0.9, 'z': 0.0}, 'Cabinet|+02.00|+00.90|+00.00': {'x': 2.0, 'y': 0.9, 'z': 0.0}, 'CounterTop|+01.50|+00.90|+00.00': {'x': 1.5, 'y': 0.9, 'z': 0.0}, 'Sink|+01.00|+00.90|+00.00': {'x': 1.0, 'y': 0.9, 'z': 1.0}, 'SinkBasin|+01.05|+00.90|+00.00': {'x': 1.05, 'y': 0.9, 'z': 1.0}, 'Faucet|+01.10|+00.90|+00.00': {'x': 1.1, 'y': 0.9, 'z': 1.0}, 'Drawer|+02.70|+00.90|+00.00': {'x': 2.7, 'y': 0.9, 'z': 0.5}, 'Fridge|+02.50|+00.90|+00.00': {'x': 2.5, 'y': 0.9, 'z': -0.5}, 'DeskLamp|+03.00|+00.90|+00.00': {'x': 3.0, 'y': 0.9, 'z': 1.0}}
        for obj in profile['objects']:
            if obj['object_id'] in object_positions:
                obj['position'] = object_positions[obj['object_id']]
        profile['objects'].append(_obj('SideTable|+00.75|+00.90|+00.25', 'SideTable', receptacle=True, position={'x': 0.75, 'y': 0.9, 'z': 0.25}))
        tasks, _ = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1, max_tasks_per_house=4, max_tasks_per_category_per_house=4)
        self.assertEqual(len(tasks), 1)
        override = tasks[0]['init']['object_pose_overrides'][0]
        self.assertEqual(override['mode'], 'near_primary_object_floor')
        self.assertEqual(override['to_floor_anchor_role'], 'primary_object')
        self.assertEqual(override['validation'], 'interactable_poses')
        self.assertNotIn('to_receptacle_role', override)
        self.assertGreater(len(override['floor_position_candidates']), 0)
        self.assertNotIn('incident_dish_relocate_target', tasks[0]['roles'])

    def test_composite_wash_put_away_relocates_dish_to_storage_room(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'composite_wash_then_put_away_001.json'))
        profile = _scene_profile()
        profile['rooms'].append({'room_id': 'room_1', 'room_type': 'LivingRoom'})
        for obj in profile['objects']:
            if obj['object_type'] in {'Bowl', 'Plate'}:
                obj['room_id'] = 'room_1'
                obj['position'] = {'x': 4.0, 'y': 0.9, 'z': 0.0}
            elif obj['object_type'] in {'Cabinet', 'CounterTop', 'Drawer', 'Faucet', 'Sink', 'SinkBasin'}:
                obj['room_id'] = 'room_0'
        tasks, summary = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1, max_tasks_per_house=4, max_tasks_per_category_per_house=4)
        self.assertEqual(summary['num_tasks'], 1)
        override = tasks[0]['init']['object_pose_overrides'][0]
        self.assertEqual(override['mode'], 'same_room_as')
        self.assertEqual(override['object_role'], 'target_object')
        self.assertEqual(override['to_receptacle_role'], 'target_object_relocate_target')
        self.assertIn('target_object_relocate_target', tasks[0]['roles'])
        validate_task_bank(tasks)

    def test_cross_room_food_relocates_food_near_source_anchor(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'put_away_food_cross_room.json'))
        profile = _scene_profile()
        profile['rooms'].append({'room_id': 'room_1', 'room_type': 'LivingRoom'})
        for obj in profile['objects']:
            if obj['object_type'] in {'Apple', 'Fridge'}:
                obj['room_id'] = 'room_0'
            if obj['object_type'] == 'CounterTop':
                obj['room_id'] = 'room_0'
        profile['objects'].append(_obj('DiningTable|+04.00|+00.90|+00.00', 'DiningTable', room_id='room_1', receptacle=True, position={'x': 4.0, 'y': 0.9, 'z': 0.0}))
        profile['reachable']['positions'].extend([{'x': 3.25, 'y': 0.9, 'z': 0.0, 'room_id': 'room_1'}, {'x': 3.5, 'y': 0.9, 'z': 0.0, 'room_id': 'room_1'}, {'x': 4.75, 'y': 0.9, 'z': 0.0, 'room_id': 'room_1'}])
        tasks, summary = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1, max_tasks_per_house=4, max_tasks_per_category_per_house=4)
        self.assertEqual(summary['num_tasks'], 1)
        task = tasks[0]
        override = task['init']['object_pose_overrides'][0]
        self.assertEqual(override['mode'], 'near_primary_object_floor')
        self.assertEqual(override['object_role'], 'target_food')
        self.assertEqual(override['to_floor_anchor_role'], 'source_anchor')
        self.assertGreater(len(override['floor_position_candidates']), 0)
        self.assertEqual(task['roles']['source_anchor']['object_type'], 'DiningTable')
        validate_task_bank(tasks)

    def test_mid_execution_re_dirty_clears_existing_sink_clutter(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'mid_execution_state_re_dirty_001.json'))
        profile = _scene_profile()
        for obj in profile['objects']:
            if obj['object_id'] == 'Bowl|+00.22|+00.90|+00.10':
                obj['visible_from_initial_pose'] = True
            if obj['object_id'] == 'SinkBasin|+01.05|+00.90|+00.00':
                obj['position'] = {'x': 1.05, 'y': 0.9, 'z': 1.0}
            if obj['object_id'] == 'CounterTop|+01.50|+00.90|+00.00':
                obj['position'] = {'x': 1.5, 'y': 0.9, 'z': 0.0}
        profile['objects'].append(_obj('DishSponge|+01.05|+00.90|+00.10', 'DishSponge', pickupable=True, parent_receptacles=['SinkBasin|+01.05|+00.90|+00.00']))
        tasks, summary = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1, max_tasks_per_house=4, max_tasks_per_category_per_house=4)
        self.assertEqual(summary['num_tasks'], 1)
        cleanup_overrides = [override for override in tasks[0]['init']['object_pose_overrides'] if override['type'] == 'clear_receptacle_clutter']
        self.assertEqual(len(cleanup_overrides), 1)
        cleanup = cleanup_overrides[0]
        self.assertEqual(cleanup['receptacle_role'], 'sink_basin')
        self.assertEqual(cleanup['object_ids'], ['DishSponge|+01.05|+00.90|+00.10'])
        self.assertNotIn('to_receptacle_object_id', cleanup)
        self.assertNotIn('to_receptacle_object_type', cleanup)
        validate_task_bank(tasks)

    def test_recover_dirty_dish_allows_cross_room_sink_support(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'recover_dirty_dish_then_simple_place_001.json'))
        profile = _scene_profile()
        profile['rooms'].append({'room_id': 'room_1', 'room_type': 'Bathroom'})
        for obj in profile['objects']:
            if obj['object_id'] in {'Sink|+01.00|+00.90|+00.00', 'SinkBasin|+01.05|+00.90|+00.00', 'Faucet|+01.10|+00.90|+00.00', 'Drawer|+02.70|+00.90|+00.00'}:
                obj['room_id'] = 'room_1'
        profile['objects'].append(_obj('SideTable|+00.75|+00.90|+00.25', 'SideTable', receptacle=True, position={'x': 0.75, 'y': 0.9, 'z': 0.25}))
        for obj in profile['objects']:
            if obj['object_id'] == 'SideTable|+00.75|+00.90|+00.25':
                continue
            if obj['object_id'] == 'CounterTop|+01.50|+00.90|+00.00':
                obj['position'] = {'x': 1.5, 'y': 0.9, 'z': 0.0}
            elif obj['object_id'] == 'Apple|+00.00|+00.90|+00.00':
                obj['position'] = {'x': 0.0, 'y': 0.9, 'z': 0.0}
            elif obj['object_id'] == 'SinkBasin|+01.05|+00.90|+00.00':
                obj['position'] = {'x': 4.0, 'y': 0.9, 'z': 0.0}
            elif obj['object_id'] == 'Faucet|+01.10|+00.90|+00.00':
                obj['position'] = {'x': 4.1, 'y': 0.9, 'z': 0.0}
            elif obj['object_id'] == 'Drawer|+02.70|+00.90|+00.00':
                obj['position'] = {'x': 4.2, 'y': 0.9, 'z': 0.0}
        tasks, _ = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1, max_tasks_per_house=4, max_tasks_per_category_per_house=4)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task['roles']['incident_sink']['object_type'], 'SinkBasin')
        self.assertIn(task['roles']['incident_storage']['object_type'], {'Cabinet', 'Drawer', 'Shelf'})
        self.assertIn(task['init']['object_pose_overrides'][0]['mode'], {'near_primary_object_floor'})

    def test_recover_dirty_dish_allows_cross_room_storage_support(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'recover_dirty_dish_then_simple_place_001.json'))
        profile = _scene_profile()
        profile['rooms'].append({'room_id': 'room_1', 'room_type': 'Bathroom'})
        for obj in profile['objects']:
            if obj['object_id'] in {'Sink|+01.00|+00.90|+00.00', 'SinkBasin|+01.05|+00.90|+00.00', 'Faucet|+01.10|+00.90|+00.00'}:
                obj['room_id'] = 'room_1'
        profile['objects'].append(_obj('SideTable|+00.75|+00.90|+00.25', 'SideTable', receptacle=True, position={'x': 0.75, 'y': 0.9, 'z': 0.25}))
        for obj in profile['objects']:
            if obj['object_id'] == 'SideTable|+00.75|+00.90|+00.25':
                continue
            if obj['object_id'] == 'CounterTop|+01.50|+00.90|+00.00':
                obj['position'] = {'x': 1.5, 'y': 0.9, 'z': 0.0}
            elif obj['object_id'] == 'Apple|+00.00|+00.90|+00.00':
                obj['position'] = {'x': 0.0, 'y': 0.9, 'z': 0.0}
            elif obj['object_id'] == 'SinkBasin|+01.05|+00.90|+00.00':
                obj['position'] = {'x': 4.0, 'y': 0.9, 'z': 0.0}
            elif obj['object_id'] == 'Faucet|+01.10|+00.90|+00.00':
                obj['position'] = {'x': 4.1, 'y': 0.9, 'z': 0.0}
            elif obj['object_id'] == 'Drawer|+02.70|+00.90|+00.00':
                obj['position'] = {'x': 2.7, 'y': 0.9, 'z': 0.0}
        tasks, _ = generate_candidate_tasks([profile], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1, max_tasks_per_house=4, max_tasks_per_category_per_house=4)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task['roles']['incident_sink']['object_type'], 'SinkBasin')
        room_by_object_id = {obj['object_id']: obj.get('room_id') for obj in profile['objects']}
        self.assertEqual(room_by_object_id[task['roles']['incident_sink']['object_id']], 'room_1')
        self.assertNotEqual(room_by_object_id[task['roles']['incident_storage']['object_id']], 'room_1')
        self.assertEqual(task['init']['object_pose_overrides'][0]['mode'], 'near_primary_object_floor')

    def test_completion_timing_sets_require_model_stop(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'completion_timing.json'))
        tasks, _ = generate_candidate_tasks([_scene_profile()], [template], taxonomy, seed=0, max_tasks_per_template_per_house=1)
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]['goal']['require_model_stop'])

    def test_generation_is_deterministic_for_fixed_seed(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        templates = load_task_templates(TEMPLATE_DIR)
        first, first_summary = generate_candidate_tasks([_scene_profile()], templates, taxonomy, seed=11, max_tasks_per_template_per_house=2)
        second, second_summary = generate_candidate_tasks([_scene_profile()], templates, taxonomy, seed=11, max_tasks_per_template_per_house=2)
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)

    def test_limits_are_applied_per_house_and_category(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        templates = load_task_templates(TEMPLATE_DIR)
        tasks, summary = generate_candidate_tasks([_scene_profile()], templates, taxonomy, seed=0, max_tasks_per_house=3, max_tasks_per_category_per_house=1, max_tasks_per_template_per_house=5)
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all((count == 1 for count in summary['category_counts'].values())))

    def test_max_tasks_per_category_caps_numbered_output(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        templates = load_task_templates(TEMPLATE_DIR)
        tasks, summary = generate_candidate_tasks([_scene_profile('house_a'), _scene_profile('house_b')], templates, taxonomy, seed=0, max_tasks_per_template_per_house=2, max_tasks_per_category=1)
        self.assertTrue(all((count <= 1 for count in summary['category_counts'].values())))
        ids = [task['id'] for task in tasks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_write_task_bank_round_trip(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        templates = load_task_templates(TEMPLATE_DIR)
        tasks, _ = generate_candidate_tasks([_scene_profile()], templates, taxonomy, seed=0, max_tasks_per_template_per_house=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'task_bank.candidates.jsonl')
            write_task_bank(tasks, output_path)
            loaded = load_task_bank(output_path)
            self.assertEqual(list(loaded.values()), tasks)

    def test_load_scene_profiles_from_jsonl(self):
        profile = _scene_profile()
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, 'profiles.jsonl')
            with open(profile_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(profile) + '\n')
            loaded = load_scene_profiles(profile_path)
            self.assertEqual(loaded, [profile])

    def test_missing_template_candidates_are_reported_as_rejections(self):
        taxonomy = load_object_taxonomy(TAXONOMY_PATH)
        template = load_task_template(os.path.join(TEMPLATE_DIR, 'dynamic_state_change.json'))
        profile = _scene_profile()
        profile['objects'] = [obj for obj in profile['objects'] if obj['object_type'] not in {'Faucet', 'DeskLamp'}]
        tasks, summary = generate_candidate_tasks([profile], [template], taxonomy, seed=0)
        self.assertEqual(tasks, [])
        self.assertEqual(summary['num_tasks'], 0)
        self.assertEqual(summary['rejections'][0]['category'], 'dynamic_state_change')
        self.assertIn('No candidates', summary['rejections'][0]['reason'])
if __name__ == '__main__':
    unittest.main()
