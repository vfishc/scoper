from scoper.benchmark.common.container_inspection.definition import *

def same_contents(before, after):
    before = {x['id']: x for x in before}
    after = {x['id']: x for x in after}
    if before.keys() != after.keys():
        return False
    for key, obj in before.items():
        other = after[key]
        if obj['type'] != other['type']:
            return False
        a, b = (obj['position'], other['position'])
        if isinstance(a, dict):
            a, b = ([a[k] for k in ('x', 'y', 'z')], [b[k] for k in ('x', 'y', 'z')])
        if sum(((x - y) ** 2 for x, y in zip(a, b))) > 0.01 ** 2:
            return False
    return True

def is_dishware(category):
    value = str(category).lower().replace('_', ' ').replace('-', ' ')
    return value in {'cup', 'mug', 'bowl', 'plate', 'coffee cup', 'drinking cup', 'coffee mug'} or value.startswith(('bowl.n.', 'plate.n.', 'cup.n.', 'mug.n.'))

def inspect_contents(env, backend):
    if backend == 'procthor':
        raw = env.online_env if hasattr(env, 'online_env') else env
        if not hasattr(raw, 'role_object_ids'):
            raw = env._online_env
        objects = raw._objects_by_id()
        cid = raw.role_object_ids['cabinet']
        cabinet = objects[cid]
        names = set(cabinet.get('receptacleObjectIds') or [])
        names.update((k for k, o in objects.items() if cid in (o.get('parentReceptacles') or [])))
        contents = [{'id': k, 'type': objects[k].get('objectType'), 'position': objects[k].get('position')} for k in sorted(names) if k in objects]
        return {'contents': contents, 'is_open': bool(cabinet.get('isOpen')), 'has_dishware': any((is_dishware(x['type']) for x in contents))}
    from scoper.benchmark.behavior.goal_checker import _is_inside
    cabinet = env.core.objects[env.role_map['cabinet']]
    contents = [{'id': k, 'type': obj.category, 'position': list(obj.center)} for k, obj in env.core.objects.items() if k != cabinet.name and _is_inside(obj, cabinet) and ('pickupable' in obj.abilities or 'pickable' in obj.abilities)]
    return {'contents': contents, 'is_open': bool(cabinet.states.get('open', False)), 'has_dishware': any((is_dishware(x['type']) for x in contents))}

def teacher_class(backend, output=None):
    if backend == 'behavior':
        from scoper.benchmark.behavior.brains.base import BaseBrain, ToolCall
        from scoper.benchmark.behavior.brains.teachers.dispatcher import TeacherDispatcher
    else:
        from scoper.benchmark.procthor.brains.base import BaseBrain, ToolCall
        from scoper.benchmark.procthor.brains.teachers.dispatcher import TeacherDispatcher

    class InspectTeacher(BaseBrain):

        def on_episode_start(self, env, task_spec):
            self.env, self.task = (env, task_spec)
            self.events = []
            self.last_call = None
            self.visible_dishware = False
            initial = inspect_contents(env, backend)
            write(Path(output) / 'initial_contents_audit.json', initial) if output is not None else None
            assert initial['is_open'] is False, 'INITIAL_CABINET_NOT_CLOSED'
            expected = task_spec['conditional_storage_policy']['expected_branch'] == 'dishware'
            assert initial['has_dishware'] == expected, 'INITIAL_CONTENTS_BRANCH_MISMATCH'
            self.initial = initial
            self.calls = [('navigate_to', {'target': 'cabinet'}), ('open', {'target': 'cabinet'})]
            if backend == 'behavior':
                self.calls.append(('scan', {'num_views': 4}))
            if expected:
                if backend == 'behavior':
                    self.calls.append(('navigate_to', {'target': 'cabinet'}))
                self.calls.append(('close', {'target': 'cabinet'}))
            self.calls += [('stop', {})]
            self.save()

        def save(self):
            if output is None:
                return
            write(Path(output) / 'conditional_oracle_evidence.json', {'initial': self.initial, 'actions': self.events, 'dishware_visible_after_open': self.visible_dishware, 'policy': self.task['conditional_storage_policy']})

        def act(self, brain_input):
            if self.last_call:
                transition = brain_input.last_transition or {}
                if transition.get('success') is False:
                    raise RuntimeError('CONDITIONAL_ORACLE_TOOL_FAILED:' + self.last_call)
                current = inspect_contents(self.env, backend)
                assert same_contents(current['contents'], self.initial['contents']), 'CONTENTS_CHANGED_DURING_INSPECTION'
                if self.last_call == 'open':
                    assert current['is_open'], 'CABINET_OPEN_NOT_APPLIED'
                    rgb = (brain_input.observation or {}).get('head_rgb')
                    if backend == 'behavior' and self.env.renderer is not None:
                        pose = brain_input.observation.get('observation_pose') or self.env.core.ego_state.pose
                        capture = self.env.renderer.capture_observation_at_pose(pose, min_pixels=256)
                        rgb = capture['head_rgb']
                        evidence = capture['visibility']
                        write(Path(output) / 'opened_visibility.json', evidence) if output is not None else None
                        import re
                        normalize = lambda name: re.sub('[^A-Za-z0-9_]', '_', str(name))
                        visible = {normalize(n) for n in evidence.get('visible_object_ids', [])}
                        for item in current['contents']:
                            if is_dishware(item['type']) and normalize(item['id']) in visible:
                                self.visible_dishware = True
                    elif backend == 'procthor':
                        visible = self.env._objects_by_id()
                        self.visible_dishware = any((visible.get(x['id'], {}).get('visible') for x in current['contents'] if is_dishware(x['type'])))
                    if rgb is not None and output is not None:
                        from PIL import Image
                        Image.fromarray(rgb).save(Path(output) / 'opened_cabinet.png')
                self.events.append({'tool': self.last_call, 'success': transition.get('success'), 'state': current})
                self.save()
            # The oracle may answer evaluator-owned QA after physical stop.
            # Private keys stay inside this teacher and never enter VLM inputs.
            qa = getattr(self.env, '_human_qa_state', {})
            pending = qa.get('pending', {})
            if pending:
                question_id = next(iter(pending))
                keys = qa.get('evaluator_keys', {})
                if question_id not in keys:
                    raise RuntimeError('ORACLE_QA_KEY_MISSING')
                self.last_call = 'answer'
                return ToolCall('answer', {'question_id': question_id, 'value': keys[question_id]}, reasoning='oracle answer to pending assessment')
            if not self.calls:
                raise RuntimeError('ORACLE_CALLED_AFTER_COMPLETION')
            tool, arguments = self.calls.pop(0)
            if tool == 'stop' and self.initial['has_dishware']:
                if backend == 'procthor' or getattr(self.env, 'renderer', None) is not None:
                    assert self.visible_dishware, 'CONTENTS_NOT_VISUALLY_INSPECTABLE'
            self.last_call = tool
            return ToolCall(tool, arguments, reasoning='oracle conditional cabinet inspection')
    return InspectTeacher
