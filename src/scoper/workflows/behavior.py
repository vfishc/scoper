"""behavior workflows; simulator imports are lazy."""
import os
from scoper.benchmark.common.evaluation_policy import step_budget

def generate(a, save, template_path):
    from scoper.benchmark.behavior.scene.profiler import profile_scene
    from scoper.benchmark.behavior.task_generation.scene_matcher import load_procthor_templates, match_scene_to_templates
    from scoper.benchmark.behavior.task_generation.profile_task_generator import generate_stage4_tasks
    from scoper.benchmark.behavior.task_generation.stage5_validator import validate_stage5_reset_mutation
    from scoper.benchmark.behavior.task_generation.stage6_executor import validate_stage6_execution
    profile = profile_scene(a.layout, layout_variant_id=a.layout + '_var0000')
    templates = load_procthor_templates(template_path)
    matches = match_scene_to_templates(profile, templates, max_bindings_per_template=a.limit, candidate_seed=a.seed)
    save('matches.json', matches)
    generated = generate_stage4_tasks(profiles=[profile], templates=templates, match_records=matches, template_ids=[t['template_id'] for t in templates], max_tasks_per_template_scene=a.limit, seed=a.seed)
    tasks = generated['tasks']
    checks = []
    for task in tasks:
        stage5 = validate_stage5_reset_mutation(task, rng_seed=a.seed, scene_layout_mode='source_json_full')
        stage6 = validate_stage6_execution(task, stage5_result=stage5, rng_seed=a.seed, max_high_level_steps=24, scene_layout_mode='source_json_full')
        checks.append({'task_id': task['id'], 'symbolic_admission': stage5, 'symbolic_oracle': stage6})
    save('symbolic_checks.json', checks)
    admitted = [t for t, c in zip(tasks, checks) if c['symbolic_admission']['stage5_reset_mutation_ok'] and c['symbolic_oracle']['goal_success']]
    return profile, generated, tasks, admitted

def oracle(args, task):
    os.environ.setdefault('EB_BEHAVIOR_GPU', '1')
    os.environ.setdefault('OMNIGIBSON_HEADLESS', '1')
    from scoper.benchmark.behavior.tools.stage9_render_video_report import run_stage9_episode
    result = run_stage9_episode(task, output_dir=args.output, rng_seed=0, max_high_level_steps=step_budget(task), render_requested=True, render_env={'can_attempt_render': True}, require_render=True, video_fps=6, renders_per_frame=2, render_nav_max_frames=12, navigation_speed_mps=None, navigation_turn_speed_radps=None)
    return result

def oracle_passed(args, result):
    manifest = result.get('artifacts', {}).get('render_manifest', 'execution.manifest.json')
    passed = result.get('task_success') == 1 and result.get('render_error') is None and (result.get('render_status') == 'passed') and (result.get('video_quality_status') == 'passed') and (args.output / manifest).is_file()
    return passed

EVALUATION_MODULE = 'scoper.benchmark.behavior.tools.run_lightweight_evaluation'
