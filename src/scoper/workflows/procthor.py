"""procthor workflows; simulator imports are lazy."""

def generate(a, save, template_path):
    data = template_path.parent.parent
    from scoper.benchmark.procthor.scene_profiler import profile_house
    from scoper.benchmark.procthor.task_templates import load_task_templates
    from scoper.benchmark.procthor.task_generator import generate_candidate_tasks
    from scoper.benchmark.procthor.object_taxonomy import load_object_taxonomy
    from scoper.benchmark.procthor.procthor_dataset_assets import parse_prior_house_uri
    split, index = parse_prior_house_uri(a.layout)
    entry = {'house_id': f'{split}_{index:06d}', 'house_path': a.layout, 'split': split}
    profile = profile_house(entry, gpu_device=a.gpu, seed=a.seed)
    templates = load_task_templates(str(template_path))
    tasks, summary = generate_candidate_tasks([profile], templates, load_object_taxonomy(str(data / 'object_taxonomy.json')), seed=a.seed, max_tasks_per_house=a.limit, max_tasks_per_category_per_house=a.limit, max_tasks_per_template_per_house=a.limit)
    generated = {'tasks': tasks, 'summary': summary}
    admitted = []
    return profile, generated, tasks, admitted

def oracle(args, task):
    from scoper.benchmark.procthor.oracle_tool_video_runner import run_oracle_tool_video_recording
    result = run_oracle_tool_video_recording(str(args.bank), str(args.output), task_ids=[args.task_id], gpu_device=args.gpu, width=512, height=512, fps=6, save_frames=False, burn_in_subtitles=False, manipulathor_arm=True)
    return result

def oracle_passed(args, result):
    passed = result[1].get('num_passed') == 1 and result[1].get('total_video_frames', 0) > 0
    return passed

EVALUATION_MODULE = 'scoper.benchmark.procthor.lightweight_evaluation'
