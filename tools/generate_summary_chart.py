import os
import sys
import json
from datetime import datetime

# Ensure project root is on sys.path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from process_optimization_agent.models import load_process_from_json, Schedule, ScheduleEntry
from process_optimization_agent.visualizer import Visualizer


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_path = os.path.join(base_dir, 'examples', 'software_project.json')
    sched_path = os.path.join(base_dir, 'rl_optimization_results', 'results', 'base_case_schedule.json')
    output_dir = os.path.join(base_dir, 'rl_optimization_results')
    output_file = os.path.join(output_dir, 'summary_comparison.png')

    if not os.path.exists(proc_path):
        raise FileNotFoundError(f"Process file not found: {proc_path}")
    if not os.path.exists(sched_path):
        raise FileNotFoundError(f"Schedule file not found: {sched_path}")

    # Load process
    process = load_process_from_json(proc_path)

    # Load schedule JSON and reconstruct Schedule
    with open(sched_path, 'r') as f:
        data = json.load(f)

    entries = []
    for e in data.get('entries', []):
        st = datetime.fromisoformat(e['start_time']) if e.get('start_time') else None
        et = datetime.fromisoformat(e['end_time']) if e.get('end_time') else None
        entries.append(ScheduleEntry(
            task_id=e['task_id'],
            resource_id=e['resource_id'],
            start_time=st,
            end_time=et,
            cost=float(e.get('cost', 0.0) or 0.0)
        ))

    schedule = Schedule(
        process_id=data.get('process_id', process.id),
        entries=entries,
        total_duration_hours=float(data.get('total_duration_hours', 0.0) or 0.0),
        total_cost=float(data.get('total_cost', 0.0) or 0.0),
    )

    # If duration/cost missing, compute basics
    if (not schedule.total_duration_hours) and schedule.entries:
        start = min(e.start_time for e in schedule.entries if e.start_time)
        end = max(e.end_time for e in schedule.entries if e.end_time)
        schedule.total_duration_hours = (end - start).total_seconds() / 3600.0
    if (not schedule.total_cost) and schedule.entries:
        schedule.total_cost = sum(e.cost for e in schedule.entries)

    # Prepare before/after dicts
    before = {
        'duration_hours': sum(t.duration_hours for t in process.tasks),
        'peak_people': len(process.resources),
        'total_resources': len(process.resources),
        'total_cost': 0.0  # Unknown for sequential baseline; leave as 0 for comparison
    }

    after = {
        'duration_hours': schedule.total_duration_hours,
        'total_cost': schedule.total_cost,
        'schedule': schedule
    }

    viz = Visualizer()
    os.makedirs(output_dir, exist_ok=True)
    out_path = viz.plot_summary_comparison(before, after, title='Before vs After Optimization', output_file=output_file, show=False)
    print('Saved chart to:', out_path)


if __name__ == '__main__':
    main()
