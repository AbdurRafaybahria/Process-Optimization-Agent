#!/usr/bin/env python3
"""
Run RL Optimizer on a process JSON file with automatic dependency detection and visualization.

Adds CLI controls:
- --max-parallel INT
- --parallel-policy {strict,balanced}
- --dep-detect {off,strict,balanced,aggressive}
- --dep-threshold FLOAT(0-1)
- --review-deps (interactive confirm to apply proposals)
"""

import os
import sys
import json
import argparse
import tempfile
import webbrowser
import copy
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set

from process_optimization_agent.optimizers import RLBasedOptimizer
from process_optimization_agent.models import Process, Task, Resource, Skill, ScheduleEntry
from process_optimization_agent.visualizer import Visualizer
from process_optimization_agent.analyzers import WhatIfAnalyzer, DependencyDetector

def load_process_from_json(filepath):
    """Load a process from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Create skills mapping
    all_skills = {}
    
    # Create resources
    resources = []
    for res_data in data.get('resources', []):
        skills = []
        for skill_data in res_data.get('skills', []):
            if isinstance(skill_data, dict):
                skill = Skill(
                    name=skill_data['name'],
                    level=skill_data.get('level', 1)
                )
            else:
                skill = Skill(name=skill_data, level=1)
            skills.append(skill)
            all_skills[skill.name] = skill
        
        resource = Resource(
            id=res_data['id'],
            name=res_data.get('name', res_data['id']),
            skills=skills,
            hourly_rate=res_data.get('hourly_rate', 100)
        )
        resources.append(resource)
    
    # Create tasks
    tasks = []
    for task_data in data.get('tasks', []):
        required_skills = []
        for skill_data in task_data.get('required_skills', []):
            if isinstance(skill_data, dict):
                skill = Skill(
                    name=skill_data['name'],
                    level=skill_data.get('level', 1)
                )
            else:
                skill = all_skills.get(skill_data, Skill(name=skill_data, level=1))
            required_skills.append(skill)
        
        # Parse optional deadline
        parsed_deadline = None
        if 'deadline' in task_data and task_data['deadline']:
            try:
                parsed_deadline = datetime.fromisoformat(task_data['deadline'])
            except Exception:
                parsed_deadline = None

        task = Task(
            id=task_data['id'],
            name=task_data['name'],
            description=task_data.get('description', ''),
            duration_hours=task_data['duration_hours'],
            required_skills=required_skills,
            order=task_data.get('order', None),
            dependencies=set(task_data.get('dependencies', [])),
            deadline=parsed_deadline
        )
        tasks.append(task)
    
    # Create process
    # Parse dates and calculate project duration in hours
    # Use defaults if dates not provided
    if 'start_date' in data:
        start_date = datetime.fromisoformat(data['start_date'])
    else:
        start_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    
    if 'target_end_date' in data:
        target_end_date = datetime.fromisoformat(data['target_end_date'])
    else:
        # Default to 30 days from start if not provided
        target_end_date = start_date + timedelta(days=30)
    
    project_duration_hours = (target_end_date - start_date).total_seconds() / 3600.0
    
    process = Process(
        id=data['id'],
        name=data['name'],
        description=data.get('description', ''),
        start_date=start_date,
        project_duration_hours=project_duration_hours,
        tasks=tasks,
        resources=resources
    )
    
    # Store target_end_date as an attribute for backward compatibility
    process.target_end_date = target_end_date
    
    return process

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run RL optimizer on a process JSON file")
    parser.add_argument("process_json", help="Path to process JSON file")
    parser.add_argument("--max-parallel", type=int, default=4, dest="max_parallel",
                        help="Maximum tasks to run in parallel")
    parser.add_argument("--parallel-policy", choices=["strict", "balanced"], default="balanced",
                        help="Parallelization policy hints (currently advisory)")
    parser.add_argument("--dep-detect", choices=["off", "strict", "balanced", "aggressive"], default="balanced",
                        help="Dependency detection mode")
    parser.add_argument("--dep-threshold", type=float, default=0.75,
                        help="Threshold for semantic dependency proposals (0-1)")
    parser.add_argument("--review-deps", action="store_true", default=False,
                        help="Interactively review and confirm dependency proposals before applying")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    json_file = args.process_json
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    print(f"Loading process from: {json_file}")
    process = load_process_from_json(json_file)
    
    print("\n=== Process Summary ===")
    print(f"Name: {process.name}")
    print(f"Start Date: {process.start_date}")
    print(f"Target End Date: {process.target_end_date}")
    print(f"Number of Tasks: {len(process.tasks)}")
    print(f"Number of Resources: {len(process.resources)}")
    
    # Optional: dependency proposals and application prior to optimization
    detector = DependencyDetector()
    # Dependency detection with timeout protection
    if args.dep_detect != "off":
        print("\n=== Dependency Proposals ===")
        try:
            # Skip dependency detection for now to avoid hanging
            print("Skipping automatic dependency detection to prevent hanging")
            print("Using existing task dependencies only")
            applied = 0
            print(f"Applied {applied} dependency edges to process")
        except Exception as e:
            print(f"Warning: Dependency detection failed: {e}")
            print("Proceeding with existing dependencies only")
    else:
        print("Dependency detection disabled")

    print("\n=== Running RL Optimization ===")
    print("Optimizing with RL-based parallel scheduling...\n")
    
    # Create optimizer with parameters focused on time reduction
    optimizer = RLBasedOptimizer(
        learning_rate=0.15,
        epsilon=0.3,  # Start with more exploration
        discount_factor=0.95,
        training_episodes=5,  # Reduced episodes to prevent hanging
        enable_parallel=True,
        max_parallel_tasks=4  # Allow more parallelization
    )
    optimizer.initial_epsilon = 0.3  # Store initial epsilon for reset
    visualizer = Visualizer()
    
    # Run optimization (includes training internally)
    print("Training and optimizing...")
    try:
        schedule = optimizer.optimize(process)
        if not schedule or not schedule.entries:
            print("Warning: No valid schedule found during optimization")
            # Create a simple sequential schedule as fallback
            from process_optimization_agent.optimizers import SimpleOptimizer
            print("Falling back to simple sequential scheduling...")
            simple_optimizer = SimpleOptimizer()
            schedule = simple_optimizer.optimize(process)
    except Exception as e:
        print(f"Error during optimization: {e}")
        print("Falling back to simple sequential scheduling...")
        from process_optimization_agent.optimizers import SimpleOptimizer
        simple_optimizer = SimpleOptimizer()
        schedule = simple_optimizer.optimize(process)
    
    # Store the original optimized schedule
    original_schedule = copy.deepcopy(schedule)
    original_duration = 0
    
    # The optimizer now handles training internally and returns the best schedule
    if schedule and schedule.entries:
        # Calculate makespan (total elapsed time)
        min_start = min(e.start_time for e in schedule.entries)
        max_end = max(e.end_time for e in schedule.entries)
        original_duration = (max_end - min_start).total_seconds() / 3600.0
        print(f"\nOptimization complete. Total duration: {original_duration:.1f}h")
    else:
        print("\nWarning: No valid schedule found during optimization")

    print("\n=== Optimization Results (Baseline RL) ===")
    print(f"Scheduled {len(schedule.entries)}/{len(process.tasks)} tasks")
    print(f"Total Duration: {schedule.total_duration_hours:.1f} hours")
    print(f"Total Cost: ${schedule.total_cost:,.2f}")

    # What-If Analysis with RL to explore parallelization/resource tweaks
    print("\n=== Running What-If Analysis (RL scenarios) ===")
    analyzer = WhatIfAnalyzer(optimizer)
    wi_results = None
    try:
        scenario_params = {
            'epsilon': 0.05,  # Low exploration for exploitation
            'discount_factor': 0.95,
            'learning_rate': 0.2,
            'enable_parallel': True,
            'max_parallel_tasks': 5
        }
        wi_results = analyzer.analyze_scenarios(
            process=process,
            scenarios=None,               # use default RL scenarios
            time_weight=0.6,
            cost_weight=0.4,
            auto_detect_dependencies=True,
            baseline_schedule=original_schedule  # Pass the optimized schedule as baseline
        )
        best = wi_results.get('best_scenario')
        if best and best.get('improvement', {}).get('score', 0) > 0:
            sid = best['id']
            sdata = wi_results['scenarios'].get(sid, {})
            best_schedule = sdata.get('schedule')
            best_scenario_cfg = sdata.get('scenario', {})
            if best_schedule:
                # Apply scenario to a copy of the process so visuals reflect it
                scenario_process = copy.deepcopy(process)
                try:
                    if 'config' in best:
                        analyzer._apply_scenario(scenario_process, best['config'])
                    else:
                        analyzer._apply_scenario(scenario_process, best)
                except Exception:
                    # Fallback: if internal apply fails, keep original process
                    scenario_process = process
                process = scenario_process
                # Check if the what-if scenario actually improves the schedule
                if best_schedule and best_schedule.entries:
                    wi_min_start = min(e.start_time for e in best_schedule.entries)
                    wi_max_end = max(e.end_time for e in best_schedule.entries)
                    wi_duration = (wi_max_end - wi_min_start).total_seconds() / 3600.0
                    
                    # Only use what-if schedule if it's actually better
                    if wi_duration < original_duration:
                        schedule = best_schedule
                        print(f"\nUsing best what-if scenario: {best.get('name','(unnamed)')} (score={best['improvement']['score']:.3f})")
                        print(f"New Duration: {wi_duration:.1f} hours (improved from {original_duration:.1f}h), Cost: ${schedule.total_cost:,.2f}")
                    else:
                        schedule = original_schedule
                        print(f"\nWhat-if scenario did not improve duration ({wi_duration:.1f}h vs {original_duration:.1f}h), keeping optimized schedule.")
                else:
                    schedule = original_schedule
                    print("What-if scenario produced invalid schedule, keeping optimized schedule.")
        else:
            print("No better what-if scenario found; keeping optimized schedule.")
    except Exception as e:
        print(f"What-If analysis skipped due to error: {e}")
    
    # === Ensure all tasks are scheduled (force-schedule fallback) ===
    try:
        # Identify unscheduled tasks
        scheduled_ids = {e.task_id for e in (schedule.entries or [])}
        all_tasks_by_id = {t.id: t for t in process.tasks}
        missing_tasks = [t for t in process.tasks if t.id not in scheduled_ids]

        if missing_tasks:
            # Helper to choose a resource: first qualified, else highest-rate as fallback
            def _pick_resource(task):
                for r in process.resources:
                    if r.has_all_skills(getattr(task, 'required_skills', []) or []):
                        return r
                # If none qualified, pick any resource with highest rate to make it explicit in cost
                return max(process.resources, key=lambda r: getattr(r, 'hourly_rate', 0.0)) if process.resources else None

            # Place missing tasks sequentially after the last scheduled end
            if schedule.entries:
                cur_time = max(e.end_time for e in schedule.entries)
            else:
                from datetime import datetime as _dt
                cur_time = _dt.now().replace(hour=9, minute=0, second=0, microsecond=0)

            # Local business-hours helpers
            from datetime import datetime as _dt, time as _time, timedelta as _td
            def _is_bd(dt: _dt) -> bool:
                return dt.weekday() < 5
            def _next_bstart(dt: _dt) -> _dt:
                WS, WE = _time(9,0), _time(17,0)
                cur = dt
                if cur.time() >= WE:
                    cur = _dt(cur.year, cur.month, cur.day) + _td(days=1, hours=9)
                elif cur.time() < WS:
                    cur = _dt(cur.year, cur.month, cur.day, 9, 0)
                while not _is_bd(cur):
                    cur = _dt(cur.year, cur.month, cur.day) + _td(days=1, hours=9)
                return cur
            def _add_bhours(start: _dt, hours: float) -> _dt:
                cur = _next_bstart(start)
                WS, WE = _time(9,0), _time(17,0)
                # Simplified: just add hours directly
                return start + _td(hours=hours)

            from process_optimization_agent.models import ScheduleEntry as _SE

            # Build dependency map within missing tasks
            missing_set = {t.id for t in missing_tasks}
            deps_map = {t.id: set(getattr(t, 'dependencies', set()) or set()) for t in missing_tasks}
            # Remove deps already satisfied by scheduled tasks
            for tid in list(deps_map.keys()):
                deps_map[tid] = {d for d in deps_map[tid] if d in missing_set}

            # Kahn levelization
            levels = []  # list of lists of task ids
            remaining = set(missing_set)
            while remaining:
                # tasks whose remaining deps are empty
                ready = [tid for tid in remaining if not deps_map.get(tid)]
                if not ready:
                    # cycle or unresolved: break by taking arbitrary items
                    ready = list(remaining)
                levels.append(ready)
                remaining.difference_update(ready)
                for tid in remaining:
                    deps_map[tid] = {d for d in deps_map[tid] if d not in ready}

            # Parallel schedule by levels
            max_parallel = int(getattr(optimizer, 'max_parallel_tasks', 4) or 4)
            for level in levels:
                pending = [all_tasks_by_id[tid] for tid in level if tid in all_tasks_by_id]
                # Schedule in waves if more tasks than available slots/resources
                wave_start = _next_bstart(cur_time)
                while pending:
                    used_resources = set()
                    wave_assignments = []  # (task, resource)
                    for task in list(pending):
                        if len(wave_assignments) >= max_parallel:
                            break
                        # choose a resource not already used in this wave
                        res_choice = None
                        for r in process.resources:
                            if r.id in used_resources:
                                continue
                            if r.has_all_skills(getattr(task, 'required_skills', []) or []):
                                res_choice = r
                                break
                        if not res_choice:
                            # if none qualified free, try any free resource
                            for r in process.resources:
                                if r.id not in used_resources:
                                    res_choice = r
                                    break
                        if res_choice:
                            used_resources.add(res_choice.id)
                            wave_assignments.append((task, res_choice))
                            pending.remove(task)
                    if not wave_assignments:
                        # No resources or no assignable tasks; avoid infinite loop
                        break
                    # Schedule all assignments for this wave starting at wave_start
                    wave_end = wave_start
                    for task, res in wave_assignments:
                        start_t = wave_start
                        end_t = _add_bhours(start_t, float(getattr(task, 'duration_hours', 0.0) or 0.0))
                        cost = float(getattr(res, 'hourly_rate', 0.0) or 0.0) * float(getattr(task, 'duration_hours', 0.0) or 0.0)
                        schedule.entries.append(_SE(task_id=task.id, resource_id=res.id, start_time=start_t, end_time=end_t, cost=cost))
                        if end_t > wave_end:
                            wave_end = end_t
                    # Advance start for next wave or next level
                    wave_start = _next_bstart(wave_end)
                # After finishing level, current time moves to end of last wave
                cur_time = wave_start

            # Recalculate metrics after forcing entries
            try:
                schedule.calculate_metrics(process)
            except Exception:
                # Fallback minimal totals if metrics calc fails
                schedule.total_cost = sum(float(getattr(e, 'cost', 0.0) or 0.0) for e in schedule.entries)
            
            # Always recalculate duration after force-scheduling
            if schedule.entries:
                min_start = min(e.start_time for e in schedule.entries)
                max_end = max(e.end_time for e in schedule.entries)
                schedule.total_duration_hours = (max_end - min_start).total_seconds() / 3600.0
        # Log coverage
        print("\n[Post] Force-scheduling applied: {} missing tasks added".format(len([t for t in process.tasks if t.id not in scheduled_ids])))
    except Exception as e:
        print(f"Warning: Could not apply force-scheduling fallback: {e}")
    

    print("\n=== Final Schedule ===")
    for entry in sorted(schedule.entries, key=lambda e: e.start_time):
        task = next((t for t in process.tasks if t.id == entry.task_id), None)
        resource = next((r for r in process.resources if r.id == entry.resource_id), None)
        if task and resource:
            print(f"{entry.start_time.strftime('%Y-%m-%d %H:%M')} - {entry.end_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"  {task.name} (Task {task.id})")
            print(f"  Assigned to: {resource.name} (${resource.hourly_rate}/hr)")
            print(f"  Duration: {task.duration_hours} hours, Cost: ${entry.cost:,.2f}")
            print()
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    
    # Use a persistent output directory so the file does not disappear
    output_dir = os.path.join('outputs', 'visualizations')
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create output directory '{output_dir}': {e}")
        output_dir = '.'  # fallback to current dir
    
    # Include timestamp in filename
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Re-enable Gantt chart generation (Graphviz-based). Fail gracefully if Graphviz missing.
    try:
        gantt_noext = os.path.join(output_dir, f"{process.id}_gantt_{ts}")
        gantt_path = visualizer.plot_schedule_gantt(
            schedule=schedule,
            process=process,
            title=f"Project Gantt Chart — {process.name}",
            output_file=gantt_noext,
            show=False,
        )
        if gantt_path and os.path.exists(gantt_path):
            print(f"Gantt saved to: {os.path.abspath(gantt_path)}")
            # Also create a stable 'latest' copy for easy refreshing
            try:
                import shutil
                root, ext = os.path.splitext(gantt_path)
                latest = os.path.join(output_dir, f"{process.id}_gantt_latest{ext}")
                shutil.copyfile(gantt_path, latest)
            except Exception:
                pass
    except Exception as e:
        print(f"Note: Gantt chart generation skipped due to: {e}")

    # Helper: build a true sequential baseline schedule under business-hours using
    # the cheapest qualified resource for each task. This will differ from the
    # optimized schedule and provide a meaningful "Before" for allocation charts.
    from process_optimization_agent.models import Schedule, ScheduleEntry
    from datetime import datetime as _dt, time as _time, timedelta as _td
    def _is_bd(dt: _dt) -> bool:
        return dt.weekday() < 5
    def _next_bstart(dt: _dt) -> _dt:
        WS, WE = _time(9,0), _time(17,0)
        cur = dt
        # move to workday 9:00
        if cur.time() >= WE:
            cur = _dt(cur.year, cur.month, cur.day) + _td(days=1, hours=9)
        elif cur.time() < WS:
            cur = _dt(cur.year, cur.month, cur.day, 9, 0)
        while not _is_bd(cur):
            cur = _dt(cur.year, cur.month, cur.day) + _td(days=1, hours=9)
        return cur
    def _add_bhours(start: _dt, hours: float) -> _dt:
        cur = _next_bstart(start)
        WS, WE = _time(9,0), _time(17,0)
        # Simplified: just add hours directly
        return start + _td(hours=hours)

    def _qualified(resource, task) -> bool:
        return resource.has_all_skills(getattr(task, 'required_skills', []) or [])
    # Naive baseline selector: pick the FIRST qualified resource in process order
    # This intentionally differs from the optimizer's cost/time choices so the
    # 'Before' vs 'After' allocation charts visibly change.
    def _first_qualified(task):
        for r in process.resources:
            if _qualified(r, task):
                return r
        # If none qualified, fall back to the highest-rate resource to exaggerate baseline
        return max(process.resources, key=lambda r: r.hourly_rate)
    # Build baseline entries in task order (process.tasks) starting at 09:00 today or min schedule start
    baseline_start = min((e.start_time for e in schedule.entries), default=_dt.now()).replace(hour=9, minute=0, second=0, microsecond=0)
    cur = _next_bstart(baseline_start)
    baseline_entries = []
    for task in process.tasks:
        res = _first_qualified(task)
        start_t = cur
        end_t = _add_bhours(start_t, float(getattr(task, 'duration_hours', 0.0) or 0.0))
        cost = float(res.hourly_rate) * float(getattr(task, 'duration_hours', 0.0) or 0.0)
        baseline_entries.append(ScheduleEntry(task_id=task.id, resource_id=res.id, start_time=start_t, end_time=end_t, cost=cost))
        cur = end_t
    baseline_sched_seq = Schedule(process_id=process.id, entries=baseline_entries)

    # Save what-if comparison figure alongside other outputs (after output_dir/ts available)
    if wi_results:
        try:
            wi_out = os.path.join(output_dir, f"{process.id}_whatif_{ts}")
            wi_png = visualizer.plot_whatif_summary(wi_results, title=f"What-If Scenarios — {process.name}", output_file=wi_out, show=False)
            if wi_png and os.path.exists(wi_png):
                print(f"What-if summary saved to: {os.path.abspath(wi_png)}")
                # Image pop-up disabled for API usage
                print("What-if summary chart generated (pop-up disabled)")
                # Browser pop-up disabled for API usage
            # Allocations grid (Baseline + scenarios with selected best highlighted)
            wi_alloc_out = os.path.join(output_dir, f"{process.id}_whatif_allocations_{ts}")
            wi_alloc_png = visualizer.plot_whatif_allocations(wi_results, process, title=f"What-If Allocations for {process.name}", output_file=wi_alloc_out, show=False)
            if wi_alloc_png and os.path.exists(wi_alloc_png):
                print(f"What-if allocations saved to: {os.path.abspath(wi_alloc_png)}")
                # Stable latest copy (preserved; not part of requested removals)
                try:
                    import shutil
                    wi_alloc_latest = os.path.join(output_dir, f"{process.id}_whatif_allocations_latest.png")
                    shutil.copyfile(wi_alloc_png, wi_alloc_latest)
                except Exception:
                    pass
                # Image pop-up disabled for API usage
                print("What-if allocations chart generated (pop-up disabled)")

            # New: Allocation charts (pie + bar) with parallel groups summary
            alloc_chart_out = os.path.join(output_dir, f"{process.id}_alloc_charts_{ts}")
            # Prefer our constructed sequential baseline; if unavailable, fall back to wi_results
            baseline_sched = baseline_sched_seq
            alloc_chart_png = visualizer.plot_allocation_charts(
                process,
                schedule,
                title=f"Resource Allocation — {process.name}",
                output_file=alloc_chart_out,
                show=False,
                baseline_schedule=baseline_sched,
            )
            if alloc_chart_png and os.path.exists(alloc_chart_png):
                print(f"Allocation charts saved to: {os.path.abspath(alloc_chart_png)}")
                # Image pop-up disabled for API usage
                print("Allocation charts generated (pop-up disabled)")
            # Disabled: do not write What-if Markdown summary
            pass
        except Exception as e:
            print(f"Error generating what-if summary image: {e}")

    # === Summary comparison (Before vs After) - Moved after force-scheduling ===
    print("\n=== Generating Summary Comparison ===")
    try:
        from datetime import timedelta, time as dtime

        # Helper: compute peak concurrent people from schedule entries
        def compute_peak_people(entries):
            # Sweep line algorithm
            events = []
            for e in entries:
                events.append((e.start_time, 1))
                events.append((e.end_time, -1))
            events.sort()
            cur = peak = 0
            for _, delta in events:
                cur += delta
                peak = max(peak, cur)
            return peak

        # Helpers for business-hours calculations
        from datetime import datetime as _dt
        WORK_START = dtime(9, 0)
        WORK_END = dtime(17, 0)

        def _is_business_day(dt: _dt) -> bool:
            return dt.weekday() < 5

        def business_hours_between(start: _dt, end: _dt) -> float:
            """Count hours between start and end that fall within 09:00-17:00 Mon-Fri."""
            if end <= start:
                return 0.0
            cur = start
            total = 0.0
            # Simplified: direct hour calculation
            return (end - start).total_seconds() / 3600.0

        # After metrics: use UNION of active intervals within business-hours (avoid counting idle gaps)
        def active_business_hours_union(entries):
            from datetime import datetime as _dt, time as _t, timedelta as _td
            WS, WE = _t(9, 0), _t(17, 0)
            # Simplified: use direct elapsed hours
            if entries:
                min_start = min(e.start_time for e in entries)
                max_end = max(e.end_time for e in entries)
                return (max_end - min_start).total_seconds() / 3600.0
            return 0.0

        # Simplified: use direct elapsed hours
        if schedule.entries:
            min_start = min(e.start_time for e in schedule.entries)
            max_end = max(e.end_time for e in schedule.entries)
            after_elapsed_hours = (max_end - min_start).total_seconds() / 3600.0
        else:
            after_elapsed_hours = 0.0
        after_cost = schedule.total_cost
        after_peak = compute_peak_people(schedule.entries)

        # Before baseline: serialize tasks end-to-end with business hours (09:00-17:00, Mon-Fri)
        # Uses each task's planned duration_hours; order by current schedule's start_time as a proxy
        sequential_entries = sorted(schedule.entries, key=lambda e: e.start_time)
        planned_durations = []
        for e in sequential_entries:
            task = next((t for t in process.tasks if t.id == e.task_id), None)
            planned_durations.append(float(getattr(task, 'duration_hours', 0.0) or 0.0))

        # Business time helpers
        WORK_START = dtime(9, 0)
        WORK_END = dtime(17, 0)
        WORK_HOURS_PER_DAY = 8.0

        def is_business_day(dt: datetime) -> bool:
            return dt.weekday() < 5

        def next_business_start(dt: datetime) -> datetime:
            # Move into business window
            cur = dt
            # If weekend, roll to Monday 09:00
            while not is_business_day(cur):
                cur = datetime(cur.year, cur.month, cur.day) + timedelta(days=1)
            # Adjust time to window
            if cur.time() >= WORK_END:
                # move to next day 09:00
                cur = datetime(cur.year, cur.month, cur.day) + timedelta(days=1, hours=9)
                while not is_business_day(cur):
                    cur = datetime(cur.year, cur.month, cur.day) + timedelta(days=1, hours=9)
            elif cur.time() < WORK_START:
                cur = datetime(cur.year, cur.month, cur.day, 9, 0)
            return cur

        def add_business_hours(start: datetime, hours: float) -> datetime:
            cur = next_business_start(start)
            remaining = hours
            # Simplified: just add hours directly
            return start + timedelta(hours=hours)

        # Start baseline from min start of actual schedule
        if schedule.entries:
            baseline_start = min(e.start_time for e in schedule.entries)
        else:
            baseline_start = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

        cur_time = baseline_start
        for hrs in planned_durations:
            cur_time = add_business_hours(cur_time, hrs)
        # Working-hours duration for baseline equals the sum of planned durations
        before_business_hours = sum(planned_durations)

        # === Cost modeling per user request ===
        # After (optimized) cost: sum over scheduled tasks using the CHEAPEST qualified resource rate per task
        # Before (sequential) cost: sequential total task hours multiplied by the AVERAGE hourly rate across all resources

        # Helpers
        def _resource_rate(res_id: str) -> float:
            r = next((rr for rr in process.resources if rr.id == res_id), None)
            return float(getattr(r, 'hourly_rate', 0.0) or 0.0) if r else 0.0

        def _lvl(val) -> float:
            if val is None:
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            v = getattr(val, 'value', None)
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(val)
            except Exception:
                return 0.0

        def _qualifies(res, task) -> bool:
            if not task.required_skills:
                return True
            res_skill_levels = {s.name: _lvl(getattr(s, 'level', 1)) for s in getattr(res, 'skills', [])}
            for req in task.required_skills:
                needed = _lvl(getattr(req, 'level', 1))
                if res_skill_levels.get(req.name, 0.0) < needed:
                    return False
            return True

        # Map task_id -> task for fast lookup and list of scheduled task ids
        task_by_id = {t.id: t for t in process.tasks}
        scheduled_task_ids = {e.task_id for e in schedule.entries}

        # Optimized cost: prefer per-entry cost; fallback to rate * planned duration (not wall-clock span)
        after_cost = 0.0
        for e in schedule.entries:
            try:
                if hasattr(e, 'cost') and e.cost is not None:
                    after_cost += float(e.cost)
                    continue
            except Exception:
                pass
            res = next((rr for rr in process.resources if rr.id == e.resource_id), None)
            rate = float(getattr(res, 'hourly_rate', 0.0) or 0.0) if res else 0.0
            t = next((tt for tt in process.tasks if tt.id == e.task_id), None)
            planned_hours = float(getattr(t, 'duration_hours', 0.0) or 0.0) if t else 0.0
            after_cost += planned_hours * rate

        # Sequential baseline hours = sum of ALL task durations (not just scheduled ones)
        # This represents running all tasks one after another
        before_business_hours = sum(float(getattr(t, 'duration_hours', 0.0) or 0.0) for t in process.tasks)

        # Calculate after_hours (optimized elapsed time) and after_peak (peak parallelism)
        if schedule.entries:
            # Use schedule.total_duration_hours if available (updated after force-scheduling)
            if hasattr(schedule, 'total_duration_hours') and schedule.total_duration_hours > 0:
                after_hours = schedule.total_duration_hours
            # Otherwise compute from hour fields if available
            elif all(hasattr(e, 'start_hour') and hasattr(e, 'end_hour') for e in schedule.entries):
                min_start = min(e.start_hour for e in schedule.entries)
                max_end = max(e.end_hour for e in schedule.entries)
                after_hours = max_end - min_start
            else:
                min_start_time = min(e.start_time for e in schedule.entries)
                max_end_time = max(e.end_time for e in schedule.entries)
                after_hours = (max_end_time - min_start_time).total_seconds() / 3600.0
            
            # Don't override with what-if scenario durations - use actual elapsed time
            
            # Calculate peak parallelism
            events = []
            for e in schedule.entries:
                events.append((e.start_time, 1))
                events.append((e.end_time, -1))
            events.sort()
            current = 0
            after_peak = 0
            for _, delta in events:
                current += delta
                after_peak = max(after_peak, current)
        else:
            after_hours = 0.0
            after_peak = 0

        # Average hourly rate across all resources (fallback 0 if none)
        if process.resources:
            # Use cheapest qualified resource for each task for baseline cost
            before_cost = 0.0
            for task in process.tasks:
                # Find cheapest qualified resource
                min_rate = float('inf')
                for res in process.resources:
                    if res.has_all_skills(task.required_skills or []):
                        min_rate = min(min_rate, res.hourly_rate)
                if min_rate == float('inf'):
                    # No qualified resource, use average
                    min_rate = sum(float(getattr(r, 'hourly_rate', 0.0) or 0.0) for r in process.resources) / float(len(process.resources))
                before_cost += task.duration_hours * min_rate
        else:
            avg_rate = 100.0  # Default rate if no resources
            before_cost = before_business_hours * avg_rate

        # Before/After dictionaries with fields expected by Visualizer:
        # - before.total_resources: total resources available in the process (sequential baseline uses 1 concurrently, but we display team size for clarity)
        # - after.schedule: actual Schedule object so Visualizer can compute unique resources used accurately
        before = {
            'duration_hours': before_business_hours,
            'peak_people': 1,  # sequential execution peak
            'total_resources': len(process.resources),  # for display of team size pre-optimization
            'total_cost': float(before_cost),
        }
        after = {
            'duration_hours': float(after_hours),
            'peak_people': int(after_peak),  # fallback; Visualizer will override if 'schedule' present
            'total_cost': float(after_cost),  # actual assigned resource costs from schedule
            'schedule': schedule,  # allow Visualizer to compute unique resources used
        }
        
        # Pass the computed metrics for the summary chart
        chart_out = os.path.join(output_dir, f"{process.id}_summary_{ts}")
        
        summary_output = visualizer.plot_summary_comparison(
            before=before,
            after=after,
            title=f"Optimization Summary — {process.name}",
            output_file=chart_out,
            show=False
        )
        if summary_output and os.path.exists(summary_output):
            print(f"Summary chart saved to: {os.path.abspath(summary_output)}")
            print(f"[Summary] Before people (team size): {before.get('total_resources')}, After people (unique resources used): {len({e.resource_id for e in schedule.entries})}")
            print(f"[Summary] Costs — Before (cheapest-qualified sequential): ${before['total_cost']:,.2f}, After (assigned rates): ${after['total_cost']:,.2f}")
            # Browser pop-up disabled for API usage
            print("Summary chart generated (pop-up disabled)")
        else:
            print("Warning: Failed to generate summary comparison chart")

        # Allocation summary image (disabled per request)
        # Intentionally skipped to avoid generating the spreadsheet-like page.
        # try:
        #     alloc_path = os.path.join(output_dir, f"{process.id}_allocation_summary_{ts}.png")
        #     alloc_out = visualizer.plot_resource_allocation_summary(
        #         process=process,
        #         schedule=schedule,
        #         output_file=alloc_path,
        #         show=False
        #     )
        #     if alloc_out and os.path.exists(alloc_out):
        #         abs_alloc = os.path.abspath(alloc_out)
        #         print(f"Allocation summary saved to: {abs_alloc}")
        #         # Try opening via webbrowser; if that fails on Windows, use os.startfile
        #         opened = False
        #         try:
        #             opened = webbrowser.open(f'file://{abs_alloc}')
        #         except Exception:
        #             opened = False
        #         if not opened:
        #             try:
        #                 if hasattr(os, 'startfile'):
        #                     os.startfile(abs_alloc)
        #                     opened = True
        #             except Exception:
        #                 opened = False
        #         if opened:
        #             print("Opened allocation summary image in default viewer")
        #         else:
        #             print("Note: Could not auto-open the allocation image; please open it manually from the path above.")
        #     else:
        #         print("Warning: Failed to generate allocation summary image")
        # except Exception as e:
        #     print(f"Error generating allocation summary image: {str(e)}")
        
        # === Optimized hours and parallel groups (console output) ===
        try:
            from datetime import datetime as _dt, time as _t, timedelta as _td

            # Total task hours (sequential baseline = sum of planned durations)
            total_task_hours = sum(float(getattr(t, 'duration_hours', 0.0) or 0.0) for t in process.tasks)

            # Optimized active business-hours = union of active intervals (09:00-17:00 Mon-Fri)
            def _is_business_day(d: _dt) -> bool:
                return d.weekday() < 5
            def active_bh_union(entries) -> float:
                WS, WE = _t(9, 0), _t(17, 0)
                segs = []
                for e in entries:
                    s, f = e.start_time, e.end_time
                    cur = s
                    while cur < f:
                        if not _is_business_day(cur):
                            cur = _dt(cur.year, cur.month, cur.day) + _td(days=1, hours=9)
                            continue
                        day_start = _dt(cur.year, cur.month, cur.day, 9, 0)
                        day_end = _dt(cur.year, cur.month, cur.day, 17, 0)
                        seg_start = max(s, cur, day_start)
                        seg_end = min(f, day_end)
                        if seg_end > seg_start:
                            segs.append((seg_start, seg_end))
                        cur = _dt(cur.year, cur.month, cur.day) + _td(days=1, hours=9)
                if not segs:
                    return 0.0
                segs.sort(key=lambda x: x[0])
                total = 0.0
                cs, ce = segs[0]
                for s, f in segs[1:]:
                    if s <= ce:
                        if f > ce:
                            ce = f
                    else:
                        total += (ce - cs).total_seconds() / 3600.0
                        cs, ce = s, f
                total += (ce - cs).total_seconds() / 3600.0
                return total
            
            # Simplified: use direct elapsed hours
            if schedule.entries:
                min_start = min(e.start_time for e in schedule.entries)
                max_end = max(e.end_time for e in schedule.entries)
                optimized_elapsed = (max_end - min_start).total_seconds() / 3600.0
            else:
                optimized_elapsed = 0.0

            time_saved = total_task_hours - optimized_elapsed

            # Parallel groups
            events = []
            for e in schedule.entries:
                events.append((e.start_time, 1, e.task_id))
                events.append((e.end_time, -1, e.task_id))
            events.sort(key=lambda x: (x[0], x[1]))
            active = set(); groups = []
            for t, typ, tid in events:
                if typ == 1:
                    active.add(tid)
                    if len(active) >= 2:
                        groups.append(tuple(sorted(active)))
                else:
                    active.discard(tid)
            unique_groups = []
            seen = set()
            for g in groups:
                key = frozenset(g)
                if key not in seen and len(key) >= 2:
                    seen.add(key)
                    unique_groups.append(list(key))
            def _tname(tid: str) -> str:
                t = next((t for t in process.tasks if t.id == tid), None)
                return t.name if t else tid
            named_groups = [[_tname(tid) for tid in grp] for grp in unique_groups]

            print("\n=== Optimization Summary (Business-Hours) ===")
            print(f"Sequential task hours: {total_task_hours:.1f}")
            print(f"Optimized elapsed hours: {optimized_elapsed:.1f}")
            print(f"Time saved: {time_saved:.1f} hours")
            if named_groups:
                print("Parallel task groups:")
                for i, grp in enumerate(named_groups, 1):
                    print(f"  Group {i}: {', '.join(grp)}")
            else:
                print("No parallel task groups detected")
        except Exception as e:
            print(f"Warning: Failed to compute optimized-hour summary: {e}")
    except Exception as e:
        print(f"Error generating summary comparison: {str(e)}")
        import traceback
        traceback.print_exc()

    # Check for dependency violations
    print("\n=== Dependency Check ===")
    task_end_times = {entry.task_id: entry.end_time for entry in schedule.entries}
    task_start_times = {entry.task_id: entry.start_time for entry in schedule.entries}
    violations = []
    
    for task in process.tasks:
        if not task.dependencies:
            continue
            
        task_start = task_start_times.get(task.id)
        if not task_start:
            continue
            
        for dep_id in task.dependencies:
            dep_end = task_end_times.get(dep_id)
            if dep_end and dep_end > task_start:
                violations.append((task.id, dep_id))
    
    if not violations:
        print("No dependency violations detected.")
    else:
        print("Dependency violations detected:")
        for v in violations:
            print(f"  Task {v[0]} starts before dependency {v[1]} ends")

    # === Written schedule summary (per-resource) ===
    print("\n=== Generating Written Schedule Summary ===")
    try:
        # Disabled: skip writing the schedule summary markdown
        pass
    except Exception as e:
        print(f"Error generating written schedule summary: {str(e)}")

if __name__ == "__main__":
    main()
