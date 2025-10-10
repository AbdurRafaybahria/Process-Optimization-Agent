"""
Optimization engines for the Process Optimization Agent
"""

import random
import numpy as np
import time
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import heapq
import json
import pickle

from .models import Task, Resource, Process, Schedule, ScheduleEntry, Skill, SkillLevel
from .analyzers import DependencyDetector, DeadlockDetector


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers"""
    
    def __init__(self):
        self.dependency_detector = DependencyDetector()
        self.deadlock_detector = DeadlockDetector()
    
    @abstractmethod
    def optimize(self, process: Process) -> Schedule:
        """Optimize the process and return a schedule"""
        pass
    
    def _detect_and_apply_dependencies(self, process: Process):
        """Detect and apply dependencies to tasks"""
        detected_deps = self.dependency_detector.detect_dependencies(process.tasks)
        validated_deps = self.dependency_detector.validate_dependencies(process.tasks, detected_deps)
        
        # Apply detected dependencies to tasks
        for task in process.tasks:
            if not hasattr(task, 'dependencies') or task.dependencies is None:
                task.dependencies = set()
                
            if task.id in validated_deps:
                # Convert dependencies to a set if they're not already
                deps = validated_deps[task.id]
                if isinstance(deps, (list, set)):
                    task.dependencies.update(deps)
                else:
                    task.dependencies.add(deps)


class ProcessOptimizer(BaseOptimizer):
    """Greedy rule-based optimizer for baseline scheduling"""
    
    def __init__(self, optimization_strategy: str = "balanced"):
        """
        Initialize optimizer with strategy
        
        Strategies:
        - "time": Minimize total duration
        - "cost": Minimize total cost  
        - "balanced": Balance time and cost
        """
        super().__init__()
        self.strategy = optimization_strategy
    
    def optimize(self, process: Process, max_retries: int = 3) -> Schedule:
        """
        Optimize using simplified sequential or parallel scheduling
        
        Args:
            process: The process to optimize
            max_retries: Maximum number of retries for deadlock resolution
            
        Returns:
            Schedule: The optimized schedule
        """
        # Detect and apply dependencies first
        self._detect_and_apply_dependencies(process)
        
        # Initialize schedule
        schedule = Schedule(process_id=process.id)
        
        # Track resource availability in hours from project start
        resource_next_available = {r.id: 0.0 for r in process.resources}  # Hour when resource is next available
        resource_workload = {r.id: 0.0 for r in process.resources}  # Total hours assigned
        completed_tasks = set()
        scheduled_tasks = set()
        
        # Priority queue for ready tasks (priority, task_id)
        ready_tasks = []
        
        # Track which tasks we've tried to schedule
        failed_attempts = defaultdict(int)
        
        # Initialize with tasks that have no dependencies
        for task in process.tasks:
            if not task.dependencies:  # No dependencies
                priority = self._calculate_task_priority(task, process)
                heapq.heappush(ready_tasks, (priority, task.id))
        
        retry_count = 0
        
        while (ready_tasks or len(scheduled_tasks) < len(process.tasks)) and retry_count < max_retries:
            if not ready_tasks:
                # No ready tasks - check for deadlocks
                deadlocks = self.deadlock_detector.detect_deadlocks(process, schedule)
                if deadlocks:
                    print(f"[WARNING] Detected {len(deadlocks)} deadlocks")
                    schedule.deadlocks_detected = [d.get('task_id', 'unknown') for d in deadlocks]
                    
                    # Try to resolve deadlocks
                    if retry_count < max_retries - 1:
                        print(f"[INFO] Attempting to resolve deadlocks (attempt {retry_count + 1}/{max_retries})")
                        # Break some dependencies to resolve deadlocks
                        def log_callback(task_id, dep_id):
                            print(f"[AUTO-FIX] Removing dependency: {task_id} -> {dep_id}")
                        
                        if self.deadlock_detector.resolve_dependency_deadlocks(process, log_callback):
                            # Reset and retry with updated dependencies
                            retry_count += 1
                            completed_tasks.clear()
                            scheduled_tasks.clear()
                            ready_tasks = []
                            resource_availability = {r.id: process.start_date for r in process.resources}
                            
                            # Re-initialize ready tasks
                            for task in process.tasks:
                                if not task.dependencies:
                                    priority = self._calculate_task_priority(task, process)
                                    heapq.heappush(ready_tasks, (priority, task.id))
                            continue
                break
            
            # Get highest priority ready task
            _, task_id = heapq.heappop(ready_tasks)
            task = process.get_task_by_id(task_id)
            
            if not task:
                print(f"[DEBUG] Task with ID {task_id} not found in process")
                continue
                
            if task_id in scheduled_tasks:
                print(f"[DEBUG] Task {task_id} already scheduled, skipping")
                continue
            
            # Skip if we've tried this task too many times
            if failed_attempts[task_id] > 2:
                print(f"[WARNING] Skipping task {task_id} after multiple failed attempts"
                      f" (required skills: {[f'{s.name}:{s.level.name}' for s in task.required_skills]})")
                continue
            
            # Find best resource for this task using simplified approach
            print(f"[DEBUG] Finding best resource for task {task_id} (required skills: {[f'{s.name}:{s.level.name}' for s in task.required_skills]})")
            best_resource, start_hour = self._find_best_resource_simple(
                task, process, resource_next_available, resource_workload
            )
            
            if best_resource:
                print(f"[DEBUG] Found resource {best_resource.id} for task {task_id}")
                # Use task's duration directly
                duration_hours = task.duration_hours
                end_hour = start_hour + duration_hours
                cost = duration_hours * best_resource.hourly_rate
                
                # Create schedule entry with both datetime (for compatibility) and hour values
                entry = ScheduleEntry(
                    task_id=task.id,
                    resource_id=best_resource.id,
                    start_time=process.start_date + timedelta(hours=start_hour),
                    end_time=process.start_date + timedelta(hours=end_hour),
                    start_hour=start_hour,
                    end_hour=end_hour,
                    cost=cost
                )
                
                schedule.entries.append(entry)
                scheduled_tasks.add(task.id)
                
                # Update resource tracking
                resource_next_available[best_resource.id] = end_hour
                resource_workload[best_resource.id] += duration_hours
                
                # Mark task as completed for dependency checking
                completed_tasks.add(task.id)
                failed_attempts.pop(task_id, None)  # Clear any failed attempts
                
                # Check for newly ready tasks
                for other_task in process.tasks:
                    if (other_task.id not in scheduled_tasks and 
                        other_task.id not in [t[1] for t in ready_tasks] and
                        other_task.can_start(completed_tasks)):
                        priority = self._calculate_task_priority(other_task, process)
                        heapq.heappush(ready_tasks, (priority, other_task.id))
            else:
                # No available resource - try again later
                failed_attempts[task_id] = failed_attempts.get(task_id, 0) + 1
                if failed_attempts[task_id] <= 2:  # Only retry a few times
                    priority = self._calculate_task_priority(task, process)
                    heapq.heappush(ready_tasks, (priority, task.id))
                    print(f"[DEBUG] No resource found for task {task_id}, will retry (attempt {failed_attempts[task_id]}/2)")
                else:
                    print(f"[WARNING] Failed to schedule task {task_id} after multiple attempts")
                    print(f"[DEBUG] Required skills: {[f'{s.name}:{s.level.name}' for s in task.required_skills]}")
                    print(f"[DEBUG] Available resources: {[r.id for r in process.resources]}")
        
        # Calculate final metrics
        schedule.calculate_metrics(process)
        
        # Detect remaining deadlocks
        deadlocks = self.deadlock_detector.detect_deadlocks(process, schedule)
        if deadlocks:
            for deadlock in deadlocks:
                if deadlock.get('task_id') and deadlock['task_id'] not in schedule.deadlocks_detected:
                    schedule.deadlocks_detected.append(deadlock['task_id'])
        
        return schedule
    
    def _calculate_task_priority(self, task: Task, process: Process) -> float:
        """Return scheduling key based strictly on explicit order.
        Lower number = scheduled earlier. If order is missing, push to the end.
        """
        order_val = getattr(task, 'order', None)
        return float(order_val) if order_val is not None else float('inf')
    
    def _find_best_resource_simple(self, task: Task, process: Process, 
                                  resource_next_available: Dict[str, float],
                                  resource_workload: Dict[str, float]) -> Tuple[Optional[Resource], Optional[float]]:
        """Find the best available resource using simplified logic"""
        best_resource = None
        best_start_hour = float('inf')
        best_score = float('-inf')
        
        required_skills = task.required_skills
        
        for resource in process.resources:
            # Check if resource has required skills
            if required_skills and not resource.has_all_skills(required_skills):
                continue
            
            # Check if resource has capacity
            if resource_workload[resource.id] + task.duration_hours > resource.total_available_hours:
                continue
                
            # Get earliest available hour for this resource
            available_hour = resource_next_available.get(resource.id, 0.0)
            
            # Calculate score based on optimization strategy
            score = self._calculate_resource_score_simple(task, resource, available_hour, resource_workload)
            
            # Select based on strategy
            if self.strategy == "time":
                # Prefer earliest available resource
                if available_hour < best_start_hour:
                    best_resource = resource
                    best_start_hour = available_hour
                    best_score = score
            elif self.strategy == "cost":
                # Prefer lower cost resources
                if score > best_score:
                    best_resource = resource
                    best_start_hour = available_hour
                    best_score = score
            else:
                # Balanced: consider both time and cost
                if score > best_score:
                    best_resource = resource
                    best_start_hour = available_hour
                    best_score = score
        
        return best_resource, best_start_hour if best_resource else None
    
    def _calculate_resource_score_simple(self, task: Task, resource: Resource, 
                                        available_hour: float, resource_workload: Dict[str, float]) -> float:
        """Calculate resource assignment score for simplified scheduling"""
        score = 0.0
        
        if self.strategy == "cost":
            # Lower cost is better (invert for higher score)
            score = 1000.0 / (resource.hourly_rate + 1.0)
        elif self.strategy == "time":
            # Earlier availability is better
            score = 1000.0 / (available_hour + 1.0)
        else:  # balanced
            # Balance cost and time
            cost_score = 500.0 / (resource.hourly_rate + 1.0)
            time_score = 500.0 / (available_hour + 1.0)
            score = cost_score + time_score
        
        # Bonus for matching skills
        if task.required_skills:
            skill_score = resource.get_skill_score(task.required_skills)
            score += skill_score * 100.0
        
        # Penalty for high workload (prefer balanced workload)
        workload_ratio = resource_workload[resource.id] / max(resource.total_available_hours, 1.0)
        score -= workload_ratio * 50.0
        
        return score
        
    def _get_next_available_slot(self, resource: Resource, timestamp: datetime) -> datetime:
        """Get next available time slot considering working hours and availability"""
        current = timestamp
        # Derive working hours with sane defaults if not present on resource
        start_hour = getattr(resource, 'working_hours_start', 9)
        end_hour = getattr(resource, 'working_hours_end', 17)
        try:
            start_hour = int(start_hour)
            end_hour = int(end_hour)
        except Exception:
            start_hour, end_hour = 9, 17

        # Normalize misconfigured hours
        if end_hour <= start_hour:
            end_hour = min(start_hour + 8, 24)

        # Check if current time is outside working hours
        if current.hour >= end_hour:
            # Move to next working day at start_hour
            next_day = current.date() + timedelta(days=1)
            current = datetime.combine(
                next_day,
                datetime.min.time().replace(hour=start_hour, minute=0, second=0, microsecond=0)
            )
        # If before working hours, move to start of working hours
        elif current.hour < start_hour:
            current = current.replace(
                hour=start_hour,
                minute=0,
                second=0,
                microsecond=0
            )

        return current
    
    def _calculate_reward_simple(self, task: Task, resource: Resource, 
                                start_hour: float, resource_workload: Dict[str, float]) -> float:
        """Calculate reward prioritizing time reduction and parallelization"""
        # Time component: STRONGLY prefer earlier completion and parallelization
        # Higher reward for tasks that start earlier (enables parallelization)
        time_penalty = start_hour + task.duration_hours  # Total completion time
        time_factor = 100.0 / (1.0 + time_penalty)  # Strong preference for early completion
        
        # Parallelization bonus: reward if task runs in parallel with others
        parallel_bonus = 0.0
        min_resource_available = min(resource_workload.values()) if resource_workload else 0
        if start_hour <= min_resource_available + 5:  # Task starts within 5 hours of earliest availability
            parallel_bonus = 20.0  # Significant bonus for parallelization
        
        # Cost component: moderate penalty for expensive resources
        max_rate = 200.0
        cost_penalty = (resource.hourly_rate / max_rate) * 10.0  # Moderate cost consideration
        
        # Load balancing: prefer balanced workload
        avg_workload = sum(resource_workload.values()) / len(resource_workload) if resource_workload else 0
        current_workload = resource_workload.get(resource.id, 0)
        load_factor = 5.0 / (1.0 + abs(current_workload - avg_workload) * 0.1)
        
        # Skill match: prefer resources with better skill match
        skill_factor = 1.0
        if task.required_skills and resource.skills:
            matched_skills = 0
            total_skills = len(task.required_skills)
            resource_skill_names = {s.name for s in resource.skills}
            for req_skill in task.required_skills:
                if req_skill.name in resource_skill_names:
                    matched_skills += 1
            skill_factor = (matched_skills / total_skills) * 10.0 if total_skills > 0 else 5.0
        
        # Combine factors with heavy emphasis on time reduction
        reward = (
            time_factor * 2.0 +  # Double weight for time
            parallel_bonus +     # Bonus for parallelization
            load_factor +
            skill_factor -
            cost_penalty * 0.3   # Small cost consideration
        )
        
        return max(1.0, reward)
    
    def _calculate_resource_score(self, task: Task, resource: Resource, start_time: datetime, process: Process) -> float:
        """Calculate resource assignment score (lower is better)"""
        base_score = 0.0
        
        if self.strategy == "cost":
            # Prioritize cheaper resources
            base_score = resource.hourly_rate
        elif self.strategy == "time":
            # Prioritize faster/more skilled resources
            skill_score = resource.get_skill_score(task.required_skills)
            base_score = 100.0 - (skill_score * 100.0)  # Higher skill = lower score
        else:  # balanced
            # Balance cost and skill
            cost_factor = resource.hourly_rate * 0.01
            skill_score = resource.get_skill_score(task.required_skills)
            skill_factor = (100.0 - skill_score * 100.0) * 0.01
            base_score = cost_factor + skill_factor
        
        # Add penalty for delayed start relative to the process start time (not wall-clock)
        delay_hours = (start_time - process.start_date).total_seconds() / 3600.0
        base_score += max(0.0, delay_hours * 0.1)
        
        return base_score
    
    def _get_state_simple(self, process: Process, completed_tasks: Set[str],
                         resource_next_available: Dict[str, float],
                         resource_workload: Dict[str, float]) -> np.ndarray:
        """Get simplified state representation"""
        state = []
        
        # Task features (for each task: completed, ready to start)
        for task in process.tasks:
            state.append(1.0 if task.id in completed_tasks else 0.0)
            state.append(1.0 if task.can_start(completed_tasks) else 0.0)
        
        # Resource features (for each resource: next available hour and workload)
        for resource in process.resources:
            # Normalize next available hour to [0, 1]
            next_hour = resource_next_available.get(resource.id, 0.0)
            state.append(min(1.0, next_hour / process.project_duration_hours))
            
            # Normalize workload to [0, 1]
            workload = resource_workload.get(resource.id, 0.0)
            state.append(min(1.0, workload / resource.total_available_hours))
        
        return np.array(state)
    
    def _get_state(self, process: Process, completed_tasks: Set[str],
                   resource_availability: Dict[str, datetime]) -> np.ndarray:
        """Get state representation for Q-learning"""
        # Get all tasks that are ready to be scheduled
        ready_tasks = [
            task.id for task in process.tasks 
            if task.id not in completed_tasks and task.can_start(completed_tasks)
        ]
        ready_tasks.sort()
            
        # Get resource states (available time)
        resource_states = [
            f"{r.id}:{resource_availability[r.id].timestamp()}" 
            for r in process.resources
        ]
            
        resource_states.sort()
            
        state = f"tasks:{','.join(ready_tasks)}|resources:{','.join(resource_states)}"
        return state
            
    def _get_available_actions_simple(self, process: Process, completed_tasks: Set[str],
                                     resource_workload: Dict[str, float]) -> List[Tuple[str, str]]:
        """Get available actions using simplified logic with parallelization priority"""
        actions = []
        
        # Find all ready tasks (dependencies satisfied)
        ready_tasks = []
        for task in process.tasks:
            if task.id in completed_tasks:
                continue
            
            # Check dependencies
            deps_satisfied = True
            if hasattr(task, 'dependencies') and task.dependencies:
                deps_satisfied = all(dep_id in completed_tasks for dep_id in task.dependencies)
            
            if deps_satisfied:
                ready_tasks.append(task)
        
        # Sort ready tasks by duration (longer first for better parallelization)
        ready_tasks.sort(key=lambda t: t.duration_hours, reverse=True)
        
        # Create actions for all ready tasks with all compatible resources
        for task in ready_tasks:
            qualified_resources = []
            for resource in process.resources:
                # Check if resource has required skills
                if resource.has_all_skills(task.required_skills or []):
                    qualified_resources.append(resource)
                elif not task.required_skills:  # Task has no skill requirements
                    qualified_resources.append(resource)
            
            # Sort resources by current workload (least loaded first)
            qualified_resources.sort(key=lambda r: resource_workload.get(r.id, 0))
            
            # Add actions prioritizing least loaded resources
            for resource in qualified_resources:
                actions.append((task.id, resource.id))
        
        return actions
    
    def _get_available_actions(self, process: Process, completed_tasks: Set[str],
                              resource_availability: Dict[str, datetime]) -> List[Tuple[str, str]]:
        """Get available (task_id, resource_id) actions with dynamic parallel execution support
        
        Determines schedulable tasks based on dependencies satisfied by
        `completed_tasks`, then pairs them with resources that have the
        required skills, capping distinct tasks by computed parallelism.
        
        Args:
            process: The process being optimized
            completed_tasks: Set of completed task IDs
            resource_availability: Current availability time for each resource
        
        Returns:
            List[Tuple[str, str]]: List of (task_id, resource_id) actions that can be taken
        """
        available_actions: List[Tuple[str, str]] = []
        
        # Tasks ready if all dependencies are completed
        schedulable_tasks = [
            t for t in process.tasks
            if t.id not in completed_tasks and all(dep in completed_tasks for dep in getattr(t, 'dependencies', []))
        ]
        
        if not schedulable_tasks:
            return []
        
        # Calculate maximum parallelism
        max_parallel = self._calculate_max_parallelism(process, schedulable_tasks, completed_tasks)
        
        # Compute DAG levels for schedulable tasks and restrict to the earliest level window
        levels = self._compute_task_levels(process.tasks)
        min_level = min(levels.get(t.id, 0) for t in schedulable_tasks)
        level_window = [t for t in schedulable_tasks if levels.get(t.id, 0) == min_level]

        # Within the level, strictly prioritize by explicit order (lower first), fallback to ID
        def _order_key(t: Task):
            try:
                return (int(t.order) if getattr(t, 'order', None) is not None else float('inf'), t.id)
            except Exception:
                return (float('inf'), t.id)
        sorted_tasks = sorted(level_window, key=_order_key)
        
        selected_task_ids: Set[str] = set()
        # Deterministic assignment: pick the single best resource per task
        # Priority: highest level surplus over required -> earliest availability -> lowest cost
        
        # Track resources already chosen for this selection to prevent proposing
        # parallel actions that would contend for the same resource.
        used_resources: Set[str] = set()

        # Precompute eligible resources per task in this window for lookahead
        window_eligibles: Dict[str, List[Resource]] = {
            t.id: [r for r in process.resources if r.has_all_skills(getattr(t, 'required_skills', []) or [])]
            for t in sorted_tasks
        }

        for task in sorted_tasks:
            # Stop if we've reached parallel limit (distinct tasks)
            if len(selected_task_ids) >= max_parallel:
                break
            
            # Find resources that can perform the task
            suitable_resources = [
                r for r in process.resources
                if r.has_all_skills(getattr(task, 'required_skills', []) or [])
            ]
            
            if not suitable_resources:
                continue

            # Compute a deterministic ranking:
            # - Level surplus: sum(max(0, res_level - req_level)) across required skills with exact name match
            # - Exact level match count as secondary signal
            def resource_rank(r: Resource):
                surplus = 0
                exact_matches = 0
                for req in getattr(task, 'required_skills', []) or []:
                    # find matching skill on resource by name (case-insensitive)
                    match = next((s for s in r.skills if s.name.lower() == req.name.lower()), None)
                    if match:
                        delta = (match.level.value if hasattr(match.level, 'value') else int(match.level)) - (
                            req.level.value if hasattr(req.level, 'value') else int(req.level)
                        )
                        if delta >= 0:
                            surplus += delta
                            if delta == 0:
                                exact_matches += 1
                # Earlier availability preferred; lower rate preferred
                avail = resource_availability.get(r.id)
                rate = getattr(r, 'hourly_rate', 0.0)
                # Sort by: higher surplus (-surplus), more exact matches (-exact_matches), earlier avail, lower rate
                return (-surplus, -exact_matches, avail, rate)

            # Pick the single best resource for this task
            ranked_resources = sorted(suitable_resources, key=resource_rank)

            # Small lookahead heuristic: avoid consuming a resource that is unique for
            # another remaining task when we have alternatives.
            # Compute resources that are the only eligible option for any other task.
            remaining_task_ids = [t.id for t in sorted_tasks if t.id not in selected_task_ids and t.id != task.id]
            unique_resources_for_others: Set[str] = set()
            for tid in remaining_task_ids:
                elig = [r.id for r in window_eligibles.get(tid, [])]
                if len(elig) == 1:
                    unique_resources_for_others.add(elig[0])

            # Pick the first resource that is not already used and preferably not unique for others
            best_resource = None
            for r in ranked_resources:
                if r.id in used_resources:
                    continue
                if r.id in unique_resources_for_others:
                    # if we have other options, skip this to preserve parallelism for others
                    continue
                best_resource = r
                break
            # If all candidates are either used or unique-for-others, fall back to the best available not used
            if best_resource is None:
                best_resource = next((r for r in ranked_resources if r.id not in used_resources), None)
            if best_resource is None:
                continue

            # If this resource already selected for another task in this window and
            # we aim for multiple parallel tasks, skip to avoid resource contention.
            if best_resource.id in used_resources:
                continue

            available_actions.append((task.id, best_resource.id))
            selected_task_ids.add(task.id)
            used_resources.add(best_resource.id)
        
        return available_actions
        
    def _get_resource_load(self, resource_id: str) -> float:
        """Calculate the current load of a resource based on their assigned tasks
        
        Args:
            resource_id: The ID of the resource
            
        Returns:
            float: The resource load as a value between 0 (no load) and 1 (fully loaded)
        """
        # Initialize resource load tracking if not exists
        if not hasattr(self, '_resource_loads'):
            self._resource_loads = {}
            
        # Return cached load if available
        if resource_id in self._resource_loads:
            return self._resource_loads[resource_id]
            
        # Default to no load
        return 0.0
        
    def _update_resource_load(self, resource_id: str, load_change: float) -> None:
        """Update the load of a resource
        
        Args:
            resource_id: The ID of the resource
            load_change: The change in load (positive or negative)
        """
        if not hasattr(self, '_resource_loads'):
            self._resource_loads = {}
            
        current_load = self._resource_loads.get(resource_id, 0.0)
        self._resource_loads[resource_id] = max(0.0, min(1.0, current_load + load_change))
    
    def _calculate_reward(self, task: Task, resource: Resource, start_time: datetime, 
                         process: Process) -> float:
        """Calculate reward for an action with enhanced parallel execution support
        
        Enhanced reward function that considers:
        - Task duration and critical path impact
        - Resource utilization and skill matching
        - Parallel execution opportunities and efficiency
        - Dependency chain optimization
        - What-if scenario benefits (if enabled)
        
        Args:
            task: The task being scheduled
            resource: The resource being assigned
            start_time: When the task will start
            process: The overall process
            
        Returns:
            float: The calculated reward, higher is better
        """
        # Scenario parameters (applied by WhatIfAnalyzer as optimizer attributes)
        time_weight = float(getattr(self, 'time_weight', 0.5))
        cost_weight = float(getattr(self, 'cost_weight', 0.5))
        load_balancing_factor = float(getattr(self, 'load_balancing_factor', 1.0))

        # Base: shorter is better (normalize roughly to 0..1 for typical 8..56h)
        duration_reward = 1.0 / max(1e-3, task.duration_hours)

        # Resource utilization: prefer less-loaded resources
        resource_load = self._get_resource_load(resource.id)
        utilization_reward = (1.0 - resource_load)  # 0..1
        
        # Skill matching reward (better matches get higher rewards)
        skill_match_reward = 0.0
        skill_mismatch_penalty = 0.0
        
        for req_skill in task.required_skills:
            skill_matched = False
            for res_skill in resource.skills:
                if res_skill.name == req_skill.name:
                    # Higher skill levels get higher rewards
                    skill_match = (res_skill.level.value / 5.0)  # Scale to 0-1 range
                    skill_match_reward += skill_match
                    skill_matched = True
                    break
            if not skill_matched:
                skill_mismatch_penalty += 0.2  # Penalty for missing skills
        
        # Parallel execution bonus
        parallel_info = self.task_parallel_info.get(task.id, {})
        parallel_bonus = 0.0
        
        if parallel_info.get('can_parallelize', False):
            # Base parallel bonus
            parallel_bonus = 0.3
            
            # Bonus for tasks in parallel groups
            parallel_group = parallel_info.get('parallel_group', [])
            if parallel_group:
                parallel_bonus += 0.2
                
                # Bonus for starting parallel tasks together
                parallel_group_active = any(
                    t_id not in self.completed_tasks and 
                    t_id != task.id
                    for t_id in parallel_info['parallel_group']
                )
                
                if parallel_group_active:
                    parallel_bonus += 0.2
        
        # Critical path awareness - prioritize tasks on the critical path
        critical_path_bonus = 0.0
        if hasattr(task, 'is_critical') and task.is_critical:
            critical_path_bonus = 0.4
        
        # Dependency chain optimization - prioritize tasks with many dependents
        dependents_count = sum(1 for t in process.tasks if task.id in t.dependencies)
        dependents_bonus = min(0.2, dependents_count * 0.05)
        
        # Combine rewards with weights (tuned for parallel execution)
        # Cost penalty (normalize): higher rate and longer duration cost more
        # 200.0 is a soft scale so penalty ~0..2 for typical rates/durations
        cost_penalty = (resource.hourly_rate * task.duration_hours) / 2000.0

        # Compose reward with scenario-driven weights
        # - time_weight emphasizes speed (shorter durations)
        # - cost_weight emphasizes cheaper resources
        # - load_balancing_factor scales utilization spreading across team
        reward = (
            (time_weight) * duration_reward +
            (0.2 * load_balancing_factor) * utilization_reward +
            0.15 * skill_match_reward +
            0.25 * parallel_bonus +
            0.1 * critical_path_bonus +
            0.1 * dependents_bonus
        )
        # Apply cost penalty using cost_weight
        reward -= cost_weight * cost_penalty
        
        # What-if scenario bonus (if enabled and scenarios are defined)
        if self.enable_what_if and self.what_if_scenarios:
            scenario_bonus = self._calculate_what_if_bonus(task, resource, process)
            reward += scenario_bonus * 0.25

        # Keep reward positive but do not clamp to a constant floor that hides differences
        return max(1e-6, float(reward))
    
    def _update_parallel_groups(self, process: Process, ready_tasks: List[Task]) -> None:
        """Update parallel execution groups based on task dependencies and types
        
        Args:
            process: The process being optimized
            ready_tasks: List of tasks that are ready to be scheduled
        """
        # Group tasks by type for potential parallel execution
        task_groups = defaultdict(list)
        
        for task in ready_tasks:
            # Skip if already processed
            if task.id in self.task_parallel_info:
                continue
                
            # Initialize task info
            self.task_parallel_info[task.id] = {
                'can_parallelize': False,
                'parallel_group': [],
                'in_progress': False
            }
            
            # Check if task can be parallelized based on type and dependencies
            task_text = f"{task.name} {task.description}".lower()
            can_parallelize = False
            
            # Check task patterns for parallel potential
            for task_type, pattern_info in self.dependency_detector.task_patterns.items():
                if re.search(pattern_info['pattern'], task_text, re.IGNORECASE):
                    can_parallelize = pattern_info['parallel']
                    break
            
            # Check for explicit parallel keywords in description
            if not can_parallelize and hasattr(task, 'description') and task.description:
                for keyword in self.dependency_detector.dependency_keywords['parallel']:
                    if keyword in task.description.lower():
                        can_parallelize = True
                        break
            
            # Update task info
            self.task_parallel_info[task.id]['can_parallelize'] = can_parallelize
            
            # Add to appropriate group if parallelizable
            if can_parallelize:
                # Find or create a parallel group for this task type
                group_key = f"{task_type}_group"
                task_groups[group_key].append(task.id)
        
        # Update parallel groups
        for group_tasks in task_groups.values():
            if len(group_tasks) > 1:  # Only create groups with multiple tasks
                for task_id in group_tasks:
                    self.task_parallel_info[task_id]['parallel_group'] = group_tasks
    
    def _calculate_earliest_start_time(self, task: Task, resource: Resource, process: Process,
                                     completed_tasks: Set[str], 
                                     resource_availability: Dict[str, datetime]) -> datetime:
        """Calculate the earliest start time for a task considering dependencies and resources
        
        Args:
            task: The task to schedule
            resource: The resource being considered
            process: The process being optimized
            completed_tasks: Set of completed task IDs
            resource_availability: Current availability of resources
            
        Returns:
            datetime: Earliest possible start time for the task
        """
        # Start with resource's next available time
        start_time = resource_availability[resource.id]
        
        # Consider task dependencies
        for dep_id in task.dependencies:
            dep_task = next((t for t in process.tasks if t.id == dep_id), None)
            if dep_task and dep_task.end_time and dep_task.end_time > start_time:
                start_time = dep_task.end_time
        
        # Consider parallel group constraints
        task_info = self.task_parallel_info.get(task.id, {})
        if task_info.get('in_progress', False):
            # If another task in the same group is already in progress,
            # this task cannot start until it's done
            parallel_group = task_info.get('parallel_group', [])
            for t_id in parallel_group:
                if t_id != task.id and t_id in [t.id for t in process.tasks]:
                    t = next(t for t in process.tasks if t.id == t_id)
                    if t.end_time and t.end_time > start_time:
                        start_time = t.end_time
        
        return start_time
    
    def _calculate_task_score(self, task: Task, process: Process, 
                            completed_tasks: Set[str]) -> float:
        """Calculate a score for task prioritization
        
        Args:
            task: The task to score
            process: The process being optimized
            completed_tasks: Set of completed task IDs
            
        Returns:
            float: Score for task prioritization (higher is better)
        """
        # Base score from explicit order (lower order => higher base)
        score = 50
        if getattr(task, 'order', None) is not None:
            try:
                order_val = int(task.order)
                score = max(0, 100 - order_val * 5)
            except Exception:
                pass
        
        # Additional small bias to reinforce order preference
        if getattr(task, 'order', None) is not None:
            try:
                order_val = int(task.order)
                score += max(0, 20 - order_val * 2)
            except Exception:
                pass
        
        # Bonus for tasks with many dependents (critical path)
        dependents = sum(1 for t in process.tasks if task.id in t.dependencies)
        score += dependents * 5
        
        # Bonus for parallel tasks
        task_info = self.task_parallel_info.get(task.id, {})
        if task_info.get('can_parallelize', False):
            score += 20
            
            # Additional bonus for parallel groups
            if task_info.get('parallel_group'):
                score += 10
        
        # Penalty for tasks that block others
        blocking = any(
            dep_id not in completed_tasks and dep_id != task.id
            for t in process.tasks
            for dep_id in t.dependencies
        )
        
        if blocking:
            score += 15
        
        return score
    
    def _update_parallel_metrics(self, process: Process, 
                               scheduled_actions: List[Tuple[str, str]]) -> None:
        """Update parallel execution metrics
        
        Args:
            process: The process being optimized
            scheduled_actions: List of (task_id, resource_id) actions being taken
        """
        if not hasattr(self, 'parallelization_metrics'):
            self.parallelization_metrics = {
                'tasks_executed_in_parallel': 0,
                'total_tasks': 0,
                'parallel_efficiency': 0.0,
                'critical_path_length': 0
            }
        
        # Count parallel tasks in this scheduling step
        parallel_count = len(scheduled_actions)
        
        if parallel_count > 1:
            self.parallelization_metrics['tasks_executed_in_parallel'] += parallel_count
        
        # Update total tasks
        self.parallelization_metrics['total_tasks'] = len(process.tasks)
        
        # Calculate parallel efficiency (tasks / max_parallel)
        if self.parallelization_metrics['total_tasks'] > 0:
            max_possible_parallel = min(
                len(process.resources),
                self.max_parallel_tasks,
                len([t for t in process.tasks if self.task_parallel_info.get(t.id, {}).get('can_parallelize', False)])
            )
            
            if max_possible_parallel > 0:
                self.parallelization_metrics['parallel_efficiency'] = (
                    self.parallelization_metrics['tasks_executed_in_parallel'] /
                    (self.parallelization_metrics['total_tasks'] * max_possible_parallel)
                )
    
    def _get_parallelization_bonus(self, task: Task, process: Process, start_time: datetime) -> float:
        """Calculate bonus for parallel execution of independent tasks
        
        Enhanced to consider parallel groups and resource constraints.
        
        Args:
            task: The task being scheduled
            process: The process being optimized
            start_time: Scheduled start time for the task
            
        Returns:
            float: Bonus reward for enabling parallel execution
        """
        task_info = self.task_parallel_info.get(task.id, {})
        
        # Base bonus for parallel execution
        bonus = 0.0
    
    def _update_q_table(self, states: List[str], actions: List[Tuple[str, str]], 
                       rewards: List[float]):
        """Update Q-table using Q-learning"""
        for i in range(len(states)):
            state = states[i]
            action = str(actions[i])
            reward = rewards[i]
            
            # Q-learning update
            old_q = self.q_table[state][action]
            
            # Future reward (simplified - use average of remaining rewards)
            future_reward = 0
            if i < len(rewards) - 1:
                future_reward = sum(rewards[i+1:]) / len(rewards[i+1:])
            
            new_q = old_q + self.learning_rate * (reward + self.discount_factor * future_reward - old_q)
            self.q_table[state][action] = new_q
    
    def save_model(self, filepath: str):
        """Save the learned Q-table"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'training_episodes': self.training_episodes,
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                'discount_factor': self.discount_factor
            }, f)
    
    def load_model(self, filepath: str):
        """Load a saved Q-table"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
                self.training_episodes = data['training_episodes']
                self.learning_rate = data['learning_rate']
                self.epsilon = data['epsilon']
                self.discount_factor = data['discount_factor']
        except FileNotFoundError:
            print(f"Model file {filepath} not found. Starting with empty Q-table.")


class RLBasedOptimizer(BaseOptimizer):
    """Reinforcement Learning based optimizer using Q-learning with simplified scheduling"""
    
    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.3,
                 discount_factor: float = 0.9, training_episodes: int = 100,
                 enable_parallel: bool = True, max_parallel_tasks: int = 3,
                 enable_what_if: bool = False, what_if_scenarios: Optional[List[Dict]] = None):
        """Initialize RL optimizer with enhanced parallel processing
        
        Args:
            learning_rate: Q-learning learning rate
            epsilon: Exploration vs exploitation parameter
            discount_factor: Future reward discount factor
            training_episodes: Number of training episodes
            enable_parallel: Whether to enable parallel task execution
            max_parallel_tasks: Maximum number of tasks that can run in parallel
            enable_what_if: Whether to enable what-if scenario analysis
            what_if_scenarios: List of what-if scenarios to consider
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.training_episodes = training_episodes
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.dependency_detector = DependencyDetector()
        self.deadlock_detector = DeadlockDetector()
        
        # Parallel execution parameters
        self.enable_parallel = enable_parallel
        self.max_parallel_tasks = max_parallel_tasks
        self.task_parallel_info = {}  # Track parallel execution info for each task
        self.parallelization_metrics = {}
        
        # What-if scenario parameters
        self.enable_what_if = enable_what_if
        self.what_if_scenarios = what_if_scenarios or []
        
        # Track completed tasks for reward calculation
        self.completed_tasks = set()
    
    def _get_state_simple(self, process: Process, completed_tasks: Set[str],
                         resource_next_available: Dict[str, float],
                         resource_workload: Dict[str, float]) -> str:
        """Get simplified state representation for Q-learning"""
        state = []
        
        # Task features (for each task: completed, ready to start)
        for task in process.tasks:
            state.append(1.0 if task.id in completed_tasks else 0.0)
            state.append(1.0 if task.can_start(completed_tasks) else 0.0)
        
        # Resource features (for each resource: next available hour and workload)
        for resource in process.resources:
            # Normalize next available hour to [0, 1]
            next_hour = resource_next_available.get(resource.id, 0.0)
            state.append(min(1.0, next_hour / max(process.project_duration_hours, 1.0)))
            
            # Normalize workload to [0, 1]
            workload = resource_workload.get(resource.id, 0.0)
            state.append(min(1.0, workload / max(resource.total_available_hours, 1.0)))
        
        # Convert to string for Q-table key
        return str(state)
    
    def _get_available_actions_simple(self, process: Process, completed_tasks: Set[str],
                                     resource_workload: Dict[str, float]) -> List[Tuple[str, str]]:
        """Get available actions using simplified logic with parallelization priority"""
        actions = []
        
        # Find all ready tasks (dependencies satisfied)
        ready_tasks = []
        for task in process.tasks:
            if task.id in completed_tasks:
                continue
            
            # Check dependencies
            deps_satisfied = True
            if hasattr(task, 'dependencies') and task.dependencies:
                deps_satisfied = all(dep_id in completed_tasks for dep_id in task.dependencies)
            
            if deps_satisfied:
                ready_tasks.append(task)
        
        # Sort ready tasks by duration (longer first for better parallelization)
        ready_tasks.sort(key=lambda t: t.duration_hours, reverse=True)
        
        # Create actions for all ready tasks with all compatible resources
        for task in ready_tasks:
            qualified_resources = []
            for resource in process.resources:
                # Check if resource has required skills
                if resource.has_all_skills(task.required_skills or []):
                    qualified_resources.append(resource)
                elif not task.required_skills:  # Task has no skill requirements
                    qualified_resources.append(resource)
            
            # Sort resources by current workload (least loaded first)
            qualified_resources.sort(key=lambda r: resource_workload.get(r.id, 0))
            
            # Add actions prioritizing least loaded resources
            for resource in qualified_resources:
                actions.append((task.id, resource.id))
        
        return actions
    
    def _select_action(self, state: str, available_actions: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Select action using epsilon-greedy strategy"""
        import random
        
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(available_actions)
        else:
            # Exploit: best known action
            if state in self.q_table:
                # Find best action from Q-table that's available
                best_action = None
                best_value = float('-inf')
                
                for action in available_actions:
                    action_str = str(action)
                    if action_str in self.q_table[state]:
                        value = self.q_table[state][action_str]
                        if value > best_value:
                            best_value = value
                            best_action = action
                
                if best_action:
                    return best_action
            
            # Default to random if no Q-value exists
            return random.choice(available_actions)
    
    def _calculate_reward_simple(self, task: Task, resource: Resource, 
                                start_hour: float, resource_workload: Dict[str, float]) -> float:
        """Calculate reward for scheduling decision"""
        # Base reward for completing a task
        reward = 100.0
        
        # Penalize late starts
        if start_hour > 0:
            reward -= start_hour * 0.5
        
        # Reward good resource utilization
        workload = resource_workload.get(resource.id, 0.0)
        utilization = workload / max(resource.total_available_hours, 1.0)
        if 0.3 <= utilization <= 0.8:
            reward += 20.0  # Good utilization
        elif utilization > 0.8:
            reward -= 10.0  # Over-utilized
        else:
            reward -= 5.0   # Under-utilized
        
        # Skill match bonus
        if resource.has_all_skills(task.required_skills):
            reward += 10.0
        
        return max(1.0, reward)
    
    def _update_q_table(self, states: List[str], actions: List[Tuple[str, str]], 
                       rewards: List[float]) -> None:
        """Update Q-table based on experience"""
        for i in range(len(states)):
            state = states[i]
            action = str(actions[i])  # Convert action to string for Q-table key
            reward = rewards[i]
            
            # Initialize Q-value if not exists
            if state not in self.q_table:
                self.q_table[state] = {}
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0
            
            # Q-learning update
            old_value = self.q_table[state][action]
            next_value = 0.0
            if i + 1 < len(states):
                next_state = states[i + 1]
                if next_state in self.q_table:
                    next_value = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
            
            # Q-value update formula
            self.q_table[state][action] = old_value + self.learning_rate * (
                reward + self.discount_factor * next_value - old_value
            )
    
    def optimize(self, process: Process) -> Schedule:
        """Optimize using simplified hour-based scheduling logic with parallelization"""
        return self._train_and_optimize(process)
    
    def _get_available_actions_for_generation(self, process: Process, completed_tasks: Set[str],
                                             resource_workload: Dict[str, float]) -> List[Tuple[str, str]]:
        """Get available actions for schedule generation with relaxed dependencies"""
        actions = []
        
        # Find tasks that can potentially run (relaxed dependency checking)
        ready_tasks = []
        for task in process.tasks:
            if task.id in completed_tasks:
                continue
            
            # For schedule generation, include tasks even if some dependencies aren't complete
            # The Q-table has learned which tasks can actually run in parallel
            ready_tasks.append(task)
        
        # Sort ready tasks by order if available, then by duration
        ready_tasks.sort(key=lambda t: (getattr(t, 'order', 999), -t.duration_hours))
        
        # Create actions for all ready tasks with all compatible resources
        for task in ready_tasks:
            for resource in process.resources:
                if resource.has_all_skills(task.required_skills or []):
                    actions.append((task.id, resource.id))
        
        return actions
    
    def generate_schedule_from_model(self, process: Process) -> Schedule:
        """Generate a schedule using the trained Q-table without retraining"""
        print("[generate_schedule_from_model] Using trained Q-table to generate schedule")
        print(f"[generate_schedule_from_model] Q-table has {len(self.q_table)} states")
        
        # Initialize tracking variables
        schedule = Schedule(process_id=process.id)
        completed_tasks = set()
        resource_next_available = {r.id: 0 for r in process.resources}
        resource_workload = {r.id: 0 for r in process.resources}
        task_completion_times = {}
        
        # Generate schedule using Q-table knowledge
        while len(completed_tasks) < len(process.tasks):
            # Get current state
            state = self._get_state_simple(process, completed_tasks, 
                                          resource_next_available, resource_workload)
            
            # Get all possible actions - use relaxed method for generation
            possible_actions = self._get_available_actions_for_generation(process, completed_tasks, resource_workload)
            if not possible_actions:
                break
            
            # Build batch of parallel actions based on Q-values
            scheduled_in_batch = []
            used_resources = set()
            scheduled_tasks = set()
            
            # Sort actions by Q-value (or heuristic if not in Q-table)
            action_scores = []
            for task_id, resource_id in possible_actions:
                action_str = str((task_id, resource_id))
                
                # Get Q-value or use heuristic
                if state in self.q_table and action_str in self.q_table[state]:
                    score = self.q_table[state][action_str]
                else:
                    # Use heuristic for unseen states
                    task = next(t for t in process.tasks if t.id == task_id)
                    resource = next(r for r in process.resources if r.id == resource_id)
                    
                    # Score based on: early start, load balance, skill match
                    start_time = resource_next_available[resource_id]
                    workload = resource_workload.get(resource_id, 0)
                    
                    score = 100.0
                    score -= start_time * self.time_weight  # Prefer early starts
                    score -= (workload / max(resource.total_available_hours, 1)) * 20 * self.load_balancing_factor
                    if resource.has_all_skills(task.required_skills or []):
                        score += 10
                    
                action_scores.append((score, task_id, resource_id))
            
            # Sort by score (higher is better)
            action_scores.sort(reverse=True, key=lambda x: x[0])
            
            # Schedule multiple high-scoring actions in parallel
            for score, task_id, resource_id in action_scores:
                if task_id in scheduled_tasks or resource_id in used_resources:
                    continue
                
                task = next(t for t in process.tasks if t.id == task_id)
                resource = next(r for r in process.resources if r.id == resource_id)
                
                # Calculate timing
                dep_completion_time = 0
                if hasattr(task, 'dependencies') and task.dependencies:
                    for dep_id in task.dependencies:
                        dep_completion_time = max(dep_completion_time, 
                                                 task_completion_times.get(dep_id, 0))
                
                start_time = max(resource_next_available[resource_id], dep_completion_time)
                end_time = start_time + task.duration_hours
                
                scheduled_in_batch.append({
                    'task': task,
                    'resource': resource,
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                used_resources.add(resource_id)
                scheduled_tasks.add(task_id)
                
                # Limit batch size
                if len(scheduled_in_batch) >= self.max_parallel_tasks:
                    break
            
            # If no tasks can be scheduled, break
            if not scheduled_in_batch:
                break
            
            print(f"[Batch] Scheduled {len(scheduled_in_batch)} tasks in parallel:")
            for task_info in scheduled_in_batch:
                print(f"  - Task {task_info['task'].id} on {task_info['resource'].name} at t={task_info['start_time']}")
            
            # Now actually schedule the batch
            batch_updates = []
            for task_info in scheduled_in_batch:
                task = task_info['task']
                resource = task_info['resource']
                task_id = task.id
                resource_id = resource.id
                earliest_start = task_info['start_time']
                
                # Create schedule entry
                start_time = datetime(2025, 8, 20, 9, 0) + timedelta(hours=earliest_start)
                end_time = start_time + timedelta(hours=task.duration_hours)
                cost = task.duration_hours * resource.hourly_rate
                
                entry = ScheduleEntry(
                    task_id=task_id,
                    resource_id=resource_id,
                    start_time=start_time,
                    end_time=end_time,
                    cost=cost,
                    start_hour=earliest_start,
                    end_hour=earliest_start + task.duration_hours
                )
                
                schedule.entries.append(entry)
                completed_tasks.add(task_id)
                task_completion_times[task_id] = earliest_start + task.duration_hours
                resource_workload[resource_id] += task.duration_hours
                
                # Update resource utilization
                if resource_id not in schedule.resource_utilization:
                    schedule.resource_utilization[resource_id] = 0
                schedule.resource_utilization[resource_id] += task.duration_hours
                
                # Store updates to apply after batch
                batch_updates.append((resource_id, earliest_start + task.duration_hours))
            
            # Update resource availability AFTER scheduling the entire batch
            for resource_id, next_available in batch_updates:
                resource_next_available[resource_id] = max(resource_next_available[resource_id], next_available)
        
        # Calculate final metrics
        schedule.calculate_metrics(process)
        
        return schedule
    
    def _train_and_optimize(self, process: Process) -> Schedule:
        """Train the RL model and return the best schedule found"""
        # Detect dependencies first
        self._detect_and_apply_dependencies(process)
        
        best_schedule = None
        best_duration = float('inf')
        
        # Run multiple training episodes
        for episode in range(self.training_episodes):
            # Initialize schedule for this episode
            schedule = Schedule(
                process_id=process.id,
                entries=[],
                total_cost=0.0,
                total_duration_hours=0.0,
                resource_utilization={}
            )
            
            # Track resource availability and workload in hours from project start
            resource_next_available = {r.id: 0.0 for r in process.resources}
            resource_workload = {r.id: 0.0 for r in process.resources}
            
            # Track completed tasks and their completion times
            completed_tasks = set()
            task_completion_times = {}
            
            # Main scheduling loop
            states = []
            actions = []
            rewards = []
            
            while len(completed_tasks) < len(process.tasks):
                # Get current state
                state = self._get_state_simple(process, completed_tasks, resource_next_available, resource_workload)
                
                # Get available actions
                available_actions = self._get_available_actions_simple(process, completed_tasks, resource_workload)
                
                if not available_actions:
                    break  # No more actions available
                
                # Select action (task, resource) using epsilon-greedy
                action = self._select_action(state, available_actions)
                task_id, resource_id = action
                
                # Get task and resource
                task = process.get_task_by_id(task_id)
                resource = process.get_resource_by_id(resource_id)
                
                if not task or not resource:
                    continue
                
                # Calculate earliest start time considering dependencies
                earliest_start = resource_next_available[resource_id]
                
                # Check dependencies and update earliest start
                if hasattr(task, 'dependencies') and task.dependencies:
                    for dep_id in task.dependencies:
                        if dep_id in task_completion_times:
                            earliest_start = max(earliest_start, task_completion_times[dep_id])
                
                # Schedule the task for parallelization
                start_hour = earliest_start
                end_hour = start_hour + task.duration_hours
                
                # Convert to datetime for compatibility
                start_time = process.start_date + timedelta(hours=start_hour)
                end_time = process.start_date + timedelta(hours=end_hour)
                
                # Calculate cost
                cost = task.duration_hours * resource.hourly_rate
                
                # Create schedule entry with both hour and datetime fields
                entry = ScheduleEntry(
                    task_id=task_id,
                    resource_id=resource_id,
                    start_time=start_time,
                    end_time=end_time,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    cost=cost
                )
                
                schedule.entries.append(entry)
                
                # Update resource availability and workload
                resource_next_available[resource_id] = end_hour
                resource_workload[resource_id] += task.duration_hours
                
                # Mark task as completed and track completion time
                completed_tasks.add(task_id)
                task_completion_times[task_id] = end_hour
                
                # Calculate reward
                reward = self._calculate_reward_simple(task, resource, start_hour, resource_workload)
                
                # Store for learning
                states.append(state)
                actions.append(action)
                rewards.append(reward)
            
            # Update Q-table if we have experience
            if states and actions and rewards:
                self._update_q_table(states, actions, rewards)
            
            # Calculate schedule metrics
            schedule.calculate_metrics(process)
            
            # Track best schedule
            if schedule.entries:
                max_end_hour = max(e.end_hour for e in schedule.entries)
                if max_end_hour < best_duration:
                    best_duration = max_end_hour
                    best_schedule = schedule
            
            # Decay epsilon after each episode
            self.epsilon *= 0.95
        
        # Reset epsilon for future runs
        self.epsilon = self.initial_epsilon if hasattr(self, 'initial_epsilon') else 0.3
        
        return best_schedule if best_schedule else Schedule(process_id=process.id)


class GeneticOptimizer(BaseOptimizer):
    """Genetic Algorithm-based optimizer"""
    
    def __init__(self, population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        """Initialize genetic algorithm parameters"""
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(self, process: Process) -> Schedule:
        """Optimize using genetic algorithm"""
        # Detect dependencies first
        self._detect_and_apply_dependencies(process)
        
        # Initialize population
        population = self._initialize_population(process)
        
        best_schedule = None
        best_fitness = float('inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                schedule = self._decode_individual(individual, process)
                fitness = self._calculate_fitness(schedule, process)
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_schedule = schedule
            
            # Selection, crossover, and mutation
            new_population = []
            
            # Elitism - keep best individuals
            elite_count = max(1, self.population_size // 10)
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elite_count]
            for i in elite_indices:
                new_population.append(population[i].copy())
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                if random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, process)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, process)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return best_schedule if best_schedule else Schedule(process_id=process.id)
    
    def _initialize_population(self, process: Process) -> List[List[Tuple[str, str]]]:
        """Initialize random population of schedules"""
        population = []
        
        for _ in range(self.population_size):
            individual = []
            
            # Create random task-resource assignments
            for task in process.tasks:
                suitable_resources = [r for r in process.resources 
                                    if r.has_all_skills(task.required_skills)]
                if suitable_resources:
                    resource = random.choice(suitable_resources)
                    individual.append((task.id, resource.id))
                else:
                    # Assign to first available resource (may cause issues)
                    individual.append((task.id, process.resources[0].id))
            
            population.append(individual)
        
        return population
    
    def _decode_individual(self, individual: List[Tuple[str, str]], process: Process) -> Schedule:
        """Convert individual to schedule"""
        schedule = Schedule(process_id=process.id)
        
        # Sort tasks by dependencies (topological sort)
        task_order = self._get_dependency_order(process.tasks)
        
        resource_availability = {r.id: process.start_date for r in process.resources}
        assignment_map = {task_id: resource_id for task_id, resource_id in individual}
        
        for task_id in task_order:
            task = process.get_task_by_id(task_id)
            resource_id = assignment_map.get(task_id)
            resource = process.get_resource_by_id(resource_id)
            
            if task and resource:
                start_time = resource_availability[resource_id]
                end_time = start_time + timedelta(hours=task.duration_hours)
                cost = task.duration_hours * resource.hourly_rate
                
                entry = ScheduleEntry(
                    task_id=task_id,
                    resource_id=resource_id,
                    start_time=start_time,
                    end_time=end_time,
                    start_hour=(start_time - process.start_date).total_seconds() / 3600.0,
                    end_hour=(end_time - process.start_date).total_seconds() / 3600.0,
                    cost=cost
                )
                
                schedule.entries.append(entry)
                resource_availability[resource_id] = end_time
        
        # Calculate final schedule metrics
        if schedule.entries:
            # Calculate elapsed time using hour-based fields if available
            if all(hasattr(e, 'start_hour') and hasattr(e, 'end_hour') for e in schedule.entries):
                min_start_hour = min(e.start_hour for e in schedule.entries)
                max_end_hour = max(e.end_hour for e in schedule.entries)
                schedule.total_duration_hours = max_end_hour - min_start_hour
            else:
                # Fallback to datetime calculation
                min_start = min(e.start_time for e in schedule.entries)
                max_end = max(e.end_time for e in schedule.entries)
                schedule.total_duration_hours = (max_end - min_start).total_seconds() / 3600.0
            
            # Calculate total cost
            schedule.total_cost = sum(e.cost for e in schedule.entries)
            
            # Calculate resource utilization
            for entry in schedule.entries:
                if entry.resource_id not in schedule.resource_utilization:
                    schedule.resource_utilization[entry.resource_id] = 0.0
                # Use task duration for utilization calculation
                task = process.get_task_by_id(entry.task_id)
                if task:
                    schedule.resource_utilization[entry.resource_id] += task.duration_hours
        
        return schedule
    
    def _get_dependency_order(self, tasks: List[Task]) -> List[str]:
        """Get topological order of tasks based on dependencies"""
        from collections import deque
        
        # Calculate in-degrees
        in_degree = {task.id: len(task.dependencies) for task in tasks}
        
        # Initialize queue with tasks that have no dependencies
        queue = deque([task.id for task in tasks if len(task.dependencies) == 0])
        result = []
        
        while queue:
            task_id = queue.popleft()
            result.append(task_id)
            
            # Update in-degrees
            for task in tasks:
                if task_id in task.dependencies:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)
        
        return result
    
    def _calculate_fitness(self, schedule: Schedule, process: Process) -> float:
        """Calculate fitness score (lower is better)"""
        fitness = 0.0
        
        # Duration penalty
        fitness += schedule.total_duration_hours * 1.0
        
        # Cost penalty
        fitness += schedule.total_cost * 0.01
        
        # Deadline penalty
        for entry in schedule.entries:
            task = process.get_task_by_id(entry.task_id)
            if task and task.deadline and entry.end_time > task.deadline:
                days_late = (entry.end_time - task.deadline).days
                fitness += days_late * 100
        
        # Resource utilization penalty (prefer balanced utilization)
        if hasattr(schedule, 'resource_utilization') and schedule.resource_utilization:
            # Extract numeric utilization values, filtering out any non-numeric values
            utilizations = [float(u) for u in schedule.resource_utilization.values() 
                          if isinstance(u, (int, float))]
            
            if utilizations:  # Only calculate if we have valid utilization data
                avg_util = sum(utilizations) / len(utilizations)
                util_variance = sum((u - avg_util) ** 2 for u in utilizations) / len(utilizations)
                fitness += util_variance * 0.1
        
        return fitness
    
    def _tournament_selection(self, population: List[List[Tuple[str, str]]], 
                             fitness_scores: List[float], tournament_size: int = 3) -> List[Tuple[str, str]]:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_index = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index]
    
    def _crossover(self, parent1: List[Tuple[str, str]], 
                  parent2: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Single-point crossover"""
        if len(parent1) != len(parent2):
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutate(self, individual: List[Tuple[str, str]], process: Process) -> List[Tuple[str, str]]:
        """Mutate individual by changing random assignments"""
        mutated = individual.copy()
        
        # Mutate random assignments
        for i in range(len(mutated)):
            if random.random() < 0.1:  # 10% chance per gene
                task_id, _ = mutated[i]
                task = process.get_task_by_id(task_id)
                
                if task:
                    suitable_resources = [r for r in process.resources 
                                        if r.has_all_skills(task.required_skills)]
                    if suitable_resources:
                        new_resource = random.choice(suitable_resources)
                        mutated[i] = (task_id, new_resource.id)
        
        return mutated
