"""
Optimization engines for the Process Optimization Agent
"""

import random
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import heapq
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
    """Greedy rule-based optimizer with optional bin-packing for cost optimization"""
    
    # Bin-packing configuration
    ENABLE_BIN_PACKING = True
    BIN_PACKING_MODE = "balanced"  # conservative | balanced | aggressive
    
    # Mode-specific settings
    BIN_PACKING_CONFIGS = {
        "conservative": {
            "respect_cms_defaults": True,
            "min_skill_match": 0.95,
            "max_utilization": 0.85,
            "min_savings_threshold": 0.03,  # 3% minimum savings
        },
        "balanced": {
            "respect_cms_defaults": True,  # Honor critical tasks only
            "min_skill_match": 0.90,
            "max_utilization": 0.90,
            "min_savings_threshold": 0.05,  # 5% minimum savings
        },
        "aggressive": {
            "respect_cms_defaults": False,
            "min_skill_match": 0.85,
            "max_utilization": 0.95,
            "min_savings_threshold": 0.02,  # 2% minimum savings
        }
    }
    
    def __init__(self, optimization_strategy: str = "balanced", cms_data: Optional[Dict[str, Any]] = None, 
                 bin_packing_mode: str = "balanced", bin_packing_enabled: bool = True):
        """
        Initialize optimizer with strategy
        
        Strategies:
        - "time": Minimize total duration
        - "cost": Minimize total cost  
        - "balanced": Balance time and cost
        
        Args:
            optimization_strategy: Optimization strategy to use
            cms_data: Original CMS data for fallback assignments
            bin_packing_mode: Bin-packing configuration mode (conservative, balanced, aggressive)
            bin_packing_enabled: Whether to enable bin-packing optimization
        """
        super().__init__()
        self.strategy = optimization_strategy
        self.cms_data = cms_data
        self._original_assignments = self._extract_original_assignments(cms_data) if cms_data else {}
        
        # Bin-packing configuration
        self._bin_packing_enabled = bin_packing_enabled
        self._bin_packing_mode = bin_packing_mode
        self._bin_packing_config = self.BIN_PACKING_CONFIGS.get(bin_packing_mode, self.BIN_PACKING_CONFIGS["balanced"])
    
    def _extract_original_assignments(self, cms_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract original task-to-resource assignments from CMS data
        
        Args:
            cms_data: Original CMS process data
            
        Returns:
            Dict mapping task_id to resource_id from original CMS assignments
        """
        assignments = {}
        
        print(f"[FALLBACK-EXTRACT] Starting to extract original assignments from CMS data")
        
        # Extract from process_task array
        process_tasks = cms_data.get('process_task', [])
        print(f"[FALLBACK-EXTRACT] Found {len(process_tasks)} process tasks in CMS data")
        
        for pt_wrapper in process_tasks:
            task_data = pt_wrapper.get('task', {})
            task_id = task_data.get('task_id')
            
            # Get assigned jobs from jobTasks array (CMS uses camelCase)
            task_jobs = task_data.get('jobTasks', [])
            print(f"[FALLBACK-EXTRACT] Task {task_id} has {len(task_jobs)} job assignments")
            
            if task_jobs and task_id:
                # Use the first assigned job as the default
                first_job = task_jobs[0]
                job_data = first_job.get('job', {})
                job_id = job_data.get('job_id')
                if job_id:
                    assignments[str(task_id)] = str(job_id)
                    print(f"[FALLBACK-EXTRACT] Stored original assignment: Task {task_id} -> Resource {job_id}")
                else:
                    print(f"[FALLBACK-EXTRACT] Task {task_id}: job_id is None in job_data: {job_data}")
            elif task_id:
                print(f"[FALLBACK-EXTRACT] Task {task_id} has no job assignments - skipping")
        
        print(f"[FALLBACK-EXTRACT] Total original assignments extracted: {len(assignments)}")
        return assignments
    
    def _categorize_tasks(self, process: Process) -> Tuple[List[Task], List[Task]]:
        """Categorize tasks into CMS-critical and flexible based on assignments
        
        Returns:
            Tuple of (critical_tasks, flexible_tasks)
        """
        critical_tasks = []
        flexible_tasks = []
        
        for task in process.tasks:
            # Task is critical if it has a CMS default assignment
            # indicating it may have regulatory/business requirements
            task_id_str = str(task.id)
            if task_id_str in self._original_assignments:
                # Has CMS assignment - treat as critical in conservative/balanced modes
                if self._bin_packing_config["respect_cms_defaults"]:
                    critical_tasks.append(task)
                else:
                    flexible_tasks.append(task)
            else:
                # No CMS assignment - safe to optimize
                flexible_tasks.append(task)
        
        print(f"[BIN-PACKING] Categorized: {len(critical_tasks)} critical, {len(flexible_tasks)} flexible tasks")
        return critical_tasks, flexible_tasks
    
    def _validate_assignment(self, task: Task, resource: Resource, 
                           resource_workload: Dict[str, float]) -> Tuple[bool, str]:
        """Validate that an assignment is safe before applying
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check 1: Skill match
        if task.required_skills:
            skill_score = resource.get_skill_score(task.required_skills)
            min_match = self._bin_packing_config["min_skill_match"]
            if skill_score < min_match:
                return False, f"Insufficient skill match ({skill_score*100:.1f}% < {min_match*100:.1f}%)"
        
        # Check 2: Capacity
        needed_capacity = resource_workload.get(resource.id, 0.0) + task.duration_hours
        max_util = self._bin_packing_config["max_utilization"]
        if needed_capacity > resource.total_available_hours * max_util:
            return False, f"Capacity exceeded ({needed_capacity:.1f}h > {resource.total_available_hours * max_util:.1f}h)"
        
        # Check 3: CMS protection (if in conservative/balanced mode)
        if self._bin_packing_config["respect_cms_defaults"]:
            task_id_str = str(task.id)
            if task_id_str in self._original_assignments:
                expected_resource_id = str(self._original_assignments[task_id_str])
                if str(resource.id) != expected_resource_id:
                    return False, f"CMS protected assignment (expected Resource {expected_resource_id})"
        
        return True, "OK"
    
    def _bin_pack_tasks(self, tasks: List[Task], process: Process,
                       resource_workload: Dict[str, float]) -> Optional[Dict[int, Tuple[Resource, float]]]:
        """Apply First-Fit Decreasing (FFD) bin-packing algorithm
        
        Returns:
            Dict mapping task_id to (resource, start_hour) or None if failed
        """
        try:
            print(f"[BIN-PACKING] Starting FFD algorithm for {len(tasks)} tasks")
            
            # Step 1: Sort tasks by duration (longest first) - key to FFD
            sorted_tasks = sorted(tasks, key=lambda t: t.duration_hours, reverse=True)
            print(f"[BIN-PACKING] Tasks sorted by duration: {[(t.id, t.duration_hours) for t in sorted_tasks[:5]]}...")
            
            # Step 2: Get resources with skill matches, sort by cost (cheapest first)
            assignments = {}
            temp_workload = resource_workload.copy()
            
            for task in sorted_tasks:
                # Find resources that can handle this task
                eligible_resources = []
                
                for resource in process.resources:
                    # Check skill match
                    if task.required_skills:
                        skill_score = resource.get_skill_score(task.required_skills)
                        if skill_score < self._bin_packing_config["min_skill_match"]:
                            continue
                    else:
                        skill_score = 1.0
                    
                    # Check capacity
                    needed_capacity = temp_workload.get(resource.id, 0.0) + task.duration_hours
                    if needed_capacity > resource.total_available_hours * self._bin_packing_config["max_utilization"]:
                        continue
                    
                    eligible_resources.append((resource, skill_score))
                
                if not eligible_resources:
                    print(f"[BIN-PACKING] ❌ No eligible resources for task {task.id} - aborting")
                    return None
                
                # Sort by: skill match (highest), then cost (lowest)
                eligible_resources.sort(key=lambda x: (-x[1], x[0].hourly_rate))
                
                # Assign to best resource (first fit)
                best_resource, skill_score = eligible_resources[0]
                start_hour = temp_workload.get(best_resource.id, 0.0)
                
                assignments[task.id] = (best_resource, start_hour)
                temp_workload[best_resource.id] = start_hour + task.duration_hours
                
                print(f"[BIN-PACKING]   Task {task.id} → Resource {best_resource.id} ({skill_score*100:.0f}% match, ${best_resource.hourly_rate:.2f}/hr)")
            
            print(f"[BIN-PACKING] ✅ Successfully packed {len(assignments)} tasks")
            return assignments
            
        except Exception as e:
            print(f"[BIN-PACKING] ❌ Exception: {str(e)}")
            return None
    
    def optimize(self, process: Process, max_retries: int = 3) -> Schedule:
        """
        Optimize using hybrid bin-packing with 4-level fallback:
        Level 1: Bin-packing with strict constraints (high skill match, cost savings threshold)
        Level 2: Bin-packing with relaxed constraints (lower skill threshold)
        Level 3: Greedy sequential scheduling (current proven method)
        Level 4: Pure CMS defaults (handled within greedy)
        
        Args:
            process: The process to optimize
            max_retries: Maximum number of retries for deadlock resolution
            
        Returns:
            Schedule: The optimized schedule
        """
        # Detect and apply dependencies first
        self._detect_and_apply_dependencies(process)
        
        # Try bin-packing optimization first (Levels 1-2)
        if self._bin_packing_enabled:
            print(f"\n[BIN-PACKING] Attempting optimization with mode: {self._bin_packing_mode}")
            
            # Level 1: Strict constraints
            print(f"[BIN-PACKING] Level 1: Strict constraints (skill≥{self._bin_packing_config['min_skill_match']*100:.0f}%, savings≥{self._bin_packing_config['min_savings_threshold']*100:.0f}%)")
            binpack_schedule = self._try_bin_packing_optimization(process)
            
            if binpack_schedule:
                # Calculate greedy baseline for comparison
                greedy_schedule = self._optimize_greedy(process, max_retries)
                greedy_cost = sum(e.cost for e in greedy_schedule.entries)
                binpack_cost = sum(e.cost for e in binpack_schedule.entries)
                
                savings_pct = (greedy_cost - binpack_cost) / greedy_cost if greedy_cost > 0 else 0.0
                
                print(f"[BIN-PACKING] Cost comparison: Greedy=${greedy_cost:.2f} vs Bin-pack=${binpack_cost:.2f} (savings: {savings_pct*100:.1f}%)")
                
                if savings_pct >= self._bin_packing_config["min_savings_threshold"]:
                    print(f"[BIN-PACKING] ✓ Level 1 SUCCESS - Using bin-packed schedule (saves ${greedy_cost - binpack_cost:.2f})")
                    return binpack_schedule
                else:
                    print(f"[BIN-PACKING] Level 1 insufficient savings ({savings_pct*100:.1f}% < {self._bin_packing_config['min_savings_threshold']*100:.0f}%)")
            else:
                print(f"[BIN-PACKING] Level 1 failed to produce valid schedule")
            
            # Level 2: Relaxed constraints
            if self._bin_packing_mode != "conservative":  # Only try relaxed for balanced/aggressive
                print(f"[BIN-PACKING] Level 2: Relaxed constraints (skill≥85%, savings≥2%)")
                relaxed_config = self._bin_packing_config.copy()
                relaxed_config["min_skill_match"] = 0.85
                relaxed_config["min_savings_threshold"] = 0.02
                
                # Temporarily swap config
                original_config = self._bin_packing_config
                self._bin_packing_config = relaxed_config
                
                binpack_schedule = self._try_bin_packing_optimization(process)
                self._bin_packing_config = original_config  # Restore
                
                if binpack_schedule:
                    greedy_schedule = self._optimize_greedy(process, max_retries)
                    greedy_cost = sum(e.cost for e in greedy_schedule.entries)
                    binpack_cost = sum(e.cost for e in binpack_schedule.entries)
                    savings_pct = (greedy_cost - binpack_cost) / greedy_cost if greedy_cost > 0 else 0.0
                    
                    if savings_pct >= relaxed_config["min_savings_threshold"]:
                        print(f"[BIN-PACKING] ✓ Level 2 SUCCESS - Using relaxed bin-packed schedule (saves ${greedy_cost - binpack_cost:.2f})")
                        return binpack_schedule
                    else:
                        print(f"[BIN-PACKING] Level 2 insufficient savings ({savings_pct*100:.1f}% < 2%)")
                else:
                    print(f"[BIN-PACKING] Level 2 failed to produce valid schedule")
            
            print(f"[BIN-PACKING] Falling back to Level 3: Greedy sequential scheduling")
        
        # Level 3: Greedy scheduling (current proven method)
        return self._optimize_greedy(process, max_retries)
    
    def _try_bin_packing_optimization(self, process: Process) -> Optional[Schedule]:
        """
        Attempt bin-packing optimization with current configuration.
        Returns Schedule if successful, None if failed.
        """
        try:
            # Categorize tasks
            critical_tasks, flexible_tasks = self._categorize_tasks(process)
            print(f"[BIN-PACKING] Task categorization: {len(critical_tasks)} critical (CMS-locked), {len(flexible_tasks)} flexible")
            
            # Bin-pack flexible tasks
            if not flexible_tasks:
                print(f"[BIN-PACKING] No flexible tasks to optimize, using greedy")
                return None
            
            # Initialize resource workload tracking
            resource_workload = {r.id: 0.0 for r in process.resources}
            
            assignments = self._bin_pack_tasks(flexible_tasks, process, resource_workload)
            
            if not assignments:
                print(f"[BIN-PACKING] Bin-packing failed to assign all tasks")
                return None
            
            # Validate all assignments
            validation_workload = {r.id: 0.0 for r in process.resources}
            for task_id, (resource, start_hour) in assignments.items():
                task = process.get_task_by_id(task_id)
                is_valid, reason = self._validate_assignment(task, resource, validation_workload)
                if not is_valid:
                    print(f"[BIN-PACKING] Validation failed for task {task_id}: {reason}")
                    return None
                # Update workload for next validation
                validation_workload[resource.id] += task.duration_hours
            
            # Create schedule from assignments
            schedule = Schedule(process_id=process.id)
            resource_workload = {r.id: 0.0 for r in process.resources}
            
            # Add critical tasks first (use greedy for these)
            for task in critical_tasks:
                # Use simple assignment for critical tasks
                resource_next_available = {r.id: 0.0 for r in process.resources}
                best_resource, start_hour = self._find_best_resource_simple(
                    task, process, resource_next_available, resource_workload
                )
                
                if best_resource:
                    duration_hours = task.duration_hours
                    end_hour = start_hour + duration_hours
                    cost = duration_hours * best_resource.hourly_rate
                    
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
                    resource_workload[best_resource.id] += duration_hours
                else:
                    print(f"[BIN-PACKING] Failed to assign critical task {task.id}")
                    return None
            
            # Add bin-packed flexible tasks
            for task_id, (resource, start_hour) in assignments.items():
                task = process.get_task_by_id(task_id)
                duration_hours = task.duration_hours
                end_hour = start_hour + duration_hours
                cost = duration_hours * resource.hourly_rate
                
                entry = ScheduleEntry(
                    task_id=task.id,
                    resource_id=resource.id,
                    start_time=process.start_date + timedelta(hours=start_hour),
                    end_time=process.start_date + timedelta(hours=end_hour),
                    start_hour=start_hour,
                    end_hour=end_hour,
                    cost=cost
                )
                schedule.entries.append(entry)
                resource_workload[resource.id] += duration_hours
            
            # Calculate metrics
            schedule.calculate_metrics(process)
            
            print(f"[BIN-PACKING] Successfully created schedule with {len(schedule.entries)} entries")
            return schedule
            
        except Exception as e:
            print(f"[BIN-PACKING] Exception during optimization: {e}")
            return None
    
    def _optimize_greedy(self, process: Process, max_retries: int = 3) -> Schedule:
        """
        Greedy sequential scheduling (Level 3 fallback).
        This is the current proven working method.
        """
        print(f"[GREEDY] Starting greedy sequential scheduling")
        
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
        
        # Log workload distribution summary
        print(f"\n[WORKLOAD-SUMMARY] Final Resource Utilization:")
        for resource in process.resources:
            workload = resource_workload.get(resource.id, 0.0)
            utilization = (workload / max(resource.total_available_hours, 1.0)) * 100
            status = "OPTIMAL" if 40 <= utilization <= 70 else ("LOW" if utilization < 40 else "HIGH")
            tasks_count = sum(1 for entry in schedule.entries if entry.resource_id == resource.id)
            print(f"[WORKLOAD-SUMMARY]   Resource {resource.id} ({resource.name}): {utilization:.1f}% utilized ({workload:.1f}h/{resource.total_available_hours:.1f}h), {tasks_count} tasks [{status}]")
        
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
        """
        Find the best available resource using cascading logic with workload balancing:
        1. Exact Match (100% skill match) + Workload Balancing + Cost Optimization
        2. Partial Match (≥50% skill match) + Workload Balancing + Cost Savings
        3. CMS Default Fallback (if capacity allows)
        """
        required_skills = task.required_skills
        PARTIAL_MATCH_THRESHOLD = 0.50  # 50% minimum skill match
        
        # Workload balancing thresholds
        LOW_UTILIZATION = 0.40      # < 40% - underutilized, prefer these
        TARGET_UTILIZATION = 0.70   # 40-70% - optimal range
        HIGH_UTILIZATION = 0.85     # 70-85% - approaching capacity
        MAX_UTILIZATION = 0.95      # > 95% - prevent overload (leave 5% buffer)
        
        # Get CMS default resource for this task (for fallback and cost comparison)
        cms_default_resource = None
        cms_default_cost = float('inf')
        
        print(f"[DEBUG-CMS] Task {task.id}: _original_assignments exists? {bool(self._original_assignments)}")
        print(f"[DEBUG-CMS] Task {task.id}: _original_assignments content: {self._original_assignments}")
        
        if self._original_assignments:
            # Use string key to match dictionary format
            task_id_str = str(task.id)
            original_resource_id = self._original_assignments.get(task_id_str)
            print(f"[DEBUG-CMS] Task {task.id}: Looking up '{task_id_str}' -> Resource ID: {original_resource_id}")
            
            if original_resource_id:
                # Find resource by matching ID (handle both string and int)
                print(f"[DEBUG-CMS] Task {task.id}: Searching for resource with ID matching '{original_resource_id}'")
                print(f"[DEBUG-CMS] Available resource IDs: {[r.id for r in process.resources]}")
                cms_default_resource = next((r for r in process.resources if str(r.id) == str(original_resource_id)), None)
                if cms_default_resource:
                    cms_default_cost = cms_default_resource.hourly_rate
                    print(f"[DEBUG-CMS] Task {task.id}: Found CMS default resource {cms_default_resource.id} (${cms_default_cost:.2f}/hr)")
                else:
                    print(f"[DEBUG-CMS] Task {task.id}: Resource {original_resource_id} NOT FOUND in process.resources!")
            else:
                print(f"[DEBUG-CMS] Task {task.id}: No original assignment found for task ID '{task_id_str}'")
        
        # Candidates for each tier
        exact_match_candidates = []
        partial_match_candidates = []
        
        print(f"[RESOURCE-MATCH] Finding resource for task {task.id} ({task.name})")
        print(f"[RESOURCE-MATCH] Required skills: {[f'{s.name}:{s.level.name}' for s in required_skills]}")
        print(f"[RESOURCE-MATCH] CMS default: Resource {cms_default_resource.id if cms_default_resource else 'None'} (${cms_default_cost:.2f}/hr)")
        
        for resource in process.resources:
            # Check capacity with workload balancing - prevent overutilization
            current_workload = resource_workload[resource.id]
            needed_capacity = current_workload + task.duration_hours
            utilization_ratio = needed_capacity / max(resource.total_available_hours, 1.0)
            
            # Hard limit: Don't exceed MAX_UTILIZATION threshold
            if utilization_ratio > MAX_UTILIZATION:
                print(f"[RESOURCE-MATCH]   Resource {resource.id} ({resource.name}) - SKIP: would exceed {MAX_UTILIZATION*100:.0f}% utilization ({utilization_ratio*100:.1f}%)")
                continue
            
            # Calculate skill match percentage
            skill_match = resource.get_skill_score(required_skills) if required_skills else 1.0
            available_hour = resource_next_available.get(resource.id, 0.0)
            
            # Determine utilization status for logging
            if utilization_ratio < LOW_UTILIZATION:
                util_status = "LOW"
            elif utilization_ratio < TARGET_UTILIZATION:
                util_status = "OPTIMAL"
            elif utilization_ratio < HIGH_UTILIZATION:
                util_status = "HIGH"
            else:
                util_status = "CRITICAL"
            
            print(f"[RESOURCE-MATCH]   Resource {resource.id} ({resource.name}) - Skill: {skill_match*100:.1f}%, Cost: ${resource.hourly_rate:.2f}/hr, Utilization: {utilization_ratio*100:.1f}% ({util_status})")
            
            # Categorize into exact or partial match with utilization score
            if skill_match >= 1.0:
                # Exact match (100% skills)
                exact_match_candidates.append((resource, available_hour, skill_match, utilization_ratio))
            elif skill_match >= PARTIAL_MATCH_THRESHOLD:
                # Partial match (≥50% skills)
                partial_match_candidates.append((resource, available_hour, skill_match, utilization_ratio))
        
        # TIER 1: Exact Match + Workload Balancing + Cost Optimization
        if exact_match_candidates:
            print(f"[RESOURCE-MATCH] TIER 1: Found {len(exact_match_candidates)} exact match(es)")
            
            # Prefer underutilized resources (better workload distribution)
            underutilized = [(r, h, s, u) for r, h, s, u in exact_match_candidates if u < LOW_UTILIZATION]
            optimal_util = [(r, h, s, u) for r, h, s, u in exact_match_candidates if LOW_UTILIZATION <= u < TARGET_UTILIZATION]
            high_util = [(r, h, s, u) for r, h, s, u in exact_match_candidates if u >= TARGET_UTILIZATION]
            
            # Priority: underutilized > optimal > high (for balanced distribution)
            best_candidate = None
            selection_reason = ""
            
            if underutilized:
                # Sort by cost (lowest first) among underutilized resources
                underutilized.sort(key=lambda x: (x[0].hourly_rate, x[3]))  # cost, then utilization
                best_candidate = underutilized[0]
                selection_reason = f"underutilized ({best_candidate[3]*100:.1f}%), cheapest"
            elif optimal_util:
                # Sort by cost among optimally utilized resources
                optimal_util.sort(key=lambda x: (x[0].hourly_rate, x[3]))
                best_candidate = optimal_util[0]
                selection_reason = f"optimal utilization ({best_candidate[3]*100:.1f}%)"
            elif high_util:
                # Last resort: high utilization but still below max
                high_util.sort(key=lambda x: (x[0].hourly_rate, x[3]))
                best_candidate = high_util[0]
                selection_reason = f"high utilization ({best_candidate[3]*100:.1f}%), no better option"
            
            if best_candidate:
                best_resource, best_start_hour, match_score, util_ratio = best_candidate
                print(f"[RESOURCE-MATCH] ✅ Selected Resource {best_resource.id} ({best_resource.name}) - Exact match, ${best_resource.hourly_rate:.2f}/hr, {selection_reason}")
                return best_resource, best_start_hour
            
            # Fallback: original behavior if categorization fails
            exact_match_candidates.sort(key=lambda x: (x[0].hourly_rate, x[1]))
            best_resource, best_start_hour, match_score, util_ratio = exact_match_candidates[0]
            print(f"[RESOURCE-MATCH] ✅ Selected Resource {best_resource.id} ({best_resource.name}) - Exact match, ${best_resource.hourly_rate:.2f}/hr")
            return best_resource, best_start_hour
        
        # TIER 2: Partial Match + Workload Balancing + Cost Savings
        if partial_match_candidates:
            print(f"[RESOURCE-MATCH] TIER 2: Found {len(partial_match_candidates)} partial match(es)")
            # Filter: must be cheaper than CMS default
            affordable_candidates = [
                (r, h, s, u) for r, h, s, u in partial_match_candidates 
                if r.hourly_rate < cms_default_cost
            ]
            
            if affordable_candidates:
                # Prefer underutilized resources, then sort by skill match and cost
                underutilized = [(r, h, s, u) for r, h, s, u in affordable_candidates if u < LOW_UTILIZATION]
                normal_util = [(r, h, s, u) for r, h, s, u in affordable_candidates if u >= LOW_UTILIZATION]
                
                best_list = underutilized if underutilized else normal_util
                # Sort by skill match (highest first), then by utilization (lowest first), then cost
                best_list.sort(key=lambda x: (-x[2], x[3], x[0].hourly_rate))
                best_resource, best_start_hour, match_score, util_ratio = best_list[0]
                
                util_note = "balanced workload" if util_ratio < TARGET_UTILIZATION else "acceptable load"
                print(f"[RESOURCE-MATCH] ✅ Selected Resource {best_resource.id} ({best_resource.name}) - {match_score*100:.1f}% match, ${best_resource.hourly_rate:.2f}/hr, {util_ratio*100:.1f}% util ({util_note})")
                return best_resource, best_start_hour
            else:
                print(f"[RESOURCE-MATCH] No partial matches cheaper than CMS default (${cms_default_cost:.2f}/hr)")
        
        # TIER 3: CMS Default Fallback (with workload check)
        if cms_default_resource:
            current_workload = resource_workload[cms_default_resource.id]
            needed_capacity = current_workload + task.duration_hours
            utilization_ratio = needed_capacity / max(cms_default_resource.total_available_hours, 1.0)
            
            # Check if CMS default can handle the load
            if utilization_ratio <= MAX_UTILIZATION:
                available_hour = resource_next_available.get(cms_default_resource.id, 0.0)
                util_warning = "⚠️ HIGH LOAD" if utilization_ratio > HIGH_UTILIZATION else ""
                print(f"[RESOURCE-MATCH] ✅ FALLBACK: Using CMS default Resource {cms_default_resource.id} ({cms_default_resource.name}) - ${cms_default_resource.hourly_rate:.2f}/hr, {utilization_ratio*100:.1f}% util {util_warning}")
                return cms_default_resource, available_hour
            else:
                print(f"[RESOURCE-MATCH] ❌ CMS default resource would exceed {MAX_UTILIZATION*100:.0f}% utilization ({utilization_ratio*100:.1f}%)")
        
        print(f"[RESOURCE-MATCH] ❌ No suitable resource found for task {task.id}")
        return None, None
    
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
