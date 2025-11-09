"""
User Journey Optimizer - Optimizes processes from the user's perspective
Focuses on minimizing user waiting time and improving experience
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import heapq

from .models import Process, Task, Resource, Schedule, ScheduleEntry
from .optimizers import BaseOptimizer

@dataclass
class UserJourneyMetrics:
    """Metrics for user journey optimization"""
    total_journey_time: float  # Total time from user's first to last involvement
    active_time: float  # Time user is actively engaged
    waiting_time: float  # Time user spends waiting
    task_transitions: int  # Number of task transitions
    resource_switches: int  # Number of times user switches between resources
    critical_path_length: float  # Length of the critical path
    efficiency_ratio: float  # Active time / Total time
    # New fields for separating patient vs admin timeline
    patient_involved_tasks: int = 0  # Number of tasks where patient is involved
    admin_only_tasks: int = 0  # Number of admin-only tasks
    total_admin_time: float = 0.0  # Total time including admin tasks
    patient_start_time: float = 0.0  # When patient first gets involved
    patient_end_time: float = 0.0  # When patient last involvement ends
    admin_overhead_time: float = 0.0  # Time spent on admin tasks (before/after patient)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_journey_time': self.total_journey_time,
            'active_time': self.active_time,
            'waiting_time': self.waiting_time,
            'task_transitions': self.task_transitions,
            'resource_switches': self.resource_switches,
            'critical_path_length': self.critical_path_length,
            'efficiency_ratio': self.efficiency_ratio,
            'patient_involved_tasks': self.patient_involved_tasks,
            'admin_only_tasks': self.admin_only_tasks,
            'total_admin_time': self.total_admin_time,
            'patient_start_time': self.patient_start_time,
            'patient_end_time': self.patient_end_time,
            'admin_overhead_time': self.admin_overhead_time
        }

@dataclass
class UserJourneyStep:
    """Represents a step in the user's journey"""
    task_id: str
    task_name: str
    resource_id: str
    resource_name: str
    start_time: float
    end_time: float
    duration: float
    waiting_time_before: float
    is_critical: bool = False

class UserJourneyOptimizer(BaseOptimizer):
    """
    Optimizer focused on single-user journey through a process
    Minimizes waiting time and optimizes the user experience
    """
    
    def __init__(self, 
                 minimize_waiting: bool = True,
                 enforce_continuity: bool = True,
                 optimize_critical_path: bool = True):
        """
        Initialize the User Journey Optimizer
        
        Args:
            minimize_waiting: Whether to minimize waiting time between tasks
            enforce_continuity: Try to keep the same resource for consecutive tasks
            optimize_critical_path: Focus optimization on critical path
        """
        super().__init__()
        self.minimize_waiting = minimize_waiting
        self.enforce_continuity = enforce_continuity
        self.optimize_critical_path = optimize_critical_path
        self.journey_steps: List[UserJourneyStep] = []
        self.critical_path: List[str] = []
    
    def optimize(self, process: Process) -> Schedule:
        """
        Optimize the process from user's perspective
        
        Args:
            process: The process to optimize
            
        Returns:
            Optimized schedule with minimal user waiting time
        """
        # Create a schedule
        schedule = Schedule(
            process_id=process.id
        )
        
        # Calculate critical path first if enabled
        if self.optimize_critical_path:
            self.critical_path = self._calculate_critical_path(process)
        
        # Sort tasks by dependencies to ensure correct order
        sorted_tasks = self._topological_sort(process.tasks)
        
        # Track completion times for dependencies
        task_completion_times = {}
        last_resource_id = None
        current_time = 0.0
        
        # Schedule each task sequentially for the user
        for task in sorted_tasks:
            # Determine earliest start time based on dependencies
            earliest_start = self._calculate_earliest_start(
                task, 
                task_completion_times
            )
            
            # Find best resource for this task
            best_resource = self._select_best_resource(
                task, 
                process.resources,
                earliest_start,
                last_resource_id if self.enforce_continuity else None
            )
            
            if not best_resource:
                print(f"Warning: No suitable resource found for task {task.name}")
                continue
            
            # Calculate actual start time (may need to wait for resource)
            actual_start = max(earliest_start, current_time)
            
            # Check if resource is available at this time
            resource_available_time = self._get_resource_available_time(
                best_resource, 
                schedule
            )
            actual_start = max(actual_start, resource_available_time)
            
            # Calculate waiting time
            waiting_time = actual_start - current_time if current_time > 0 else 0
            
            # Create schedule entry
            # Note: ScheduleEntry expects datetime objects, but we're using float hours
            # We'll convert float hours to datetime for compatibility
            from datetime import datetime, timedelta
            base_time = datetime.now()
            
            entry = ScheduleEntry(
                task_id=task.id,
                resource_id=best_resource.id,
                start_time=base_time + timedelta(hours=actual_start),
                end_time=base_time + timedelta(hours=actual_start + task.duration_hours),
                start_hour=actual_start,
                end_hour=actual_start + task.duration_hours
            )
            
            schedule.add_entry(entry)
            
            # Track journey step
            is_critical = task.id in self.critical_path
            journey_step = UserJourneyStep(
                task_id=task.id,
                task_name=task.name,
                resource_id=best_resource.id,
                resource_name=best_resource.name,
                start_time=actual_start,
                end_time=actual_start + task.duration_hours,
                duration=task.duration_hours,
                waiting_time_before=waiting_time,
                is_critical=is_critical
            )
            self.journey_steps.append(journey_step)
            
            # Update tracking variables
            task_completion_times[task.id] = actual_start + task.duration_hours
            current_time = actual_start + task.duration_hours
            last_resource_id = best_resource.id
            
            # Update task assignment
            task.assigned_resource = best_resource.id
            task.start_hour = actual_start
            task.end_hour = actual_start + task.duration_hours
        
        # Calculate and attach metrics
        schedule.metrics = self._calculate_metrics(schedule, process)
        
        return schedule
    
    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """
        Sort tasks in topological order based on dependencies
        Ensures tasks are scheduled in valid order
        """
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        task_map = {task.id: task for task in tasks}
        
        for task in tasks:
            in_degree[task.id] = len(task.dependencies)
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    graph[dep_id].append(task.id)
        
        # Start with tasks that have no dependencies
        queue = [task_id for task_id in task_map.keys() if in_degree[task_id] == 0]
        sorted_tasks = []
        
        while queue:
            task_id = queue.pop(0)
            sorted_tasks.append(task_map[task_id])
            
            # Update in-degrees
            for dependent_id in graph[task_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        # If we couldn't sort all tasks, there's a cycle
        if len(sorted_tasks) != len(tasks):
            print("Warning: Cycle detected in task dependencies")
            # Return remaining tasks in original order
            remaining = [t for t in tasks if t not in sorted_tasks]
            sorted_tasks.extend(remaining)
        
        return sorted_tasks
    
    def _calculate_earliest_start(self, 
                                  task: Task,
                                  completion_times: Dict[str, float]) -> float:
        """Calculate earliest possible start time based on dependencies"""
        if not task.dependencies:
            return 0.0
        
        earliest = 0.0
        for dep_id in task.dependencies:
            if dep_id in completion_times:
                earliest = max(earliest, completion_times[dep_id])
        
        return earliest
    
    def _select_best_resource(self, 
                             task: Task,
                             resources: List[Resource],
                             earliest_start: float,
                             preferred_resource_id: Optional[str] = None) -> Optional[Resource]:
        """
        Select the best resource for a task
        
        Criteria:
        1. Has required skills
        2. Available at earliest_start time
        3. Prefers continuity (same resource as previous task if possible)
        4. Lowest cost among qualified resources
        """
        qualified_resources = []
        
        for resource in resources:
            if resource.has_all_skills(task.required_skills):
                # Calculate resource score
                score = 0.0
                
                # Skill match score
                skill_score = resource.get_skill_score(task.required_skills)
                score += skill_score * 10
                
                # Cost factor (lower is better)
                cost_score = 1.0 / (resource.hourly_rate / 50.0)  # Normalized to base rate
                score += cost_score * 5
                
                # Continuity bonus
                if self.enforce_continuity and resource.id == preferred_resource_id:
                    score += 20  # Strong preference for same resource
                
                qualified_resources.append((score, resource))
        
        if not qualified_resources:
            return None
        
        # Sort by score (highest first)
        qualified_resources.sort(key=lambda x: x[0], reverse=True)
        
        return qualified_resources[0][1]
    
    def _get_resource_available_time(self, 
                                     resource: Resource,
                                     schedule: Schedule) -> float:
        """Get the earliest time a resource is available"""
        latest_end = 0.0
        
        for entry in schedule.entries:
            if entry.resource_id == resource.id:
                # Use end_hour (float) instead of end_time (datetime)
                latest_end = max(latest_end, entry.end_hour)
        
        return latest_end
    
    def _calculate_critical_path(self, process: Process) -> List[str]:
        """
        Calculate the critical path through the process
        Uses CPM (Critical Path Method) algorithm
        """
        task_map = {task.id: task for task in process.tasks}
        
        # Forward pass - calculate earliest start and finish times
        earliest_start = {}
        earliest_finish = {}
        
        # Topological sort for forward pass
        sorted_tasks = self._topological_sort(process.tasks)
        
        for task in sorted_tasks:
            if not task.dependencies:
                earliest_start[task.id] = 0
            else:
                earliest_start[task.id] = max(
                    earliest_finish.get(dep_id, 0) 
                    for dep_id in task.dependencies
                )
            
            earliest_finish[task.id] = earliest_start[task.id] + task.duration_hours
        
        # Find project completion time
        project_completion = max(earliest_finish.values()) if earliest_finish else 0
        
        # Backward pass - calculate latest start and finish times
        latest_start = {}
        latest_finish = {}
        
        # Reverse topological order for backward pass
        for task in reversed(sorted_tasks):
            # Find tasks that depend on this task
            dependents = [
                t for t in process.tasks 
                if task.id in t.dependencies
            ]
            
            if not dependents:
                latest_finish[task.id] = project_completion
            else:
                latest_finish[task.id] = min(
                    latest_start.get(dep.id, project_completion)
                    for dep in dependents
                )
            
            latest_start[task.id] = latest_finish[task.id] - task.duration_hours
        
        # Identify critical path (tasks with zero slack)
        critical_path = []
        for task in process.tasks:
            slack = latest_start.get(task.id, 0) - earliest_start.get(task.id, 0)
            if abs(slack) < 0.001:  # Account for floating point precision
                critical_path.append(task.id)
        
        return critical_path
    
    def _calculate_metrics(self, schedule: Schedule, process: Process) -> UserJourneyMetrics:
        """Calculate user journey metrics with separation of patient vs admin timeline"""
        if not schedule.entries:
            return UserJourneyMetrics(0, 0, 0, 0, 0, 0, 0)
        
        from .models import UserInvolvement
        
        # Sort entries by start time (use start_hour for sorting)
        sorted_entries = sorted(schedule.entries, key=lambda e: e.start_hour)
        
        # Separate patient-involved tasks from admin-only tasks
        patient_entries = []
        admin_only_entries = []
        
        for entry in sorted_entries:
            task = process.get_task_by_id(entry.task_id)
            if task and task.user_involvement in [UserInvolvement.DIRECT, UserInvolvement.PASSIVE]:
                patient_entries.append(entry)
            else:
                admin_only_entries.append(entry)
        
        # Calculate PATIENT TIMELINE (only tasks where patient is involved)
        if patient_entries:
            patient_start_time = patient_entries[0].start_hour
            patient_end_time = patient_entries[-1].end_hour
            total_journey_time = patient_end_time - patient_start_time
            active_time = sum(entry.end_hour - entry.start_hour for entry in patient_entries)
            waiting_time = total_journey_time - active_time
        else:
            patient_start_time = 0
            patient_end_time = 0
            total_journey_time = 0
            active_time = 0
            waiting_time = 0
        
        # Calculate ADMINISTRATIVE TIMELINE (all tasks)
        total_admin_time = sorted_entries[-1].end_hour - sorted_entries[0].start_hour
        admin_overhead_time = sum(entry.end_hour - entry.start_hour for entry in admin_only_entries)
        
        # Count resource switches (only for patient-involved tasks)
        resource_switches = 0
        last_resource_id = None
        for entry in patient_entries:
            if last_resource_id and entry.resource_id != last_resource_id:
                resource_switches += 1
            last_resource_id = entry.resource_id
        
        # Calculate critical path length
        critical_path_length = sum(
            process.get_task_by_id(task_id).duration_hours
            for task_id in self.critical_path
            if process.get_task_by_id(task_id)
        ) if self.critical_path else total_journey_time
        
        # Calculate efficiency ratio (for patient journey only)
        efficiency_ratio = active_time / total_journey_time if total_journey_time > 0 else 0
        
        return UserJourneyMetrics(
            total_journey_time=total_journey_time,
            active_time=active_time,
            waiting_time=waiting_time,
            task_transitions=len(patient_entries) - 1 if patient_entries else 0,
            resource_switches=resource_switches,
            critical_path_length=critical_path_length,
            efficiency_ratio=efficiency_ratio,
            patient_involved_tasks=len(patient_entries),
            admin_only_tasks=len(admin_only_entries),
            total_admin_time=total_admin_time,
            patient_start_time=patient_start_time,
            patient_end_time=patient_end_time,
            admin_overhead_time=admin_overhead_time
        )
    
    def get_journey_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for visualizing the user journey
        
        Returns:
            Dictionary with journey visualization data
        """
        return {
            'steps': [
                {
                    'task_id': step.task_id,
                    'task_name': step.task_name,
                    'resource_name': step.resource_name,
                    'start_time': step.start_time,
                    'end_time': step.end_time,
                    'duration': step.duration,
                    'waiting_time': step.waiting_time_before,
                    'is_critical': step.is_critical
                }
                for step in self.journey_steps
            ],
            'critical_path': self.critical_path,
            'total_waiting_time': sum(step.waiting_time_before for step in self.journey_steps),
            'total_active_time': sum(step.duration for step in self.journey_steps)
        }
    
    def suggest_improvements(self, schedule: Schedule, process: Process) -> List[str]:
        """
        Suggest improvements to reduce user waiting time
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Analyze waiting times
        high_wait_steps = [
            step for step in self.journey_steps 
            if step.waiting_time_before > 0.5  # More than 30 minutes
        ]
        
        if high_wait_steps:
            suggestions.append(
                f"High waiting times detected at {len(high_wait_steps)} steps. "
                "Consider adding more resources or adjusting schedules."
            )
        
        # Check resource switches
        resource_switches = len(set(step.resource_id for step in self.journey_steps)) - 1
        if resource_switches > len(self.journey_steps) * 0.5:
            suggestions.append(
                "Frequent resource switches detected. "
                "Consider assigning dedicated resources for better continuity."
            )
        
        # Critical path analysis
        if self.critical_path:
            critical_tasks = [
                process.get_task_by_id(tid) 
                for tid in self.critical_path 
                if process.get_task_by_id(tid)
            ]
            longest_critical = max(critical_tasks, key=lambda t: t.duration_hours)
            suggestions.append(
                f"Critical path task '{longest_critical.name}' takes {longest_critical.duration_hours} hours. "
                "Optimizing this task would directly reduce total journey time."
            )
        
        return suggestions
