"""
Manufacturing/Production-specific optimization logic
Focuses on maximizing throughput and minimizing cycle time
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from datetime import datetime, timedelta
import heapq
from ...Optimization.models import Process, Task, Resource, Schedule, ScheduleEntry
from .manufacturing_models import (
    ManufacturingProcess, ManufacturingMetrics, ProductionTask,
    ManufacturingProcessType, ProductionStage
)


class ManufacturingOptimizer:
    """
    Optimizer specialized for manufacturing/production processes
    Focuses on maximizing parallelization and throughput
    """
    
    def __init__(self):
        self.optimization_goals = {
            'minimize_cycle_time': True,
            'maximize_throughput': True,
            'maximize_parallelization': True,
            'balance_workload': True,
            'minimize_cost': True
        }
    
    def optimize_production(
        self,
        process: Process,
        manufacturing_process: Optional[ManufacturingProcess] = None
    ) -> Tuple[Schedule, ManufacturingMetrics]:
        """
        Optimize manufacturing process focusing on throughput and efficiency
        
        Returns:
            Tuple of (optimized schedule, manufacturing metrics)
        """
        schedule = Schedule()
        
        # Identify parallel task groups
        parallel_groups = self._identify_parallel_tasks(process)
        
        # Schedule tasks with maximum parallelization
        resource_availability = {r.id: 0.0 for r in process.resources}
        
        self._schedule_with_parallelization(
            process, schedule, parallel_groups, resource_availability
        )
        
        # Calculate manufacturing-specific metrics
        metrics = self._calculate_manufacturing_metrics(
            schedule, process, parallel_groups
        )
        
        return schedule, metrics
    
    def _identify_parallel_tasks(self, process: Process) -> List[Set[str]]:
        """Identify groups of tasks that can run in parallel"""
        parallel_groups = []
        
        # Group tasks by dependency level
        levels = self._get_dependency_levels(process)
        
        for level_tasks in levels.values():
            if len(level_tasks) > 1:
                parallel_groups.append(set(level_tasks))
        
        return parallel_groups
    
    def _get_dependency_levels(self, process: Process) -> Dict[int, List[str]]:
        """Get tasks grouped by dependency level"""
        levels = {}
        task_levels = {}
        
        def get_level(task_id: str) -> int:
            if task_id in task_levels:
                return task_levels[task_id]
            
            task = process.get_task_by_id(task_id)
            if not task or not task.dependencies:
                task_levels[task_id] = 0
                return 0
            
            max_dep_level = max(get_level(dep_id) for dep_id in task.dependencies)
            task_levels[task_id] = max_dep_level + 1
            return max_dep_level + 1
        
        for task in process.tasks:
            level = get_level(task.id)
            if level not in levels:
                levels[level] = []
            levels[level].append(task.id)
        
        return levels
    
    def _schedule_with_parallelization(
        self,
        process: Process,
        schedule: Schedule,
        parallel_groups: List[Set[str]],
        resource_availability: Dict[str, float]
    ):
        """Schedule tasks maximizing parallel execution"""
        scheduled = set()
        ready_queue = []
        
        # Initialize with tasks that have no dependencies
        for task in process.tasks:
            if not task.dependencies:
                priority = -task.duration_hours  # Shorter tasks first
                heapq.heappush(ready_queue, (priority, task.id))
        
        while ready_queue:
            _, task_id = heapq.heappop(ready_queue)
            
            if task_id in scheduled:
                continue
            
            task = process.get_task_by_id(task_id)
            if not task:
                continue
            
            # Find best resource
            resource = self._find_best_manufacturing_resource(
                task, process, resource_availability
            )
            
            if resource:
                start_time = resource_availability[resource.id]
                end_time = start_time + task.duration_hours
                
                entry = ScheduleEntry(
                    task_id=task.id,
                    resource_id=resource.id,
                    start_time=process.start_date + timedelta(hours=start_time),
                    end_time=process.start_date + timedelta(hours=end_time),
                    start_hour=start_time,
                    end_hour=end_time,
                    cost=task.duration_hours * resource.hourly_rate
                )
                
                schedule.entries.append(entry)
                scheduled.add(task.id)
                resource_availability[resource.id] = end_time
                
                # Add newly ready tasks
                for next_task in process.tasks:
                    if (next_task.id not in scheduled and 
                        next_task.can_start(scheduled)):
                        priority = -next_task.duration_hours
                        heapq.heappush(ready_queue, (priority, next_task.id))
    
    def _find_best_manufacturing_resource(
        self,
        task: Task,
        process: Process,
        resource_availability: Dict[str, float]
    ) -> Optional[Resource]:
        """Find best resource for manufacturing task"""
        best_resource = None
        best_time = float('inf')
        
        for resource in process.resources:
            if resource.can_perform(task):
                available_time = resource_availability.get(resource.id, 0.0)
                
                # Prefer resources that are available sooner
                if available_time < best_time:
                    best_time = available_time
                    best_resource = resource
        
        return best_resource
    
    def _calculate_manufacturing_metrics(
        self,
        schedule: Schedule,
        process: Process,
        parallel_groups: List[Set[str]]
    ) -> ManufacturingMetrics:
        """Calculate manufacturing-specific metrics"""
        metrics = ManufacturingMetrics()
        
        if not schedule.entries:
            return metrics
        
        # Calculate cycle time (makespan)
        start_times = [entry.start_hour for entry in schedule.entries]
        end_times = [entry.end_hour for entry in schedule.entries]
        
        metrics.cycle_time = max(end_times) - min(start_times)
        
        # Calculate throughput
        if metrics.cycle_time > 0:
            metrics.throughput = len(process.tasks) / metrics.cycle_time
        
        # Calculate costs
        metrics.production_cost = sum(entry.cost for entry in schedule.entries)
        
        # Calculate parallelization metrics
        time_slots = {}
        for entry in schedule.entries:
            start = int(entry.start_hour)
            if start not in time_slots:
                time_slots[start] = 0
            time_slots[start] += 1
        
        if time_slots:
            metrics.max_parallel_tasks = max(time_slots.values())
            parallel_time = sum(1 for count in time_slots.values() if count > 1)
            metrics.parallel_time_percentage = (parallel_time / len(time_slots)) * 100
        
        # Calculate resource utilization
        total_resource_hours = sum(entry.end_hour - entry.start_hour 
                                   for entry in schedule.entries)
        total_available_hours = sum(r.total_available_hours for r in process.resources)
        
        if total_available_hours > 0:
            metrics.resource_utilization = (total_resource_hours / total_available_hours) * 100
        
        metrics.total_resource_hours = total_resource_hours
        
        # Calculate time efficiency
        theoretical_min_time = sum(task.duration_hours for task in process.tasks) / len(process.resources)
        if theoretical_min_time > 0:
            metrics.time_efficiency = min(1.0, theoretical_min_time / metrics.cycle_time) * 100
        
        # Calculate workload balance
        resource_workload = {}
        for entry in schedule.entries:
            if entry.resource_id not in resource_workload:
                resource_workload[entry.resource_id] = 0
            resource_workload[entry.resource_id] += (entry.end_hour - entry.start_hour)
        
        if resource_workload:
            workloads = list(resource_workload.values())
            avg_workload = sum(workloads) / len(workloads)
            if avg_workload > 0:
                variance = sum((w - avg_workload) ** 2 for w in workloads) / len(workloads)
                std_dev = variance ** 0.5
                # Lower std dev = better balance (score 0-100)
                metrics.workload_balance_score = max(0, 100 - (std_dev / avg_workload * 100))
        
        return metrics
