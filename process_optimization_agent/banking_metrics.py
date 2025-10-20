"""
Banking Performance Metrics Module
Implements FR8-FR10: Performance measurement and multi-objective optimization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta


class OptimizationGoal(Enum):
    """Optimization objectives (FR9)"""
    MINIMIZE_WAITING_TIME = "minimize_waiting_time"
    MINIMIZE_PROCESSING_TIME = "minimize_processing_time"
    MINIMIZE_COST = "minimize_cost"
    BALANCE_WORKLOAD = "balance_workload"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_CUSTOMER_EFFORT = "minimize_customer_effort"


@dataclass
class TaskMetrics:
    """Metrics for individual tasks (FR8)"""
    task_id: str
    task_name: str
    customer_waiting_time: float = 0.0  # Hours customer waits
    processing_time: float = 0.0  # Actual task duration
    resource_hours: float = 0.0  # Employee hours used
    cost: float = 0.0  # Total cost
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    assigned_resource: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "customer_waiting_time": self.customer_waiting_time,
            "processing_time": self.processing_time,
            "resource_hours": self.resource_hours,
            "cost": self.cost,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "assigned_resource": self.assigned_resource
        }


@dataclass
class ResourceMetrics:
    """Metrics for resource utilization (FR8)"""
    resource_id: str
    resource_name: str
    total_hours_assigned: float = 0.0
    total_hours_available: float = 0.0
    utilization_rate: float = 0.0  # Percentage
    total_cost: float = 0.0
    tasks_assigned: int = 0
    idle_time: float = 0.0
    
    def calculate_utilization(self):
        """Calculate utilization rate"""
        if self.total_hours_available > 0:
            self.utilization_rate = (self.total_hours_assigned / self.total_hours_available) * 100
        else:
            self.utilization_rate = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "total_hours_assigned": self.total_hours_assigned,
            "total_hours_available": self.total_hours_available,
            "utilization_rate": self.utilization_rate,
            "total_cost": self.total_cost,
            "tasks_assigned": self.tasks_assigned,
            "idle_time": self.idle_time
        }


@dataclass
class ProcessMetrics:
    """Overall process performance metrics (FR8, FR9)"""
    process_id: str
    process_name: str
    
    # Time metrics
    total_customer_waiting_time: float = 0.0  # Total time customers wait
    total_processing_time: float = 0.0  # Total active processing time
    total_duration: float = 0.0  # End-to-end process duration
    average_task_duration: float = 0.0
    
    # Cost metrics
    total_cost: float = 0.0
    average_cost_per_task: float = 0.0
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Resource metrics
    total_resource_hours: float = 0.0
    average_resource_utilization: float = 0.0
    workload_balance_score: float = 0.0  # 0-100, higher is better balanced
    
    # Task metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    critical_path_length: float = 0.0
    
    # Customer experience metrics
    customer_touchpoints: int = 0  # Number of times customer is involved
    customer_effort_score: float = 0.0  # Lower is better
    
    # Detailed metrics
    task_metrics: List[TaskMetrics] = field(default_factory=list)
    resource_metrics: List[ResourceMetrics] = field(default_factory=list)
    
    def calculate_averages(self):
        """Calculate average metrics"""
        if self.total_tasks > 0:
            self.average_task_duration = self.total_processing_time / self.total_tasks
            self.average_cost_per_task = self.total_cost / self.total_tasks
        
        if self.resource_metrics:
            total_util = sum(rm.utilization_rate for rm in self.resource_metrics)
            self.average_resource_utilization = total_util / len(self.resource_metrics)
            
            # Calculate workload balance (lower variance = better balance)
            utilizations = [rm.utilization_rate for rm in self.resource_metrics]
            if len(utilizations) > 1:
                mean_util = sum(utilizations) / len(utilizations)
                variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)
                # Convert to 0-100 score (lower variance = higher score)
                self.workload_balance_score = max(0, 100 - variance)
            else:
                self.workload_balance_score = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "process_id": self.process_id,
            "process_name": self.process_name,
            "total_customer_waiting_time": self.total_customer_waiting_time,
            "total_processing_time": self.total_processing_time,
            "total_duration": self.total_duration,
            "average_task_duration": self.average_task_duration,
            "total_cost": self.total_cost,
            "average_cost_per_task": self.average_cost_per_task,
            "cost_breakdown": self.cost_breakdown,
            "total_resource_hours": self.total_resource_hours,
            "average_resource_utilization": self.average_resource_utilization,
            "workload_balance_score": self.workload_balance_score,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "critical_path_length": self.critical_path_length,
            "customer_touchpoints": self.customer_touchpoints,
            "customer_effort_score": self.customer_effort_score,
            "task_metrics": [tm.to_dict() for tm in self.task_metrics],
            "resource_metrics": [rm.to_dict() for rm in self.resource_metrics]
        }


@dataclass
class OptimizationObjective:
    """Multi-objective optimization configuration (FR10)"""
    goal: OptimizationGoal
    weight: float = 1.0  # Relative importance (0-1)
    target_value: Optional[float] = None  # Optional target
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal.value,
            "weight": self.weight,
            "target_value": self.target_value
        }


class BankingMetricsCalculator:
    """Calculate performance metrics for banking processes (FR8)"""
    
    def __init__(self):
        pass
    
    def calculate_task_metrics(
        self, 
        task: Any, 
        schedule_entry: Any,
        previous_task_end_time: Optional[float] = None
    ) -> TaskMetrics:
        """Calculate metrics for a single task (FR8)"""
        metrics = TaskMetrics(
            task_id=task.id,
            task_name=task.name,
            processing_time=task.duration_hours,
            resource_hours=task.duration_hours,
            cost=schedule_entry.cost if hasattr(schedule_entry, 'cost') else 0.0,
            assigned_resource=schedule_entry.resource_id if hasattr(schedule_entry, 'resource_id') else None
        )
        
        # Calculate customer waiting time
        if previous_task_end_time is not None and hasattr(schedule_entry, 'start_hour'):
            metrics.customer_waiting_time = max(0, schedule_entry.start_hour - previous_task_end_time)
        
        # Set times
        if hasattr(schedule_entry, 'start_time'):
            metrics.start_time = schedule_entry.start_time
        if hasattr(schedule_entry, 'end_time'):
            metrics.end_time = schedule_entry.end_time
        
        return metrics
    
    def calculate_resource_metrics(
        self, 
        resource: Any, 
        assigned_tasks: List[Any]
    ) -> ResourceMetrics:
        """Calculate metrics for a resource (FR8)"""
        metrics = ResourceMetrics(
            resource_id=resource.id,
            resource_name=resource.name,
            total_hours_available=resource.total_available_hours if hasattr(resource, 'total_available_hours') else 160.0,
            tasks_assigned=len(assigned_tasks)
        )
        
        # Calculate total hours and cost
        for task in assigned_tasks:
            metrics.total_hours_assigned += task.duration_hours
            metrics.total_cost += task.duration_hours * resource.hourly_rate
        
        # Calculate idle time
        metrics.idle_time = max(0, metrics.total_hours_available - metrics.total_hours_assigned)
        
        # Calculate utilization
        metrics.calculate_utilization()
        
        return metrics
    
    def calculate_process_metrics(
        self, 
        process: Any, 
        schedule: Any,
        banking_process: Optional[Any] = None
    ) -> ProcessMetrics:
        """Calculate overall process metrics (FR8)"""
        metrics = ProcessMetrics(
            process_id=process.id,
            process_name=process.name,
            total_tasks=len(process.tasks)
        )
        
        # Track resource assignments
        resource_tasks = {r.id: [] for r in process.resources}
        task_dict = {t.id: t for t in process.tasks}
        
        # Calculate task-level metrics
        previous_end_time = None
        for entry in schedule.entries:
            task = task_dict.get(entry.task_id)
            if task:
                task_metrics = self.calculate_task_metrics(task, entry, previous_end_time)
                metrics.task_metrics.append(task_metrics)
                
                # Update totals
                metrics.total_customer_waiting_time += task_metrics.customer_waiting_time
                metrics.total_processing_time += task_metrics.processing_time
                metrics.total_resource_hours += task_metrics.resource_hours
                metrics.total_cost += task_metrics.cost
                metrics.completed_tasks += 1
                
                # Track resource assignments
                if entry.resource_id in resource_tasks:
                    resource_tasks[entry.resource_id].append(task)
                
                # Update previous end time for waiting time calculation
                if hasattr(entry, 'end_hour'):
                    previous_end_time = entry.end_hour
                
                # Count customer touchpoints
                if hasattr(task, 'user_involvement'):
                    from .models import UserInvolvement
                    if task.user_involvement == UserInvolvement.DIRECT:
                        metrics.customer_touchpoints += 1
        
        # Calculate resource-level metrics
        for resource in process.resources:
            resource_metrics = self.calculate_resource_metrics(
                resource, 
                resource_tasks.get(resource.id, [])
            )
            metrics.resource_metrics.append(resource_metrics)
        
        # Calculate total duration
        if schedule.entries:
            max_end_hour = max(
                entry.end_hour for entry in schedule.entries 
                if hasattr(entry, 'end_hour')
            )
            metrics.total_duration = max_end_hour
        
        # Calculate critical path length
        metrics.critical_path_length = self._calculate_critical_path(process, schedule)
        
        # Calculate customer effort score (combination of waiting time and touchpoints)
        if metrics.completed_tasks > 0:
            avg_waiting = metrics.total_customer_waiting_time / metrics.completed_tasks
            metrics.customer_effort_score = avg_waiting + (metrics.customer_touchpoints * 0.5)
        
        # Calculate cost breakdown
        for resource_metric in metrics.resource_metrics:
            metrics.cost_breakdown[resource_metric.resource_name] = resource_metric.total_cost
        
        # Calculate averages
        metrics.calculate_averages()
        
        return metrics
    
    def _calculate_critical_path(self, process: Any, schedule: Any) -> float:
        """Calculate critical path length"""
        if not schedule.entries:
            return 0.0
        
        # Simple approach: longest path from start to end
        task_end_times = {}
        for entry in schedule.entries:
            if hasattr(entry, 'end_hour'):
                task_end_times[entry.task_id] = entry.end_hour
        
        if task_end_times:
            return max(task_end_times.values())
        return 0.0
    
    def compare_metrics(
        self, 
        baseline_metrics: ProcessMetrics, 
        optimized_metrics: ProcessMetrics
    ) -> Dict[str, Any]:
        """Compare baseline vs optimized metrics"""
        comparison = {
            "time_improvement": {
                "waiting_time_reduction": baseline_metrics.total_customer_waiting_time - optimized_metrics.total_customer_waiting_time,
                "waiting_time_reduction_pct": self._calculate_percentage_change(
                    baseline_metrics.total_customer_waiting_time,
                    optimized_metrics.total_customer_waiting_time
                ),
                "duration_reduction": baseline_metrics.total_duration - optimized_metrics.total_duration,
                "duration_reduction_pct": self._calculate_percentage_change(
                    baseline_metrics.total_duration,
                    optimized_metrics.total_duration
                )
            },
            "cost_improvement": {
                "cost_reduction": baseline_metrics.total_cost - optimized_metrics.total_cost,
                "cost_reduction_pct": self._calculate_percentage_change(
                    baseline_metrics.total_cost,
                    optimized_metrics.total_cost
                )
            },
            "resource_improvement": {
                "utilization_improvement": optimized_metrics.average_resource_utilization - baseline_metrics.average_resource_utilization,
                "workload_balance_improvement": optimized_metrics.workload_balance_score - baseline_metrics.workload_balance_score
            },
            "customer_experience_improvement": {
                "effort_reduction": baseline_metrics.customer_effort_score - optimized_metrics.customer_effort_score,
                "touchpoint_reduction": baseline_metrics.customer_touchpoints - optimized_metrics.customer_touchpoints
            }
        }
        
        return comparison
    
    def _calculate_percentage_change(self, baseline: float, optimized: float) -> float:
        """Calculate percentage change"""
        if baseline == 0:
            return 0.0
        return ((baseline - optimized) / baseline) * 100


class MultiObjectiveOptimizer:
    """Handle multi-objective optimization (FR10)"""
    
    def __init__(self, objectives: List[OptimizationObjective]):
        self.objectives = objectives
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0"""
        total_weight = sum(obj.weight for obj in self.objectives)
        if total_weight > 0:
            for obj in self.objectives:
                obj.weight = obj.weight / total_weight
    
    def calculate_fitness_score(self, metrics: ProcessMetrics) -> float:
        """Calculate weighted fitness score based on objectives (FR10)"""
        score = 0.0
        
        for objective in self.objectives:
            objective_score = self._calculate_objective_score(objective, metrics)
            score += objective_score * objective.weight
        
        return score
    
    def _calculate_objective_score(
        self, 
        objective: OptimizationObjective, 
        metrics: ProcessMetrics
    ) -> float:
        """Calculate score for a single objective (higher is better)"""
        if objective.goal == OptimizationGoal.MINIMIZE_WAITING_TIME:
            # Lower waiting time = higher score
            if metrics.total_customer_waiting_time == 0:
                return 100.0
            return max(0, 100 - metrics.total_customer_waiting_time)
        
        elif objective.goal == OptimizationGoal.MINIMIZE_PROCESSING_TIME:
            # Lower duration = higher score
            if metrics.total_duration == 0:
                return 100.0
            return max(0, 100 - metrics.total_duration)
        
        elif objective.goal == OptimizationGoal.MINIMIZE_COST:
            # Lower cost = higher score
            if metrics.total_cost == 0:
                return 100.0
            # Normalize cost to 0-100 scale
            return max(0, 100 - (metrics.total_cost / 100))
        
        elif objective.goal == OptimizationGoal.BALANCE_WORKLOAD:
            # Use workload balance score directly
            return metrics.workload_balance_score
        
        elif objective.goal == OptimizationGoal.MAXIMIZE_THROUGHPUT:
            # More tasks completed in less time = higher score
            if metrics.total_duration > 0:
                throughput = metrics.completed_tasks / metrics.total_duration
                return min(100, throughput * 10)
            return 0.0
        
        elif objective.goal == OptimizationGoal.MINIMIZE_CUSTOMER_EFFORT:
            # Lower effort score = higher score
            if metrics.customer_effort_score == 0:
                return 100.0
            return max(0, 100 - metrics.customer_effort_score)
        
        return 0.0
    
    def get_objectives_summary(self) -> Dict[str, Any]:
        """Get summary of optimization objectives"""
        return {
            "objectives": [obj.to_dict() for obj in self.objectives],
            "total_weight": sum(obj.weight for obj in self.objectives)
        }
