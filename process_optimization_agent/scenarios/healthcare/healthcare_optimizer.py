"""
Healthcare-specific optimization logic
Focuses on patient journey optimization and healthcare workflows
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from datetime import datetime, timedelta
from ...Optimization.models import Process, Task, Resource, Schedule, ScheduleEntry, UserInvolvement
from .healthcare_models import (
    HealthcareProcess, HealthcareMetrics, PatientTouchpoint,
    HealthcareProcessType, PatientJourneyStage
)


class HealthcareOptimizer:
    """
    Optimizer specialized for healthcare processes
    Focuses on minimizing patient waiting time while maintaining quality
    """
    
    def __init__(self):
        self.optimization_goals = {
            'minimize_patient_waiting': True,
            'minimize_total_journey_time': True,
            'maximize_patient_satisfaction': True,
            'maintain_clinical_quality': True
        }
    
    def optimize_patient_journey(
        self, 
        process: Process,
        healthcare_process: Optional[HealthcareProcess] = None
    ) -> Tuple[Schedule, HealthcareMetrics]:
        """
        Optimize healthcare process focusing on patient experience
        
        Returns:
            Tuple of (optimized schedule, healthcare metrics)
        """
        schedule = Schedule()
        
        # Identify patient-facing tasks
        patient_tasks = self._identify_patient_tasks(process)
        admin_tasks = self._identify_admin_tasks(process)
        
        # Schedule patient-facing tasks sequentially to minimize waiting
        patient_touchpoints = self._schedule_patient_tasks(
            patient_tasks, process, schedule
        )
        
        # Schedule administrative tasks in parallel where possible
        self._schedule_admin_tasks(admin_tasks, process, schedule)
        
        # Calculate healthcare-specific metrics
        metrics = self._calculate_healthcare_metrics(
            schedule, process, patient_touchpoints
        )
        
        return schedule, metrics
    
    def _identify_patient_tasks(self, process: Process) -> List[Task]:
        """Identify tasks that involve direct patient interaction"""
        patient_tasks = []
        for task in process.tasks:
            if task.user_involvement in [UserInvolvement.DIRECT, UserInvolvement.PASSIVE]:
                patient_tasks.append(task)
        return patient_tasks
    
    def _identify_admin_tasks(self, process: Process) -> List[Task]:
        """Identify administrative tasks that don't require patient presence"""
        admin_tasks = []
        for task in process.tasks:
            if task.user_involvement == UserInvolvement.ADMIN:
                admin_tasks.append(task)
        return admin_tasks
    
    def _schedule_patient_tasks(
        self,
        tasks: List[Task],
        process: Process,
        schedule: Schedule
    ) -> List[PatientTouchpoint]:
        """Schedule patient-facing tasks to minimize waiting time"""
        touchpoints = []
        current_time = 0.0
        
        # Sort by dependencies to maintain proper order
        sorted_tasks = self._topological_sort(tasks)
        
        for task in sorted_tasks:
            # Find best available resource
            resource = self._find_best_healthcare_resource(task, process)
            
            if resource:
                end_time = current_time + task.duration_hours
                
                entry = ScheduleEntry(
                    task_id=task.id,
                    resource_id=resource.id,
                    start_time=process.start_date + timedelta(hours=current_time),
                    end_time=process.start_date + timedelta(hours=end_time),
                    start_hour=current_time,
                    end_hour=end_time,
                    cost=task.duration_hours * resource.hourly_rate
                )
                
                schedule.entries.append(entry)
                
                # Track touchpoint
                touchpoint = PatientTouchpoint(
                    task_id=task.id,
                    task_name=task.name,
                    resource_id=resource.id,
                    resource_name=resource.name,
                    start_time=current_time,
                    duration=task.duration_hours,
                    interaction_type=task.user_involvement.value
                )
                touchpoints.append(touchpoint)
                
                current_time = end_time
        
        return touchpoints
    
    def _schedule_admin_tasks(
        self,
        tasks: List[Task],
        process: Process,
        schedule: Schedule
    ):
        """Schedule administrative tasks, can run in parallel"""
        for task in tasks:
            resource = self._find_best_healthcare_resource(task, process)
            
            if resource:
                # Admin tasks can start immediately (parallel to patient journey)
                start_time = 0.0
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
    
    def _find_best_healthcare_resource(
        self,
        task: Task,
        process: Process
    ) -> Optional[Resource]:
        """Find best resource for healthcare task"""
        best_resource = None
        best_score = -1
        
        for resource in process.resources:
            if resource.can_perform(task):
                # Score based on skill match and availability
                score = resource.get_skill_match_score(task)
                
                if score > best_score:
                    best_score = score
                    best_resource = resource
        
        return best_resource
    
    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks by dependencies"""
        sorted_tasks = []
        visited = set()
        
        def visit(task: Task):
            if task.id in visited:
                return
            visited.add(task.id)
            
            # Visit dependencies first
            for dep_id in task.dependencies:
                dep_task = next((t for t in tasks if t.id == dep_id), None)
                if dep_task:
                    visit(dep_task)
            
            sorted_tasks.append(task)
        
        for task in tasks:
            visit(task)
        
        return sorted_tasks
    
    def _calculate_healthcare_metrics(
        self,
        schedule: Schedule,
        process: Process,
        touchpoints: List[PatientTouchpoint]
    ) -> HealthcareMetrics:
        """Calculate healthcare-specific metrics"""
        metrics = HealthcareMetrics()
        
        if not touchpoints:
            return metrics
        
        # Calculate patient journey times
        patient_start = min(tp.start_time for tp in touchpoints)
        patient_end = max(tp.start_time + tp.duration for tp in touchpoints)
        
        metrics.total_patient_journey_time = patient_end - patient_start
        metrics.patient_active_time = sum(tp.duration for tp in touchpoints)
        metrics.patient_waiting_time = metrics.total_patient_journey_time - metrics.patient_active_time
        
        # Calculate touchpoints
        metrics.number_of_touchpoints = len(touchpoints)
        metrics.number_of_resource_changes = len(set(tp.resource_id for tp in touchpoints)) - 1
        
        # Calculate costs
        metrics.total_cost = sum(entry.cost for entry in schedule.entries)
        
        # Calculate satisfaction
        if metrics.total_patient_journey_time > 0:
            metrics.patient_satisfaction_score = 1.0 - min(1.0, 
                metrics.patient_waiting_time / metrics.total_patient_journey_time
            )
        
        return metrics
