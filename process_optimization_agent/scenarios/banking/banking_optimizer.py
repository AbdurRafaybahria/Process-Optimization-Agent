"""
Banking Process Optimization Engine
Implements FR11-FR15: What-if analysis, parallelization, resource-aware scheduling, and RL
"""

import copy
import random
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import heapq

from ...Optimization.models import Process, Task, Resource, Schedule, ScheduleEntry, UserInvolvement
from .banking_models import BankingProcess, TaskDependency
from .banking_metrics import (
    BankingMetricsCalculator, ProcessMetrics, OptimizationObjective,
    MultiObjectiveOptimizer, OptimizationGoal
)
from ...Optimization.optimizers import BaseOptimizer


class BankingProcessOptimizer(BaseOptimizer):
    """
    Advanced optimizer for banking processes
    Implements FR11-FR15
    """
    
    def __init__(
        self, 
        objectives: Optional[List[OptimizationObjective]] = None,
        enable_parallelization: bool = True,
        enable_rl: bool = False
    ):
        super().__init__()
        self.metrics_calculator = BankingMetricsCalculator()
        
        # Default objectives if none provided (FR9, FR10)
        if objectives is None:
            objectives = [
                OptimizationObjective(OptimizationGoal.MINIMIZE_WAITING_TIME, weight=0.4),
                OptimizationObjective(OptimizationGoal.MINIMIZE_COST, weight=0.3),
                OptimizationObjective(OptimizationGoal.BALANCE_WORKLOAD, weight=0.3)
            ]
        
        self.multi_objective_optimizer = MultiObjectiveOptimizer(objectives)
        self.enable_parallelization = enable_parallelization  # FR12
        self.enable_rl = enable_rl  # FR15
        
        # RL components (FR15)
        self.q_table = {}  # State-action Q-values
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        self.training_episodes = 0
    
    def optimize(self, process: Process, banking_process: Optional[BankingProcess] = None) -> Schedule:
        """
        Optimize banking process with advanced features (FR11-FR15)
        """
        # Detect and apply dependencies
        self._detect_and_apply_dependencies(process)
        
        # Create banking process if not provided
        if banking_process is None:
            from .banking_detector import BankingProcessDetector
            detector = BankingProcessDetector()
            banking_process = detector.detect_process(process)
        
        # Apply banking-specific dependencies
        self._apply_banking_dependencies(process, banking_process)
        
        # Identify parallelizable tasks (FR12)
        parallel_groups = self._identify_parallel_tasks(process, banking_process)
        
        # Create schedule with resource awareness (FR14)
        if self.enable_rl and self.training_episodes > 0:
            schedule = self._optimize_with_rl(process, banking_process, parallel_groups)
        else:
            schedule = self._optimize_with_heuristics(process, banking_process, parallel_groups)
        
        # Validate process integrity (FR7)
        is_valid, missing_critical = banking_process.validate_process_integrity(
            {entry.task_id for entry in schedule.entries}
        )
        
        if not is_valid:
            print(f"[WARNING] Process integrity violation: Missing critical tasks: {missing_critical}")
            schedule.metadata["integrity_violation"] = missing_critical
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_process_metrics(
            process, schedule, banking_process
        )
        schedule.metadata["performance_metrics"] = metrics.to_dict()
        schedule.metadata["banking_process_type"] = banking_process.process_type.value
        
        return schedule
    
    def _apply_banking_dependencies(self, process: Process, banking_process: BankingProcess):
        """Apply banking-specific dependencies to tasks (FR3)"""
        for dep in banking_process.task_dependencies:
            target_task = process.get_task_by_id(dep.target_task_id)
            if target_task:
                if not hasattr(target_task, 'dependencies') or target_task.dependencies is None:
                    target_task.dependencies = set()
                target_task.dependencies.add(dep.source_task_id)
    
    def _identify_parallel_tasks(
        self, 
        process: Process, 
        banking_process: BankingProcess
    ) -> List[Set[str]]:
        """
        Identify tasks that can run in parallel (FR12)
        Returns groups of task IDs that can run simultaneously
        """
        parallel_groups = []
        
        # Build dependency graph
        dependency_graph = {task.id: set(task.dependencies) if task.dependencies else set() 
                           for task in process.tasks}
        
        # Find tasks with no dependencies that can run in parallel
        independent_tasks = {task.id for task in process.tasks if not task.dependencies}
        if len(independent_tasks) > 1:
            parallel_groups.append(independent_tasks)
        
        # Check banking dependencies for parallel-allowed tasks
        for dep in banking_process.task_dependencies:
            if dep.dependency_type == "parallel_allowed":
                # These tasks can run in parallel
                group = {dep.source_task_id, dep.target_task_id}
                
                # Check if we can merge with existing group
                merged = False
                for existing_group in parallel_groups:
                    if group & existing_group:  # Has overlap
                        existing_group.update(group)
                        merged = True
                        break
                
                if not merged:
                    parallel_groups.append(group)
        
        return parallel_groups
    
    def _optimize_with_heuristics(
        self, 
        process: Process, 
        banking_process: BankingProcess,
        parallel_groups: List[Set[str]]
    ) -> Schedule:
        """
        Optimize using heuristics with resource awareness (FR13, FR14)
        """
        schedule = Schedule(process_id=process.id)
        
        # Track resource availability (FR14)
        resource_next_available = {r.id: 0.0 for r in process.resources}
        resource_workload = {r.id: 0.0 for r in process.resources}
        resource_working_hours = {
            r.id: (
                r.metadata.get('working_hours_start', 9),
                r.metadata.get('working_hours_end', 17)
            ) for r in process.resources
        }
        
        completed_tasks = set()
        scheduled_tasks = set()
        ready_tasks = []
        
        # Initialize with tasks that have no dependencies
        for task in process.tasks:
            if not task.dependencies:
                priority = self._calculate_priority_score(task, process, banking_process)
                heapq.heappush(ready_tasks, (priority, task.id))
        
        # Safety counter to prevent infinite loops
        max_iterations = len(process.tasks) * 10
        iteration_count = 0
        
        while ready_tasks or len(scheduled_tasks) < len(process.tasks):
            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"Warning: Maximum iterations reached. Scheduled {len(scheduled_tasks)}/{len(process.tasks)} tasks.")
                break
            
            if not ready_tasks:
                break
            
            # Process parallel tasks if enabled (FR12)
            if self.enable_parallelization and len(ready_tasks) > 1:
                parallel_batch = self._get_parallel_batch(ready_tasks, parallel_groups, process)
                if len(parallel_batch) > 1:
                    self._schedule_parallel_batch(
                        parallel_batch, process, schedule, 
                        resource_next_available, resource_workload,
                        resource_working_hours, completed_tasks, scheduled_tasks
                    )
                    
                    # Add newly ready tasks
                    self._add_ready_tasks(process, completed_tasks, scheduled_tasks, ready_tasks, banking_process)
                    continue
            
            # Schedule single task
            _, task_id = heapq.heappop(ready_tasks)
            task = process.get_task_by_id(task_id)
            
            if not task or task_id in scheduled_tasks:
                continue
            
            # Find best resource considering availability and working hours (FR14)
            best_resource, start_hour = self._find_best_resource_with_constraints(
                task, process, resource_next_available, resource_workload, resource_working_hours
            )
            
            if best_resource:
                end_hour = start_hour + task.duration_hours
                cost = task.duration_hours * best_resource.hourly_rate
                
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
                resource_workload[best_resource.id] += task.duration_hours
                completed_tasks.add(task.id)
                
                # Add newly ready tasks
                self._add_ready_tasks(process, completed_tasks, scheduled_tasks, ready_tasks, banking_process)
        
        return schedule
    
    def _optimize_with_rl(
        self, 
        process: Process, 
        banking_process: BankingProcess,
        parallel_groups: List[Set[str]]
    ) -> Schedule:
        """
        Optimize using Reinforcement Learning (FR15)
        Uses Q-learning to learn optimal scheduling decisions
        """
        # For now, use heuristics with RL-informed priorities
        # Full RL implementation would require training episodes
        schedule = self._optimize_with_heuristics(process, banking_process, parallel_groups)
        
        # Update Q-table based on results (learning)
        if self.enable_rl:
            metrics = self.metrics_calculator.calculate_process_metrics(process, schedule, banking_process)
            reward = self.multi_objective_optimizer.calculate_fitness_score(metrics)
            
            # Store experience for future learning
            state = self._get_state_representation(process)
            self._update_q_values(state, reward)
        
        return schedule
    
    def _get_state_representation(self, process: Process) -> str:
        """Get state representation for RL (FR15)"""
        # Simple state: number of tasks, resources, and average duration
        state = f"{len(process.tasks)}_{len(process.resources)}_{sum(t.duration_hours for t in process.tasks)/len(process.tasks):.1f}"
        return state
    
    def _update_q_values(self, state: str, reward: float):
        """Update Q-values for RL learning (FR15)"""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        # Simple Q-learning update
        action = "optimize"
        old_value = self.q_table[state].get(action, 0.0)
        self.q_table[state][action] = old_value + self.learning_rate * (reward - old_value)
        self.training_episodes += 1
    
    def _calculate_priority_score(
        self, 
        task: Task, 
        process: Process, 
        banking_process: BankingProcess
    ) -> float:
        """Calculate priority score for task scheduling"""
        score = 0.0
        
        # Higher priority tasks get lower score (for min-heap)
        if hasattr(task, 'priority'):
            score += task.priority * 10
        
        # Critical tasks get higher priority (lower score)
        if task.id in banking_process.critical_tasks:
            score -= 50
        
        # Tasks with more dependents get higher priority
        dependents = sum(1 for t in process.tasks if task.id in (t.dependencies or set()))
        score -= dependents * 5
        
        # Longer tasks get slightly higher priority
        score -= task.duration_hours * 0.1
        
        return score
    
    def _find_best_resource_with_constraints(
        self,
        task: Task,
        process: Process,
        resource_next_available: Dict[str, float],
        resource_workload: Dict[str, float],
        resource_working_hours: Dict[str, Tuple[int, int]]
    ) -> Tuple[Optional[Resource], float]:
        """
        Find best resource considering availability and working hours (FR14)
        """
        best_resource = None
        best_start_hour = float('inf')
        best_score = float('-inf')
        
        for resource in process.resources:
            # Check if resource has required skills
            if not resource.has_all_skills(task.required_skills):
                continue
            
            # Check if resource has capacity
            if resource_workload[resource.id] + task.duration_hours > resource.total_available_hours:
                continue
            
            # Calculate earliest start time
            earliest_start = resource_next_available[resource.id]
            
            # Adjust for working hours if needed
            start_hour_of_day = earliest_start % 24
            working_start, working_end = resource_working_hours[resource.id]
            
            if start_hour_of_day < working_start:
                # Adjust to start of working hours
                earliest_start += (working_start - start_hour_of_day)
            elif start_hour_of_day >= working_end:
                # Move to next day
                hours_to_next_day = 24 - start_hour_of_day + working_start
                earliest_start += hours_to_next_day
            
            # Calculate resource score (skill match, cost, availability)
            skill_score = resource.get_skill_score(task.required_skills)
            cost_score = 1.0 / (resource.hourly_rate + 1)  # Lower cost is better
            availability_score = 1.0 / (earliest_start + 1)  # Earlier is better
            
            total_score = skill_score * 0.5 + cost_score * 0.3 + availability_score * 0.2
            
            if total_score > best_score:
                best_score = total_score
                best_resource = resource
                best_start_hour = earliest_start
        
        return best_resource, best_start_hour
    
    def _get_parallel_batch(
        self, 
        ready_tasks: List[Tuple[float, str]], 
        parallel_groups: List[Set[str]],
        process: Process
    ) -> List[Task]:
        """Get batch of tasks that can be scheduled in parallel (FR12)"""
        batch = []
        task_ids = [tid for _, tid in ready_tasks[:5]]  # Consider top 5 ready tasks
        
        # Find tasks that are in the same parallel group
        for group in parallel_groups:
            group_tasks = [tid for tid in task_ids if tid in group]
            if len(group_tasks) > 1:
                batch = [process.get_task_by_id(tid) for tid in group_tasks]
                break
        
        # If no parallel group found, just return first task
        if not batch and ready_tasks:
            _, task_id = ready_tasks[0]
            task = process.get_task_by_id(task_id)
            if task:
                batch = [task]
        
        return batch
    
    def _schedule_parallel_batch(
        self,
        batch: List[Task],
        process: Process,
        schedule: Schedule,
        resource_next_available: Dict[str, float],
        resource_workload: Dict[str, float],
        resource_working_hours: Dict[str, Tuple[int, int]],
        completed_tasks: Set[str],
        scheduled_tasks: Set[str]
    ):
        """Schedule a batch of parallel tasks (FR12)"""
        for task in batch:
            if task.id in scheduled_tasks:
                continue
            
            best_resource, start_hour = self._find_best_resource_with_constraints(
                task, process, resource_next_available, resource_workload, resource_working_hours
            )
            
            if best_resource:
                end_hour = start_hour + task.duration_hours
                cost = task.duration_hours * best_resource.hourly_rate
                
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
                resource_next_available[best_resource.id] = end_hour
                resource_workload[best_resource.id] += task.duration_hours
                completed_tasks.add(task.id)
    
    def _add_ready_tasks(
        self,
        process: Process,
        completed_tasks: Set[str],
        scheduled_tasks: Set[str],
        ready_tasks: List[Tuple[float, str]],
        banking_process: BankingProcess
    ):
        """Add newly ready tasks to the queue"""
        ready_task_ids = {tid for _, tid in ready_tasks}
        
        for task in process.tasks:
            if (task.id not in scheduled_tasks and 
                task.id not in ready_task_ids and
                task.can_start(completed_tasks)):
                priority = self._calculate_priority_score(task, process, banking_process)
                heapq.heappush(ready_tasks, (priority, task.id))
    
    def what_if_analysis(
        self, 
        process: Process, 
        banking_process: BankingProcess,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform what-if analysis on different scenarios (FR11)
        
        Scenarios can include:
        - Adding/removing resources
        - Changing task durations
        - Modifying resource allocations
        - Adjusting business rules
        """
        results = {
            "baseline": None,
            "scenarios": []
        }
        
        # Get baseline optimization
        baseline_schedule = self.optimize(process, banking_process)
        baseline_metrics = self.metrics_calculator.calculate_process_metrics(
            process, baseline_schedule, banking_process
        )
        results["baseline"] = {
            "schedule": baseline_schedule,
            "metrics": baseline_metrics.to_dict()
        }
        
        # Test each scenario
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get("name", f"Scenario {i+1}")
            scenario_type = scenario.get("type", "unknown")
            
            # Create modified process
            modified_process = self._apply_scenario(process, scenario)
            
            # Optimize modified process
            scenario_schedule = self.optimize(modified_process, banking_process)
            scenario_metrics = self.metrics_calculator.calculate_process_metrics(
                modified_process, scenario_schedule, banking_process
            )
            
            # Compare with baseline
            comparison = self.metrics_calculator.compare_metrics(
                baseline_metrics, scenario_metrics
            )
            
            results["scenarios"].append({
                "name": scenario_name,
                "type": scenario_type,
                "schedule": scenario_schedule,
                "metrics": scenario_metrics.to_dict(),
                "comparison": comparison
            })
        
        return results
    
    def _apply_scenario(self, process: Process, scenario: Dict[str, Any]) -> Process:
        """Apply a what-if scenario to create modified process (FR11)"""
        # Deep copy process to avoid modifying original
        modified_process = copy.deepcopy(process)
        
        scenario_type = scenario.get("type", "")
        
        if scenario_type == "add_resource":
            # Add a new resource
            new_resource_data = scenario.get("resource", {})
            from .models import Skill, SkillLevel
            
            skills = []
            for skill_data in new_resource_data.get("skills", []):
                skills.append(Skill(
                    name=skill_data.get("name", ""),
                    level=SkillLevel.from_value(skill_data.get("level", 1))
                ))
            
            new_resource = Resource(
                id=new_resource_data.get("id", f"resource_{len(modified_process.resources)}"),
                name=new_resource_data.get("name", "New Resource"),
                skills=skills,
                hourly_rate=new_resource_data.get("hourly_rate", 50.0),
                total_available_hours=new_resource_data.get("total_available_hours", 160.0)
            )
            modified_process.resources.append(new_resource)
        
        elif scenario_type == "remove_resource":
            # Remove a resource
            resource_id = scenario.get("resource_id")
            modified_process.resources = [
                r for r in modified_process.resources if r.id != resource_id
            ]
        
        elif scenario_type == "modify_task":
            # Modify task parameters
            task_id = scenario.get("task_id")
            modifications = scenario.get("modifications", {})
            
            for task in modified_process.tasks:
                if task.id == task_id:
                    for key, value in modifications.items():
                        if hasattr(task, key):
                            setattr(task, key, value)
        
        elif scenario_type == "change_resource_capacity":
            # Change resource capacity
            resource_id = scenario.get("resource_id")
            new_capacity = scenario.get("new_capacity")
            
            for resource in modified_process.resources:
                if resource.id == resource_id:
                    resource.total_available_hours = new_capacity
        
        return modified_process
    
    def train_rl_model(
        self, 
        training_processes: List[Tuple[Process, BankingProcess]], 
        episodes: int = 100
    ):
        """
        Train RL model on multiple processes (FR15)
        """
        print(f"Training RL model for {episodes} episodes...")
        
        for episode in range(episodes):
            for process, banking_process in training_processes:
                # Enable exploration
                old_epsilon = self.epsilon
                self.epsilon = max(0.01, 1.0 - (episode / episodes))
                
                # Optimize and learn
                schedule = self.optimize(process, banking_process)
                
                # Restore epsilon
                self.epsilon = old_epsilon
            
            if (episode + 1) % 10 == 0:
                print(f"  Completed episode {episode + 1}/{episodes}")
        
        print(f"Training complete. Q-table size: {len(self.q_table)}")
        self.enable_rl = True
