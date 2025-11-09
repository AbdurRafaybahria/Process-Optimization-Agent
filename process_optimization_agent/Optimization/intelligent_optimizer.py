"""
Intelligent Process Optimizer - Integrates process detection, user journey optimization,
and enhanced dependency detection for smart process optimization
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .models import Process, Task, Resource, Schedule, ScheduleEntry
from .optimizers import BaseOptimizer, ProcessOptimizer, RLBasedOptimizer
from .process_intelligence import ProcessIntelligence, ProcessType, OptimizationStrategy
from .user_journey_optimizer import UserJourneyOptimizer, UserJourneyMetrics
from .analyzers import DependencyDetector

# Domain-specific optimizers
from ..scenarios.healthcare.healthcare_optimizer import HealthcareOptimizer
from ..scenarios.manufacturing.manufacturing_optimizer import ManufacturingOptimizer
from ..scenarios.insurance.insurance_optimizer import InsuranceProcessOptimizer

@dataclass
class IntelligentOptimizationResult:
    """Result of intelligent optimization"""
    schedule: Schedule
    process_type: ProcessType
    optimization_strategy: OptimizationStrategy
    user_metrics: Optional[UserJourneyMetrics]
    admin_metrics: Dict[str, Any]
    dual_metrics: Dict[str, Any]
    recommendations: List[str]
    confidence: float
    optimizer: Optional[Any] = None  # Reference to the optimizer used

class IntelligentOptimizer:
    """
    Main intelligent optimization engine that automatically detects process type
    and applies appropriate optimization strategies
    """
    
    def __init__(self):
        self.process_intelligence = ProcessIntelligence()
        self.optimizers = {}
        self._initialize_optimizers()
    
    def _initialize_optimizers(self):
        """Initialize different optimizers for different process types"""
        # User journey optimizer for healthcare/single-user processes
        self.optimizers[OptimizationStrategy.SEQUENTIAL_USER] = UserJourneyOptimizer(
            minimize_waiting=True,
            enforce_continuity=True,
            optimize_critical_path=True
        )
        
        # Insurance workflow optimizer
        self.optimizers[OptimizationStrategy.INSURANCE_WORKFLOW] = InsuranceProcessOptimizer()
        
        # Standard optimizer for manufacturing (parallel production)
        self.optimizers[OptimizationStrategy.PARALLEL_PRODUCTION] = ProcessOptimizer(
            optimization_strategy="time"  # Focus on minimizing time through parallelization
        )
        
        # Balanced optimizer for banking (conditional approval)
        self.optimizers[OptimizationStrategy.CONDITIONAL_APPROVAL] = ProcessOptimizer(
            optimization_strategy="balanced"  # Balance time and cost
        )
        
        # Mixed optimizer for academic processes
        self.optimizers[OptimizationStrategy.MIXED_ACADEMIC] = ProcessOptimizer(
            optimization_strategy="cost"  # Often cost-sensitive
        )
        
        # Default balanced optimizer
        self.optimizers[OptimizationStrategy.BALANCED] = ProcessOptimizer(
            optimization_strategy="balanced"
        )
    
    def optimize(self, process: Process, 
                 force_type: Optional[ProcessType] = None,
                 dual_optimization: bool = True) -> IntelligentOptimizationResult:
        """
        Intelligently optimize a process
        
        Args:
            process: The process to optimize
            force_type: Force a specific process type (optional)
            dual_optimization: Whether to perform dual optimization (user + admin)
            
        Returns:
            IntelligentOptimizationResult with optimized schedule and metrics
        """
        # Step 1: Detect process type
        if force_type:
            classification = self.process_intelligence.detect_process_type(process)
            classification.process_type = force_type
            classification.optimization_strategy = self.process_intelligence._determine_strategy(force_type)
        else:
            classification = self.process_intelligence.detect_process_type(process)
        
        print(f"\n=== Process Type Detection ===")
        print(f"Detected Type: {classification.process_type.value}")
        print(f"Confidence: {classification.confidence:.2%}")
        print(f"Strategy: {classification.optimization_strategy.value}")
        print(f"Reasoning: {', '.join(classification.reasoning[:3])}")
        
        # Step 2: Apply enhanced dependency detection based on process type
        self._enhance_dependencies(process, classification.process_type)
        
        # Step 3: Get optimization parameters
        optimization_params = self.process_intelligence.get_optimization_parameters(classification)
        
        # Step 4: Select appropriate optimizer
        optimizer = self.optimizers.get(
            classification.optimization_strategy,
            self.optimizers[OptimizationStrategy.BALANCED]
        )
        
        # Step 5: Perform optimization
        print(f"\n=== Optimizing Process ===")
        print(f"Using optimizer: {optimizer.__class__.__name__}")
        
        schedule = optimizer.optimize(process)
        
        # Step 6: Calculate metrics based on process type
        user_metrics = None
        admin_metrics = {}
        dual_metrics = {}
        recommendations = []
        
        if classification.process_type == ProcessType.HEALTHCARE:
            # Calculate user journey metrics
            if isinstance(optimizer, UserJourneyOptimizer):
                user_metrics = optimizer._calculate_metrics(schedule, process)
                recommendations = optimizer.suggest_improvements(schedule, process)
            
            # Calculate administrative metrics
            admin_metrics = self._calculate_admin_metrics(schedule, process)
            
            # Calculate dual optimization metrics
            if dual_optimization:
                dual_metrics = self._calculate_dual_metrics(
                    user_metrics, 
                    admin_metrics,
                    classification.process_type
                )
        
        elif classification.process_type == ProcessType.MANUFACTURING:
            # Focus on production metrics
            admin_metrics = self._calculate_production_metrics(schedule, process)
            recommendations = self._suggest_production_improvements(schedule, process)
        
        elif classification.process_type == ProcessType.INSURANCE:
            # Focus on insurance workflow metrics
            if isinstance(optimizer, InsuranceProcessOptimizer):
                # Insurance optimizer returns InsuranceOptimizationResult, not Schedule
                # We need to handle this differently
                insurance_result = schedule  # The optimize() already returned the result
                
                # Use the schedule from the insurance result
                schedule = insurance_result.optimized_schedule
                
                # Extract metrics from insurance result
                admin_metrics = {
                    'scenario_type': insurance_result.scenario_type.value,
                    'current_time': insurance_result.current_metrics.total_process_time,
                    'optimized_time': insurance_result.optimized_metrics.total_process_time,
                    'time_savings': insurance_result.optimized_metrics.time_savings_percent,
                    'time_savings_minutes': insurance_result.optimized_metrics.time_savings_minutes,
                    'current_cost': insurance_result.current_metrics.total_labor_cost,
                    'optimized_cost': insurance_result.optimized_metrics.total_labor_cost,
                    'bottlenecks': len(insurance_result.bottlenecks),
                    'parallelization_opportunities': len(insurance_result.parallelization_opportunities),
                    'resource_utilization': insurance_result.current_metrics.resource_utilization
                }
                recommendations = [rec.title + ": " + rec.description for rec in insurance_result.recommendations]
                
                # Store the full insurance result for later use
                schedule.optimization_metrics['insurance_result'] = insurance_result
            else:
                admin_metrics = self._calculate_admin_metrics(schedule, process)
        
        elif classification.process_type == ProcessType.BANKING:
            # Focus on approval workflow metrics
            admin_metrics = self._calculate_approval_metrics(schedule, process)
            recommendations = self._suggest_approval_improvements(schedule, process)
        
        else:
            # Generic metrics
            admin_metrics = self._calculate_admin_metrics(schedule, process)
        
        # Step 7: Create result
        result = IntelligentOptimizationResult(
            schedule=schedule,
            process_type=classification.process_type,
            optimization_strategy=classification.optimization_strategy,
            user_metrics=user_metrics,
            admin_metrics=admin_metrics,
            dual_metrics=dual_metrics,
            recommendations=recommendations,
            confidence=classification.confidence,
            optimizer=optimizer  # Store optimizer reference for accessing journey_steps
        )
        
        # Print summary
        self._print_optimization_summary(result)
        
        return result
    
    def _enhance_dependencies(self, process: Process, process_type: ProcessType):
        """Enhance task dependencies based on process type"""
        detector = DependencyDetector(
            use_nlp=True,
            process_type=process_type.value
        )
        
        if process_type == ProcessType.HEALTHCARE:
            # Detect sequential dependencies for healthcare
            sequential_deps = detector.detect_sequential_dependencies(process.tasks)
            
            # Apply detected dependencies
            for task_id, deps in sequential_deps.items():
                task = process.get_task_by_id(task_id)
                if task:
                    task.dependencies.update(deps)
            
            print(f"Applied {len(sequential_deps)} sequential dependencies")
        
        elif process_type == ProcessType.MANUFACTURING:
            # Detect parallel opportunities
            parallel_groups = detector.detect_parallel_opportunities(process.tasks)
            
            # Mark tasks that can run in parallel
            for group in parallel_groups:
                for task_id in group:
                    task = process.get_task_by_id(task_id)
                    if task and 'parallel_group' not in task.metadata:
                        task.metadata['parallel_group'] = group
            
            print(f"Identified {len(parallel_groups)} parallel task groups")
        
        # Detect critical sequence for all types
        critical_sequence = detector.detect_critical_sequence(process.tasks)
        if critical_sequence:
            print(f"Critical sequence: {' -> '.join(critical_sequence[:5])}...")
    
    def _calculate_admin_metrics(self, schedule: Schedule, process: Process) -> Dict[str, Any]:
        """Calculate administrative metrics"""
        if not schedule.entries:
            return {}
        
        total_cost = 0
        resource_utilization = {}
        
        for entry in schedule.entries:
            task = process.get_task_by_id(entry.task_id)
            resource = process.get_resource_by_id(entry.resource_id)
            
            if task and resource:
                cost = task.duration_hours * resource.hourly_rate
                total_cost += cost
                
                if resource.id not in resource_utilization:
                    resource_utilization[resource.id] = 0
                resource_utilization[resource.id] += task.duration_hours
        
        # Calculate utilization percentages
        avg_utilization = 0
        if resource_utilization:
            total_available = sum(r.total_available_hours for r in process.resources)
            total_used = sum(resource_utilization.values())
            avg_utilization = (total_used / total_available * 100) if total_available > 0 else 0
        
        return {
            'total_cost': total_cost,
            'avg_resource_utilization': avg_utilization,
            'resource_hours': resource_utilization,
            'total_duration': max(e.end_time for e in schedule.entries) if schedule.entries else 0
        }
    
    def _calculate_production_metrics(self, schedule: Schedule, process: Process) -> Dict[str, Any]:
        """Calculate production-specific metrics focused on time and cost efficiency"""
        metrics = self._calculate_admin_metrics(schedule, process)
        
        # Add production-specific metrics
        parallel_tasks = 0
        max_parallel = 0
        idle_time = 0
        
        # Analyze parallelization using hour-based values
        time_slots = {}
        for entry in schedule.entries:
            start = int(entry.start_hour)
            end = int(entry.end_hour) + 1
            for t in range(start, end):
                if t not in time_slots:
                    time_slots[t] = 0
                time_slots[t] += 1
                max_parallel = max(max_parallel, time_slots[t])
        
        if time_slots:
            parallel_tasks = sum(1 for count in time_slots.values() if count > 1)
            # Calculate idle time (time slots with less than max resources)
            idle_time = sum(max_parallel - count for count in time_slots.values())
        
        # Calculate cycle time (makespan)
        if schedule.entries:
            cycle_time = max(e.end_hour for e in schedule.entries) - min(e.start_hour for e in schedule.entries)
        else:
            cycle_time = 0
        
        # Calculate throughput (tasks per hour)
        throughput = len(schedule.entries) / cycle_time if cycle_time > 0 else 0
        
        # Calculate cost per task
        cost_per_task = metrics['total_cost'] / len(schedule.entries) if schedule.entries else 0
        
        # Calculate time efficiency (actual work time vs total time)
        total_work_time = sum(e.end_hour - e.start_hour for e in schedule.entries)
        time_efficiency = (total_work_time / (cycle_time * len(process.resources))) * 100 if cycle_time > 0 and process.resources else 0
        
        metrics.update({
            'cycle_time': cycle_time,  # Total time from start to finish (makespan)
            'max_parallel_tasks': max_parallel,
            'parallel_time_percentage': (parallel_tasks / len(time_slots) * 100) if time_slots else 0,
            'throughput': throughput,  # Tasks completed per hour
            'cost_per_task': cost_per_task,
            'time_efficiency': time_efficiency,  # Percentage of time resources are actually working
            'idle_time_hours': idle_time,
            'total_tasks': len(schedule.entries)
        })
        
        return metrics
    
    def _calculate_approval_metrics(self, schedule: Schedule, process: Process) -> Dict[str, Any]:
        """Calculate approval workflow metrics"""
        metrics = self._calculate_admin_metrics(schedule, process)
        
        # Identify approval tasks
        approval_tasks = []
        for entry in schedule.entries:
            task = process.get_task_by_id(entry.task_id)
            if task and any(keyword in task.name.lower() 
                          for keyword in ['approval', 'review', 'verify']):
                approval_tasks.append(entry)
        
        if approval_tasks:
            # Calculate using hours instead of datetime objects
            total_approval_hours = sum((e.end_hour - e.start_hour) for e in approval_tasks)
            avg_approval_time = total_approval_hours / len(approval_tasks)
            metrics['avg_approval_time'] = avg_approval_time
            metrics['total_approvals'] = len(approval_tasks)
        
        return metrics
    
    def _calculate_dual_metrics(self, 
                               user_metrics: Optional[UserJourneyMetrics],
                               admin_metrics: Dict[str, Any],
                               process_type: ProcessType) -> Dict[str, Any]:
        """Calculate dual optimization metrics balancing user and admin perspectives"""
        if not user_metrics:
            return admin_metrics
        
        # Calculate balance score
        user_efficiency = user_metrics.efficiency_ratio
        cost_efficiency = 1.0 - (admin_metrics.get('total_cost', 0) / 10000)  # Normalized
        
        # Weight based on process type
        if process_type == ProcessType.HEALTHCARE:
            # User experience is more important
            user_weight = 0.7
            admin_weight = 0.3
        else:
            # Balanced weights
            user_weight = 0.5
            admin_weight = 0.5
        
        balance_score = (user_efficiency * user_weight + cost_efficiency * admin_weight)
        
        return {
            'user_efficiency': user_efficiency,
            'cost_efficiency': cost_efficiency,
            'balance_score': balance_score,
            'user_journey_time': user_metrics.total_journey_time,
            'user_waiting_time': user_metrics.waiting_time,
            'total_cost': admin_metrics.get('total_cost', 0),
            'resource_utilization': admin_metrics.get('avg_resource_utilization', 0)
        }
    
    def _suggest_production_improvements(self, schedule: Schedule, process: Process) -> List[str]:
        """Suggest improvements for production processes focused on time and cost reduction"""
        suggestions = []
        
        if not schedule.entries:
            return suggestions
        
        # Calculate metrics for analysis
        cycle_time = max(e.end_hour for e in schedule.entries) - min(e.start_hour for e in schedule.entries)
        
        # Check for low parallelization
        time_slots = {}
        for entry in schedule.entries:
            start = int(entry.start_hour)
            end = int(entry.end_hour) + 1
            for t in range(start, end):
                time_slots[t] = time_slots.get(t, 0) + 1
        
        max_parallel = max(time_slots.values()) if time_slots else 0
        avg_parallel = sum(time_slots.values()) / len(time_slots) if time_slots else 0
        
        if avg_parallel < max_parallel * 0.5:
            suggestions.append(
                f"Low parallelization detected (avg: {avg_parallel:.1f} tasks, max: {max_parallel}). "
                f"Restructure dependencies to run more tasks simultaneously and reduce cycle time by up to {((max_parallel - avg_parallel) / max_parallel * 100):.0f}%."
            )
        
        # Check for resource bottlenecks
        resource_loads = {}
        for entry in schedule.entries:
            if entry.resource_id not in resource_loads:
                resource_loads[entry.resource_id] = 0
            resource_loads[entry.resource_id] += entry.end_hour - entry.start_hour
        
        if resource_loads:
            max_load = max(resource_loads.values())
            avg_load = sum(resource_loads.values()) / len(resource_loads)
            
            if max_load > avg_load * 1.5:
                bottleneck_resource = max(resource_loads, key=resource_loads.get)
                resource = process.get_resource_by_id(bottleneck_resource)
                potential_savings = (max_load - avg_load) * resource.hourly_rate if resource else 0
                
                suggestions.append(
                    f"Resource bottleneck detected on '{resource.name if resource else bottleneck_resource}' "
                    f"({max_load:.1f}h vs avg {avg_load:.1f}h). Adding one more resource could save "
                    f"${potential_savings:.2f} and reduce cycle time by {((max_load - avg_load) / cycle_time * 100):.0f}%."
                )
        
        # Check for idle time
        total_work_time = sum(e.end_hour - e.start_hour for e in schedule.entries)
        total_available_time = cycle_time * len(process.resources)
        idle_percentage = ((total_available_time - total_work_time) / total_available_time * 100) if total_available_time > 0 else 0
        
        if idle_percentage > 30:
            suggestions.append(
                f"High idle time detected ({idle_percentage:.0f}% of available time). "
                f"Consider reducing resources or increasing workload to improve cost efficiency."
            )
        
        # Check for long tasks that could be split
        long_tasks = [e for e in schedule.entries if (e.end_hour - e.start_hour) > cycle_time * 0.3]
        if long_tasks:
            for entry in long_tasks[:2]:  # Show top 2
                task = process.get_task_by_id(entry.task_id)
                if task:
                    suggestions.append(
                        f"Task '{task.name}' takes {(entry.end_hour - entry.start_hour):.1f}h "
                        f"({((entry.end_hour - entry.start_hour) / cycle_time * 100):.0f}% of cycle time). "
                        f"Consider breaking it into smaller subtasks for better parallelization."
                    )
        
        return suggestions
    
    def _suggest_approval_improvements(self, schedule: Schedule, process: Process) -> List[str]:
        """Suggest improvements for approval workflows"""
        suggestions = []
        
        # Check for sequential approvals that could be parallel
        approval_entries = []
        for entry in schedule.entries:
            task = process.get_task_by_id(entry.task_id)
            if task and 'approval' in task.name.lower():
                approval_entries.append(entry)
        
        if len(approval_entries) > 1:
            # Check if they're sequential
            sequential = True
            for i in range(len(approval_entries) - 1):
                if approval_entries[i].end_time > approval_entries[i+1].start_time:
                    sequential = False
                    break
            
            if sequential:
                suggestions.append(
                    "Sequential approvals detected. Consider parallelizing "
                    "independent approval tasks to reduce cycle time."
                )
        
        return suggestions
    
    def _print_optimization_summary(self, result: IntelligentOptimizationResult):
        """Print a summary of the optimization results"""
        print(f"\n=== Optimization Summary ===")
        print(f"Process Type: {result.process_type.value}")
        print(f"Strategy: {result.optimization_strategy.value}")
        print(f"Confidence: {result.confidence:.2%}")
        
        if result.user_metrics:
            print(f"\nUser Metrics:")
            print(f"  Total Journey Time: {result.user_metrics.total_journey_time:.1f} hours")
            print(f"  Waiting Time: {result.user_metrics.waiting_time:.1f} hours")
            print(f"  Efficiency: {result.user_metrics.efficiency_ratio:.2%}")
        
        if result.admin_metrics:
            print(f"\nAdmin Metrics:")
            # Check if this is insurance-specific metrics
            if 'scenario_type' in result.admin_metrics:
                print(f"  Insurance Scenario: {result.admin_metrics.get('scenario_type', 'unknown')}")
                print(f"  Current Process Time: {result.admin_metrics.get('current_time', 0):.1f} minutes")
                print(f"  Optimized Process Time: {result.admin_metrics.get('optimized_time', 0):.1f} minutes")
                print(f"  Time Savings: {result.admin_metrics.get('time_savings', 0):.1f}% ({result.admin_metrics.get('time_savings_minutes', 0):.1f} minutes)")
                print(f"  Current Cost: ${result.admin_metrics.get('current_cost', 0):.2f}")
                print(f"  Optimized Cost: ${result.admin_metrics.get('optimized_cost', 0):.2f}")
                print(f"  Bottlenecks Identified: {result.admin_metrics.get('bottlenecks', 0)}")
                print(f"  Parallelization Opportunities: {result.admin_metrics.get('parallelization_opportunities', 0)}")
            else:
                print(f"  Total Cost: ${result.admin_metrics.get('total_cost', 0):.2f}")
                print(f"  Resource Utilization: {result.admin_metrics.get('avg_resource_utilization', 0):.1f}%")
        
        if result.dual_metrics:
            print(f"\nDual Optimization:")
            print(f"  Balance Score: {result.dual_metrics.get('balance_score', 0):.2%}")
        
        if result.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(result.recommendations[:3], 1):
                print(f"  {i}. {rec}")
