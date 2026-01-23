"""
Parallel Gateway Detection Module
Analyzes process workflows to detect opportunities for parallel execution
and suggests gateway configurations compatible with CMS structure
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import json

from .gateway_base import GatewayDetectorBase, GatewayBranch, GatewaySuggestion


class ParallelGatewayDetector:
    """
    Detects opportunities for parallel gateway execution in process workflows
    """
    
    def __init__(self, min_confidence: float = 0.7):
        """
        Initialize the parallel gateway detector
        
        Args:
            min_confidence: Minimum confidence score to suggest a gateway (0.0-1.0)
        """
        self.min_confidence = min_confidence
    
    def analyze_process(self, cms_data: Dict[str, Any]) -> List[GatewaySuggestion]:
        """
        Analyze a CMS process and detect parallel gateway opportunities
        
        Args:
            cms_data: Process data (can include task_assignments with start_time for accurate detection)
            
        Returns:
            List of gateway suggestions
        """
        suggestions = []
        used_parallel_tasks = set()  # Track tasks already used in parallel gateways
        
        # CHECK FOR task_assignments WITH start_time (from optimization)
        # This is the most accurate way to detect parallel execution
        task_assignments = cms_data.get('task_assignments', [])
        
        if task_assignments:
            print(f"[PARALLEL-DEBUG] Found {len(task_assignments)} task assignments with timing data")
            
            # Group tasks by start_time
            tasks_by_start_time = {}
            for task in task_assignments:
                start_time = task.get('start_time', -1)
                if start_time not in tasks_by_start_time:
                    tasks_by_start_time[start_time] = []
                tasks_by_start_time[start_time].append(task)
            
            print(f"[PARALLEL-DEBUG] Tasks grouped by start time: {len(tasks_by_start_time)} groups")
            
            # Debug: Show all start time groups
            for start_time in sorted(tasks_by_start_time.keys()):
                task_count = len(tasks_by_start_time[start_time])
                print(f"[PARALLEL-DEBUG]   Start time {start_time}h: {task_count} task(s)")
                if task_count < 2:
                    for task in tasks_by_start_time[start_time]:
                        print(f"[PARALLEL-DEBUG]     - Task {task.get('task_id')}: {task.get('task_name')}")
            
            # Find groups with multiple tasks (parallel execution)
            for start_time, tasks_at_time in sorted(tasks_by_start_time.items()):
                if len(tasks_at_time) >= 2:
                    print(f"[PARALLEL-DEBUG] Found {len(tasks_at_time)} tasks starting at t={start_time}h")
                    for task in tasks_at_time:
                        print(f"[PARALLEL-DEBUG]   - Task {task.get('task_id')}: {task.get('task_name')}")
                    
                    # Filter out tasks already used in previous parallel gateways
                    # Convert task IDs to int for comparison (handle string/int mismatch and sub-task IDs like "720_1")
                    available_tasks = []
                    for task in tasks_at_time:
                        task_id = task.get('task_id')
                        # Normalize to int for comparison - handle sub-task IDs like "720_1"
                        if isinstance(task_id, str):
                            # Extract base task ID (before underscore if present)
                            base_id_str = task_id.split('_')[0]
                            task_id_normalized = int(base_id_str) if base_id_str.isdigit() else task_id
                        else:
                            task_id_normalized = task_id
                        
                        if task_id_normalized not in used_parallel_tasks:
                            available_tasks.append(task)
                        else:
                            print(f"[PARALLEL-DEBUG]   ⊗ Task {task_id} already used in previous gateway")
                    
                    if len(available_tasks) < len(tasks_at_time):
                        excluded_count = len(tasks_at_time) - len(available_tasks)
                        print(f"[PARALLEL-DEBUG] ⚠️ Excluded {excluded_count} task(s) already used in previous parallel gateways")
                    
                    # SKIP if fewer than 2 available tasks remain
                    if len(available_tasks) < 2:
                        print(f"[PARALLEL-DEBUG] ⚠️ Skipping gateway at t={start_time}h - only {len(available_tasks)} available task(s) after filtering")
                        continue
                    
                    # Create parallel gateway suggestion
                    if start_time == 0:
                        # Gateway at the start of the process
                        suggestion = self._create_start_gateway_from_assignments(available_tasks, cms_data)
                    else:
                        # Gateway after a specific task
                        # Find the task that completed just before this start_time
                        predecessor = self._find_predecessor_task(start_time, task_assignments)
                        
                        # Check if predecessor is part of the same parallel group
                        # (this prevents creating duplicate gateways for tasks that started together)
                        if predecessor:
                            pred_id = predecessor.get('task_id')
                            # Handle sub-task IDs like "720_1"
                            if isinstance(pred_id, str):
                                base_id_str = pred_id.split('_')[0]
                                pred_id_normalized = int(base_id_str) if base_id_str.isdigit() else pred_id
                            else:
                                pred_id_normalized = pred_id
                            pred_start_time = predecessor.get('start_time', -1)
                            
                            # Check if any of the current parallel tasks started at the same time as predecessor
                            parallel_with_pred = any(
                                task.get('start_time') == pred_start_time 
                                for task in available_tasks
                            )
                            
                            if parallel_with_pred:
                                print(f"[PARALLEL-DEBUG] ⚠️ Skipping redundant gateway - predecessor task {pred_id_normalized} is part of same parallel group")
                                continue
                            
                            suggestion = self._create_gateway_from_assignments(
                                predecessor, available_tasks, cms_data
                            )
                        else:
                            suggestion = None
                    
                    if suggestion and suggestion.confidence_score >= self.min_confidence:
                        suggestions.append(suggestion)
                        # Mark all tasks in this gateway as used (normalize to int, handle sub-task IDs)
                        for task in available_tasks:
                            task_id = task.get('task_id')
                            if task_id:
                                # Handle sub-task IDs like "720_1"
                                if isinstance(task_id, str):
                                    base_id_str = task_id.split('_')[0]
                                    task_id_normalized = int(base_id_str) if base_id_str.isdigit() else task_id
                                else:
                                    task_id_normalized = task_id
                                used_parallel_tasks.add(task_id_normalized)
                        print(f"[PARALLEL-DEBUG] ✅ Added parallel gateway at t={start_time}h with {len(available_tasks)} tasks")
                        print(f"[PARALLEL-DEBUG] Used parallel tasks so far: {used_parallel_tasks}")
            
            print(f"[PARALLEL-DEBUG] Total parallel gateway suggestions: {len(suggestions)}")
            return suggestions
        
        # FALLBACK: Use process_task with order field (less accurate)
        print("[PARALLEL-DEBUG] No task_assignments found, using process_task with order field")
        process_tasks = cms_data.get('process_task', [])
        if not process_tasks:
            print("[PARALLEL-DEBUG] No process_task found in CMS data")
            return suggestions
        
        print(f"[PARALLEL-DEBUG] Found {len(process_tasks)} process tasks")
        
        # Sort by order
        process_tasks_sorted = sorted(process_tasks, key=lambda x: x.get('order', 0))
        
        # CHECK FOR PARALLEL TASKS AT START (from Start Event)
        first_order = process_tasks_sorted[0].get('order', 0)
        start_tasks = [pt for pt in process_tasks_sorted if pt.get('order', 0) == first_order]
        
        print(f"[PARALLEL-DEBUG] First order: {first_order}, Tasks at start: {len(start_tasks)}")
        for task in start_tasks:
            task_data = task.get('task', {})
            print(f"[PARALLEL-DEBUG]   - Task {task_data.get('task_id')}: {task_data.get('task_name')}")
        
        if len(start_tasks) >= 2:
            print(f"[PARALLEL-DEBUG] Creating start gateway suggestion for {len(start_tasks)} parallel tasks")
            suggestion = self._create_start_gateway_suggestion(start_tasks, cms_data)
            if suggestion:
                print(f"[PARALLEL-DEBUG] Start gateway confidence: {suggestion.confidence_score}")
                if suggestion.confidence_score >= self.min_confidence:
                    suggestions.append(suggestion)
                    print(f"[PARALLEL-DEBUG] ✅ Added start gateway suggestion")
                else:
                    print(f"[PARALLEL-DEBUG] ❌ Start gateway confidence too low: {suggestion.confidence_score} < {self.min_confidence}")
            else:
                print("[PARALLEL-DEBUG] ❌ Failed to create start gateway suggestion")
        
        # Analyze each task to find parallel opportunities
        for i, current_task_wrapper in enumerate(process_tasks_sorted):
            current_task = current_task_wrapper.get('task', {})
            current_task_id = current_task.get('task_id')
            
            if not current_task_id:
                continue
            
            # Look ahead to find tasks that could run in parallel
            parallel_candidates = self._find_parallel_candidates(
                current_task_wrapper,
                process_tasks_sorted[i+1:],
                cms_data
            )
            
            # If we found 2 or more independent tasks, suggest a gateway
            if len(parallel_candidates) >= 2:
                suggestion = self._create_gateway_suggestion(
                    current_task_wrapper,
                    parallel_candidates,
                    cms_data
                )
                
                if suggestion and suggestion.confidence_score >= self.min_confidence:
                    suggestions.append(suggestion)
        
        print(f"[PARALLEL-DEBUG] Total parallel gateway suggestions: {len(suggestions)}")
        return suggestions
    
    def _find_parallel_candidates(
        self, 
        current_task_wrapper: Dict[str, Any],
        remaining_tasks: List[Dict[str, Any]],
        cms_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find tasks that can potentially run in parallel after the current task
        
        Args:
            current_task_wrapper: The current task being analyzed
            remaining_tasks: Remaining tasks in the process
            cms_data: Full process data
            
        Returns:
            List of task wrappers that can run in parallel
        """
        if not remaining_tasks:
            return []
        
        candidates = []
        current_order = current_task_wrapper.get('order', 0)
        
        # Group consecutive tasks with same implied predecessor
        # In CMS, tasks right after each other might be independent if they:
        # 1. Have consecutive order numbers (order, order+1, order+2, etc.)
        # 2. Use different job roles
        # 3. Don't have explicit dependencies in their descriptions
        
        # Start with tasks immediately following the current one
        next_order = current_order + 1
        for task_wrapper in remaining_tasks:
            task_order = task_wrapper.get('order', 0)
            
            # Only consider tasks at the next order level
            if task_order == next_order:
                candidates.append(task_wrapper)
            elif task_order > next_order:
                # Stop looking once we pass the next order level
                break
        
        # If we didn't find multiple tasks at the same order, 
        # look for tasks at consecutive orders that might be independent
        if len(candidates) < 2:
            candidates = []
            for task_wrapper in remaining_tasks[:3]:  # Look at next 3 tasks max
                task = task_wrapper.get('task', {})
                
                # Check if this task might be independent
                if self._could_be_independent(current_task_wrapper, task_wrapper, cms_data):
                    candidates.append(task_wrapper)
                else:
                    # Stop if we hit a dependent task
                    break
        
        # Filter candidates to ensure they're truly independent
        independent_candidates = []
        for candidate in candidates:
            if self._check_independence(candidate, candidates, cms_data):
                independent_candidates.append(candidate)
        
        return independent_candidates
    
    def _could_be_independent(
        self,
        task1_wrapper: Dict[str, Any],
        task2_wrapper: Dict[str, Any],
        cms_data: Dict[str, Any]
    ) -> bool:
        """
        Check if two tasks could potentially be independent
        
        Args:
            task1_wrapper: First task
            task2_wrapper: Second task
            cms_data: Full process data
            
        Returns:
            True if tasks might be independent
        """
        task1 = task1_wrapper.get('task', {})
        task2 = task2_wrapper.get('task', {})
        
        # Check 1: Different job roles (strong indicator of independence)
        jobs1 = {jt['job_id'] for jt in task1.get('jobTasks', [])}
        jobs2 = {jt['job_id'] for jt in task2.get('jobTasks', [])}
        
        if jobs1 and jobs2 and not jobs1.intersection(jobs2):
            # Different jobs - likely independent
            return True
        
        # Check 2: Look for dependency keywords in task names/descriptions
        task1_name = task1.get('task_name', '').lower()
        task2_name = task2.get('task_name', '').lower()
        
        # If task2 explicitly mentions task1, they're likely dependent
        dependency_indicators = ['after', 'following', 'based on', 'using', 'from']
        for indicator in dependency_indicators:
            if indicator in task2_name and any(word in task2_name for word in task1_name.split()):
                return False
        
        return True
    
    def _check_independence(
        self,
        task_wrapper: Dict[str, Any],
        all_candidates: List[Dict[str, Any]],
        cms_data: Dict[str, Any]
    ) -> bool:
        """
        Verify that a task is independent from other candidates
        
        Args:
            task_wrapper: Task to check
            all_candidates: All parallel candidate tasks
            cms_data: Full process data
            
        Returns:
            True if task is independent from all other candidates
        """
        task = task_wrapper.get('task', {})
        task_id = task.get('task_id')
        task_jobs = {jt['job_id'] for jt in task.get('jobTasks', [])}
        
        # NEW: Check if this task depends on any other candidate tasks
        # Look through ALL candidates to see if there's a dependency
        for other_wrapper in all_candidates:
            if other_wrapper == task_wrapper:
                continue
            
            other_task = other_wrapper.get('task', {})
            other_task_id = other_task.get('task_id')
            
            # Check if task depends on other_task through gateway structure
            if self._has_dependency(task_id, other_task_id, cms_data):
                return False  # Task depends on another candidate - not independent!
            
            # Check if other_task depends on this task
            if self._has_dependency(other_task_id, task_id, cms_data):
                return False  # Another candidate depends on this task - not independent!
            
            other_jobs = {jt['job_id'] for jt in other_task.get('jobTasks', [])}
            
            # If tasks share jobs, they might conflict
            if task_jobs.intersection(other_jobs):
                # Could still be independent if resources are sufficient
                # For now, we'll allow it but note it in confidence score
                pass
        
        return True
    
    def _has_dependency(
        self,
        task_id: int,
        depends_on_task_id: int,
        cms_data: Dict[str, Any]
    ) -> bool:
        """
        Check if task_id depends on depends_on_task_id
        
        This checks existing gateway structures to find dependencies
        
        Args:
            task_id: The task to check
            depends_on_task_id: The potential dependency
            cms_data: Full process data
            
        Returns:
            True if task_id depends on depends_on_task_id
        """
        # Check if there's an existing gateway that shows this dependency
        gateways = cms_data.get('gateways', [])
        
        for gateway in gateways:
            branches = gateway.get('branches', [])
            for branch in branches:
                # Check if branch shows: "after depends_on_task_id, execute task_id"
                if branch.get('end_task_id') == depends_on_task_id:
                    # This branch ends after depends_on_task_id
                    # Check if it leads to task_id
                    if branch.get('target_task_id') == task_id:
                        return True
        
        # Also check task order - if depends_on_task comes significantly before task_id
        # with a large gap, there might be an implicit dependency
        all_tasks = cms_data.get('process_task', [])
        task_order = None
        depends_on_order = None
        
        for task_wrapper in all_tasks:
            if task_wrapper.get('task', {}).get('task_id') == task_id:
                task_order = task_wrapper.get('order')
            if task_wrapper.get('task', {}).get('task_id') == depends_on_task_id:
                depends_on_order = task_wrapper.get('order')
        
        # If depends_on_task comes right before task_id, there might be dependency
        if task_order and depends_on_order:
            # For now, we'll be conservative: if task comes right after depends_on_task (order+1)
            # we'll assume possible dependency
            if task_order == depends_on_order + 1:
                # Check task names for dependency hints
                task_name = None
                depends_on_name = None
                for tw in all_tasks:
                    if tw.get('task', {}).get('task_id') == task_id:
                        task_name = tw.get('task', {}).get('task_name', '').lower()
                    if tw.get('task', {}).get('task_id') == depends_on_task_id:
                        depends_on_name = tw.get('task', {}).get('task_name', '').lower()
                
                if task_name and depends_on_name:
                    # Look for keywords suggesting dependency
                    if any(keyword in task_name for keyword in ['approval', 'communication', 'final']):
                        # Tasks with these keywords often depend on documentation/processing
                        if any(keyword in depends_on_name for keyword in ['documentation', 'processing', 'preparation']):
                            return True
        
        return False
    
    def _find_predecessor_task(self, start_time: float, task_assignments: List[Dict]) -> Optional[Dict]:
        """Find the task that completed just before this start_time"""
        predecessors = [t for t in task_assignments if t.get('end_time', 0) == start_time]
        if predecessors:
            # Return the one with highest task_id (last one to complete at that time)
            return max(predecessors, key=lambda t: t.get('task_id', 0))
        return None
    
    def _create_start_gateway_from_assignments(
        self,
        start_tasks: List[Dict[str, Any]],
        cms_data: Dict[str, Any]
    ) -> Optional[GatewaySuggestion]:
        """
        Create a parallel gateway suggestion for tasks starting at the beginning (using task_assignments)
        """
        branches = []
        total_time = 0
        max_time = 0
        
        for idx, task in enumerate(start_tasks, start=1):
            task_id = task.get('task_id')
            task_name = task.get('task_name', f'Task {task_id}')
            duration_minutes = task.get('duration_minutes', 0)
            
            total_time += duration_minutes
            max_time = max(max_time, duration_minutes)
            
            branch = GatewayBranch(
                branch_id=str(idx),
                target_task_id=task_id,
                task_name=task_name,
                condition=None,
                condition_expression=None,
                is_default=False,
                probability=1.0 / len(start_tasks),
                end_task_id=None,
                end_event_name=None,
                description=f"Execute {task_name} ({int(duration_minutes)} min)",
                assigned_jobs=[],
                duration_minutes=float(duration_minutes)
            )
            branches.append(branch)
        
        # Calculate time savings
        time_saved = total_time - max_time
        efficiency_gain = (time_saved / total_time * 100) if total_time > 0 else 0
        
        # High confidence for start tasks running in parallel
        confidence = 0.95
        
        # Build justification
        justification = {
            "why_parallel": f"{len(start_tasks)} tasks start simultaneously at the beginning of the process",
            "independence_factors": [
                "Tasks start at the same time (t=0h)",
                "Use different resources",
                "No dependencies between these initial tasks"
            ],
            "resource_availability": "Different resources enable simultaneous execution"
        }
        
        # Build benefits
        benefits = {
            "time_saved_minutes": time_saved,
            "before_duration_minutes": total_time,
            "after_duration_minutes": max_time,
            "efficiency_gain_percent": round(efficiency_gain, 2),
            "critical_path": f"{branches[0].task_name} ({int(max_time)} minutes)" if branches else "Unknown",
            "resource_utilization": f"{len(branches)} resources working simultaneously from the start"
        }
        
        # Build implementation notes
        implementation_notes = {
            "gateway_type": "PARALLEL (AND)",
            "placement": "After Start Event",
            "behavior": "All branches execute simultaneously",
            "synchronization": "Process waits for all branches to complete before continuing",
            "bpmn_symbol": "Diamond with + symbol",
            "convergence_required": True,
            "convergence_point": "Before next sequential task"
        }
        
        return GatewaySuggestion(
            suggestion_id=0,
            gateway_type="PARALLEL",
            after_task_id=None,
            after_task_name="Start Event",
            branches=branches,
            confidence_score=confidence,
            justification=justification,
            benefits=benefits,
            implementation_notes=implementation_notes
        )
    
    def _create_gateway_from_assignments(
        self,
        predecessor: Dict[str, Any],
        parallel_tasks: List[Dict[str, Any]],
        cms_data: Dict[str, Any]
    ) -> Optional[GatewaySuggestion]:
        """
        Create a parallel gateway suggestion after a predecessor task (using task_assignments)
        """
        branches = []
        total_time = 0
        max_time = 0
        
        for idx, task in enumerate(parallel_tasks, start=1):
            task_id = task.get('task_id')
            task_name = task.get('task_name', f'Task {task_id}')
            duration_minutes = task.get('duration_minutes', 0)
            
            total_time += duration_minutes
            max_time = max(max_time, duration_minutes)
            
            branch = GatewayBranch(
                branch_id=str(idx),
                target_task_id=task_id,
                task_name=task_name,
                condition=None,
                condition_expression=None,
                is_default=False,
                probability=1.0 / len(parallel_tasks),
                end_task_id=None,
                end_event_name=None,
                description=f"Execute {task_name} ({int(duration_minutes)} min)",
                assigned_jobs=[],
                duration_minutes=float(duration_minutes)
            )
            branches.append(branch)
        
        # Calculate time savings
        time_saved = total_time - max_time
        efficiency_gain = (time_saved / total_time * 100) if total_time > 0 else 0
        
        confidence = 0.90
        
        predecessor_name = predecessor.get('task_name', 'Unknown Task')
        
        # Build justification
        justification = {
            "why_parallel": f"{len(parallel_tasks)} tasks start simultaneously after '{predecessor_name}'",
            "independence_factors": [
                f"All start at t={parallel_tasks[0].get('start_time')}h",
                "Use different resources",
                "Can execute independently"
            ],
            "resource_availability": "Different resources enable simultaneous execution"
        }
        
        # Build benefits
        benefits = {
            "time_saved_minutes": time_saved,
            "before_duration_minutes": total_time,
            "after_duration_minutes": max_time,
            "efficiency_gain_percent": round(efficiency_gain, 2),
            "critical_path": f"{branches[0].task_name} ({int(max_time)} minutes)" if branches else "Unknown",
            "resource_utilization": f"{len(branches)} resources working simultaneously"
        }
        
        # Build implementation notes
        implementation_notes = {
            "gateway_type": "PARALLEL (AND)",
            "placement": f"After {predecessor_name}",
            "behavior": "All branches execute simultaneously",
            "synchronization": "Process waits for all branches to complete before continuing",
            "bpmn_symbol": "Diamond with + symbol",
            "convergence_required": True,
            "convergence_point": "Before next sequential task"
        }
        
        return GatewaySuggestion(
            suggestion_id=0,
            gateway_type="PARALLEL",
            after_task_id=predecessor.get('task_id'),
            after_task_name=predecessor_name,
            branches=branches,
            confidence_score=confidence,
            justification=justification,
            benefits=benefits,
            implementation_notes=implementation_notes
        )
    
    def _create_start_gateway_suggestion(
        self,
        start_tasks: List[Dict[str, Any]],
        cms_data: Dict[str, Any]
    ) -> Optional[GatewaySuggestion]:
        """
        Create a parallel gateway suggestion for tasks starting at the beginning of the process
        
        Args:
            start_tasks: Tasks that all start at order 1 (parallel from start)
            cms_data: Full process data
            
        Returns:
            GatewaySuggestion or None
        """
        # Create branches for each start task
        branches = []
        total_time = 0
        max_time = 0
        
        for idx, task_wrapper in enumerate(start_tasks, start=1):
            task = task_wrapper.get('task', {})
            task_id = task.get('task_id')
            task_name = task.get('task_name', f'Task {task_id}')
            duration_minutes = task.get('task_capacity_minutes', 0)
            job_tasks = task.get('jobTasks', [])
            assigned_jobs = [jt['job_id'] for jt in job_tasks]
            
            total_time += duration_minutes
            max_time = max(max_time, duration_minutes)
            
            branch = GatewayBranch(
                branch_id=str(idx),
                target_task_id=task_id,
                task_name=task_name,
                condition=None,
                condition_expression=None,
                is_default=False,
                probability=1.0 / len(start_tasks),
                end_task_id=None,
                end_event_name=None,
                description=f"Execute {task_name} ({duration_minutes} min)",
                assigned_jobs=assigned_jobs,
                duration_minutes=float(duration_minutes)
            )
            branches.append(branch)
        
        # Calculate time savings (if they ran sequentially vs parallel)
        time_saved = total_time - max_time
        efficiency_gain = (time_saved / total_time * 100) if total_time > 0 else 0
        
        # High confidence for start tasks running in parallel
        confidence = 0.95
        
        # Build justification
        justification = {
            "why_parallel": f"{len(start_tasks)} tasks start simultaneously at the beginning of the process",
            "independence_factors": [
                "Tasks start at the same time (order 1)",
                "Use different job roles/resources",
                "No dependencies between these initial tasks"
            ],
            "resource_availability": "Different job roles enable simultaneous execution"
        }
        
        # Build benefits
        benefits = {
            "time_saved_minutes": time_saved,
            "before_duration_minutes": total_time,
            "after_duration_minutes": max_time,
            "efficiency_gain_percent": round(efficiency_gain, 2),
            "critical_path": f"{branches[0].task_name} ({max_time} minutes)" if branches else "Unknown",
            "resource_utilization": f"{len(branches)} specialists working simultaneously from the start"
        }
        
        # Build implementation notes
        implementation_notes = {
            "gateway_type": "PARALLEL (AND)",
            "placement": "After Start Event",
            "behavior": "All branches execute simultaneously",
            "synchronization": "Process waits for all branches to complete before continuing",
            "bpmn_symbol": "Diamond with + symbol",
            "convergence_required": True,
            "convergence_point": "Before next sequential task"
        }
        
        return GatewaySuggestion(
            suggestion_id=0,
            gateway_type="PARALLEL",
            after_task_id=None,
            after_task_name="Start Event",
            branches=branches,
            confidence_score=confidence,
            justification=justification,
            benefits=benefits,
            implementation_notes=implementation_notes
        )
    
    def _create_gateway_suggestion(
        self,
        current_task_wrapper: Dict[str, Any],
        parallel_candidates: List[Dict[str, Any]],
        cms_data: Dict[str, Any]
    ) -> Optional[GatewaySuggestion]:
        """
        Create a gateway suggestion from parallel candidates
        
        Args:
            current_task_wrapper: The task after which the gateway should be placed
            parallel_candidates: Tasks that can run in parallel
            cms_data: Full process data
            
        Returns:
            GatewaySuggestion or None
        """
        current_task = current_task_wrapper.get('task', {})
        current_task_id = current_task.get('task_id')
        current_task_name = current_task.get('task_name', 'Unknown Task')
        
        # Create branches for each parallel candidate
        branches = []
        total_sequential_time = 0
        max_parallel_time = 0
        
        for idx, candidate_wrapper in enumerate(parallel_candidates, start=1):
            candidate_task = candidate_wrapper.get('task', {})
            task_id = candidate_task.get('task_id')
            task_name = candidate_task.get('task_name', f'Task {task_id}')
            duration_minutes = candidate_task.get('task_capacity_minutes', 0)
            job_tasks = candidate_task.get('jobTasks', [])
            assigned_jobs = [jt['job_id'] for jt in job_tasks]
            
            total_sequential_time += duration_minutes
            max_parallel_time = max(max_parallel_time, duration_minutes)
            
            branch = GatewayBranch(
                branch_id=str(idx),
                target_task_id=task_id,
                task_name=task_name,
                condition=None,
                condition_expression=None,
                is_default=False,
                probability=1.0 / len(parallel_candidates),
                end_task_id=None,
                end_event_name=None,
                description=f"Execute {task_name} ({duration_minutes} min)",
                assigned_jobs=assigned_jobs,
                duration_minutes=float(duration_minutes)
            )
            branches.append(branch)
        
        # Add convergence branch
        # The convergence waits for all branches to complete
        last_task_id = max(b.target_task_id for b in branches if b.target_task_id)
        convergence_branch = GatewayBranch(
            branch_id=str(len(branches) + 1),
            target_task_id=None,
            task_name="Convergence Point",
            condition="all_complete",
            condition_expression=None,
            is_default=False,
            probability=1.0,
            end_task_id=last_task_id,
            end_event_name="parallel_convergence",
            description="Wait for all parallel branches to complete",
            assigned_jobs=[],
            duration_minutes=0.0
        )
        branches.append(convergence_branch)
        
        # Calculate time savings
        time_saved = total_sequential_time - max_parallel_time
        efficiency_gain = (time_saved / total_sequential_time * 100) if total_sequential_time > 0 else 0
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            current_task_wrapper,
            parallel_candidates,
            cms_data
        )
        
        # Build justification
        justification = {
            "why_parallel": f"Tasks following {current_task_name} can execute in parallel as they have no interdependencies",
            "independence_factors": self._get_independence_factors(parallel_candidates),
            "resource_availability": "Different job roles assigned to each parallel task",
            "downstream_impact": self._analyze_downstream_impact(parallel_candidates, cms_data)
        }
        
        # Build benefits
        benefits = {
            "time_saved_minutes": time_saved,
            "before_duration_minutes": total_sequential_time,
            "after_duration_minutes": max_parallel_time,
            "efficiency_gain_percent": round(efficiency_gain, 2),
            "critical_path": f"Task {branches[0].task_name} ({max_parallel_time} minutes)" if branches else "Unknown",
            "resource_utilization": f"{len(branches)-1} specialists working simultaneously instead of sequentially"
        }
        
        # Find next task after parallel group
        next_task_id, next_task_name = self._find_next_task(parallel_candidates, cms_data)
        
        # Build implementation notes
        implementation_notes = {
            "next_task_id": next_task_id,
            "next_task_name": next_task_name,
            "next_task_prerequisites": [b.target_task_id for b in branches if b.target_task_id],
            "task_dependency_update": f"Task {next_task_id} must have predecessors {[b.target_task_id for b in branches if b.target_task_id]} to enforce implicit convergence",
            "workflow_engine_behavior": f"Task {next_task_id} should not start until all parallel branches complete",
            "visualization_hint": "Show PARALLEL (AND) gateway with multiple paths merging before next task"
        }
        
        return GatewaySuggestion(
            suggestion_id=0,
            gateway_type="PARALLEL",
            after_task_id=current_task_id,
            after_task_name=current_task_name,
            branches=branches,
            confidence_score=confidence,
            justification=justification,
            benefits=benefits,
            implementation_notes=implementation_notes
        )
    
    def _get_independence_factors(self, candidates: List[Dict[str, Any]]) -> List[str]:
        """Get list of factors showing why tasks are independent"""
        factors = []
        
        # Check job diversity
        all_jobs = set()
        for candidate in candidates:
            task = candidate.get('task', {})
            job_tasks = task.get('jobTasks', [])
            task_jobs = {jt['job_id'] for jt in job_tasks}
            all_jobs.update(task_jobs)
        
        if len(all_jobs) >= len(candidates):
            factors.append("Different job roles (each task uses different specialists)")
        
        # Check task types
        task_types = set()
        for candidate in candidates:
            task = candidate.get('task', {})
            task_name = task.get('task_name', '').lower()
            if 'decision' in task_name:
                task_types.add('decision-making')
            elif 'pricing' in task_name or 'cost' in task_name:
                task_types.add('pricing')
            elif 'document' in task_name:
                task_types.add('documentation')
            elif 'assessment' in task_name or 'analysis' in task_name:
                task_types.add('assessment')
        
        if len(task_types) >= 2:
            factors.append(f"Different task types: {', '.join(task_types)}")
        
        # Add general factor
        factors.append("No shared data modifications between tasks")
        factors.append("All tasks can execute with only predecessor output")
        
        return factors
    
    def _analyze_downstream_impact(
        self,
        candidates: List[Dict[str, Any]],
        cms_data: Dict[str, Any]
    ) -> str:
        """Analyze what happens after parallel tasks complete"""
        next_task_id, next_task_name = self._find_next_task(candidates, cms_data)
        
        if next_task_id:
            candidate_ids = [c.get('task', {}).get('task_id') for c in candidates]
            return f"All parallel tasks required before Task {next_task_id} ({next_task_name}) can begin"
        else:
            return "Parallel tasks can complete independently (multiple end points)"
    
    def _find_next_task(
        self,
        parallel_candidates: List[Dict[str, Any]],
        cms_data: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the task that comes after the parallel group
        
        Returns:
            Tuple of (task_id, task_name) or (None, None)
        """
        # Get the maximum order from parallel candidates
        max_order = max(c.get('order', 0) for c in parallel_candidates)
        
        # Find the task with order = max_order + 1
        all_tasks = cms_data.get('process_task', [])
        for task_wrapper in all_tasks:
            if task_wrapper.get('order', 0) == max_order + 1:
                task = task_wrapper.get('task', {})
                return task.get('task_id'), task.get('task_name', 'Unknown')
        
        return None, None
    
    def _calculate_confidence(
        self,
        current_task_wrapper: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        cms_data: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for the gateway suggestion
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 1.0
        
        # Factor 1: Job diversity (different jobs = higher confidence)
        all_jobs = []
        for candidate in candidates:
            task = candidate.get('task', {})
            job_tasks = task.get('jobTasks', [])
            all_jobs.extend([jt['job_id'] for jt in job_tasks])
        
        unique_jobs = len(set(all_jobs))
        total_jobs = len(all_jobs)
        
        if total_jobs > 0:
            job_diversity = unique_jobs / total_jobs
            confidence *= job_diversity
        else:
            confidence *= 0.5  # Lower confidence if no job info
        
        # Factor 2: Number of parallel tasks (2-3 is ideal, 4+ might be risky)
        num_tasks = len(candidates)
        if num_tasks == 2:
            confidence *= 1.0  # Perfect
        elif num_tasks == 3:
            confidence *= 0.95  # Good
        elif num_tasks >= 4:
            confidence *= 0.85  # Okay but complex
        
        # Factor 3: Time savings potential
        total_time = sum(c.get('task', {}).get('task_capacity_minutes', 0) for c in candidates)
        max_time = max(c.get('task', {}).get('task_capacity_minutes', 0) for c in candidates)
        
        if total_time > 0:
            time_savings_ratio = (total_time - max_time) / total_time
            if time_savings_ratio > 0.3:  # Good savings
                confidence *= 1.0
            elif time_savings_ratio > 0.15:  # Moderate savings
                confidence *= 0.9
            else:  # Low savings
                confidence *= 0.7
        
        return round(confidence, 2)
    
    def format_for_cms(self, suggestion: GatewaySuggestion, process_id: int) -> Dict[str, Any]:
        """
        Format a gateway suggestion for CMS database structure
        
        Args:
            suggestion: The gateway suggestion
            process_id: The process ID
            
        Returns:
            Dict in CMS gateway format
        """
        branches = []
        for branch in suggestion.branches:
            branches.append({
                "is_default": False,
                "target_task_id": branch.target_task_id,
                "end_task_id": branch.end_task_id,
                "end_event_name": branch.end_event_name,
                "condition": branch.condition,
                "description": branch.description,
                "assigned_jobs": branch.assigned_jobs
            })
        
        return {
            "process_id": process_id,
            "gateway_type": suggestion.gateway_type,
            "after_task_id": suggestion.after_task_id,
            "name": f"{suggestion.after_task_name} - Parallel Execution",
            "branches": branches,
            "confidence_score": suggestion.confidence_score,
            "justification": suggestion.justification,
            "benefits": suggestion.benefits,
            "implementation_notes": suggestion.implementation_notes
        }
    
    def format_suggestions_for_api(
        self,
        suggestions: List[GatewaySuggestion],
        cms_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format all suggestions for API response
        
        Args:
            suggestions: List of gateway suggestions
            cms_data: Original CMS process data
            
        Returns:
            Dict with gateway suggestions and analysis
        """
        process_id = cms_data.get('process_id')
        process_name = cms_data.get('process_name', 'Unknown Process')
        
        # Calculate current and optimized timelines
        all_tasks = cms_data.get('process_task', [])
        current_total_time = sum(
            t.get('task', {}).get('task_capacity_minutes', 0) 
            for t in all_tasks
        )
        
        # Calculate optimized time (considering parallel execution)
        optimized_total_time = current_total_time
        total_time_saved = 0
        
        for suggestion in suggestions:
            time_saved = suggestion.benefits.get('time_saved_minutes', 0)
            total_time_saved += time_saved
            optimized_total_time -= time_saved
        
        efficiency_improvement = (total_time_saved / current_total_time * 100) if current_total_time > 0 else 0
        
        # Format suggestions
        formatted_suggestions = []
        for suggestion in suggestions:
            formatted_suggestions.append({
                "suggestion_id": len(formatted_suggestions) + 1,
                "confidence_score": suggestion.confidence_score,
                "gateway_definition": self.format_for_cms(suggestion, process_id),
                "location": f"After Task {suggestion.after_task_id} ({suggestion.after_task_name})",
                "justification": suggestion.justification,
                "benefits": suggestion.benefits,
                "implementation_notes": suggestion.implementation_notes
            })
        
        return {
            "process_id": process_id,
            "process_name": process_name,
            "parallel_gateway_analysis": {
                "opportunities_found": len(suggestions),
                "current_capacity_minutes": current_total_time,
                "optimized_capacity_minutes": optimized_total_time,
                "total_time_saved_minutes": total_time_saved,
                "efficiency_improvement_percent": round(efficiency_improvement, 2)
            },
            "gateway_suggestions": formatted_suggestions
        }

