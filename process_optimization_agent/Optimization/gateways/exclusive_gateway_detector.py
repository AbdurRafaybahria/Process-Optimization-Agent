"""
Exclusive Gateway (XOR) Detector for Process Optimization.

Detects opportunities for exclusive (XOR) gateways where only ONE path executes
based on conditions/decisions. This is used for decision points like:
- Approval/Rejection decisions
- Threshold-based routing
- Type-based processing paths
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import re

from .gateway_base import GatewayDetectorBase, GatewayBranch, GatewaySuggestion


@dataclass
class DecisionPoint:
    """Represents a detected decision point in the process"""
    task_id: Any
    task_name: str
    decision_type: str  # approval, threshold, classification, validation
    confidence: float
    keywords_found: List[str]
    possible_outcomes: List[str]
    successor_tasks: List[Dict[str, Any]]


@dataclass
class ExclusiveBranch:
    """Branch for exclusive gateway"""
    branch_id: str
    target_task_id: int
    task_name: str
    condition: str
    condition_expression: str
    outcome_type: str  # positive, negative, neutral
    probability: float
    is_default: bool
    duration_minutes: float
    assigned_jobs: List[int] = field(default_factory=list)


class ExclusiveGatewayDetector(GatewayDetectorBase):
    """
    Detects exclusive (XOR) gateway opportunities in process workflows.
    
    XOR gateways are decision points where exactly ONE path is taken based on
    a condition. Examples:
    - Approved → Process / Rejected → Cancel
    - High Risk → Manual Review / Low Risk → Auto Approve
    - Valid → Continue / Invalid → Correction
    """
    
    def __init__(self, min_confidence: float = 0.7):
        """
        Initialize exclusive gateway detector
        
        Args:
            min_confidence: Minimum confidence score (0.0-1.0) for suggestions
        """
        super().__init__(min_confidence)
        
        # Decision type patterns - more comprehensive detection
        self.decision_patterns = {
            'approval': {
                'keywords': ['approval', 'approve', 'authorize', 'authorise', 'sanction',
                            'consent', 'permission', 'sign-off', 'signoff', 'sign off'],
                'outcomes': ['approved', 'rejected', 'pending'],
                'condition_template': "{field} == '{outcome}'"
            },
            'review': {
                'keywords': ['review', 'evaluate', 'assess', 'examine', 'inspect',
                            'audit', 'check', 'verify', 'appraise'],
                'outcomes': ['passed', 'failed', 'needs_revision'],
                'condition_template': "{field}_status == '{outcome}'"
            },
            'decision': {
                'keywords': ['decision', 'decide', 'determine', 'conclude', 'judgment',
                            'ruling', 'verdict', 'resolution'],
                'outcomes': ['proceed', 'halt', 'defer'],
                'condition_template': "{field}_result == '{outcome}'"
            },
            'validation': {
                'keywords': ['validate', 'valid', 'verify', 'confirm', 'authenticate',
                            'certify', 'attest'],
                'outcomes': ['valid', 'invalid', 'incomplete'],
                'condition_template': "is_{field} == {outcome}"
            },
            'classification': {
                'keywords': ['classify', 'categorize', 'categorise', 'type', 'class',
                            'grade', 'rank', 'tier', 'level'],
                'outcomes': ['standard', 'premium', 'special', 'custom'],
                'condition_template': "{field}_type == '{outcome}'"
            },
            'threshold': {
                'keywords': ['score', 'rating', 'amount', 'value', 'level', 'limit',
                            'threshold', 'calculate', 'compute'],
                'outcomes': ['high', 'medium', 'low'],
                'condition_template': "{field}_level == '{outcome}'"
            },
            'risk': {
                'keywords': ['risk', 'hazard', 'exposure', 'liability', 'threat'],
                'outcomes': ['high_risk', 'medium_risk', 'low_risk'],
                'condition_template': "risk_category == '{outcome}'"
            },
            'eligibility': {
                'keywords': ['eligible', 'eligibility', 'qualify', 'qualified', 
                            'meet', 'criteria', 'requirement'],
                'outcomes': ['eligible', 'ineligible', 'conditional'],
                'condition_template': "is_eligible == {outcome}"
            }
        }
        
        # Successor task patterns indicating different outcomes
        self.successor_patterns = {
            'positive_flow': [
                'process', 'proceed', 'continue', 'next', 'forward', 'complete',
                'finalize', 'execute', 'implement', 'issue', 'generate', 'create',
                'send', 'dispatch', 'deliver'
            ],
            'negative_flow': [
                'reject', 'deny', 'cancel', 'terminate', 'stop', 'decline',
                'refuse', 'abort', 'close', 'end', 'halt', 'block'
            ],
            'neutral_flow': [
                'revise', 'correct', 'update', 'modify', 'amend', 'change',
                'resubmit', 'review', 'escalate', 'hold', 'pending', 'wait'
            ]
        }
        
        # Task type indicators
        self.task_type_indicators = {
            'notification': ['notify', 'notification', 'alert', 'inform', 'communicate', 'email', 'message'],
            'processing': ['process', 'execute', 'perform', 'handle', 'manage'],
            'documentation': ['document', 'record', 'log', 'register', 'file'],
            'approval': ['approve', 'authorize', 'sanction', 'sign'],
            'rejection': ['reject', 'deny', 'decline', 'refuse']
        }
    
    def analyze_process(self, process_data: Dict[str, Any]) -> List[GatewaySuggestion]:
        """
        Analyze process and detect exclusive gateway opportunities
        
        Args:
            process_data: CMS process data with tasks, jobs, etc.
            
        Returns:
            List of exclusive gateway suggestions
        """
        suggestions = []
        
        # Extract tasks from process data
        tasks = self._extract_tasks(process_data)
        print(f"[XOR-DEBUG] Extracted {len(tasks)} tasks from process data")
        if not tasks:
            print("[XOR-DEBUG] No tasks found in process data")
            return suggestions
        
        # Build task sequence and dependency map
        task_sequence = self._build_task_sequence(tasks)
        dependency_map = self._build_dependency_map(process_data)
        
        # Step 1: Find all potential decision points
        decision_points = self._find_decision_points(tasks, task_sequence, dependency_map)
        print(f"[XOR-DEBUG] Found {len(decision_points)} decision points")
        for dp in decision_points:
            print(f"[XOR-DEBUG]   - Task {dp.task_id}: '{dp.task_name}' (type: {dp.decision_type}, confidence: {dp.confidence:.2f})")
        
        # Step 2: For each decision point, analyze for XOR opportunities
        for decision_point in decision_points:
            # Step 3: Identify mutually exclusive branches
            branches = self._identify_exclusive_branches(
                decision_point, tasks, task_sequence, dependency_map
            )
            
            # Need at least 2 branches for XOR
            if len(branches) < 2:
                continue
            
            # Step 4: Verify mutual exclusivity
            if not self._verify_mutual_exclusivity(branches, tasks, process_data):
                continue
            
            # Step 5: Calculate confidence score
            confidence_result = self._calculate_confidence(
                decision_point, branches, tasks, process_data
            )
            
            if confidence_result['score'] < self.min_confidence:
                continue
            
            # Step 6: Create gateway suggestion
            suggestion = self._create_xor_suggestion(
                decision_point, branches, confidence_result, tasks, process_data
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _build_task_sequence(self, tasks: List[Dict]) -> Dict[Any, int]:
        """Build task sequence order map"""
        sequence = {}
        for idx, task in enumerate(tasks):
            task_id = task.get('task_id')
            sequence[str(task_id)] = idx
            # Handle split task IDs (e.g., "13_1")
            base_id = str(task_id).split('_')[0]
            if base_id not in sequence:
                sequence[base_id] = idx
        return sequence
    
    def _build_dependency_map(self, process_data: Dict) -> Dict[str, List[str]]:
        """Build task dependency map from process data"""
        dependency_map = {}
        
        # Check parallel execution data for dependencies
        if 'parallel_execution' in process_data:
            parallel_groups = process_data['parallel_execution'].get('parallel_groups', [])
            for group in parallel_groups:
                for task_detail in group.get('task_details', []):
                    task_id = str(task_detail.get('task_id'))
                    dep_chain = task_detail.get('dependency_chain', [])
                    dependency_map[task_id] = [str(d) for d in dep_chain]
        
        # Check network diagram for dependencies
        if 'network_diagram' in process_data:
            for stage in process_data['network_diagram'].get('stages', []):
                for task in stage.get('tasks', []):
                    task_id = str(task.get('task_id'))
                    if task.get('has_dependencies') and task_id not in dependency_map:
                        dependency_map[task_id] = []
        
        # Check constraints
        if 'constraints' in process_data:
            for dep in process_data['constraints'].get('dependencies', []):
                if isinstance(dep, dict):
                    from_task = str(dep.get('from'))
                    to_task = str(dep.get('to'))
                    if to_task not in dependency_map:
                        dependency_map[to_task] = []
                    dependency_map[to_task].append(from_task)
        
        return dependency_map
    
    def _find_decision_points(self, tasks: List[Dict], task_sequence: Dict,
                              dependency_map: Dict) -> List[DecisionPoint]:
        """
        Find all potential decision points in the process
        
        Decision points are tasks that:
        1. Have decision-related keywords (approval, review, validation, etc.)
        2. Have multiple possible successors
        3. Result in conditional branching
        """
        decision_points = []
        
        for task in tasks:
            task_id = task.get('task_id')
            task_name = task.get('task_name', '')
            
            # Check 1: Task name contains decision keywords
            decision_type, keywords_found = self._detect_decision_type(task_name)
            
            if not decision_type:
                # Check 2: Task might be implicit decision point
                decision_type, keywords_found = self._detect_implicit_decision(
                    task, tasks, task_sequence
                )
            
            if decision_type:
                # Get possible outcomes based on decision type
                possible_outcomes = self._get_possible_outcomes(decision_type)
                
                # Find successor tasks
                successors = self._find_successor_tasks(task_id, tasks, task_sequence, dependency_map)
                
                # Calculate decision point confidence
                confidence = self._calculate_decision_point_confidence(
                    task, decision_type, keywords_found, successors
                )
                
                if confidence >= 0.5:  # Lower threshold for initial detection
                    decision_points.append(DecisionPoint(
                        task_id=task_id,
                        task_name=task_name,
                        decision_type=decision_type,
                        confidence=confidence,
                        keywords_found=keywords_found,
                        possible_outcomes=possible_outcomes,
                        successor_tasks=successors
                    ))
        
        return decision_points
    
    def _detect_decision_type(self, task_name: str) -> Tuple[Optional[str], List[str]]:
        """Detect decision type from task name"""
        task_lower = task_name.lower()
        
        best_match = None
        best_keywords = []
        max_matches = 0
        
        for decision_type, pattern_info in self.decision_patterns.items():
            keywords_found = [kw for kw in pattern_info['keywords'] if kw in task_lower]
            
            if len(keywords_found) > max_matches:
                max_matches = len(keywords_found)
                best_match = decision_type
                best_keywords = keywords_found
        
        return best_match, best_keywords
    
    def _detect_implicit_decision(self, task: Dict, tasks: List[Dict],
                                  task_sequence: Dict) -> Tuple[Optional[str], List[str]]:
        """
        Detect implicit decision points based on:
        - Task position in workflow
        - Successor task patterns
        - Resource/job patterns
        """
        task_name = task.get('task_name', '').lower()
        resource_name = task.get('resource_name', '').lower()
        
        # Check resource name for decision indicators
        for decision_type, pattern_info in self.decision_patterns.items():
            keywords_found = [kw for kw in pattern_info['keywords'] 
                            if kw in resource_name]
            if keywords_found:
                return decision_type, keywords_found
        
        # Check if task precedes diverging paths (multiple independent successors)
        # This would indicate a potential decision point
        
        return None, []
    
    def _find_successor_tasks(self, task_id: Any, tasks: List[Dict],
                              task_sequence: Dict, dependency_map: Dict) -> List[Dict]:
        """Find successor tasks that follow the given task"""
        successors = []
        task_id_str = str(task_id)
        base_id = task_id_str.split('_')[0]
        
        current_order = task_sequence.get(task_id_str, task_sequence.get(base_id, -1))
        
        for task in tasks:
            other_id = str(task.get('task_id'))
            other_order = task_sequence.get(other_id, -1)
            
            # Check if this task comes after and potentially depends on current task
            if other_order > current_order:
                # Check if this is an immediate successor
                deps = dependency_map.get(other_id, [])
                
                # Successor criteria (in priority order):
                # 1. Has explicit dependency on current task
                is_explicit_successor = (
                    task_id_str in deps or 
                    base_id in deps
                )
                
                # 2. Is immediate next task with no dependencies
                is_immediate_next = (other_order == current_order + 1 and len(deps) == 0)
                
                # 3. Is within reasonable range (max 2 tasks away) for decision branches
                # This allows for decision points to have multiple potential paths
                is_reasonable_successor = (other_order <= current_order + 2)
                
                # Include if it's explicit, immediate, or a reasonable potential branch
                if is_explicit_successor or is_immediate_next or is_reasonable_successor:
                    successor_info = {
                        'task_id': task.get('task_id'),
                        'task_name': task.get('task_name'),
                        'distance': other_order - current_order,
                        'has_explicit_dependency': is_explicit_successor,
                        'flow_type': self._classify_flow_type(task.get('task_name', ''))
                    }
                    successors.append(successor_info)
        
        return successors
    
    def _classify_flow_type(self, task_name: str) -> str:
        """Classify task as positive, negative, or neutral flow"""
        task_lower = task_name.lower()
        
        for flow_type, keywords in self.successor_patterns.items():
            if any(kw in task_lower for kw in keywords):
                return flow_type.replace('_flow', '')
        
        return 'neutral'
    
    def _get_possible_outcomes(self, decision_type: str) -> List[str]:
        """Get possible outcomes for a decision type"""
        if decision_type in self.decision_patterns:
            return self.decision_patterns[decision_type]['outcomes']
        return ['positive', 'negative', 'neutral']
    
    def _calculate_decision_point_confidence(self, task: Dict, decision_type: str,
                                             keywords_found: List[str],
                                             successors: List[Dict]) -> float:
        """Calculate confidence that this is a valid decision point"""
        confidence = 0.0
        
        # Factor 1: Keywords found (0-40%)
        keyword_score = min(len(keywords_found) * 0.15, 0.4)
        confidence += keyword_score
        
        # Factor 2: Decision type specificity (0-20%)
        high_specificity_types = ['approval', 'validation', 'eligibility']
        if decision_type in high_specificity_types:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # Factor 3: Successor diversity (0-25%)
        if successors:
            flow_types = set(s['flow_type'] for s in successors)
            if len(flow_types) >= 2:  # Different flow types = likely decision
                confidence += 0.25
            elif len(flow_types) == 1:
                confidence += 0.1
        
        # Factor 4: Task name contains decision indicators (0-15%)
        task_name_lower = task.get('task_name', '').lower()
        strong_indicators = ['decision', 'approval', 'review', 'evaluate']
        if any(ind in task_name_lower for ind in strong_indicators):
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _identify_exclusive_branches(self, decision_point: DecisionPoint,
                                     tasks: List[Dict], task_sequence: Dict,
                                     dependency_map: Dict) -> List[ExclusiveBranch]:
        """
        Identify exclusive branches from a decision point
        
        For XOR, branches must be:
        1. Mutually exclusive (only one executes)
        2. Cover different outcomes
        3. Have distinct conditions
        """
        branches = []
        
        # Group successors by flow type
        flow_groups = {'positive': [], 'negative': [], 'neutral': []}
        
        for successor in decision_point.successor_tasks:
            flow_type = successor['flow_type']
            if flow_type in flow_groups:
                flow_groups[flow_type].append(successor)
            else:
                flow_groups['neutral'].append(successor)
        
        # Check if we have actual successors or need to infer branches
        has_explicit_successors = any(successors for successors in flow_groups.values())
        
        if not has_explicit_successors:
            # IMPLICIT BRANCH INFERENCE
            # If no explicit successors found, infer potential branches based on decision type
            print(f"[XOR INFERENCE] No explicit branches found for decision task '{decision_point.task_name}'. Inferring branches...")
            branches = self._infer_branches_from_decision(decision_point, tasks, task_sequence)
            return branches
        
        # Create branches for each distinct flow type with tasks
        branch_id = 0
        has_default = False
        
        for flow_type, successors in flow_groups.items():
            if not successors:
                continue
            
            # Take the first/most relevant task for each flow type
            primary_successor = self._select_primary_successor(successors)
            
            if primary_successor:
                branch_id += 1
                task_data = self._get_task_by_id(tasks, primary_successor['task_id'])
                
                # Generate condition based on decision type and flow
                condition, condition_expr = self._generate_condition(
                    decision_point, flow_type
                )
                
                # Determine if this should be default branch
                is_default = flow_type == 'neutral' and not has_default
                if is_default:
                    has_default = True
                    condition = None  # Default branch has no condition
                    condition_expr = None
                
                # Calculate probability based on flow type
                probability = self._estimate_probability(flow_type)
                
                branch = ExclusiveBranch(
                    branch_id=f"xor_branch_{branch_id}",
                    target_task_id=primary_successor['task_id'],
                    task_name=primary_successor['task_name'],
                    condition=condition,
                    condition_expression=condition_expr,
                    outcome_type=flow_type,
                    probability=probability,
                    is_default=is_default,
                    duration_minutes=task_data.get('duration_minutes', 0) if task_data else 0,
                    assigned_jobs=self._get_task_jobs(task_data) if task_data else []
                )
                branches.append(branch)
        
        # If we have branches but no default, mark the last one as default
        if branches and not has_default:
            branches[-1].is_default = True
            branches[-1].condition = None
            branches[-1].condition_expression = None
        
        return branches
    
    def _select_primary_successor(self, successors: List[Dict]) -> Optional[Dict]:
        """Select the primary/most relevant successor from a group"""
        if not successors:
            return None
        
        # Prefer tasks with explicit dependencies
        explicit_deps = [s for s in successors if s.get('has_explicit_dependency')]
        if explicit_deps:
            return min(explicit_deps, key=lambda x: x['distance'])
        
        # Otherwise, prefer closest task
        return min(successors, key=lambda x: x['distance'])
    
    def _infer_branches_from_decision(self, decision_point: DecisionPoint,
                                      tasks: List[Dict], task_sequence: Dict) -> List[ExclusiveBranch]:
        """
        Infer potential XOR branches when no explicit successors found.
        Creates hypothetical branches based on decision type and business logic.
        """
        branches = []
        decision_type = decision_point.decision_type
        task_id_str = str(decision_point.task_id)
        base_id = task_id_str.split('_')[0]
        current_order = task_sequence.get(task_id_str, task_sequence.get(base_id, -1))
        
        # Find the immediate next task (if any) - this will be the "approved/positive" path
        next_task = None
        for task in tasks:
            other_id = str(task.get('task_id'))
            other_order = task_sequence.get(other_id, -1)
            if other_order == current_order + 1:
                next_task = task
                break
        
        # Generate standard branches based on decision type
        outcome_patterns = self.decision_patterns.get(decision_type, {}).get('outcomes', [])
        
        # Branch 1: POSITIVE outcome (approved/passed/valid) - uses existing next task
        if next_task:
            condition, condition_expr = self._generate_condition_for_outcome(
                decision_point, 'positive', outcome_patterns
            )
            
            branches.append(ExclusiveBranch(
                branch_id="xor_branch_1",
                target_task_id=next_task.get('task_id'),
                task_name=next_task.get('task_name', 'Continue Processing'),
                condition=condition,
                condition_expression=condition_expr,
                outcome_type='positive',
                probability=0.75,
                is_default=False,
                duration_minutes=next_task.get('duration_minutes', 0),
                assigned_jobs=self._get_task_jobs(next_task)
            ))
        
        # Branch 2: NEGATIVE outcome (rejected/failed/invalid) - should terminate process
        negative_condition, negative_expr = self._generate_condition_for_outcome(
            decision_point, 'negative', outcome_patterns
        )
        
        # Create termination path for rejection
        negative_task_name = self._infer_negative_task_name(decision_point)
        
        # Check if this should be a termination (end event) branch
        is_termination = self._is_termination_outcome(decision_point, 'negative')
        
        branches.append(ExclusiveBranch(
            branch_id="xor_branch_2",
            target_task_id=None if is_termination else f"{decision_point.task_id}_rejected",  # None for end events
            task_name=negative_task_name if not is_termination else None,  # No task name for end events
            condition=negative_condition,
            condition_expression=negative_expr,
            outcome_type='negative',
            probability=0.15,
            is_default=False,
            duration_minutes=0 if is_termination else 15,  # No duration for immediate termination
            assigned_jobs=self._get_task_jobs(None)
        ))
        
        print(f"[XOR-DEBUG] Created negative branch for {decision_point.task_name}: is_termination={is_termination}")
        
        # Branch 3: NEUTRAL outcome (pending/review) - inferred/missing task (optional, default)
        if len(outcome_patterns) >= 3:
            neutral_task_name = self._infer_neutral_task_name(decision_point)
            
            branches.append(ExclusiveBranch(
                branch_id="xor_branch_3",
                target_task_id=f"{decision_point.task_id}_pending",  # Placeholder ID
                task_name=neutral_task_name,
                condition=None,  # Default path
                condition_expression=None,
                outcome_type='neutral',
                probability=0.10,
                is_default=True,
                duration_minutes=30,  # Estimated duration for additional review
                assigned_jobs=self._get_task_jobs(None)
            ))
        elif not any(b.is_default for b in branches):
            # Make the negative branch default if no neutral branch
            # NOTE: This means rejection becomes the catch-all path, which may not be ideal
            branches[-1].is_default = True
            branches[-1].condition = None
            branches[-1].condition_expression = None
        
        print(f"[XOR-DEBUG] Total branches created: {len(branches)}, has_default: {any(b.is_default for b in branches)}")
        
        return branches
    
    def _generate_condition_for_outcome(self, decision_point: DecisionPoint,
                                       outcome_type: str, 
                                       outcome_patterns: List[str]) -> Tuple[str, str]:
        """Generate condition for a specific outcome type"""
        decision_type = decision_point.decision_type
        pattern = self.decision_patterns.get(decision_type, {})
        template = pattern.get('condition_template', "{field} == '{outcome}'")
        
        # Map outcome type to actual outcome value
        outcome_map = {
            'positive': outcome_patterns[0] if outcome_patterns else 'approved',
            'negative': outcome_patterns[1] if len(outcome_patterns) > 1 else 'rejected',
            'neutral': outcome_patterns[2] if len(outcome_patterns) > 2 else 'pending'
        }
        
        outcome = outcome_map.get(outcome_type, 'unknown')
        field_name = decision_type.replace('_', '')
        
        # Create condition
        condition = template.format(field=field_name, outcome=outcome)
        
        # Create workflow expression (BPMN-style)
        condition_expression = f"#{{task_{decision_point.task_id}_{field_name} == '{outcome}'}}"
        
        return condition, condition_expression
    
    def _infer_negative_task_name(self, decision_point: DecisionPoint) -> str:
        """Infer task name for negative outcome path"""
        decision_type = decision_point.decision_type
        base_name = decision_point.task_name.replace('Review & Approval', '').replace('Review', '').replace('Decision', '').strip()
        
        naming_map = {
            'approval': f"{base_name} Rejection Notice",
            'review': f"{base_name} Failed Review Handling",
            'validation': f"{base_name} Invalid Entry Correction",
            'eligibility': f"{base_name} Ineligibility Notification",
            'decision': f"{base_name} Rejection Processing",
            'risk': f"{base_name} High Risk Escalation"
        }
        
        return naming_map.get(decision_type, f"{base_name} - Alternative Path")
    
    def _infer_neutral_task_name(self, decision_point: DecisionPoint) -> str:
        """Infer task name for neutral/pending outcome path"""
        decision_type = decision_point.decision_type
        base_name = decision_point.task_name.replace('Review & Approval', '').replace('Review', '').replace('Decision', '').strip()
        
        naming_map = {
            'approval': f"{base_name} Additional Review",
            'review': f"{base_name} Revision Request",
            'validation': f"{base_name} Incomplete Data Follow-up",
            'eligibility': f"{base_name} Conditional Approval",
            'decision': f"{base_name} Further Investigation",
            'risk': f"{base_name} Manual Risk Assessment"
        }
        
        return naming_map.get(decision_type, f"{base_name} - Pending Review")
    
    def _is_termination_outcome(self, decision_point: DecisionPoint, outcome_type: str) -> bool:
        """
        Determine if an outcome should terminate the process (lead to END event)
        
        Args:
            decision_point: The decision being evaluated
            outcome_type: Type of outcome ('positive', 'negative', 'neutral')
            
        Returns:
            True if this outcome should end the process
        """
        # Termination keywords that indicate process should end
        termination_keywords = [
            'reject', 'rejection', 'denied', 'denial', 'deny',
            'cancel', 'cancellation', 'abort', 'terminate', 'close',
            'failed', 'failure', 'invalid', 'ineligible', 'declined', 'refused',
            'withdrawn', 'abandoned', 'expired', 'void', 'disapprove'
        ]
        
        # Only negative outcomes typically terminate
        if outcome_type != 'negative':
            return False
        
        # Check if decision type suggests termination
        task_name_lower = decision_point.task_name.lower()
        
        # Approval/Review decisions with rejection typically terminate
        termination_decision_types = ['approval', 'eligibility', 'validation', 'authorization']
        if decision_point.decision_type in termination_decision_types:
            # Check if task name contains termination keywords
            if any(keyword in task_name_lower for keyword in termination_keywords):
                return True
            # Approval/Denial in final stages often terminates
            if any(word in task_name_lower for word in ['final', 'ultimate', 'last']):
                return True
        
        return False
    
    def _generate_condition(self, decision_point: DecisionPoint,
                           flow_type: str) -> Tuple[str, str]:
        """Generate condition and expression for a branch"""
        decision_type = decision_point.decision_type
        pattern = self.decision_patterns.get(decision_type, {})
        template = pattern.get('condition_template', "{field} == '{outcome}'")
        
        # Map flow type to outcome
        outcome_map = {
            'positive': pattern.get('outcomes', ['approved'])[0] if pattern.get('outcomes') else 'approved',
            'negative': pattern.get('outcomes', ['rejected'])[1] if len(pattern.get('outcomes', [])) > 1 else 'rejected',
            'neutral': pattern.get('outcomes', ['pending'])[-1] if pattern.get('outcomes') else 'pending'
        }
        
        outcome = outcome_map.get(flow_type, 'unknown')
        
        # Generate field name from decision type
        field_name = decision_type.replace('_', '')
        
        # Create condition
        condition = template.format(field=field_name, outcome=outcome)
        
        # Create workflow expression (BPMN-style)
        condition_expression = f"#{{task_{decision_point.task_id}_{field_name} == '{outcome}'}}"
        
        return condition, condition_expression
    
    def _estimate_probability(self, flow_type: str) -> float:
        """Estimate probability for each flow type"""
        probability_map = {
            'positive': 0.75,  # Most processes succeed
            'negative': 0.15,  # Some get rejected
            'neutral': 0.10   # Few need revision/escalation
        }
        return probability_map.get(flow_type, 0.33)
    
    def _get_task_jobs(self, task: Optional[Dict]) -> List[int]:
        """Get job IDs assigned to task"""
        if not task:
            return []
        
        resource_id = task.get('resource_id')
        if resource_id:
            try:
                return [int(resource_id)]
            except (ValueError, TypeError):
                pass
        return []
    
    def _verify_mutual_exclusivity(self, branches: List[ExclusiveBranch],
                                   tasks: List[Dict], process_data: Dict) -> bool:
        """
        Verify that branches are mutually exclusive
        
        Checks:
        1. Branches have different outcome types
        2. Only one branch can execute at a time
        3. No parallel execution pattern for these branches
        """
        # Always accept if we have at least 2 branches
        if len(branches) < 2:
            return False
        
        # Check 1: Different outcome types
        outcome_types = set(b.outcome_type for b in branches)
        if len(outcome_types) < 2:
            return False  # All same type = not XOR
        
        # Check 2: If branches have inferred/placeholder task IDs, accept as valid
        # (these are hypothetical branches we're suggesting)
        inferred_branches = [b for b in branches if isinstance(b.target_task_id, str) and 
                            ('_rejected' in str(b.target_task_id) or '_pending' in str(b.target_task_id))]
        if inferred_branches:
            # This is an inferred XOR gateway - accept it
            return True
        
        # Check 3: Not already marked as parallel
        if 'parallel_execution' in process_data:
            parallel_groups = process_data['parallel_execution'].get('parallel_groups', [])
            branch_task_ids = {str(b.target_task_id) for b in branches}
            
            for group in parallel_groups:
                group_task_ids = {str(t.get('task_id')) for t in group.get('task_details', [])}
                
                # If all branch targets are in same parallel group, not XOR
                if branch_task_ids.issubset(group_task_ids):
                    return False
        
        # Check 4: Branches have mutually exclusive conditions
        conditions = [b.condition for b in branches if b.condition]
        if conditions:
            # Check for opposite conditions (approved vs rejected)
            positive_conditions = [c for c in conditions if 'approved' in c.lower() or 'valid' in c.lower() or 'passed' in c.lower()]
            negative_conditions = [c for c in conditions if 'rejected' in c.lower() or 'invalid' in c.lower() or 'failed' in c.lower()]
            
            if positive_conditions and negative_conditions:
                return True  # Clear mutual exclusivity
        
        return True  # Assume mutual exclusivity if other checks pass
    
    def _calculate_confidence(self, decision_point: DecisionPoint,
                             branches: List[ExclusiveBranch], tasks: List[Dict],
                             process_data: Dict) -> Dict[str, Any]:
        """Calculate confidence score for XOR gateway suggestion"""
        factors = []
        
        # Check if branches are inferred or explicit
        inferred_count = sum(1 for b in branches if isinstance(b.target_task_id, str) and 
                           ('_rejected' in str(b.target_task_id) or '_pending' in str(b.target_task_id)))
        is_inferred = inferred_count > 0
        
        # Factor 1: Decision point strength (0-30%)
        decision_score = decision_point.confidence * 0.3
        factors.append(('decision_point_quality', decision_score, 
                       f"Decision point '{decision_point.task_name}' has confidence {decision_point.confidence:.2f}"))
        
        # Factor 2: Branch diversity (0-25%)
        outcome_types = set(b.outcome_type for b in branches)
        branch_diversity = len(outcome_types) / 3  # Max 3 types
        diversity_score = min(branch_diversity, 1.0) * 0.25
        factors.append(('branch_diversity', diversity_score,
                       f"Found {len(outcome_types)} distinct outcome types"))
        
        # Factor 3: Condition clarity (0-20%)
        conditions_defined = sum(1 for b in branches if b.condition)
        condition_ratio = conditions_defined / max(len(branches) - 1, 1)  # Exclude default
        condition_score = condition_ratio * 0.20
        factors.append(('condition_clarity', condition_score,
                       f"{conditions_defined} branches have clear conditions"))
        
        # Factor 4: Probability distribution (0-15%)
        total_prob = sum(b.probability for b in branches)
        prob_score = min(total_prob, 1.0) * 0.15
        factors.append(('probability_coverage', prob_score,
                       f"Branch probabilities sum to {total_prob:.2f}"))
        
        # Factor 5: Mutual exclusivity evidence (0-10%)
        has_opposite_outcomes = (
            any(b.outcome_type == 'positive' for b in branches) and
            any(b.outcome_type == 'negative' for b in branches)
        )
        exclusivity_score = 0.10 if has_opposite_outcomes else 0.05
        factors.append(('mutual_exclusivity', exclusivity_score,
                       "Branches have opposite outcomes" if has_opposite_outcomes else "Branches may be exclusive"))
        
        # Factor 6: Inference penalty (if branches are inferred)
        if is_inferred:
            inference_penalty = -0.15  # Reduce confidence for inferred branches
            factors.append(('inference_adjustment', inference_penalty,
                           f"⚠️ {inferred_count} branch(es) inferred from business logic (not explicit in data)"))
        
        total_score = sum(score for _, score, _ in factors)
        
        # Add metadata about inference
        reasoning = [reason for _, _, reason in factors]
        if is_inferred:
            reasoning.append(f"NOTE: This XOR gateway includes {inferred_count} inferred branch(es) that don't exist in the current process data")
            reasoning.append("RECOMMENDATION: Validate and create the missing task(s) in CMS to implement this gateway")
        
        return {
            'score': max(0.5, min(total_score, 1.0)),  # Minimum 0.5 for inferred gateways
            'factors': factors,
            'reasoning': reasoning,
            'is_inferred': is_inferred,
            'inferred_branch_count': inferred_count
        }
    
    def _create_xor_suggestion(self, decision_point: DecisionPoint,
                               branches: List[ExclusiveBranch],
                               confidence_result: Dict,
                               tasks: List[Dict],
                               process_data: Dict) -> GatewaySuggestion:
        """Create an exclusive gateway suggestion"""
        
        # Convert ExclusiveBranch to GatewayBranch
        gateway_branches = []
        for branch in branches:
            # Determine if this branch leads to an end event
            is_end_event = branch.target_task_id is None
            
            # Generate end event name if this is a termination branch
            end_event_name = None
            if is_end_event:
                # Create descriptive end event name based on outcome
                outcome_labels = {
                    'negative': 'Rejected',
                    'neutral': 'Pending Review',
                    'positive': 'Completed'
                }
                outcome_label = outcome_labels.get(branch.outcome_type, 'Terminated')
                end_event_name = f"{decision_point.task_name} - {outcome_label}"
            
            gateway_branch = GatewayBranch(
                branch_id=branch.branch_id,
                target_task_id=branch.target_task_id,  # None for end events
                task_name=branch.task_name,  # None for end events
                condition=branch.condition,
                condition_expression=branch.condition_expression,
                is_default=branch.is_default,
                probability=branch.probability,
                end_task_id=None,
                end_event_name=end_event_name,  # Set for termination branches
                description=f"Path for {branch.outcome_type} outcome" + (f" → END: {end_event_name}" if is_end_event else f": {branch.task_name}"),
                assigned_jobs=branch.assigned_jobs,
                duration_minutes=branch.duration_minutes
            )
            gateway_branches.append(gateway_branch)
        
        # Create justification
        justification = {
            'why_exclusive': f"Task '{decision_point.task_name}' is a {decision_point.decision_type} decision point with mutually exclusive outcomes",
            'decision_type': decision_point.decision_type,
            'evidence': [
                f"Keywords found: {', '.join(decision_point.keywords_found)}",
                f"Possible outcomes: {', '.join(decision_point.possible_outcomes)}",
                f"Branch count: {len(branches)} mutually exclusive paths"
            ],
            'confidence_factors': confidence_result['reasoning']
        }
        
        # Calculate benefits
        benefits = {
            'clear_decision_routing': True,
            'branches_count': len(branches),
            'conditions_defined': sum(1 for b in branches if b.condition),
            'default_path_defined': any(b.is_default for b in branches),
            'probability_distribution': {
                b.outcome_type: b.probability for b in branches
            }
        }
        
        # Implementation notes
        non_default_branches = [b for b in branches if not b.is_default]
        default_branch = next((b for b in branches if b.is_default), None)
        
        implementation_notes = {
            'gateway_behavior': 'Exactly ONE path will be taken based on the decision outcome',
            'conditions_required': [
                {
                    'branch': b.task_name,
                    'condition': b.condition,
                    'expression': b.condition_expression
                }
                for b in non_default_branches
            ],
            'default_branch': {
                'task_name': default_branch.task_name if default_branch else 'None',
                'task_id': default_branch.target_task_id if default_branch else None,
                'behavior': 'Executes if no other condition matches'
            } if default_branch else None,
            'no_convergence_needed': 'XOR branches do not require synchronization - paths are independent',
            'visualization_hint': 'Show EXCLUSIVE (XOR) gateway with X symbol, paths should show conditions'
        }
        
        return GatewaySuggestion(
            suggestion_id=0,  # Will be set during formatting
            gateway_type='EXCLUSIVE',
            after_task_id=decision_point.task_id,
            after_task_name=decision_point.task_name,
            branches=gateway_branches,
            confidence_score=confidence_result['score'],
            justification=justification,
            benefits=benefits,
            implementation_notes=implementation_notes
        )
    
    def _validate_gateway_suggestions(self, suggestions: List[GatewaySuggestion],
                                     tasks: List[Dict]) -> List[GatewaySuggestion]:
        """
        Validate gateway suggestions and filter out illogical ones
        
        Checks:
        1. Branches should not ALL lead to very similar tasks (e.g., both "Settlement and Payment")
        2. Gateway should not route back to same type of decision (e.g., Approval → Approval)
        3. Must have at least 2 branches
        """
        validated = []
        
        for suggestion in suggestions:
            # Check 0: Must have at least 2 branches
            if len(suggestion.branches) < 2:
                print(f"[XOR-VALIDATION] ❌ Rejecting gateway after '{suggestion.after_task_name}': "
                      f"Only {len(suggestion.branches)} branch(es) - need at least 2")
                continue
            
            # Check 1: Are branch target tasks meaningfully different?
            branch_task_names = [b.task_name for b in suggestion.branches]
            
            # Extract base task names (remove suffixes like "in Household Claims Processing")
            base_names = []
            for name in branch_task_names:
                # Remove common suffixes
                base = name.replace(' in Household Claims Processing', '') \
                          .replace(' in Claims Processing', '') \
                          .replace(' in Medical Billing', '') \
                          .replace(' in Policy Underwriting', '') \
                          .strip()
                base_names.append(base.lower())
            
            # Check if ALL base names are the same (e.g., all "Verification")
            unique_bases = set(base_names)
            if len(unique_bases) == 1 and len(branch_task_names) > 1:
                # ALL branches lead to same task type - reject
                print(f"[XOR-VALIDATION] ❌ Rejecting gateway after '{suggestion.after_task_name}': "
                      f"All branches lead to same task type: {list(unique_bases)[0]}")
                continue
            
            # Check 2: Does decision task route to another decision of EXACT same type?
            # This catches cases like "Approval/Denial" → "Approval/Denial"
            decision_keywords = ['approval', 'denial', 'review', 'verification', 'assessment']
            after_task_lower = suggestion.after_task_name.lower()
            
            # Find which decision keyword is in the source task
            source_decision_type = None
            for keyword in decision_keywords:
                if keyword in after_task_lower:
                    source_decision_type = keyword
                    break
            
            if source_decision_type:
                # Check if any branch also contains this EXACT decision type
                has_same_decision = False
                for branch_name in branch_task_names:
                    branch_lower = branch_name.lower()
                    # Only reject if it's the SAME decision type (not just any decision)
                    if source_decision_type in branch_lower and branch_name != suggestion.after_task_name:
                        print(f"[XOR-VALIDATION] ❌ Rejecting gateway after '{suggestion.after_task_name}': "
                              f"Routes to same decision type ('{source_decision_type}') in '{branch_name}'")
                        has_same_decision = True
                        break
                
                if has_same_decision:
                    continue
            
            # Passed all validation checks
            print(f"[XOR-VALIDATION] ✅ Validated gateway after '{suggestion.after_task_name}': "
                  f"{len(suggestion.branches)} branches to distinct tasks ({', '.join(base_names)})")
            validated.append(suggestion)
        
        print(f"[XOR-VALIDATION] Final: {len(validated)}/{len(suggestions)} gateways passed validation")
        return validated
    
    def _get_gateway_type_name(self) -> str:
        """Return the gateway type name"""
        return 'EXCLUSIVE'
    
    def format_suggestions_for_api(self, suggestions: List[GatewaySuggestion],
                                   process_id: int, process_name: str) -> Dict[str, Any]:
        """Format XOR gateway suggestions for API response"""
        if not suggestions:
            return {
                'process_id': process_id,
                'process_name': process_name,
                'exclusive_gateway_analysis': {
                    'opportunities_found': 0,
                    'decision_points_detected': 0,
                    'total_branches': 0
                },
                'gateway_suggestions': []
            }
        
        total_branches = sum(len(s.branches) for s in suggestions)
        
        # Format suggestions
        formatted_suggestions = []
        for idx, suggestion in enumerate(suggestions, 1):
            suggestion.suggestion_id = idx
            cms_format = self.format_for_cms([suggestion], process_id, process_name)[0]
            
            formatted = {
                'suggestion_id': idx,
                'confidence_score': suggestion.confidence_score,
                'gateway_definition': cms_format,
                'location': f"After Task {suggestion.after_task_id} ({suggestion.after_task_name})",
                'decision_type': suggestion.justification.get('decision_type', 'unknown'),
                'branches_summary': [
                    {
                        'outcome': b.outcome_type if hasattr(b, 'outcome_type') else 'unknown',
                        'target_task': b.task_name,
                        'condition': b.condition,
                        'probability': b.probability,
                        'is_default': b.is_default
                    }
                    for b in suggestion.branches
                ],
                'justification': suggestion.justification,
                'benefits': suggestion.benefits,
                'implementation_notes': suggestion.implementation_notes
            }
            formatted_suggestions.append(formatted)
        
        return {
            'process_id': process_id,
            'process_name': process_name,
            'exclusive_gateway_analysis': {
                'opportunities_found': len(suggestions),
                'decision_points_detected': len(suggestions),
                'total_branches': total_branches,
                'average_branches_per_gateway': round(total_branches / len(suggestions), 1) if suggestions else 0
            },
            'gateway_suggestions': formatted_suggestions
        }
