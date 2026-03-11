"""
Inclusive Gateway (OR) Detector for Process Optimization.

Detects opportunities for inclusive (OR) gateways where ONE or MORE paths execute
based on conditions. This is the hybrid between Exclusive (XOR) and Parallel (AND):
- Multiple branches CAN execute simultaneously (unlike XOR where only ONE executes)
- NOT all branches always execute (unlike Parallel where ALL execute)
- Each branch has a condition
- At least one branch must be activated

Examples:
- Optional Requirements: Lab Work + Consent Form (one or both may be needed)
- Multi-Channel Notifications: Email + SMS + Push (send to any combination)
- Conditional Fulfillment: Physical Items + Digital Items + Subscription (order may have any combination)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .gateway_base import GatewayDetectorBase, GatewayBranch, GatewaySuggestion


@dataclass
class InclusiveBranchCandidate:
    """Candidate branch for inclusive gateway"""
    branch_id: str
    target_task_id: Any
    task_name: str
    condition: Optional[str]
    condition_keywords: List[str]
    is_optional: bool
    is_conditional: bool
    probability: float
    duration_minutes: float
    assigned_jobs: List[int] = field(default_factory=list)


class InclusiveGatewayDetector(GatewayDetectorBase):
    """
    Detects inclusive (OR) gateway opportunities in process workflows.
    
    OR gateways are conditional multi-path points where one or more branches
    execute based on runtime conditions. Key characteristics:
    - Multiple conditions can be true simultaneously
    - Not all branches always execute (that would be Parallel)
    - Not mutually exclusive (that would be Exclusive XOR)
    - Each branch has a meaningful condition
    
    Real-world patterns:
    1. Optional Requirements: "Get Lab Work" + "Get Consent" (both may be needed)
    2. Multi-Channel: "Send Email" + "Send SMS" (send via any active channels)
    3. Conditional Services: "Ship Physical" + "Send Digital" (order may have both)
    4. Assessment Types: "Financial Check" + "Legal Review" + "Tech Audit" (do applicable ones)
    """
    
    def __init__(self, min_confidence: float = 0.65):
        """
        Initialize inclusive gateway detector
        
        Args:
            min_confidence: Minimum confidence score (0.0-1.0) for suggestions
                           (Lower than XOR/Parallel since OR patterns are more nuanced)
        """
        super().__init__(min_confidence)
        
        # Patterns indicating inclusive OR opportunities
        self.inclusive_patterns = {
            'optional_requirements': {
                'keywords': ['optional', 'if needed', 'as required', 'when applicable',
                            'conditionally', 'may require', 'depending on', 'if necessary'],
                'indicators': ['lab', 'test', 'review', 'check', 'verify', 'assessment',
                              'evaluation', 'screening', 'clearance']
            },
            'multi_channel': {
                'keywords': ['notify', 'send', 'alert', 'communicate', 'inform', 
                            'contact', 'reach', 'message', 'dispatch'],
                'indicators': ['email', 'sms', 'push', 'text', 'message', 'call',
                              'notification', 'letter', 'fax', 'mail']
            },
            'fulfillment': {
                'keywords': ['fulfill', 'process', 'handle', 'deliver', 'ship',
                            'provide', 'supply', 'distribute', 'send'],
                'indicators': ['physical', 'digital', 'subscription', 'license',
                              'service', 'product', 'item', 'goods']
            },
            'assessment': {
                'keywords': ['assess', 'evaluate', 'check', 'review', 'examine',
                            'inspect', 'audit', 'verify', 'validate'],
                'indicators': ['risk', 'compliance', 'legal', 'financial', 'technical',
                              'medical', 'safety', 'quality', 'security']
            },
            'preparation': {
                'keywords': ['prepare', 'setup', 'arrange', 'organize', 'ready',
                            'configure', 'initialize', 'provision'],
                'indicators': ['equipment', 'materials', 'resources', 'documents',
                              'tools', 'workspace', 'environment']
            }
        }
        
        # Condition keywords that suggest branches are independent and conditional
        self.condition_keywords = [
            'if', 'when', 'requires', 'needs', 'has', 'contains', 'includes',
            'enabled', 'applicable', 'relevant', 'necessary', 'needed'
        ]
        
        # Keywords suggesting mutual exclusivity (these indicate XOR, not OR)
        self.mutually_exclusive_keywords = [
            'approved', 'rejected', 'accept', 'deny', 'pass', 'fail',
            'yes', 'no', 'valid', 'invalid', 'success', 'failure'
        ]
    
    def analyze_process(self, process_data: Dict[str, Any]) -> List[GatewaySuggestion]:
        """
        Analyze process and detect inclusive OR gateway opportunities
        
        Args:
            process_data: CMS process data with tasks, jobs, etc.
            
        Returns:
            List of inclusive OR gateway suggestions
        """
        suggestions = []
        suggestion_id = 1000  # Start from 1000 to avoid conflicts with other gateway types
        
        # Extract tasks from process data
        tasks = self._extract_tasks(process_data)
        print(f"[OR-DEBUG] Extracted {len(tasks)} tasks from process data")
        if not tasks:
            print("[OR-DEBUG] No tasks found in process data")
            return suggestions
        
        # Build task relationships
        task_sequence = self._build_task_sequence(tasks)
        dependency_map = self._build_dependency_map(process_data)
        
        # Step 1: Find potential split points (tasks with multiple independent successors)
        split_candidates = self._find_split_candidates(tasks, task_sequence, dependency_map, process_data)
        print(f"[OR-DEBUG] Found {len(split_candidates)} potential split points")
        
        # Step 2: For each split candidate, analyze if it's an inclusive OR pattern
        for split_task_id, successor_ids in split_candidates.items():
            if len(successor_ids) < 2:
                continue
            
            split_task = self._get_task_by_id(tasks, split_task_id)
            if not split_task:
                continue
            
            print(f"[OR-DEBUG] Analyzing split after task {split_task_id}: '{split_task.get('task_name')}'")
            print(f"[OR-DEBUG]   Successors: {successor_ids}")
            
            # Step 3: Build branch candidates
            branch_candidates = self._build_branch_candidates(
                successor_ids, tasks, split_task, process_data
            )
            
            if len(branch_candidates) < 2:
                print(f"[OR-DEBUG]   [WARNING] Only {len(branch_candidates)} valid branches - skipping")
                continue
            
            # Step 4: Validate this is truly an Inclusive OR pattern
            validation_result = self._validate_inclusive_pattern(
                branch_candidates, split_task, tasks, dependency_map
            )
            
            if not validation_result['is_valid']:
                print(f"[OR-DEBUG]   [REJECT] Not an Inclusive OR: {validation_result['reason']}")
                continue
            
            # Step 5: Calculate confidence score
            confidence_result = self._calculate_confidence(
                branch_candidates, split_task, tasks, validation_result
            )
            
            if confidence_result['score'] < self.min_confidence:
                print(f"[OR-DEBUG]   [LOW CONF] Confidence {confidence_result['score']:.2f} below threshold {self.min_confidence}")
                continue
            
            # Step 6: Create gateway suggestion
            suggestion = self._create_or_suggestion(
                suggestion_id, split_task, branch_candidates, 
                confidence_result, tasks, process_data
            )
            
            suggestions.append(suggestion)
            print(f"[OR-DEBUG]   [SUCCESS] Created Inclusive OR gateway suggestion (confidence: {confidence_result['score']:.2f})")
            suggestion_id += 1
        
        print(f"[OR-DEBUG] Total Inclusive OR gateway suggestions: {len(suggestions)}")
        return suggestions
    
    def _build_task_sequence(self, tasks: List[Dict]) -> Dict[str, int]:
        """Build task sequence order map"""
        sequence = {}
        for idx, task in enumerate(tasks):
            task_id = str(task.get('task_id'))
            sequence[task_id] = idx
            # Handle split task IDs (e.g., "13_1")
            base_id = task_id.split('_')[0]
            if base_id not in sequence:
                sequence[base_id] = idx
        return sequence
    
    def _build_dependency_map(self, process_data: Dict) -> Dict[str, List[str]]:
        """Build task dependency map"""
        dependency_map = {}
        
        # First check task-level dependencies (from extracted tasks)
        if 'process_task' in process_data:
            for pt in process_data['process_task']:
                task = pt.get('task', {})
                task_id = str(task.get('task_id'))
                deps = task.get('dependencies', [])
                if deps:
                    dependency_map[task_id] = [str(d) for d in deps]
        
        # Also check constraints/dependencies at process level
        if 'constraints' in process_data:
            for dep in process_data['constraints'].get('dependencies', []):
                if isinstance(dep, dict):
                    from_task = str(dep.get('from'))
                    to_task = str(dep.get('to'))
                    if to_task not in dependency_map:
                        dependency_map[to_task] = []
                    dependency_map[to_task].append(from_task)
        
        return dependency_map
    
    def _find_split_candidates(self, tasks: List[Dict], task_sequence: Dict,
                                dependency_map: Dict, process_data: Dict = None) -> Dict[str, List[str]]:
        """
        Find tasks that could be split points for inclusive OR gateways.
        A split point has multiple successors that are NOT mutually exclusive.
        """
        split_candidates = {}
        
        # Build successor map (which tasks follow each task)
        successor_map = {}
        for task in tasks:
            task_id = str(task.get('task_id'))
            successors = []
            
            # Find tasks that depend on this task
            for other_task in tasks:
                other_id = str(other_task.get('task_id'))
                if other_id == task_id:
                    continue
                
                # Check if other_task depends on current task
                if other_id in dependency_map:
                    if task_id in dependency_map[other_id]:
                        successors.append(other_id)
                # Also check by sequence order (simple case: next tasks in order)
                elif task_sequence.get(other_id, 999) == task_sequence.get(task_id, 0) + 1:
                    successors.append(other_id)
            
            if len(successors) >= 2:
                successor_map[task_id] = successors
        
        # If no explicit dependencies found, look for tasks with similar start conditions
        if not successor_map and process_data and 'task_assignments' in process_data:
            # Group tasks by their dependencies/start times
            task_groups = {}
            for task in process_data.get('task_assignments', []):
                deps = tuple(sorted(task.get('dependency_chain', [])))
                if deps not in task_groups:
                    task_groups[deps] = []
                task_groups[deps].append(str(task.get('task_id')))
            
            # Find groups with multiple tasks
            for deps, task_ids in task_groups.items():
                if len(task_ids) >= 2 and len(deps) > 0:
                    # These tasks share the same predecessor
                    predecessor_id = str(deps[-1]) if deps else None
                    if predecessor_id:
                        successor_map[predecessor_id] = task_ids
        
        # Filter: only keep splits where successors are not obviously mutually exclusive
        for split_id, successors in successor_map.items():
            successor_tasks = [self._get_task_by_id(tasks, sid) for sid in successors]
            successor_tasks = [t for t in successor_tasks if t]
            
            if not self._are_obviously_mutually_exclusive(successor_tasks):
                split_candidates[split_id] = successors
        
        return split_candidates
    
    def _are_obviously_mutually_exclusive(self, tasks: List[Dict]) -> bool:
        """
        Quick check if tasks are obviously mutually exclusive (XOR pattern).
        Returns True if tasks seem to be approval/rejection type branches.
        """
        task_names_lower = [t.get('task_name', '').lower() for t in tasks]
        
        # Check for opposing keywords
        has_positive = any(
            any(kw in name for kw in ['approve', 'accept', 'proceed', 'continue', 'valid'])
            for name in task_names_lower
        )
        has_negative = any(
            any(kw in name for kw in ['reject', 'deny', 'cancel', 'stop', 'invalid'])
            for name in task_names_lower
        )
        
        # If we have both positive and negative outcomes, likely XOR
        return has_positive and has_negative
    
    def _build_branch_candidates(self, successor_ids: List[str], tasks: List[Dict],
                                  split_task: Dict, process_data: Dict) -> List[InclusiveBranchCandidate]:
        """Build branch candidates for potential inclusive OR gateway"""
        candidates = []
        
        for idx, successor_id in enumerate(successor_ids):
            successor_task = self._get_task_by_id(tasks, successor_id)
            if not successor_task:
                continue
            
            task_name = successor_task.get('task_name', '')
            task_name_lower = task_name.lower()
            
            # Extract condition keywords
            condition_keywords = [kw for kw in self.condition_keywords if kw in task_name_lower]
            
            # Check if task seems optional/conditional
            is_optional = any(
                pattern_kw in task_name_lower 
                for pattern in self.inclusive_patterns.values()
                for pattern_kw in pattern['keywords']
            )
            
            is_conditional = len(condition_keywords) > 0 or is_optional
            
            # Infer condition from task name
            condition = self._infer_condition(task_name, split_task.get('task_name', ''))
            
            # Create branch candidate
            candidate = InclusiveBranchCandidate(
                branch_id=f"branch_{idx+1}",
                target_task_id=successor_id,
                task_name=task_name,
                condition=condition,
                condition_keywords=condition_keywords,
                is_optional=is_optional,
                is_conditional=is_conditional,
                probability=0.6,  # Default probability for OR branches
                duration_minutes=successor_task.get('duration_minutes', 0),
                assigned_jobs=[]
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _infer_condition(self, task_name: str, split_task_name: str) -> str:
        """
        Infer condition expression from task names.
        
        Examples:
        - "Send Email Notification" → "emailEnabled == true"
        - "Lab Work Required" → "needsLabWork == true"
        - "Ship Physical Items" → "hasPhysicalProducts == true"
        """
        task_lower = task_name.lower()
        
        # Pattern 1: "Send [Channel]" → "[channel]Enabled == true"
        for channel in ['email', 'sms', 'push', 'text', 'notification']:
            if channel in task_lower and 'send' in task_lower:
                return f"{channel}Enabled == true"
        
        # Pattern 2: "[Action] Required" or "Required [Item]"
        if 'required' in task_lower or 'needs' in task_lower or 'require' in task_lower:
            # Extract key words
            words = re.findall(r'\b\w+\b', task_lower)
            key_words = [w for w in words if w not in ['the', 'a', 'an', 'and', 'or', 'if', 'is', 'are', 'required', 'needs', 'need']]
            if key_words:
                field = ''.join([w.capitalize() for w in key_words[:2]])
                return f"needs{field} == true"
        
        # Pattern 3: "Has [Item]" or "Contains [Item]"
        if 'has' in task_lower or 'have' in task_lower or 'contains' in task_lower:
            words = re.findall(r'\b\w+\b', task_lower)
            key_words = [w for w in words if w not in ['the', 'a', 'an', 'and', 'or', 'has', 'have', 'contains']]
            if key_words:
                field = ''.join([w.capitalize() for w in key_words[:2]])
                return f"has{field} == true"
        
        # Pattern 4: "[Type] [Action]" → "is[Type] == true"
        type_keywords = ['physical', 'digital', 'subscription', 'premium', 'standard', 'custom']
        for type_kw in type_keywords:
            if type_kw in task_lower:
                return f"is{type_kw.capitalize()} == true"
        
        # Default: Generic condition based on task name
        words = re.findall(r'\b[A-Z][a-z]+\b', task_name)
        if words:
            field = ''.join(words[:2])
            return f"{field}Required == true"
        
        return "condition == true"
    
    def _validate_inclusive_pattern(self, branch_candidates: List[InclusiveBranchCandidate],
                                     split_task: Dict, tasks: List[Dict],
                                     dependency_map: Dict) -> Dict[str, Any]:
        """
        Validate that this is truly an Inclusive OR pattern and not XOR or Parallel.
        
        Returns:
            Dict with 'is_valid' (bool) and 'reason' (str)
        """
        # Rule 1: Must have at least 2 branches
        if len(branch_candidates) < 2:
            return {'is_valid': False, 'reason': 'Need at least 2 branches'}
        
        # Rule 2: Branches should NOT be mutually exclusive (that's XOR)
        if self._branches_are_mutually_exclusive(branch_candidates):
            return {'is_valid': False, 'reason': 'Branches are mutually exclusive - should be XOR'}
        
        # Rule 3: Branches should NOT all be unconditional (that's Parallel)
        conditional_count = sum(1 for b in branch_candidates if b.is_conditional or b.is_optional)
        if conditional_count < 2:
            return {'is_valid': False, 'reason': 'Not enough conditional branches - may be Parallel or Sequential'}
        
        # Rule 4: Branches should be relatively independent (no dependencies between them)
        if self._branches_have_dependencies(branch_candidates, dependency_map):
            return {'is_valid': False, 'reason': 'Branches have dependencies - cannot execute in parallel'}
        
        # Rule 5: At least one branch should match inclusive patterns
        pattern_matches = sum(1 for b in branch_candidates if self._matches_inclusive_patterns(b.task_name))
        if pattern_matches == 0:
            return {'is_valid': False, 'reason': 'No branches match known Inclusive OR patterns'}
        
        return {
            'is_valid': True,
            'reason': 'Valid Inclusive OR pattern',
            'conditional_count': conditional_count,
            'pattern_matches': pattern_matches
        }
    
    def _branches_are_mutually_exclusive(self, branches: List[InclusiveBranchCandidate]) -> bool:
        """Check if branches have mutually exclusive keywords (indicating XOR)"""
        task_names_lower = [b.task_name.lower() for b in branches]
        
        # Count positive vs negative keywords
        positive_count = sum(
            any(kw in name for kw in ['approve', 'accept', 'proceed', 'valid', 'pass'])
            for name in task_names_lower
        )
        negative_count = sum(
            any(kw in name for kw in ['reject', 'deny', 'cancel', 'invalid', 'fail'])
            for name in task_names_lower
        )
        
        # If we have opposing outcomes, it's mutually exclusive
        return positive_count >= 1 and negative_count >= 1
    
    def _branches_have_dependencies(self, branches: List[InclusiveBranchCandidate],
                                     dependency_map: Dict) -> bool:
        """Check if any branch depends on another branch"""
        branch_ids = [str(b.target_task_id) for b in branches]
        
        for branch_id in branch_ids:
            if branch_id in dependency_map:
                # Check if any dependency is another branch in this gateway
                for dep_id in dependency_map[branch_id]:
                    if dep_id in branch_ids:
                        return True
        
        return False
    
    def _matches_inclusive_patterns(self, task_name: str) -> bool:
        """Check if task name matches known inclusive OR patterns"""
        task_lower = task_name.lower()
        
        for pattern_name, pattern_data in self.inclusive_patterns.items():
            # Check keywords
            keyword_match = any(kw in task_lower for kw in pattern_data['keywords'])
            indicator_match = any(ind in task_lower for ind in pattern_data['indicators'])
            
            if keyword_match or indicator_match:
                return True
        
        return False
    
    def _calculate_confidence(self, branch_candidates: List[InclusiveBranchCandidate],
                               split_task: Dict, tasks: List[Dict],
                               validation_result: Dict) -> Dict[str, Any]:
        """
        Calculate confidence score for Inclusive OR gateway suggestion.
        
        Factors:
        - Number of conditional branches (more is better)
        - Pattern matching (matches known OR patterns)
        - Branch independence (no cross-dependencies)
        - Semantic similarity (related but not identical operations)
        - Absence of mutual exclusivity keywords
        """
        factors = {}
        score = 0.0
        
        # Factor 1: Multiple conditional branches (0.0-0.25)
        conditional_count = validation_result.get('conditional_count', 0)
        factors['conditional_branches'] = min(conditional_count * 0.1, 0.25)
        score += factors['conditional_branches']
        
        # Factor 2: Pattern matching (0.0-0.25)
        pattern_matches = validation_result.get('pattern_matches', 0)
        pattern_ratio = pattern_matches / len(branch_candidates)
        factors['pattern_matching'] = pattern_ratio * 0.25
        score += factors['pattern_matching']
        
        # Factor 3: Branch independence (0.0-0.15)
        # Already validated in _validate_inclusive_pattern
        factors['branch_independence'] = 0.15
        score += 0.15
        
        # Factor 4: Semantic similarity (related operations) (0.0-0.15)
        similarity_score = self._calculate_semantic_similarity(branch_candidates)
        factors['semantic_similarity'] = similarity_score * 0.15
        score += factors['semantic_similarity']
        
        # Factor 5: Absence of XOR keywords (0.0-0.1)
        xor_keyword_count = sum(
            1 for b in branch_candidates
            if any(kw in b.task_name.lower() for kw in self.mutually_exclusive_keywords)
        )
        xor_penalty = (xor_keyword_count / len(branch_candidates)) * 0.1
        factors['no_xor_keywords'] = 0.1 - xor_penalty
        score += factors['no_xor_keywords']
        
        # Factor 6: Condition quality (0.0-0.1)
        has_explicit_conditions = sum(1 for b in branch_candidates if b.condition and '==' in b.condition)
        condition_quality = (has_explicit_conditions / len(branch_candidates)) * 0.1
        factors['condition_quality'] = condition_quality
        score += condition_quality
        
        return {
            'score': min(max(score, 0.0), 1.0),
            'factors': factors
        }
    
    def _calculate_semantic_similarity(self, branches: List[InclusiveBranchCandidate]) -> float:
        """
        Calculate semantic similarity between branch task names.
        Inclusive OR branches should be related but not identical.
        
        Returns score 0.0-1.0 (0.5-0.8 is ideal range for OR patterns)
        """
        if len(branches) < 2:
            return 0.0
        
        task_names = [b.task_name for b in branches]
        
        try:
            # Use TF-IDF to calculate similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(task_names)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Get average pairwise similarity (excluding diagonal)
            n = len(task_names)
            total_similarity = 0.0
            count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    total_similarity += similarity_matrix[i][j]
                    count += 1
            
            avg_similarity = total_similarity / count if count > 0 else 0.0
            
            # Ideal range is 0.3-0.7 (related but not identical)
            # Map this to 0.0-1.0 score
            if 0.3 <= avg_similarity <= 0.7:
                # Optimal range gets high score
                return 1.0
            elif avg_similarity < 0.3:
                # Too different (may be unrelated tasks)
                return avg_similarity / 0.3 * 0.7
            else:
                # Too similar (may be parallel execution of same task type)
                return max(0.3, 1.0 - (avg_similarity - 0.7) / 0.3 * 0.7)
        
        except Exception as e:
            print(f"[OR-DEBUG] Error calculating semantic similarity: {e}")
            return 0.5  # Default moderate similarity
    
    def _create_or_suggestion(self, suggestion_id: int, split_task: Dict,
                              branch_candidates: List[InclusiveBranchCandidate],
                              confidence_result: Dict, tasks: List[Dict],
                              process_data: Dict) -> GatewaySuggestion:
        """Create an Inclusive OR gateway suggestion"""
        
        # Convert branch candidates to GatewayBranch objects
        gateway_branches = []
        for idx, candidate in enumerate(branch_candidates):
            gateway_branch = GatewayBranch(
                branch_id=candidate.branch_id,
                target_task_id=int(candidate.target_task_id) if str(candidate.target_task_id).isdigit() else None,
                task_name=candidate.task_name,
                condition=candidate.condition,
                condition_expression=candidate.condition,
                is_default=False,  # Inclusive OR doesn't typically have explicit default
                probability=candidate.probability,
                duration_minutes=candidate.duration_minutes,
                assigned_jobs=candidate.assigned_jobs
            )
            gateway_branches.append(gateway_branch)
        
        # Add a default/fallback branch (BPMN best practice)
        gateway_branches.append(GatewayBranch(
            branch_id=f"branch_{len(gateway_branches) + 1}_default",
            target_task_id=None,
            task_name="Default Path (Log Only)",
            condition="Default - No conditions matched",
            condition_expression="else",
            is_default=True,
            probability=0.1,
            end_event_name="No Action Taken"
        ))
        
        # Build justification
        pattern_matches = [
            pattern_name for pattern_name, pattern_data in self.inclusive_patterns.items()
            if any(
                any(kw in b.task_name.lower() for kw in pattern_data['keywords'])
                for b in branch_candidates
            )
        ]
        
        justification = {
            'pattern_type': 'Inclusive OR (One or More Branches)',
            'matched_patterns': pattern_matches,
            'branch_count': len(branch_candidates),
            'conditional_branches': confidence_result['factors'].get('conditional_branches', 0),
            'reasoning': [
                f"Task '{split_task.get('task_name')}' branches to {len(branch_candidates)} conditional paths",
                f"Multiple conditions can be true simultaneously (unlike XOR)",
                f"Not all branches always execute (unlike Parallel)",
                f"Matches inclusive patterns: {', '.join(pattern_matches)}",
                "Each branch has a condition that determines execution"
            ],
            'gateway_behavior': 'ONE or MORE paths will be taken based on which conditions are true',
            'confidence_factors': confidence_result['factors']
        }
        
        # Build benefits
        benefits = {
            'flexibility': 'Supports variable combinations of conditions',
            'efficiency': 'Only executes necessary branches based on runtime data',
            'scalability': 'Easy to add new conditional branches',
            'clarity': 'Makes conditional multi-path logic explicit in the process model'
        }
        
        # Implementation notes
        implementation_notes = {
            'bpmn_element': 'inclusiveGateway',
            'bpmn_symbol': 'Diamond with circle (○)',
            'convergence': 'Join gateway waits for all ACTIVE branches (not all branches)',
            'default_branch': 'Recommended to include default branch as safety net',
            'condition_evaluation': 'ALL conditions are evaluated; branches where condition=true execute',
            'minimum_branches': 'At least one branch must be activated (or use default)',
            'visualization_hint': 'Show INCLUSIVE (OR) gateway with ○ symbol, display conditions on each branch'
        }
        
        # Create suggestion
        return GatewaySuggestion(
            suggestion_id=suggestion_id,
            gateway_type='INCLUSIVE',
            after_task_id=int(split_task.get('task_id')) if split_task.get('task_id') else None,
            after_task_name=split_task.get('task_name', 'Process Start'),
            branches=gateway_branches,
            confidence_score=confidence_result['score'],
            justification=justification,
            benefits=benefits,
            implementation_notes=implementation_notes
        )
    
    def _get_gateway_type_name(self) -> str:
        """Return the gateway type name"""
        return 'INCLUSIVE'
