"""
Base class for all gateway detectors (Parallel, Exclusive, Inclusive).
Provides shared functionality for gateway detection and formatting.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class GatewayBranch:
    """Represents a branch in a gateway"""
    branch_id: str
    target_task_id: Optional[int]
    task_name: Optional[str]
    condition: Optional[str]
    condition_expression: Optional[str]
    is_default: bool
    probability: float
    end_task_id: Optional[int] = None
    end_event_name: Optional[str] = None
    description: Optional[str] = None
    assigned_jobs: List[int] = field(default_factory=list)
    duration_minutes: Optional[float] = None


@dataclass
class GatewaySuggestion:
    """Base gateway suggestion structure"""
    suggestion_id: int
    gateway_type: str  # PARALLEL, EXCLUSIVE, INCLUSIVE
    after_task_id: Optional[int]  # None for gateways at the start
    after_task_name: str
    branches: List[GatewayBranch]
    confidence_score: float
    justification: Dict[str, Any]
    benefits: Dict[str, Any]
    implementation_notes: Dict[str, Any]


class GatewayDetectorBase(ABC):
    """Base class for all gateway detectors"""
    
    def __init__(self, min_confidence: float = 0.7):
        """
        Initialize gateway detector
        
        Args:
            min_confidence: Minimum confidence score (0.0-1.0) for suggestions
        """
        self.min_confidence = min_confidence
        self.decision_keywords = [
            'approval', 'decision', 'review', 'evaluation', 'assessment',
            'check', 'verify', 'validate', 'authorize', 'confirm', 'inspect'
        ]
        self.outcome_keywords = {
            'positive': ['approved', 'accepted', 'passed', 'valid', 'confirmed', 
                        'success', 'complete', 'proceed', 'continue'],
            'negative': ['rejected', 'denied', 'failed', 'invalid', 'declined',
                        'error', 'cancel', 'terminate', 'stop'],
            'neutral': ['pending', 'hold', 'review', 'escalate', 'defer',
                       'revision', 'correction', 'resubmit']
        }
    
    @abstractmethod
    def analyze_process(self, process_data: Dict[str, Any]) -> List[GatewaySuggestion]:
        """
        Analyze process and detect gateway opportunities
        
        Args:
            process_data: CMS process data with tasks, jobs, etc.
            
        Returns:
            List of gateway suggestions
        """
        pass
    
    @abstractmethod
    def _calculate_confidence(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Calculate confidence score for a gateway suggestion
        
        Returns:
            Dict with 'score' and 'factors'
        """
        pass
    
    def _extract_tasks(self, process_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tasks from process data"""
        print(f"[GATEWAY-BASE] Extracting tasks. Keys in process_data: {list(process_data.keys())[:10]}")
        
        # Handle CMS raw format (process_task array with nested task objects)
        if 'process_task' in process_data:
            process_tasks = process_data.get('process_task', [])
            print(f"[GATEWAY-BASE] Found 'process_task' with {len(process_tasks)} items")
            tasks = []
            for pt_wrapper in process_tasks:
                task = pt_wrapper.get('task', {})
                if task:
                    # Extract task data and add order info from wrapper
                    task_data = {
                        'task_id': task.get('task_id'),
                        'task_name': task.get('task_name', task.get('name', '')),
                        'duration_minutes': task.get('duration_minutes', 0),
                        'duration_hours': task.get('duration_minutes', 0) / 60 if task.get('duration_minutes') else 0,
                        'order': pt_wrapper.get('order', 0)
                    }
                    
                    # Extract job assignments from process_task_job array
                    jobs = pt_wrapper.get('process_task_job', [])
                    if jobs:
                        # Use first job for now
                        first_job = jobs[0].get('job', {})
                        task_data['resource_name'] = first_job.get('job_name', first_job.get('name', ''))
                        task_data['resource_id'] = first_job.get('job_id')
                    
                    tasks.append(task_data)
            print(f"[GATEWAY-BASE] Extracted {len(tasks)} tasks from process_task array")
            return tasks
        
        # Handle optimized format (task_assignments array)
        if 'task_assignments' in process_data:
            print(f"[GATEWAY-BASE] Found 'task_assignments' with {len(process_data['task_assignments'])} items")
            tasks = []
            for task_assignment in process_data['task_assignments']:
                tasks.append({
                    'task_id': task_assignment.get('task_id'),
                    'task_name': task_assignment.get('task_name'),
                    'resource_name': task_assignment.get('resource_name'),
                    'resource_id': task_assignment.get('resource_id'),
                    'duration_minutes': task_assignment.get('duration_minutes', 0),
                    'duration_hours': task_assignment.get('duration_hours', 0)
                })
            return tasks
        elif 'tasks' in process_data:
            print(f"[GATEWAY-BASE] Found 'tasks' with {len(process_data['tasks'])} items")
            return process_data['tasks']
        
        print("[GATEWAY-BASE] No tasks or task_assignments found!")
        return []
    
    def _get_task_by_id(self, tasks: List[Dict], task_id: Any) -> Optional[Dict]:
        """Get task by ID (handles string and int IDs)"""
        task_id_str = str(task_id)
        for task in tasks:
            if str(task.get('task_id')) == task_id_str:
                return task
        return None
    
    def _has_decision_keywords(self, task_name: str) -> bool:
        """Check if task name contains decision-making keywords"""
        task_lower = task_name.lower()
        return any(keyword in task_lower for keyword in self.decision_keywords)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant keywords from text"""
        text_lower = text.lower()
        keywords = []
        
        # Check decision keywords
        keywords.extend([kw for kw in self.decision_keywords if kw in text_lower])
        
        # Check outcome keywords
        for category, outcome_list in self.outcome_keywords.items():
            keywords.extend([kw for kw in outcome_list if kw in text_lower])
        
        return list(set(keywords))
    
    def _infer_outcome_type(self, task_name: str) -> Optional[str]:
        """Infer outcome type (positive/negative/neutral) from task name"""
        keywords = self._extract_keywords(task_name)
        
        for keyword in keywords:
            if keyword in self.outcome_keywords['positive']:
                return 'positive'
            elif keyword in self.outcome_keywords['negative']:
                return 'negative'
            elif keyword in self.outcome_keywords['neutral']:
                return 'neutral'
        
        return None
    
    def format_for_cms(self, suggestions: List[GatewaySuggestion], 
                       process_id: int, process_name: str) -> List[Dict[str, Any]]:
        """
        Format gateway suggestions for CMS database structure
        
        Args:
            suggestions: List of gateway suggestions
            process_id: CMS process ID
            process_name: Process name
            
        Returns:
            List of CMS-formatted gateway definitions
        """
        cms_gateways = []
        
        for suggestion in suggestions:
            cms_gateway = {
                'process_id': process_id,
                'gateway_type': suggestion.gateway_type,
                'after_task_id': suggestion.after_task_id,
                'name': f"{suggestion.gateway_type.title()} Gateway after {suggestion.after_task_name}",
                'branches': [],
                'confidence_score': suggestion.confidence_score,
                'justification': suggestion.justification,
                'benefits': suggestion.benefits,
                'implementation_notes': suggestion.implementation_notes
            }
            
            # Format branches
            for branch in suggestion.branches:
                cms_branch = {
                    'is_default': branch.is_default,
                    'target_task_id': branch.target_task_id,
                    'end_task_id': branch.end_task_id,
                    'end_event_name': branch.end_event_name,
                    'condition': branch.condition,
                    'condition_expression': branch.condition_expression,
                    'description': branch.description,
                    'assigned_jobs': branch.assigned_jobs,
                    'probability': branch.probability
                }
                
                # Add duration for parallel gateways
                if branch.duration_minutes is not None:
                    cms_branch['duration_minutes'] = branch.duration_minutes
                
                cms_gateway['branches'].append(cms_branch)
            
            cms_gateways.append(cms_gateway)
        
        return cms_gateways
    
    def format_suggestions_for_api(self, suggestions: List[GatewaySuggestion],
                                   process_id: int, process_name: str) -> Dict[str, Any]:
        """
        Format gateway suggestions for API response
        
        Args:
            suggestions: List of gateway suggestions
            process_id: Process ID
            process_name: Process name
            
        Returns:
            Formatted API response with gateway analysis
        """
        if not suggestions:
            return {
                'process_id': process_id,
                'process_name': process_name,
                'gateway_analysis': {
                    'opportunities_found': 0,
                    'gateway_type': self._get_gateway_type_name(),
                    'total_time_saved_minutes': 0,
                    'efficiency_improvement_percent': 0
                },
                'gateway_suggestions': []
            }
        
        # Calculate total time savings
        total_time_saved = sum(
            sug.benefits.get('time_saved_minutes', 0) 
            for sug in suggestions
        )
        
        # Calculate average efficiency improvement
        efficiency_improvements = [
            sug.benefits.get('efficiency_gain_percent', 0)
            for sug in suggestions
            if sug.benefits.get('efficiency_gain_percent', 0) > 0
        ]
        avg_efficiency = (
            sum(efficiency_improvements) / len(efficiency_improvements)
            if efficiency_improvements else 0
        )
        
        # Format suggestions
        formatted_suggestions = []
        for idx, suggestion in enumerate(suggestions, 1):
            formatted = {
                'suggestion_id': idx,
                'confidence_score': suggestion.confidence_score,
                'gateway_definition': self.format_for_cms([suggestion], process_id, process_name)[0],
                'location': f"After Task {suggestion.after_task_id} ({suggestion.after_task_name})",
                'justification': suggestion.justification,
                'benefits': suggestion.benefits,
                'implementation_notes': suggestion.implementation_notes
            }
            formatted_suggestions.append(formatted)
        
        return {
            'process_id': process_id,
            'process_name': process_name,
            f'{self._get_gateway_type_name().lower()}_gateway_analysis': {
                'opportunities_found': len(suggestions),
                'total_time_saved_minutes': round(total_time_saved, 2),
                'efficiency_improvement_percent': round(avg_efficiency, 2)
            },
            'gateway_suggestions': formatted_suggestions
        }
    
    @abstractmethod
    def _get_gateway_type_name(self) -> str:
        """Return the gateway type name (PARALLEL, EXCLUSIVE, etc.)"""
        pass
    
    def _calculate_time_savings(self, before_duration: float, 
                                after_duration: float) -> Dict[str, float]:
        """Calculate time savings metrics"""
        time_saved = before_duration - after_duration
        efficiency_gain = (time_saved / before_duration * 100) if before_duration > 0 else 0
        
        return {
            'time_saved_minutes': round(time_saved, 2),
            'before_duration_minutes': round(before_duration, 2),
            'after_duration_minutes': round(after_duration, 2),
            'efficiency_gain_percent': round(efficiency_gain, 2)
        }
