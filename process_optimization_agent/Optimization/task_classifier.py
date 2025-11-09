"""
Task Classifier - Automatically detects user involvement level in tasks
"""

from typing import Dict, List
from .models import UserInvolvement


class TaskClassifier:
    """Classifies tasks based on user/patient involvement level"""
    
    def __init__(self):
        # Keywords that indicate administrative tasks (no user involvement)
        self.admin_keywords = [
            'call patient', 'contact patient', 'notify patient',
            'document', 'record', 'file', 'archive',
            'schedule internal', 'prepare', 'setup', 'clean',
            'review internally', 'process paperwork',
            'update system', 'data entry', 'billing',
            'insurance processing', 'claim', 'report generation'
        ]
        
        # Keywords that indicate direct user participation
        self.direct_keywords = [
            'consultation', 'examination', 'assessment', 'interview',
            'treatment', 'diagnosis', 'procedure', 'surgery',
            'therapy', 'counseling', 'meeting', 'discussion',
            'appointment', 'visit', 'check-up', 'screening',
            'test administration', 'sample collection'
        ]
        
        # Keywords that indicate passive user involvement (waiting)
        self.passive_keywords = [
            'waiting', 'queue', 'check-in', 'registration',
            'wait for', 'hold', 'standby'
        ]
    
    def classify_task(self, task_name: str, task_description: str = "") -> UserInvolvement:
        """
        Classify a task based on its name and description
        
        Args:
            task_name: Name of the task
            task_description: Description of the task
            
        Returns:
            UserInvolvement level
        """
        text = f"{task_name} {task_description}".lower()
        task_name_lower = task_name.lower()
        
        # Check for administrative keywords first (highest priority)
        # Be more specific with pattern matching
        admin_patterns = [
            'call patient', 'contact patient', 'notify patient',
            'document', 'documentation', 'record', 'file', 'notes',
            'data entry', 'billing', 'insurance', 'claim'
        ]
        
        for pattern in admin_patterns:
            if pattern in task_name_lower:
                return UserInvolvement.ADMINISTRATIVE
        
        for keyword in self.admin_keywords:
            if keyword in text:
                return UserInvolvement.ADMINISTRATIVE
        
        # Check for passive keywords
        for keyword in self.passive_keywords:
            if keyword in text:
                return UserInvolvement.PASSIVE
        
        # Check for direct keywords
        for keyword in self.direct_keywords:
            if keyword in text:
                return UserInvolvement.DIRECT
        
        # Default: assume direct involvement if unclear
        return UserInvolvement.DIRECT
    
    def classify_by_task_name_patterns(self, task_name: str) -> UserInvolvement:
        """
        Quick classification based on common task name patterns
        
        Args:
            task_name: Name of the task
            
        Returns:
            UserInvolvement level
        """
        task_lower = task_name.lower()
        
        # Administrative patterns
        if any(pattern in task_lower for pattern in [
            'call', 'notify', 'document', 'record', 'file',
            'prepare', 'setup', 'clean', 'process', 'update',
            'data entry', 'billing', 'insurance', 'claim'
        ]):
            return UserInvolvement.ADMINISTRATIVE
        
        # Passive patterns
        if any(pattern in task_lower for pattern in [
            'wait', 'queue', 'check-in', 'registration'
        ]):
            return UserInvolvement.PASSIVE
        
        # Direct patterns (default)
        return UserInvolvement.DIRECT
    
    def get_involvement_summary(self, tasks: List) -> Dict[str, int]:
        """
        Get summary of task involvement levels
        
        Args:
            tasks: List of tasks
            
        Returns:
            Dictionary with counts of each involvement level
        """
        summary = {
            'direct': 0,
            'passive': 0,
            'administrative': 0
        }
        
        for task in tasks:
            involvement = task.user_involvement if hasattr(task, 'user_involvement') else UserInvolvement.DIRECT
            summary[involvement.value] += 1
        
        return summary
