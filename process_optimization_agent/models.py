"""
Core data models for the Process Optimization Agent
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class SkillLevel(Enum):
    """Skill proficiency levels"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    
    def __int__(self):
        return self.value if isinstance(self.value, int) else 1
        
    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        try:
            if isinstance(value, int):
                return next(lvl for lvl in cls if lvl.value == value)
            if isinstance(value, str):
                return cls[value.upper()]
        except (ValueError, KeyError):
            return cls.BEGINNER
        return cls.BEGINNER


@dataclass
class Skill:
    """Represents a skill with proficiency level"""
    name: str
    level: SkillLevel
    
    def __post_init__(self):
        # Ensure level is a SkillLevel enum
        if not isinstance(self.level, SkillLevel):
            self.level = SkillLevel.from_value(self.level)
    
    def __str__(self):
        return f"{self.name}({self.level.name})"
        
    def to_dict(self):
        """Convert skill to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'level': int(self.level)
        }


@dataclass
class Task:
    """Represents a work task with requirements and constraints"""
    id: str
    name: str
    description: str
    duration_hours: float
    required_skills: List[Skill] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)  # Task IDs this task depends on
    order: Optional[int] = None  # Optional explicit execution order (lower runs earlier)
    deadline: Optional[float] = None  # Hour number when task should be complete
    status: TaskStatus = TaskStatus.PENDING
    assigned_resource: Optional[str] = None  # Resource ID
    start_hour: Optional[float] = None  # Start hour (0-based from project start)
    end_hour: Optional[float] = None  # End hour (0-based from project start)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure dependencies is a set"""
        if isinstance(self.dependencies, list):
            self.dependencies = set(self.dependencies)
    
    def can_start(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are completed"""
        return self.dependencies.issubset(completed_tasks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'duration_hours': self.duration_hours,
            'required_skills': [{'name': s.name, 'level': s.level.value} for s in self.required_skills],
            'dependencies': list(self.dependencies),
            'order': self.order,
            'deadline': self.deadline,
            'status': self.status.value,
            'assigned_resource': self.assigned_resource,
            'start_hour': self.start_hour,
            'end_hour': self.end_hour,
            'metadata': self.metadata
        }


@dataclass
class Resource:
    """Represents a person/resource with skills and availability
    
    Attributes:
        id: Unique identifier for the resource
    """
    id: str
    name: str
    skills: List[Skill] = field(default_factory=list)
    hourly_rate: float = 50.0
    max_hours_per_day: float = 8.0
    total_available_hours: float = 160.0  # Total hours available for the project
    current_workload: float = 0.0  # Current hours assigned
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_skill(self, required_skill: Skill) -> bool:
        """Check if resource has the required skill at sufficient level"""
        # Handle case where either skill name is None
        if not required_skill or not required_skill.name:
            return False
            
        # Normalize skill names for case-insensitive comparison
        required_name = required_skill.name.lower().strip()
        
        for skill in self.skills:
            if not skill or not skill.name:
                continue
                
            # Normalize skill name and check for match
            if skill.name.lower().strip() == required_name:
                # Check if skill level meets or exceeds required level
                return skill.level.value >= required_skill.level.value
                
        return False
    
    def has_all_skills(self, required_skills: List[Skill]) -> bool:
        """Check if resource has all required skills"""
        return all(self.has_skill(skill) for skill in required_skills)
    
    def is_available(self, start_hour: float, duration_hours: float) -> bool:
        """Check if resource is available for the given time period
        
        In the simplified model, we check if adding this task would exceed
        the resource's total available hours.
        """
        return self.current_workload + duration_hours <= self.total_available_hours
    
    def get_skill_score(self, required_skills: List[Skill]) -> float:
        """Calculate skill match score (0-1, higher is better)"""
        if not required_skills:
            return 1.0
        
        total_score = 0
        for req_skill in required_skills:
            best_match = 0
            for resource_skill in self.skills:
                if resource_skill.name.lower() == req_skill.name.lower():
                    # Score based on skill level match
                    if resource_skill.level.value >= req_skill.level.value:
                        best_match = 1.0 + (resource_skill.level.value - req_skill.level.value) * 0.1
                    else:
                        best_match = resource_skill.level.value / req_skill.level.value * 0.5
                    break
            total_score += best_match
        
        return min(total_score / len(required_skills), 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        def serialize_skill(skill):
            if isinstance(skill, dict):
                return skill
            elif hasattr(skill, 'to_dict'):
                return skill.to_dict()
            elif hasattr(skill, 'name') and hasattr(skill, 'level'):
                # Handle Skill object
                return {
                    'name': skill.name,
                    'level': skill.level.value if hasattr(skill.level, 'value') else skill.level
                }
            elif isinstance(skill, (list, tuple)) and len(skill) == 2:
                return {'name': skill[0], 'level': skill[1]}
            else:
                return {'name': str(skill), 'level': 1}  # Default level 1 for unknown formats
        
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'skills': [serialize_skill(s) for s in self.skills],
            'hourly_rate': self.hourly_rate,
            'max_hours_per_day': self.max_hours_per_day,
            'current_assignments': self.current_assignments,
            'metadata': self.metadata
        }


@dataclass
class Process:
    """Container for a complete process with tasks and resources"""
    id: str
    name: str
    description: str
    tasks: List[Task] = field(default_factory=list)
    resources: List[Resource] = field(default_factory=list)
    start_date: datetime = field(default_factory=datetime.now)  # Keep for compatibility
    project_duration_hours: float = 160.0  # Default project duration in hours
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return next((task for task in self.tasks if task.id == task_id), None)
    
    def get_resource_by_id(self, resource_id: str) -> Optional[Resource]:
        """Get resource by ID"""
        return next((resource for resource in self.resources if resource.id == resource_id), None)
    
    def get_available_resources(self, required_skills: List[Skill], 
                              start_hour: float, duration_hours: float) -> List[Resource]:
        """Get resources that can handle the task"""
        available = []
        for resource in self.resources:
            if (resource.has_all_skills(required_skills) and 
                resource.is_available(start_hour, duration_hours)):
                available.append(resource)
        return available
    
    def copy(self) -> 'Process':
        """Create a deep copy of the process"""
        # Create a new process with the same basic attributes
        new_process = Process(
            id=self.id,
            name=self.name,
            description=self.description,
            start_date=self.start_date,
            target_end_date=self.target_end_date,
            constraints=self.constraints.copy(),
            metadata=self.metadata.copy()
        )
        
        # Create deep copies of all tasks
        task_mapping = {}
        for task in self.tasks:
            new_task = Task(
                id=task.id,
                name=task.name,
                description=task.description,
                duration_hours=task.duration_hours,
                required_skills=[Skill(s.name, s.level) for s in task.required_skills],
                dependencies=set(task.dependencies),
                order=task.order,
                deadline=task.deadline,
                status=task.status,
                assigned_resource=task.assigned_resource,
                metadata=task.metadata.copy() if hasattr(task, 'metadata') else {}
            )
            new_process.tasks.append(new_task)
            task_mapping[task.id] = new_task
        
        # Create deep copies of all resources
        resource_mapping = {}
        for resource in self.resources:
            # Construct only with fields defined in the Resource dataclass
            new_resource = Resource(
                id=resource.id,
                name=resource.name,
                skills=[Skill(s.name, s.level) for s in resource.skills],
                hourly_rate=getattr(resource, 'hourly_rate', 50.0),
                max_hours_per_day=getattr(resource, 'max_hours_per_day', 8.0),
                total_available_hours=getattr(resource, 'total_available_hours', 160.0),
                current_workload=getattr(resource, 'current_workload', 0.0),
                metadata=(resource.metadata.copy() if hasattr(resource, 'metadata') and isinstance(resource.metadata, dict) else {})
            )
            # No need to copy working hours attributes in simplified model
            new_process.resources.append(new_resource)
            resource_mapping[resource.id] = new_resource
            
        return new_process
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'tasks': [task.to_dict() for task in self.tasks],
            'resources': [resource.to_dict() for resource in self.resources],
            'start_date': self.start_date.isoformat(),
            'project_duration_hours': self.project_duration_hours,
            'constraints': self.constraints,
            'metadata': self.metadata
        }


@dataclass
class ScheduleEntry:
    """Represents a single task assignment in a schedule"""
    task_id: str
    resource_id: str
    start_time: datetime  # Keep for compatibility but will convert to hours internally
    end_time: datetime    # Keep for compatibility but will convert to hours internally
    start_hour: float = 0.0  # Hour offset from project start
    end_hour: float = 0.0    # Hour offset from project start
    cost: float = 0.0
    
    @property
    def duration_hours(self) -> float:
        """Calculate and return the duration of this schedule entry in hours"""
        if self.end_hour and self.start_hour:
            return self.end_hour - self.start_hour
        elif self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            return max(0.0, duration.total_seconds() / 3600.0)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'resource_id': self.resource_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'start_hour': self.start_hour,
            'end_hour': self.end_hour,
            'duration_hours': self.duration_hours,
            'cost': self.cost
        }


@dataclass
class Schedule:
    """Complete schedule with assignments and metrics"""
    process_id: str
    entries: List[ScheduleEntry] = field(default_factory=list)
    total_duration_hours: float = 0.0
    total_cost: float = 0.0
    completion_date: Optional[datetime] = None
    resource_utilization: Dict[str, float] = field(default_factory=dict)  # resource_id -> utilization %
    critical_path: List[str] = field(default_factory=list)  # Task IDs in critical path
    deadlocks_detected: List[str] = field(default_factory=list)  # Task IDs with deadlocks
    idle_resources: Dict[str, float] = field(default_factory=dict)  # resource_id -> idle hours
    optimization_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_task_schedule(self, task_id: str) -> Optional[ScheduleEntry]:
        """Get schedule entry for a specific task"""
        return next((entry for entry in self.entries if entry.task_id == task_id), None)
    
    def get_resource_schedule(self, resource_id: str) -> List[ScheduleEntry]:
        """Get all schedule entries for a specific resource"""
        return [entry for entry in self.entries if entry.resource_id == resource_id]
    
    def calculate_metrics(self, process: Process):
        """Calculate schedule metrics using simplified hour-based calculations"""
        if not self.entries:
            return
            
        # Initialize metrics
        self.total_cost = 0
        resource_hours = {}
        resource_skill_usage = {}
        task_durations = {}
        
        # First pass: calculate basic metrics
        for entry in self.entries:
            task = process.get_task_by_id(entry.task_id)
            resource = process.get_resource_by_id(entry.resource_id)
            if not task or not resource:
                continue
                
            # Use duration_hours property which handles both hour and datetime formats
            duration = entry.duration_hours
            
            # Update total cost
            self.total_cost += entry.cost
            
            # Track resource hours
            if resource.id not in resource_hours:
                resource_hours[resource.id] = {
                    'total_hours': 0,
                    'tasks': [],
                    'skills_used': set()
                }
            resource_hours[resource.id]['total_hours'] += duration
            resource_hours[resource.id]['tasks'].append({
                'task_id': task.id,
                'task_name': task.name,
                'start_hour': entry.start_hour if entry.start_hour else 0,
                'end_hour': entry.end_hour if entry.end_hour else duration,
                'hours': duration
            })
            
            # Track skill utilization
            for skill in task.required_skills:
                resource_hours[resource.id]['skills_used'].add(f"{skill.name}({skill.level.name})")
        
        # Calculate total schedule duration using simplified hours
        if self.entries:
            # Find the maximum end_hour or calculate from datetime if needed
            max_end_hour = 0
            for entry in self.entries:
                if entry.end_hour:
                    max_end_hour = max(max_end_hour, entry.end_hour)
                elif entry.end_time:
                    # Fallback to datetime calculation if hours not set
                    hours_from_start = (entry.end_time - process.start_date).total_seconds() / 3600
                    max_end_hour = max(max_end_hour, hours_from_start)
            
            self.total_duration_hours = max_end_hour
            if max_end_hour > 0:
                self.completion_date = process.start_date + timedelta(hours=max_end_hour)
            
            # Calculate utilization metrics
            total_possible_hours = self.total_duration_hours * len(process.resources)
            total_utilized_hours = sum(r['total_hours'] for r in resource_hours.values())
            
            self.resource_utilization = {
                'overall_percentage': (total_utilized_hours / total_possible_hours) * 100 if total_possible_hours > 0 else 0,
                'resources': {
                    rid: {
                        'total_hours': data['total_hours'],
                        'utilization': (data['total_hours'] / self.total_duration_hours) * 100 if self.total_duration_hours > 0 else 0,
                        'tasks_count': len(data['tasks']),
                        'skills_used': list(data['skills_used'])
                    }
                    for rid, data in resource_hours.items()
                },
                'underutilized_resources': [
                    rid for rid, data in resource_hours.items()
                    if (data['total_hours'] / self.total_duration_hours) < 0.5  # Less than 50% utilized
                ] if self.total_duration_hours > 0 else []
            }
    
    def validate_schedule(self, process: Process) -> bool:
        """Validate the schedule against the process constraints"""
        # Check for task dependencies
        for entry in self.entries:
            task = process.get_task_by_id(entry.task_id)
            if task and task.dependencies:
                for dependency in task.dependencies:
                    dependent_entry = self.get_task_schedule(dependency)
                    if dependent_entry:
                        # Check using hour-based timing
                        dep_end = dependent_entry.end_hour if dependent_entry.end_hour else 0
                        entry_start = entry.start_hour if entry.start_hour else 0
                        if dep_end > entry_start:
                            return False
        
        # Check for resource availability using simplified hours
        for entry in self.entries:
            resource = process.get_resource_by_id(entry.resource_id)
            if resource:
                start_hour = entry.start_hour if entry.start_hour else 0
                duration = entry.duration_hours
                if not resource.is_available(start_hour, duration):
                    return False
        
        # Check for skill requirements
        for entry in self.entries:
            task = process.get_task_by_id(entry.task_id)
            resource = process.get_resource_by_id(entry.resource_id)
            if task and resource:
                if not resource.has_all_skills(task.required_skills):
                    return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'process_id': self.process_id,
            'entries': [entry.to_dict() for entry in self.entries],
            'total_duration_hours': self.total_duration_hours,
            'total_cost': self.total_cost,
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'resource_utilization': self.resource_utilization,
            'critical_path': self.critical_path,
            'deadlocks_detected': self.deadlocks_detected,
            'idle_resources': self.idle_resources,
            'optimization_metrics': self.optimization_metrics
        }


def load_process_from_json(json_path: str) -> Process:
    """Load process from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Parse tasks
    tasks = []
    for task_data in data.get('tasks', []):
        skills = [Skill(s['name'], SkillLevel(s['level'])) for s in task_data.get('required_skills', [])]
        task = Task(
            id=task_data['id'],
            name=task_data['name'],
            description=task_data['description'],
            duration_hours=task_data['duration_hours'],
            required_skills=skills,
            dependencies=set(task_data.get('dependencies', [])),
            order=task_data.get('order'),
            deadline=task_data.get('deadline')  # Now stored as hours, not datetime
        )
        tasks.append(task)
    
    # Parse resources
    resources = []
    for resource_data in data.get('resources', []):
        skills = [Skill(s['name'], SkillLevel(s['level'])) for s in resource_data.get('skills', [])]
        resource = Resource(
            id=resource_data['id'],
            name=resource_data['name'],
            description=resource_data.get('description', ''),
            skills=skills,
            hourly_rate=resource_data.get('hourly_rate', 50.0),
            max_hours_per_day=resource_data.get('max_hours_per_day', 8.0)
        )
        resources.append(resource)
    
    # Create process
    process = Process(
        id=data['id'],
        name=data['name'],
        description=data['description'],
        tasks=tasks,
        resources=resources,
        start_date=datetime.fromisoformat(data.get('start_date', datetime.now().isoformat())),
        project_duration_hours=data.get('project_duration_hours', 160.0),
        constraints=data.get('constraints', {}),
        metadata=data.get('metadata', {})
    )
    
    return process
