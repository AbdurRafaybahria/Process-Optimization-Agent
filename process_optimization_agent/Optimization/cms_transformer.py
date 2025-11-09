"""
Transformer to convert CMS data format to Process Optimization Agent format
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from .models import Process, Task, Resource, Skill, SkillLevel, UserInvolvement
from .task_classifier import TaskClassifier


class ProcessValidationError(Exception):
    """Custom exception for process validation errors"""
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class CMSDataTransformer:
    """Transform CMS process data to agent-compatible format"""
    
    def __init__(self):
        self.task_classifier = TaskClassifier()
    
    @staticmethod
    def clean_html(text: str) -> str:
        """Remove HTML tags and clean up text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove escaped quotes and newlines
        text = text.replace('\\n', ' ').replace('\\"', '"').replace('\\', '')
        # Remove data-pm-slice attributes
        text = re.sub(r'data-pm-slice="[^"]*"', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def transform_process(self, cms_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform CMS process data to agent format with robust error handling
        
        Args:
            cms_data: Raw process data from CMS API or already in agent format
            
        Returns:
            Dict in the format expected by the process optimization agent
        """
        try:
            process_data = cms_data.get("process_data", cms_data)
            
            # Check if data is already in agent format (has 'tasks' and 'resources' arrays)
            if 'tasks' in process_data and 'resources' in process_data:
                # Data is already in agent format
                # First validate the raw data before normalization fixes issues
                self._validate_process(process_data)
                # Then normalize for consistency
                normalized = self._normalize_agent_format(process_data)
                return normalized
        except ProcessValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            print(f"[WARNING] Error in transform_process: {str(e)}")
            # Try to continue with cms_data directly
            process_data = cms_data
        
        # Extract basic process information with fallbacks
        process_id = process_data.get("process_id") or process_data.get("id", "unknown")
        
        # Handle company field - can be either a dict or a string
        company_data = process_data.get("company", "")
        if isinstance(company_data, dict):
            company_name = company_data.get("name", "Unknown Company")
        else:
            company_name = str(company_data) if company_data else "Unknown Company"
        
        # Get process name with fallbacks
        process_name = (process_data.get("process_name") or 
                       process_data.get("name") or 
                       f"Process {process_id}")
        
        # Get description with fallbacks
        description = (process_data.get("description") or 
                      process_data.get("process_overview") or "")
        
        transformed = {
            "id": str(process_id),  # Add id field for optimizer (as string)
            "name": process_name,  # Add name field for optimizer
            "process_name": process_name,
            "process_id": int(process_id) if str(process_id).isdigit() else process_id,  # Keep as int for CMS compatibility
            "company": company_name,
            "description": self.clean_html(description),
            "tasks": [],
            "resources": [],
            "dependencies": []
        }
        
        # Extract tasks and resources with robust error handling
        # Handle both CMS format (process_task) and direct format (tasks)
        process_tasks = process_data.get("process_task") or process_data.get("tasks", [])
        resource_map = {}  # Track unique resources
        
        for idx, pt in enumerate(process_tasks):
            try:
                # Handle both nested and flat task structures
                if isinstance(pt, dict) and "task" in pt:
                    # CMS format: {task: {...}, order: 1}
                    task = pt.get("task", {})
                    order = pt.get("order", idx + 1)
                    job = pt.get("job")
                else:
                    # Direct format: task dict directly
                    task = pt
                    order = pt.get("order", idx + 1)
                    job = None
                
                task_transformed = self._transform_task(task, order, job)
                transformed["tasks"].append(task_transformed)
                
                # Extract resources from job tasks (nested in task)
                for job_task in task.get("jobTasks", []):
                    try:
                        job = job_task.get("job", {})
                        resource = self._extract_resource(job)
                        if resource and resource["id"] not in resource_map:
                            resource_map[resource["id"]] = resource
                    except Exception as e:
                        print(f"[WARNING] Error extracting resource from jobTask: {str(e)}")
                        continue
                
                # Also check for job at process_task level (simplified format)
                if isinstance(pt, dict) and "job" in pt:
                    try:
                        resource = self._extract_resource(pt["job"])
                        if resource and resource["id"] not in resource_map:
                            resource_map[resource["id"]] = resource
                    except Exception as e:
                        print(f"[WARNING] Error extracting resource from process_task: {str(e)}")
            except Exception as e:
                print(f"[WARNING] Error transforming task {idx}: {str(e)}")
                continue
        
        # Handle direct resources array if present
        if not resource_map and "resources" in process_data:
            for resource in process_data.get("resources", []):
                try:
                    resource_id = str(resource.get("id", ""))
                    if resource_id and resource_id not in resource_map:
                        resource_map[resource_id] = resource
                except Exception as e:
                    print(f"[WARNING] Error processing direct resource: {str(e)}")
        
        # Add unique resources
        transformed["resources"] = list(resource_map.values())
        
        # Handle dependencies - check both formats
        if "dependencies" in process_data and process_data["dependencies"]:
            transformed["dependencies"] = process_data["dependencies"]
        else:
            # Auto-detect dependencies based on task order
            transformed["dependencies"] = self._infer_dependencies(transformed["tasks"])
        
        # Final normalization to ensure consistency
        normalized = self._normalize_agent_format(transformed)
        
        # Validate the transformed process
        self._validate_process(normalized)
        
        return normalized
    
    def _transform_task(self, task_data: Dict[str, Any], order: int, process_task_job: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform a single task from CMS format to agent format with robust error handling
        
        Args:
            task_data: Raw task data from CMS
            order: Task order in process
            process_task_job: Job data from process_task level (for simplified format)
            
        Returns:
            Transformed task dictionary
        """
        # Extract task ID with fallbacks
        task_id = str(task_data.get("task_id") or task_data.get("id", f"task_{order}"))
        
        # Extract task name with fallbacks
        task_name = task_data.get("task_name") or task_data.get("name", f"Task {order}")
        
        # Extract description with fallbacks
        task_description = (task_data.get("task_overview") or 
                           task_data.get("description", ""))
        task_overview = self.clean_html(task_description)
        
        # Extract duration with fallbacks and validation
        duration_minutes = task_data.get("task_capacity_minutes") or task_data.get("duration", 60)
        try:
            duration_minutes = float(duration_minutes)
            if duration_minutes <= 0:
                duration_minutes = 60  # Default to 1 hour
        except (ValueError, TypeError):
            duration_minutes = 60
        
        duration_hours = task_data.get("duration_hours")
        if duration_hours is None:
            duration_hours = duration_minutes / 60
        
        # Extract job information for resource requirements
        job_tasks = task_data.get("jobTasks", [])
        required_skills = []
        
        # First check jobTasks in the task
        for jt in job_tasks:
            try:
                job = jt.get("job", {})
                skill = {
                    "name": job.get("name", ""),
                    "level": self._map_skill_level(job.get("job_level_id", 3))
                }
                if skill["name"]:  # Only add if name is not empty
                    required_skills.append(skill)
            except Exception as e:
                print(f"[WARNING] Error extracting skill from jobTask: {str(e)}")
                continue
        
        # If no jobTasks, use process_task level job
        if not required_skills and process_task_job:
            try:
                skill = {
                    "name": process_task_job.get("name", ""),
                    "level": self._map_skill_level(process_task_job.get("job_level_id", 3))
                }
                if skill["name"]:
                    required_skills.append(skill)
            except Exception as e:
                print(f"[WARNING] Error extracting skill from process_task job: {str(e)}")
        
        # If still no skills, check required_skills field directly
        if not required_skills and "required_skills" in task_data:
            try:
                direct_skills = task_data.get("required_skills", [])
                for skill in direct_skills:
                    if isinstance(skill, dict) and "name" in skill:
                        required_skills.append(skill)
            except Exception as e:
                print(f"[WARNING] Error extracting direct required_skills: {str(e)}")
        
        # Classify user involvement
        try:
            user_involvement = self.task_classifier.classify_task(task_name, task_overview)
            user_involvement_value = user_involvement.value
        except Exception as e:
            print(f"[WARNING] Error classifying task involvement: {str(e)}")
            # Check if user_involvement is provided directly
            user_involvement_value = task_data.get("user_involvement", "admin")
        
        # Get dependencies from task data
        task_dependencies = task_data.get("dependencies", [])
        # Ensure dependencies are strings
        try:
            task_dependencies = [str(dep) for dep in task_dependencies] if task_dependencies else []
        except Exception as e:
            print(f"[WARNING] Error processing dependencies: {str(e)}")
            task_dependencies = []
        
        # Extract task code with fallback
        task_code = task_data.get("task_code") or task_data.get("code", "")
        
        return {
            "id": task_id,
            "name": task_name,
            "description": task_overview,
            "duration": duration_minutes,  # Duration in minutes
            "duration_hours": duration_hours,  # Duration in hours
            "required_skills": required_skills,
            "dependencies": task_dependencies,
            "order": order,
            "code": task_code,
            "user_involvement": user_involvement_value
        }
    
    def _extract_resource(self, job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract resource information from job data
        
        Args:
            job_data: Job information from CMS
            
        Returns:
            Resource dictionary or None
        """
        if not job_data:
            return None
        
        return {
            "id": str(job_data.get("job_id", "")),
            "name": job_data.get("name", ""),
            "type": "human",  # CMS jobs are human resources
            "skills": [{
                "name": job_data.get("name", ""),
                "level": self._map_skill_level(job_data.get("job_level_id", 3))
            }],
            "max_hours_per_day": job_data.get("maxHoursPerDay", 8),  # Keep as hours
            "hourly_rate": job_data.get("hourlyRate", 50),
            "code": job_data.get("jobCode", "")
        }
    
    def _map_skill_level(self, job_level_id: int) -> str:
        """
        Map CMS job level ID to skill level
        
        Args:
            job_level_id: Job level ID from CMS
            
        Returns:
            Skill level string (beginner, intermediate, advanced, expert)
        """
        level_map = {
            1: "beginner",
            2: "intermediate", 
            3: "intermediate",
            4: "advanced",
            5: "expert"
        }
        return level_map.get(job_level_id, "intermediate")
    
    def _infer_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Infer task dependencies based on order and task relationships
        
        Args:
            tasks: List of transformed tasks
            
        Returns:
            List of dependency relationships
        """
        dependencies = []
        sorted_tasks = sorted(tasks, key=lambda x: x.get("order", 0))
        
        # Simple sequential dependency inference
        # Can be enhanced with more sophisticated dependency detection
        for i in range(1, len(sorted_tasks)):
            current_task = sorted_tasks[i]
            previous_task = sorted_tasks[i-1]
            
            # Create dependency from previous task to current
            dependencies.append({
                "from": previous_task["id"],
                "to": current_task["id"]
            })
            
            # Also update task's dependency list
            current_task["dependencies"].append(previous_task["id"])
        
        return dependencies
    
    def create_process_object(self, cms_data: Dict[str, Any]) -> Process:
        """
        Create a Process object from CMS data
        
        Args:
            cms_data: Raw process data from CMS API
            
        Returns:
            Process object compatible with the optimization agent
        """
        transformed_data = self.transform_process(cms_data)
        
        # Create Task objects
        tasks = []
        for task_data in transformed_data["tasks"]:
            task = Task(
                id=task_data["id"],
                name=task_data["name"],
                description=task_data["description"],
                duration_hours=task_data["duration"] / 60,  # Convert minutes to hours
                required_skills=[
                    Skill(name=s["name"], level=SkillLevel[s["level"].upper()])
                    for s in task_data["required_skills"]
                ],
                dependencies=set(task_data["dependencies"]),  # Convert to set
                user_involvement=UserInvolvement.from_string(task_data.get("user_involvement", "direct"))
            )
            tasks.append(task)
        
        # Create Resource objects
        resources = []
        for resource_data in transformed_data["resources"]:
            resource = Resource(
                id=resource_data["id"],
                name=resource_data["name"],
                skills=[
                    Skill(name=s["name"], level=SkillLevel[s["level"].upper()])
                    for s in resource_data["skills"]
                ],
                max_hours_per_day=resource_data.get("max_hours_per_day", 8),
                hourly_rate=resource_data.get("hourly_rate", 50)
            )
            resources.append(resource)
        
        # Create Process object
        process = Process(
            id=transformed_data["process_id"],
            name=transformed_data["process_name"],
            description=transformed_data["description"],
            company=transformed_data.get("company", ""),
            tasks=tasks,
            resources=resources
        )
        
        return process
    
    def _validate_process(self, data: Dict[str, Any]) -> None:
        """
        Validate process data and raise ProcessValidationError if invalid
        
        Args:
            data: Transformed process data
            
        Raises:
            ProcessValidationError: If validation fails with specific error message
        """
        process_name = data.get("name", "Unknown Process")
        
        # 1. Check if tasks exist
        tasks = data.get("tasks", [])
        if not tasks or len(tasks) == 0:
            raise ProcessValidationError(
                f"Process '{process_name}' has no tasks. At least 2 tasks are required for optimization.",
                "NO_TASKS"
            )
        
        # 2. Check if there's only one task
        if len(tasks) == 1:
            raise ProcessValidationError(
                f"Process '{process_name}' has only 1 task. At least 2 tasks are required for optimization.",
                "INSUFFICIENT_TASKS"
            )
        
        # 3. Check if resources/jobs exist
        resources = data.get("resources", [])
        if not resources or len(resources) == 0:
            raise ProcessValidationError(
                f"Process '{process_name}' has no resources/jobs assigned. At least one resource is required.",
                "NO_RESOURCES"
            )
        
        # 4. Validate task durations
        for idx, task in enumerate(tasks):
            task_name = task.get("name", f"Task {idx + 1}")
            duration = task.get("duration", 0)
            duration_hours = task.get("duration_hours", 0)
            
            if duration <= 0 and duration_hours <= 0:
                raise ProcessValidationError(
                    f"Task '{task_name}' in process '{process_name}' has invalid duration (≤ 0). All tasks must have positive duration.",
                    "NEGATIVE_DURATION"
                )
            
            # Check for negative duration
            if duration < 0 or duration_hours < 0:
                raise ProcessValidationError(
                    f"Task '{task_name}' in process '{process_name}' has negative duration. Duration must be positive.",
                    "NEGATIVE_DURATION"
                )
        
        # 5. Validate resource hourly rates
        for idx, resource in enumerate(resources):
            resource_name = resource.get("name", f"Resource {idx + 1}")
            hourly_rate = resource.get("hourly_rate", 0)
            
            if hourly_rate < 0:
                raise ProcessValidationError(
                    f"Resource '{resource_name}' in process '{process_name}' has negative hourly rate (${hourly_rate}). Hourly rate must be positive.",
                    "NEGATIVE_HOURLY_RATE"
                )
            
            if hourly_rate == 0:
                raise ProcessValidationError(
                    f"Resource '{resource_name}' in process '{process_name}' has zero hourly rate. Please provide a valid hourly rate.",
                    "ZERO_HOURLY_RATE"
                )
        
        # 6. Check if tasks have required skills
        tasks_without_skills = []
        for idx, task in enumerate(tasks):
            task_name = task.get("name", f"Task {idx + 1}")
            required_skills = task.get("required_skills", [])
            if not required_skills or len(required_skills) == 0:
                tasks_without_skills.append(task_name)
        
        if tasks_without_skills:
            task_list = ", ".join(tasks_without_skills[:3])
            if len(tasks_without_skills) > 3:
                task_list += f" and {len(tasks_without_skills) - 3} more"
            raise ProcessValidationError(
                f"Process '{process_name}' has tasks without required skills: {task_list}. All tasks must have at least one required skill.",
                "TASKS_WITHOUT_SKILLS"
            )
    
    def _normalize_agent_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize agent format data to ensure consistency
        - Convert dependencies from [{from, to}] to task.dependencies array
        - Ensure all required fields are present
        """
        normalized = data.copy()
        
        # Normalize tasks
        if 'tasks' in normalized:
            tasks = []
            for task in normalized['tasks']:
                normalized_task = task.copy()
                
                # Ensure duration_hours exists
                if 'duration_hours' not in normalized_task and 'duration' in normalized_task:
                    # Convert minutes to hours if needed
                    duration = normalized_task['duration']
                    if duration > 24:  # Likely in minutes
                        normalized_task['duration_hours'] = duration / 60
                    else:
                        normalized_task['duration_hours'] = duration
                
                # Initialize dependencies array if not present
                if 'dependencies' not in normalized_task:
                    normalized_task['dependencies'] = []
                
                tasks.append(normalized_task)
            
            normalized['tasks'] = tasks
        
        # Convert global dependencies format [{from, to}] to task-level dependencies
        if 'dependencies' in normalized and isinstance(normalized['dependencies'], list):
            if normalized['dependencies'] and isinstance(normalized['dependencies'][0], dict):
                # Build dependency map: task_id -> [prerequisite_ids]
                dep_map = {}
                for dep in normalized['dependencies']:
                    to_task = str(dep.get('to', ''))
                    from_task = str(dep.get('from', ''))
                    if to_task:
                        if to_task not in dep_map:
                            dep_map[to_task] = []
                        if from_task:
                            dep_map[to_task].append(from_task)
                
                # Apply dependencies to tasks
                for task in normalized['tasks']:
                    task_id = str(task['id'])
                    if task_id in dep_map:
                        task['dependencies'] = dep_map[task_id]
        
        # Validate and fix resources
        if 'resources' in normalized:
            resources = []
            for resource in normalized['resources']:
                normalized_resource = resource.copy()
                
                # Fix invalid hourly_rate (negative or zero)
                hourly_rate = normalized_resource.get('hourly_rate', 0)
                if hourly_rate <= 0:
                    # Use a default rate based on skill level
                    skill_level = 'intermediate'
                    if 'skills' in normalized_resource and normalized_resource['skills']:
                        skill_level = normalized_resource['skills'][0].get('level', 'intermediate')
                    
                    # Default rates by skill level
                    default_rates = {
                        'beginner': 20,
                        'intermediate': 30,
                        'advanced': 40,
                        'expert': 50
                    }
                    normalized_resource['hourly_rate'] = default_rates.get(skill_level, 30)
                
                # Fix invalid max_hours_per_day (zero or negative)
                max_hours = normalized_resource.get('max_hours_per_day', 8)
                if max_hours <= 0:
                    normalized_resource['max_hours_per_day'] = 8  # Default to 8 hours
                
                resources.append(normalized_resource)
            
            normalized['resources'] = resources
        
        return normalized
