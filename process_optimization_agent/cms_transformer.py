"""
Transformer to convert CMS data format to Process Optimization Agent format
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from .models import Process, Task, Resource, Skill, SkillLevel


class CMSDataTransformer:
    """Transform CMS process data to agent-compatible format"""
    
    def transform_process(self, cms_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform CMS process data to agent format
        
        Args:
            cms_data: Raw process data from CMS API
            
        Returns:
            Dict in the format expected by the process optimization agent
        """
        process_data = cms_data.get("process_data", cms_data)
        
        # Extract basic process information
        process_id = process_data.get("process_id", "")
        transformed = {
            "id": str(process_id),  # Add id field for optimizer (as string)
            "name": process_data.get("process_name", ""),  # Add name field for optimizer
            "process_name": process_data.get("process_name", ""),
            "process_id": int(process_id) if str(process_id).isdigit() else process_id,  # Keep as int for CMS compatibility
            "company": process_data.get("company", {}).get("name", ""),
            "description": process_data.get("process_overview", ""),
            "tasks": [],
            "resources": [],
            "dependencies": []
        }
        
        # Transform tasks
        process_tasks = process_data.get("process_tasks", [])
        resource_map = {}  # Track unique resources
        
        for pt in process_tasks:
            task = pt.get("task", {})
            task_transformed = self._transform_task(task, pt.get("order", 1))
            transformed["tasks"].append(task_transformed)
            
            # Extract resources from job tasks
            for job_task in task.get("jobTasks", []):
                job = job_task.get("job", {})
                resource = self._extract_resource(job)
                if resource and resource["id"] not in resource_map:
                    resource_map[resource["id"]] = resource
        
        # Add unique resources
        transformed["resources"] = list(resource_map.values())
        
        # Auto-detect dependencies based on task order
        transformed["dependencies"] = self._infer_dependencies(transformed["tasks"])
        
        return transformed
    
    def _transform_task(self, task_data: Dict[str, Any], order: int) -> Dict[str, Any]:
        """
        Transform a single task from CMS format to agent format
        
        Args:
            task_data: Raw task data from CMS
            order: Task order in process
            
        Returns:
            Transformed task dictionary
        """
        # Extract job information for resource requirements
        job_tasks = task_data.get("jobTasks", [])
        required_skills = []
        
        for jt in job_tasks:
            job = jt.get("job", {})
            skill = {
                "name": job.get("name", ""),
                "level": self._map_skill_level(job.get("job_level_id", 3))
            }
            required_skills.append(skill)
        
        return {
            "id": str(task_data.get("task_id", "")),
            "name": task_data.get("task_name", ""),
            "description": task_data.get("task_overview", ""),
            "duration": task_data.get("task_capacity_minutes", 60),  # Duration in minutes
            "duration_hours": task_data.get("task_capacity_minutes", 60) / 60,  # Duration in hours
            "required_skills": required_skills,
            "dependencies": [],  # Will be filled by dependency inference
            "order": order,
            "code": task_data.get("task_code", "")
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
                dependencies=set(task_data["dependencies"])  # Convert to set
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
            tasks=tasks,
            resources=resources
        )
        
        return process
