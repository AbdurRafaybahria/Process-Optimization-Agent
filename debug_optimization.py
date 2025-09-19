#!/usr/bin/env python3
"""
Debug script to identify optimization issues
"""
import json
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from process_optimization_agent.models import Process, Task, Resource, Skill, SkillLevel
from process_optimization_agent.optimizers import RLBasedOptimizer

def debug_hospital_data():
    """Debug the hospital consultation data"""
    hospital_data = {
        "id": "11",
        "name": "Outpatient Consultation",
        "process_name": "Outpatient Consultation",
        "process_id": 11,
        "company": "Maldova Hospital",
        "tasks": [
            {
                "id": "46",
                "name": "Schedule Appointment",
                "duration": 33,
                "duration_hours": 0.55,
                "required_skills": [{"name": "Scheduling Coordinator", "level": "advanced"}],
                "dependencies": [],
                "order": 1,
                "code": "HC-SA"
            },
            {
                "id": "48",
                "name": "Call Patient for Consultation",
                "duration": 5,
                "duration_hours": 0.08333333333333333,
                "required_skills": [{"name": "Care Counsellor", "level": "expert"}],
                "dependencies": ["46"],
                "order": 2,
                "code": "HC-CPC"
            },
            {
                "id": "49",
                "name": "Conduct Initial Assessment",
                "duration": 30,
                "duration_hours": 0.5,
                "required_skills": [{"name": "Medical Assistant", "level": "advanced"}],
                "dependencies": ["48"],
                "order": 3,
                "code": "HC-CIA"
            },
            {
                "id": "50",
                "name": "Perform Medical Examination",
                "duration": 40,
                "duration_hours": 0.6666666666666666,
                "required_skills": [{"name": "Doctor", "level": "expert"}],
                "dependencies": ["49"],
                "order": 4,
                "code": "HC-PME"
            },
            {
                "id": "51",
                "name": "Document Consultation Notes",
                "duration": 20,
                "duration_hours": 0.3333333333333333,
                "required_skills": [{"name": "Data Entry Officer", "level": "advanced"}],
                "dependencies": ["50"],
                "order": 5,
                "code": "HC-DC"
            }
        ],
        "resources": [
            {
                "id": "20",
                "name": "Scheduling Coordinator",
                "type": "human",
                "skills": [{"name": "Scheduling Coordinator", "level": "advanced"}],
                "max_hours_per_day": 8,
                "hourly_rate": 23,
                "code": "HC-SC"
            },
            {
                "id": "21",
                "name": "Care Counsellor",
                "type": "human",
                "skills": [{"name": "Care Counsellor", "level": "expert"}],
                "max_hours_per_day": 5,
                "hourly_rate": 55,
                "code": "HC-NCC"
            },
            {
                "id": "22",
                "name": "Medical Assistant",
                "type": "human",
                "skills": [{"name": "Medical Assistant", "level": "advanced"}],
                "max_hours_per_day": 2,
                "hourly_rate": 40,
                "code": "HC-MA"
            },
            {
                "id": "23",
                "name": "Doctor",
                "type": "human",
                "skills": [{"name": "Doctor", "level": "expert"}],
                "max_hours_per_day": 3,
                "hourly_rate": 50,
                "code": "HC-D"
            },
            {
                "id": "24",
                "name": "Data Entry Officer",
                "type": "human",
                "skills": [{"name": "Data Entry Officer", "level": "advanced"}],
                "max_hours_per_day": 3,
                "hourly_rate": 30,
                "code": "HC-DEO"
            }
        ],
        "dependencies": [
            {"from": "46", "to": "48"},
            {"from": "48", "to": "49"},
            {"from": "49", "to": "50"},
            {"from": "50", "to": "51"}
        ]
    }
    
    print("=== Debugging Hospital Consultation Process ===")
    
    # Create process object manually (like in run_rl_optimizer.py)
    from datetime import datetime, timedelta
    
    # Create tasks
    tasks = []
    for task_data in hospital_data['tasks']:
        # Convert required_skills to Skill objects
        required_skills = []
        if task_data.get('required_skills'):
            for skill_data in task_data['required_skills']:
                from process_optimization_agent.models import Skill, SkillLevel
                skill_level = SkillLevel.from_value(skill_data['level'])
                required_skills.append(Skill(name=skill_data['name'], level=skill_level))
        
        task = Task(
            id=task_data['id'],
            name=task_data['name'],
            description=task_data.get('description', ''),
            duration_hours=task_data['duration_hours'],
            required_skills=required_skills,
            dependencies=set(task_data.get('dependencies', [])),
            order=task_data.get('order', 0)
        )
        tasks.append(task)
    
    # Create resources
    resources = []
    for resource_data in hospital_data['resources']:
        # Convert skills to Skill objects
        skills = []
        if resource_data.get('skills'):
            for skill_data in resource_data['skills']:
                skill_level = SkillLevel.from_value(skill_data['level'])
                skills.append(Skill(name=skill_data['name'], level=skill_level))
        
        resource = Resource(
            id=resource_data['id'],
            name=resource_data['name'],
            skills=skills,
            max_hours_per_day=resource_data['max_hours_per_day'],
            hourly_rate=resource_data['hourly_rate']
        )
        resources.append(resource)
    
    # Create process
    start_date = datetime.now()
    target_end_date = start_date + timedelta(days=30)
    project_duration_hours = (target_end_date - start_date).total_seconds() / 3600.0
    
    process = Process(
        id=hospital_data['id'],
        name=hospital_data['name'],
        description=hospital_data.get('description', ''),
        start_date=start_date,
        project_duration_hours=project_duration_hours,
        tasks=tasks,
        resources=resources
    )
    process.target_end_date = target_end_date
    
    print(f"Process: {process.name}")
    print(f"Tasks: {len(process.tasks)}")
    print(f"Resources: {len(process.resources)}")
    
    # Check resource constraints
    print("\n=== Resource Analysis ===")
    for resource in process.resources:
        print(f"Resource: {resource.name}")
        print(f"  Max hours/day: {resource.max_hours_per_day}")
        print(f"  Skills: {[s.name for s in resource.skills]}")
        print(f"  Total available hours: {resource.total_available_hours}")
    
    # Check task requirements
    print("\n=== Task Analysis ===")
    total_duration = 0
    for task in process.tasks:
        print(f"Task: {task.name}")
        print(f"  Duration: {task.duration_hours:.2f} hours")
        print(f"  Required skills: {[s.name for s in task.required_skills] if task.required_skills else 'None'}")
        print(f"  Dependencies: {task.dependencies if hasattr(task, 'dependencies') and task.dependencies else 'None'}")
        total_duration += task.duration_hours
    
    print(f"\nTotal task duration: {total_duration:.2f} hours")
    
    # Check skill matching
    print("\n=== Skill Matching Analysis ===")
    for task in process.tasks:
        print(f"\nTask: {task.name}")
        if task.required_skills:
            for req_skill in task.required_skills:
                matching_resources = []
                for resource in process.resources:
                    if resource.has_skill(req_skill):
                        matching_resources.append(resource.name)
                print(f"  Skill '{req_skill.name}' ({req_skill.level}): {matching_resources}")
                if not matching_resources:
                    print(f"  ⚠️  NO RESOURCES FOUND for skill '{req_skill.name}'!")
        else:
            print("  No specific skills required")
    
    # Try optimization with debug info
    print("\n=== Running Optimization ===")
    optimizer = RLBasedOptimizer(
        learning_rate=0.15,
        epsilon=0.3,
        discount_factor=0.95,
        training_episodes=5,  # Reduced for debugging
        enable_parallel=True,
        max_parallel_tasks=4
    )
    
    try:
        schedule = optimizer.optimize(process)
        if schedule and schedule.entries:
            print(f"SUCCESS: Optimization successful! Scheduled {len(schedule.entries)} tasks")
            for entry in schedule.entries:
                print(f"  {entry.task_id} -> {entry.resource_id} ({entry.start_time} - {entry.end_time})")
        else:
            print("FAILED: Optimization failed - no valid schedule found")
            print("This indicates resource constraints or dependency conflicts")
            
            # Let's debug why it failed
            print("\n=== Debugging Failure ===")
            print("Checking if resources can handle individual tasks...")
            for task in process.tasks:
                print(f"\nTask {task.name} ({task.duration_hours:.2f}h):")
                compatible_resources = []
                for resource in process.resources:
                    if resource.has_all_skills(task.required_skills or []):
                        available_hours = min(resource.max_hours_per_day, resource.total_available_hours)
                        if task.duration_hours <= available_hours:
                            compatible_resources.append(f"{resource.name} ({available_hours}h available)")
                        else:
                            print(f"  {resource.name}: Has skills but insufficient time ({available_hours}h < {task.duration_hours:.2f}h needed)")
                    else:
                        missing_skills = []
                        if task.required_skills:
                            for req_skill in task.required_skills:
                                if not resource.has_skill(req_skill):
                                    missing_skills.append(req_skill.name)
                        print(f"  {resource.name}: Missing skills: {missing_skills}")
                
                if compatible_resources:
                    print(f"  Compatible resources: {compatible_resources}")
                else:
                    print(f"  NO COMPATIBLE RESOURCES FOUND!")
                    
    except Exception as e:
        print(f"ERROR: Optimization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_hospital_data()
