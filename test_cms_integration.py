"""
Test script for CMS integration with the Process Optimization Agent
"""

import json
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from process_optimization_agent.cms_transformer import CMSDataTransformer
from process_optimization_agent.models import Process

# Sample CMS data provided by user
sample_cms_data = {
    "id": 7,
    "name": "E-Commerce Platform Development",
    "description": "Build a modern e-commerce platform with user management, product, catalog, and payment processing.",
    "company": "Crystal System Pakistan",
    "tasks_count": 6,
    "type": "cms",
    "process_data": {
        "process_id": 7,
        "process_name": "E-Commerce Platform Development",
        "process_code": "ECPD-001",
        "company_id": 2,
        "capacity_requirement_minutes": 207,
        "process_overview": "Build a modern e-commerce platform with user management, product, catalog, and payment processing.",
        "parent_process_id": None,
        "parent_task_id": None,
        "created_at": "2025-09-08T06:56:14.275Z",
        "updated_at": "2025-09-08T06:56:35.970Z",
        "company": {
            "company_id": 2,
            "companyCode": "CSP",
            "name": "Crystal System Pakistan",
            "created_by": 1,
            "created_at": "2025-08-28T15:59:06.000Z",
            "updated_at": "2025-08-29T05:54:25.064Z"
        },
        "parent_process": None,
        "parent_task": None,
        "process_tasks": [
            {
                "process_id": 7,
                "task_id": 10,
                "order": 1,
                "task": {
                    "task_id": 10,
                    "task_name": "Database Design and Setup",
                    "task_code": "DBDS-01",
                    "task_company_id": 2,
                    "task_capacity_minutes": 32,
                    "task_process_id": None,
                    "task_overview": "Design database schema and set up initial database structure.",
                    "created_at": "2025-09-08T06:38:16.043Z",
                    "updated_at": "2025-09-08T06:38:16.043Z",
                    "jobTasks": [
                        {
                            "job_id": 11,
                            "task_id": 10,
                            "job": {
                                "job_id": 11,
                                "jobCode": "DBA-01",
                                "name": "Database Admin",
                                "description": "Database design and optimization specialist.",
                                "hourlyRate": 88,
                                "maxHoursPerDay": 6,
                                "function_id": 3,
                                "company_id": 2,
                                "job_level_id": 3,
                                "createdAt": "2025-09-08T06:34:02.081Z",
                                "updatedAt": "2025-09-08T06:34:02.081Z"
                            }
                        }
                    ]
                }
            },
            {
                "process_id": 7,
                "task_id": 11,
                "order": 2,
                "task": {
                    "task_id": 11,
                    "task_name": "User Authentication",
                    "task_code": "UA-01",
                    "task_company_id": 2,
                    "task_capacity_minutes": 24,
                    "task_process_id": None,
                    "task_overview": "Implement user registration login and authentication.",
                    "created_at": "2025-09-08T06:43:04.108Z",
                    "updated_at": "2025-09-08T06:43:04.108Z",
                    "jobTasks": [
                        {
                            "job_id": 6,
                            "task_id": 11,
                            "job": {
                                "job_id": 6,
                                "jobCode": "SB-01",
                                "name": "Senior Backend",
                                "description": "Lead Backend developer with API experitse.",
                                "hourlyRate": 94,
                                "maxHoursPerDay": 8,
                                "function_id": 3,
                                "company_id": 2,
                                "job_level_id": 4,
                                "createdAt": "2025-09-08T06:15:00.339Z",
                                "updatedAt": "2025-09-08T06:15:00.339Z"
                            }
                        }
                    ]
                }
            },
            {
                "process_id": 7,
                "task_id": 12,
                "order": 3,
                "task": {
                    "task_id": 12,
                    "task_name": "Product Catalog API",
                    "task_code": "PCAPI-01",
                    "task_company_id": 2,
                    "task_capacity_minutes": 40,
                    "task_process_id": None,
                    "task_overview": "Build Rest APIs for product management and catalog.",
                    "created_at": "2025-09-08T06:45:29.194Z",
                    "updated_at": "2025-09-08T06:45:29.194Z",
                    "jobTasks": [
                        {
                            "job_id": 8,
                            "task_id": 12,
                            "job": {
                                "job_id": 8,
                                "jobCode": "FSD-01",
                                "name": "Fullstack Developer",
                                "description": "Versatile developer with both frontend and backend skills.",
                                "hourlyRate": 80,
                                "maxHoursPerDay": 8,
                                "function_id": 3,
                                "company_id": 2,
                                "job_level_id": 3,
                                "createdAt": "2025-09-08T06:26:59.233Z",
                                "updatedAt": "2025-09-08T06:26:59.233Z"
                            }
                        }
                    ]
                }
            },
            {
                "process_id": 7,
                "task_id": 13,
                "order": 4,
                "task": {
                    "task_id": 13,
                    "task_name": "Frontend UI",
                    "task_code": "FUI-01",
                    "task_company_id": 2,
                    "task_capacity_minutes": 47,
                    "task_process_id": None,
                    "task_overview": "Create a responsive frontend with React/Vue.",
                    "created_at": "2025-09-08T06:47:33.469Z",
                    "updated_at": "2025-09-08T06:47:33.469Z",
                    "jobTasks": [
                        {
                            "job_id": 7,
                            "task_id": 13,
                            "job": {
                                "job_id": 7,
                                "jobCode": "FD-01",
                                "name": "Frontend Developer",
                                "description": "Frontend specialist with modern frameworks",
                                "hourlyRate": 85,
                                "maxHoursPerDay": 8,
                                "function_id": 3,
                                "company_id": 2,
                                "job_level_id": 3,
                                "createdAt": "2025-09-08T06:21:36.188Z",
                                "updatedAt": "2025-09-08T06:21:36.188Z"
                            }
                        }
                    ]
                }
            },
            {
                "process_id": 7,
                "task_id": 14,
                "order": 5,
                "task": {
                    "task_id": 14,
                    "task_name": "Payment Integration",
                    "task_code": "PI-01",
                    "task_company_id": 2,
                    "task_capacity_minutes": 28,
                    "task_process_id": None,
                    "task_overview": "Integrate Payment gateways and checkout process.",
                    "created_at": "2025-09-08T06:50:49.180Z",
                    "updated_at": "2025-09-08T06:50:49.180Z",
                    "jobTasks": [
                        {
                            "job_id": 9,
                            "task_id": 14,
                            "job": {
                                "job_id": 9,
                                "jobCode": "PS-01",
                                "name": "Payment Specialist",
                                "description": "Specialist in payment gateway integration",
                                "hourlyRate": 99,
                                "maxHoursPerDay": 6,
                                "function_id": 3,
                                "company_id": 2,
                                "job_level_id": 4,
                                "createdAt": "2025-09-08T06:29:35.322Z",
                                "updatedAt": "2025-09-08T06:29:35.322Z"
                            }
                        }
                    ]
                }
            },
            {
                "process_id": 7,
                "task_id": 15,
                "order": 6,
                "task": {
                    "task_id": 15,
                    "task_name": "Testing and QA",
                    "task_code": "TQA-01",
                    "task_company_id": 2,
                    "task_capacity_minutes": 36,
                    "task_process_id": None,
                    "task_overview": "Comprehensive testing including unit integration and system level testing.",
                    "created_at": "2025-09-08T06:52:39.166Z",
                    "updated_at": "2025-09-08T06:52:39.166Z",
                    "jobTasks": [
                        {
                            "job_id": 10,
                            "task_id": 15,
                            "job": {
                                "job_id": 10,
                                "jobCode": "QAE-01",
                                "name": "QA Engineer",
                                "description": "Quality assurance and test automation expert.",
                                "hourlyRate": 75,
                                "maxHoursPerDay": 7,
                                "function_id": 3,
                                "company_id": 2,
                                "job_level_id": 3,
                                "createdAt": "2025-09-08T06:31:50.065Z",
                                "updatedAt": "2025-09-08T06:31:50.065Z"
                            }
                        }
                    ]
                }
            }
        ]
    }
}


def test_transformation():
    """Test the CMS data transformation"""
    print("Testing CMS Data Transformation...")
    print("=" * 50)
    
    transformer = CMSDataTransformer()
    
    # Transform the CMS data
    agent_format = transformer.transform_process(sample_cms_data)
    
    # Display transformed data
    print("\n1. Basic Process Information:")
    print(f"   Process Name: {agent_format['process_name']}")
    print(f"   Process ID: {agent_format['process_id']}")
    print(f"   Company: {agent_format['company']}")
    print(f"   Description: {agent_format['description']}")
    
    print("\n2. Tasks ({} total):".format(len(agent_format['tasks'])))
    for task in agent_format['tasks']:
        print(f"   - Task {task['order']}: {task['name']} (ID: {task['id']})")
        print(f"     Duration: {task['duration']} minutes")
        print(f"     Skills Required: {[s['name'] for s in task['required_skills']]}")
        if task['dependencies']:
            print(f"     Dependencies: {task['dependencies']}")
    
    print("\n3. Resources ({} total):".format(len(agent_format['resources'])))
    for resource in agent_format['resources']:
        print(f"   - {resource['name']} (ID: {resource['id']})")
        print(f"     Cost: ${resource.get('hourly_rate', resource.get('cost_per_hour', 50))}/hour")
        print(f"     Max Hours/Day: {resource.get('max_hours_per_day', 8)} hours")
        print(f"     Skills: {[s['name'] for s in resource['skills']]}")
    
    print("\n4. Dependencies ({} total):".format(len(agent_format['dependencies'])))
    for dep in agent_format['dependencies']:
        from_task = next((t['name'] for t in agent_format['tasks'] if t['id'] == dep['from']), dep['from'])
        to_task = next((t['name'] for t in agent_format['tasks'] if t['id'] == dep['to']), dep['to'])
        print(f"   - {from_task} -> {to_task}")
    
    # Save transformed data to file
    output_file = os.path.join(PROJECT_ROOT, "outputs", "cms_transformed_sample.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(agent_format, f, indent=2)
    print(f"\n5. Transformed data saved to: {output_file}")
    
    # Test creating Process object
    try:
        process = transformer.create_process_object(sample_cms_data)
        print(f"\n6. Successfully created Process object: {process.name}")
        print(f"   - Tasks: {len(process.tasks)}")
        print(f"   - Resources: {len(process.resources)}")
    except Exception as e:
        print(f"\n6. Error creating Process object: {e}")
    
    return agent_format


if __name__ == "__main__":
    test_transformation()
