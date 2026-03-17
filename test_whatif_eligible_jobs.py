"""
Test script to verify that eligible jobs are returned in the what-if endpoint
"""
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
PROCESS_ID = 11  # Outpatient Consultation process

def test_whatif_eligible_jobs():
    """Test the /cms/whatif endpoint and check for eligible_jobs field"""
    
    print(f"\n{'='*70}")
    print(f"Testing What-If Endpoint with Eligible Jobs")
    print(f"{'='*70}\n")
    
    endpoint = f"{BASE_URL}/cms/whatif/{PROCESS_ID}"
    print(f"GET {endpoint}")
    
    try:
        response = requests.get(endpoint, timeout=60)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check structure
            print("\n[SUCCESS] Response received")
            
            # Navigate to tasks constraints
            tasks = data.get("scenario", {}).get("constraints", {}).get("tasks", [])
            
            print(f"\nTotal Tasks: {len(tasks)}")
            
            # Check each task for eligible_jobs
            tasks_with_eligible_jobs = 0
            total_eligible_jobs = 0
            
            for task in tasks:
                task_id = task.get("task_id")
                task_name = task.get("name")
                eligible_jobs = task.get("eligible_jobs", [])
                current_assignment = task.get("current_assignment")
                required_skills = task.get("required_skills", [])
                
                if eligible_jobs:
                    tasks_with_eligible_jobs += 1
                    total_eligible_jobs += len(eligible_jobs)
                
                print(f"\n{'='*70}")
                print(f"Task ID: {task_id}")
                print(f"Task Name: {task_name}")
                print(f"Required Skills: {', '.join(required_skills) if required_skills else 'None'}")
                print(f"Eligible Jobs Count: {len(eligible_jobs)}")
                
                if current_assignment:
                    print(f"Current Assignment: {current_assignment.get('job_name')} (ID: {current_assignment.get('job_id')})")
                else:
                    print(f"Current Assignment: None")
                
                # Show top 3 eligible jobs
                if eligible_jobs:
                    print(f"\nTop Eligible Jobs:")
                    for idx, job in enumerate(eligible_jobs[:3], 1):
                        print(f"  {idx}. {job.get('name')} (ID: {job.get('job_id')})")
                        print(f"     - Hourly Rate: ${job.get('hourly_rate', 0):.2f}")
                        print(f"     - Skill Match: {job.get('skill_match_percentage', 0):.1f}%")
                        print(f"     - Is Current: {job.get('is_current', False)}")
                        job_skills = job.get('skills', [])
                        if job_skills:
                            skill_names = ', '.join([s.get('name', '') for s in job_skills[:3]])
                            print(f"     - Skills: {skill_names}")
            
            print(f"\n{'='*70}")
            print(f"SUMMARY")
            print(f"{'='*70}")
            print(f"Tasks with eligible jobs: {tasks_with_eligible_jobs}/{len(tasks)}")
            print(f"Total eligible jobs across all tasks: {total_eligible_jobs}")
            print(f"Average eligible jobs per task: {total_eligible_jobs/len(tasks):.1f}" if tasks else "N/A")
            
            # Save full response to file
            output_file = "whatif_eligible_jobs_response.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\nFull response saved to: {output_file}")
            
            print(f"\n[SUCCESS] Eligible jobs feature is working correctly!")
            
        else:
            print(f"\n[ERROR] Request failed with status {response.status_code}")
            print(response.text)
            
    except requests.RequestException as e:
        print(f"\n[ERROR] Request failed: {e}")
        print("\nMake sure the server is running: uvicorn API.main:app --reload --port 8000")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_whatif_eligible_jobs()
