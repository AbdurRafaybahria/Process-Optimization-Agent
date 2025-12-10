"""Test multi-job resolver with FYP process data"""

import json
from process_optimization_agent.Optimization.multi_job_resolver import MultiJobResolver

process_data = {
    'process_id': 423,
    'process_name': 'FYP Proposal Submission and Registration',
    'process_task': [
        {
            'process_task_id': 1078,
            'process_id': 423,
            'task_id': 1150,
            'order': 1,
            'task': {
                'task_id': 1150,
                'task_capacity_minutes': 5,
                'task_name': 'Submit Proposal',
                'task_overview': 'This task involves preparing and submitting the FYP proposal. The submitter collects all required documents, fills out the proposal form, and uploads everything to the academic portal. The system records the submission and notifies the concerned department.',
                'jobTasks': [
                    {'job_id': 419, 'task_id': 1150, 'job': {'job_id': 419, 'name': 'Student', 'hourlyRate': 0, 'description': 'The project owner responsible for the creation, development, and formal submission of all project documentation.'}},
                    {'job_id': 425, 'task_id': 1150, 'job': {'job_id': 425, 'name': 'Coordinator', 'hourlyRate': 5000, 'description': 'The process manager responsible for the high-level logistical oversight.'}},
                    {'job_id': 530, 'task_id': 1150, 'job': {'job_id': 530, 'name': 'System', 'hourlyRate': 0, 'description': 'The automated component responsible for processing submitted data, validating required fields, generating tracking information, routing proposals and sending notifications.'}}
                ]
            }
        },
        {
            'process_task_id': 1079,
            'process_id': 423,
            'task_id': 1152,
            'order': 2,
            'task': {
                'task_id': 1152,
                'task_capacity_minutes': 10,
                'task_name': 'Log and Triage Proposal',
                'task_overview': 'System ingests the proposal, creates tracking ID and timestamp, performs automated validation and notifies admin; admin reviews and marks status.',
                'jobTasks': [
                    {'job_id': 425, 'task_id': 1152, 'job': {'job_id': 425, 'name': 'Coordinator', 'hourlyRate': 5000, 'description': 'The process manager responsible for the high-level logistical oversight.'}}
                ]
            }
        }
    ]
}

# Run the resolver
resolver = MultiJobResolver(best_fit_threshold=0.90)
resolved = resolver.resolve_process(process_data)

resolutions = resolved.get('_multi_job_resolutions', [])

print('='*70)
print('MULTI-JOB RESOLUTION ANALYSIS')
print('='*70)

for r in resolutions:
    print(f"\nTask: {r['task_name']} (ID: {r['task_id']})")
    print(f"Resolution: {r['resolution']}")
    print(f"Reason: {r['reason']}")
    
    if r.get('skill_analysis'):
        sa = r['skill_analysis']
        print(f"Required Skills: {sa.get('required_skills', [])}")
        for jm in sa.get('job_matches', []):
            print(f"  - {jm['job_name']}: {jm['match_percentage']}% match")
            print(f"    Matched: {jm['matched_skills']}")
            print(f"    Unmatched: {jm['unmatched_skills']}")
    
    if r.get('kept_jobs'):
        kept = [j['name'] for j in r['kept_jobs']]
        print(f"Kept Jobs: {kept}")
    if r.get('removed_jobs'):
        removed = [j['name'] for j in r['removed_jobs']]
        print(f"Removed Jobs: {removed}")
    if r.get('sub_tasks'):
        print("Sub-Tasks:")
        for st in r['sub_tasks']:
            print(f"  - {st['name']} -> {st['job']}")

print('\n' + '='*70)
print('RESOLVED TASKS')
print('='*70)

for pt in resolved.get('process_task', []):
    task = pt.get('task', {})
    jobs = task.get('jobTasks', [])
    job_name = jobs[0]['job']['name'] if jobs else 'N/A'
    is_sub = task.get('is_sub_task', False)
    print(f"\n{task.get('task_name')} (Order: {pt.get('order')})")
    print(f"  Assigned Job: {job_name}")
    print(f"  Is Sub-Task: {is_sub}")
    if task.get('parent_task_id'):
        print(f"  Parent Task: {task.get('parent_task_id')}")
    if task.get('_resolution'):
        print(f"  Resolution: {task['_resolution']}")
