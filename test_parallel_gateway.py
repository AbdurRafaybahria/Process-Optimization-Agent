"""
Test script for Parallel Gateway Detection functionality
"""

import json
from process_optimization_agent.Optimization.parallel_gateway_detector import ParallelGatewayDetector

# Sample CMS process data (Insurance Policy Underwriting)
cms_test_data = {
    "process_id": 71,
    "company_id": 3,
    "process_code": "INSPOLOW",
    "process_name": "Insurance Policy Underwriting",
    "capacity_requirement_minutes": 153,
    "process_task": [
        {
            "process_task_id": 692,
            "process_id": 71,
            "task_id": 12,
            "order": 1,
            "task": {
                "task_id": 12,
                "task_code": "InsPlRkAsm",
                "task_name": "Insurance Policy Risk Assessment",
                "task_capacity_minutes": 40,
                "jobTasks": [
                    {"job_id": 16, "task_id": 12}
                ]
            }
        },
        {
            "process_task_id": 693,
            "process_id": 71,
            "task_id": 13,
            "order": 2,
            "task": {
                "task_id": 13,
                "task_code": "InsPolDecM",
                "task_name": "Insurance Policy Decision-Making",
                "task_capacity_minutes": 30,
                "jobTasks": [
                    {"job_id": 17, "task_id": 13},
                    {"job_id": 552, "task_id": 13}
                ]
            }
        },
        {
            "process_task_id": 694,
            "process_id": 71,
            "task_id": 14,
            "order": 3,
            "task": {
                "task_id": 14,
                "task_code": "InsPolUnPc",
                "task_name": "Insurance Policy Underwriting Pricing",
                "task_capacity_minutes": 36,
                "jobTasks": [
                    {"job_id": 18, "task_id": 14},
                    {"job_id": 551, "task_id": 14}
                ]
            }
        },
        {
            "process_task_id": 695,
            "process_id": 71,
            "task_id": 15,
            "order": 4,
            "task": {
                "task_id": 15,
                "task_code": "InsPolUdDc",
                "task_name": "Insurance Policy Underwriting Documentation",
                "task_capacity_minutes": 12,
                "jobTasks": [
                    {"job_id": 19, "task_id": 15}
                ]
            }
        },
        {
            "process_task_id": 696,
            "process_id": 71,
            "task_id": 450,
            "order": 5,
            "task": {
                "task_id": 450,
                "task_code": "InsPolUdCm",
                "task_name": "Insurance Policy UW Approval & Communication",
                "task_capacity_minutes": 35,
                "jobTasks": [
                    {"job_id": 511, "task_id": 450},
                    {"job_id": 550, "task_id": 450}
                ]
            }
        }
    ]
}


def test_parallel_gateway_detection():
    """Test the parallel gateway detection functionality"""
    
    print("=" * 80)
    print("Testing Parallel Gateway Detection")
    print("=" * 80)
    print()
    
    # Initialize detector
    detector = ParallelGatewayDetector(min_confidence=0.7)
    
    # Analyze process
    print("Analyzing process: Insurance Policy Underwriting")
    print(f"Total tasks: {len(cms_test_data['process_task'])}")
    print()
    
    suggestions = detector.analyze_process(cms_test_data)
    
    print(f"Found {len(suggestions)} parallel gateway opportunities")
    print()
    
    # Display each suggestion
    for idx, suggestion in enumerate(suggestions, start=1):
        print(f"\n{'=' * 80}")
        print(f"Suggestion #{idx}")
        print(f"{'=' * 80}")
        print(f"Location: After Task {suggestion.after_task_id} - {suggestion.after_task_name}")
        print(f"Gateway Type: {suggestion.gateway_type}")
        print(f"Gateway Name: {suggestion.gateway_name}")
        print(f"Confidence Score: {suggestion.confidence_score:.2%}")
        print()
        
        print("Parallel Branches:")
        for branch in suggestion.branches:
            if branch.target_task_id:
                print(f"  - Branch {branch.branch_id}: Execute {branch.task_name}")
                print(f"    Duration: {branch.task_duration_minutes} minutes")
                print(f"    Assigned Jobs: {branch.assigned_jobs}")
            else:
                print(f"  - Branch {branch.branch_id}: {branch.description}")
                print(f"    Condition: {branch.condition}")
        print()
        
        print("Benefits:")
        benefits = suggestion.benefits
        print(f"  - Time Saved: {benefits['time_saved_minutes']} minutes")
        print(f"  - Before Duration: {benefits['before_duration_minutes']} minutes")
        print(f"  - After Duration: {benefits['after_duration_minutes']} minutes")
        print(f"  - Efficiency Gain: {benefits['efficiency_gain_percent']:.2f}%")
        print()
        
        print("Justification:")
        justification = suggestion.justification
        print(f"  Why Parallel: {justification['why_parallel']}")
        print(f"  Independence Factors:")
        for factor in justification['independence_factors']:
            print(f"    • {factor}")
        print(f"  Downstream Impact: {justification['downstream_impact']}")
        print()
        
        print("Implementation Notes:")
        impl_notes = suggestion.implementation_notes
        print(f"  Next Task: Task {impl_notes['next_task_id']} - {impl_notes['next_task_name']}")
        print(f"  Prerequisites: {impl_notes['next_task_prerequisites']}")
        print(f"  Dependency Update: {impl_notes['task_dependency_update']}")
    
    # Format for API
    print(f"\n{'=' * 80}")
    print("API Response Format")
    print(f"{'=' * 80}")
    
    api_response = detector.format_suggestions_for_api(suggestions, cms_test_data)
    print(json.dumps(api_response, indent=2))
    
    # Format for CMS database
    print(f"\n{'=' * 80}")
    print("CMS Database Format")
    print(f"{'=' * 80}")
    
    for suggestion in suggestions:
        cms_format = detector.format_for_cms(suggestion, cms_test_data['process_id'])
        print(json.dumps(cms_format, indent=2))
        print()


if __name__ == "__main__":
    test_parallel_gateway_detection()
