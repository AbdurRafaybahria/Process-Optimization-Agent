"""
Test file for Exclusive Gateway (XOR) Detection

This script tests the XOR gateway detector with sample insurance process data.
It verifies that decision points are correctly identified and appropriate
exclusive gateway suggestions are generated.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_optimization_agent.Optimization.exclusive_gateway_detector import (
    ExclusiveGatewayDetector, DecisionPoint, ExclusiveBranch
)
from process_optimization_agent.Optimization.parallel_gateway_detector import ParallelGatewayDetector
import json


# Sample CMS data for Insurance Policy Underwriting process
SAMPLE_CMS_DATA = {
    "id": 71,
    "name": "Insurance Policy Underwriting",
    "companyId": 1,
    "tasks": [
        {
            "id": 12,
            "name": "Insurance Policy Risk Assessment",
            "duration": 40,
            "order": 1,
            "processId": 71,
            "taskJobs": [
                {
                    "id": 100,
                    "jobId": 258,
                    "job": {
                        "id": 258,
                        "name": "Job Insurance Policy Risk",
                        "hourlyRate": 10
                    }
                }
            ]
        },
        {
            "id": 13,
            "name": "Insurance Policy Decision-Making - Review & Approval",
            "duration": 30,
            "order": 2,
            "processId": 71,
            "taskJobs": [
                {
                    "id": 101,
                    "jobId": 552,
                    "job": {
                        "id": 552,
                        "name": "Insurance Policy Underwriting Decision Making",
                        "hourlyRate": 45
                    }
                },
                {
                    "id": 102,
                    "jobId": 17,
                    "job": {
                        "id": 17,
                        "name": "Insurance Policy Underwriting Decision Making Specialist",
                        "hourlyRate": 35
                    }
                }
            ]
        },
        {
            "id": 14,
            "name": "Insurance Policy Underwriting Pricing Calculation",
            "duration": 36,
            "order": 3,
            "processId": 71,
            "taskJobs": [
                {
                    "id": 103,
                    "jobId": 551,
                    "job": {
                        "id": 551,
                        "name": "Insurance Policy Pricing Underwriter",
                        "hourlyRate": 48
                    }
                }
            ]
        },
        {
            "id": 15,
            "name": "Insurance Policy Underwriting Documentation",
            "duration": 12,
            "order": 4,
            "processId": 71,
            "taskJobs": [
                {
                    "id": 104,
                    "jobId": 229,
                    "job": {
                        "id": 229,
                        "name": "Insurance Policy Underwriting Documentation",
                        "hourlyRate": 31
                    }
                }
            ]
        },
        {
            "id": 450,
            "name": "Insurance Policy UW Approval & Communication",
            "duration": 35,
            "order": 5,
            "processId": 71,
            "taskJobs": [
                {
                    "id": 105,
                    "jobId": 511,
                    "job": {
                        "id": 511,
                        "name": "Policy Underwriting Apprv/Commnction Officer",
                        "hourlyRate": 58
                    }
                },
                {
                    "id": 106,
                    "jobId": 550,
                    "job": {
                        "id": 550,
                        "name": "Ins. Policy UW Approval & Communication",
                        "hourlyRate": 44
                    }
                }
            ]
        }
    ]
}

# Sample with more decision points for comprehensive testing
SAMPLE_WITH_DECISIONS = {
    "id": 72,
    "name": "Loan Application Process",
    "companyId": 1,
    "task_assignments": [
        {
            "task_id": "1",
            "task_name": "Application Submission",
            "resource_name": "Application Processor",
            "resource_id": "100",
            "duration_minutes": 15,
            "duration_hours": 0.25
        },
        {
            "task_id": "2",
            "task_name": "Initial Eligibility Check",
            "resource_name": "Eligibility Validator",
            "resource_id": "101",
            "duration_minutes": 20,
            "duration_hours": 0.33
        },
        {
            "task_id": "3",
            "task_name": "Credit Score Assessment",
            "resource_name": "Credit Analyst",
            "resource_id": "102",
            "duration_minutes": 30,
            "duration_hours": 0.5
        },
        {
            "task_id": "4",
            "task_name": "Risk Evaluation & Decision",
            "resource_name": "Risk Evaluator",
            "resource_id": "103",
            "duration_minutes": 45,
            "duration_hours": 0.75
        },
        {
            "task_id": "5",
            "task_name": "Loan Approval Processing",
            "resource_name": "Loan Processor",
            "resource_id": "104",
            "duration_minutes": 25,
            "duration_hours": 0.42
        },
        {
            "task_id": "6",
            "task_name": "Rejection Notice Preparation",
            "resource_name": "Notification Officer",
            "resource_id": "105",
            "duration_minutes": 10,
            "duration_hours": 0.17
        },
        {
            "task_id": "7",
            "task_name": "Additional Review Required",
            "resource_name": "Senior Analyst",
            "resource_id": "106",
            "duration_minutes": 60,
            "duration_hours": 1.0
        }
    ],
    "parallel_execution": {
        "parallel_groups": [
            {
                "task_details": [
                    {"task_id": "5", "has_dependencies": True, "dependency_chain": ["4"]},
                    {"task_id": "6", "has_dependencies": True, "dependency_chain": ["4"]},
                    {"task_id": "7", "has_dependencies": True, "dependency_chain": ["4"]}
                ]
            }
        ]
    }
}


def print_separator(title: str):
    """Print a visual separator"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_exclusive_gateway_detection():
    """Test the exclusive gateway detector"""
    print_separator("EXCLUSIVE GATEWAY (XOR) DETECTION TEST")
    
    # Initialize detector
    xor_detector = ExclusiveGatewayDetector(min_confidence=0.5)  # Lower threshold for testing
    
    print("\n📋 Testing with Insurance Policy Underwriting process...")
    print(f"   Process ID: {SAMPLE_CMS_DATA['id']}")
    print(f"   Process Name: {SAMPLE_CMS_DATA['name']}")
    print(f"   Tasks: {len(SAMPLE_CMS_DATA['tasks'])}")
    
    # Analyze process
    suggestions = xor_detector.analyze_process(SAMPLE_CMS_DATA)
    
    print(f"\n🔍 Analysis Results:")
    print(f"   Decision Points Found: {len(suggestions)}")
    
    if suggestions:
        for idx, suggestion in enumerate(suggestions, 1):
            print(f"\n   📌 Suggestion #{idx}:")
            print(f"      After Task: {suggestion.after_task_id} ({suggestion.after_task_name})")
            print(f"      Gateway Type: {suggestion.gateway_type}")
            print(f"      Confidence: {suggestion.confidence_score:.2%}")
            print(f"      Branches: {len(suggestion.branches)}")
            
            for branch in suggestion.branches:
                print(f"\n         🔸 Branch: {branch.branch_id}")
                print(f"            Target Task: {branch.target_task_id} ({branch.task_name})")
                print(f"            Condition: {branch.condition or '(default)'}")
                print(f"            Is Default: {branch.is_default}")
                print(f"            Probability: {branch.probability:.2%}")
            
            print(f"\n      Justification:")
            for key, value in suggestion.justification.items():
                print(f"         {key}: {value}")
    else:
        print("   ⚠️  No exclusive gateway opportunities found in this process")
    
    return suggestions


def test_with_loan_process():
    """Test with a process that has clearer decision points"""
    print_separator("LOAN APPLICATION PROCESS TEST")
    
    xor_detector = ExclusiveGatewayDetector(min_confidence=0.5)
    
    print("\n📋 Testing with Loan Application process...")
    print(f"   Process ID: {SAMPLE_WITH_DECISIONS['id']}")
    print(f"   Process Name: {SAMPLE_WITH_DECISIONS['name']}")
    print(f"   Tasks: {len(SAMPLE_WITH_DECISIONS['task_assignments'])}")
    
    suggestions = xor_detector.analyze_process(SAMPLE_WITH_DECISIONS)
    
    print(f"\n🔍 Analysis Results:")
    print(f"   Decision Points Found: {len(suggestions)}")
    
    if suggestions:
        for idx, suggestion in enumerate(suggestions, 1):
            print(f"\n   📌 XOR Gateway Suggestion #{idx}:")
            print(f"      Location: After Task {suggestion.after_task_id}")
            print(f"      Task Name: {suggestion.after_task_name}")
            print(f"      Confidence: {suggestion.confidence_score:.2%}")
            print(f"      Number of Branches: {len(suggestion.branches)}")
            
            for branch in suggestion.branches:
                default_marker = " (DEFAULT)" if branch.is_default else ""
                print(f"\n         🔀 {branch.task_name}{default_marker}")
                print(f"            Condition: {branch.condition or 'else (default path)'}")
                print(f"            Probability: {branch.probability:.2%}")
    
    return suggestions


def test_api_format():
    """Test the API response format"""
    print_separator("API RESPONSE FORMAT TEST")
    
    xor_detector = ExclusiveGatewayDetector(min_confidence=0.5)
    suggestions = xor_detector.analyze_process(SAMPLE_WITH_DECISIONS)
    
    api_response = xor_detector.format_suggestions_for_api(
        suggestions,
        SAMPLE_WITH_DECISIONS['id'],
        SAMPLE_WITH_DECISIONS['name']
    )
    
    print("\n📄 API Response Structure:")
    print(json.dumps(api_response, indent=2, default=str))
    
    return api_response


def test_cms_format():
    """Test the CMS database format"""
    print_separator("CMS DATABASE FORMAT TEST")
    
    xor_detector = ExclusiveGatewayDetector(min_confidence=0.5)
    suggestions = xor_detector.analyze_process(SAMPLE_WITH_DECISIONS)
    
    if suggestions:
        cms_format = xor_detector.format_for_cms(
            suggestions,
            SAMPLE_WITH_DECISIONS['id'],
            SAMPLE_WITH_DECISIONS['name']
        )
        
        print("\n📄 CMS Database Format:")
        print(json.dumps(cms_format, indent=2, default=str))
    else:
        print("\n⚠️  No suggestions to format for CMS")


def test_both_gateways():
    """Test both parallel and exclusive gateways together"""
    print_separator("COMBINED GATEWAY ANALYSIS")
    
    parallel_detector = ParallelGatewayDetector(min_confidence=0.7)
    xor_detector = ExclusiveGatewayDetector(min_confidence=0.5)
    
    print("\n📋 Analyzing Insurance Policy Underwriting process for ALL gateway types...")
    
    # Parallel gateway analysis
    parallel_suggestions = parallel_detector.analyze_process(SAMPLE_CMS_DATA)
    
    # Exclusive gateway analysis
    xor_suggestions = xor_detector.analyze_process(SAMPLE_CMS_DATA)
    
    print(f"\n📊 Combined Results:")
    print(f"   ⊕ Parallel Gateways (AND): {len(parallel_suggestions)} opportunities")
    print(f"   ⊗ Exclusive Gateways (XOR): {len(xor_suggestions)} opportunities")
    print(f"   Total Gateway Opportunities: {len(parallel_suggestions) + len(xor_suggestions)}")
    
    if parallel_suggestions:
        print(f"\n   ⊕ PARALLEL Gateway Details:")
        for sug in parallel_suggestions:
            print(f"      After Task {sug.after_task_id}: {len(sug.branches)} parallel branches")
    
    if xor_suggestions:
        print(f"\n   ⊗ EXCLUSIVE Gateway Details:")
        for sug in xor_suggestions:
            print(f"      After Task {sug.after_task_id}: {len(sug.branches)} exclusive branches")
    
    return parallel_suggestions, xor_suggestions


def main():
    """Run all tests"""
    print("\n" + "🚀" * 40)
    print("  EXCLUSIVE GATEWAY (XOR) DETECTOR - TEST SUITE")
    print("🚀" * 40)
    
    # Test 1: Basic detection on insurance process
    test_exclusive_gateway_detection()
    
    # Test 2: Loan process with clearer decision points
    test_with_loan_process()
    
    # Test 3: API format
    test_api_format()
    
    # Test 4: CMS format
    test_cms_format()
    
    # Test 5: Both gateway types
    test_both_gateways()
    
    print_separator("TEST SUITE COMPLETE")
    print("\n✅ All tests completed successfully!")
    print("\n📌 Summary of XOR Gateway Detection:")
    print("   - Detects decision points (approval, review, validation, etc.)")
    print("   - Identifies mutually exclusive branches")
    print("   - Generates conditions for each branch")
    print("   - Calculates probability distribution")
    print("   - Formats for both API response and CMS database")


if __name__ == "__main__":
    main()
