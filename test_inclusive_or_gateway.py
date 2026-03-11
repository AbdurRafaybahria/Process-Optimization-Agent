"""
Test Script for Inclusive OR Gateway Detection

This script creates a custom process designed to test Inclusive OR gateway detection.
The process simulates a patient discharge scenario where multiple optional preparations
may be required based on patient conditions.

Scenario: Hospital Patient Discharge Process
===========================================

Process Flow:
1. Patient Assessment (Initial evaluation)
2. ** INCLUSIVE OR GATEWAY ** - Required Preparations
   - Lab Work Required (if needsLabWork == true)
   - Consent Form Collection (if needsConsent == true)   - Pharmacy Medication Review (if hasMedications == true)
   - Physical Therapy Clearance (if needsPT == true)
3. Discharge Documentation (Final paperwork)

This tests the Inclusive OR pattern where:
- Multiple branches CAN execute simultaneously
- NOT all branches always execute (depends on conditions)
- Each branch has a meaningful condition
- At least one branch should be activated (or default path)
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from process_optimization_agent.Optimization.gateways import InclusiveGatewayDetector


def create_patient_discharge_process():
    """
    Create a test process for patient discharge with inclusive OR gateway pattern.
    
    This process has:
    - Task 1: Patient Assessment (decision point)
    - Task 2: Lab Work Required (conditional)
    - Task 3: Consent Form Collection (conditional)
    - Task 4: Pharmacy Review (conditional)
    - Task 5: Physical Therapy Clearance (conditional)
    - Task 6: Discharge Documentation (merger/end point)
    """
    process_data = {
        'process_id': 999,
        'name': 'Hospital Patient Discharge Process',
        'process_task': [
            {
                'order': 1,
                'task': {
                    'task_id': 1,
                    'task_name': 'Patient Assessment and Discharge Planning',
                    'duration_minutes': 30
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 101,
                            'job_name': 'Discharge Nurse',
                            'name': 'Discharge Nurse'
                        }
                    }
                ]
            },
            {
                'order': 2,
                'task': {
                    'task_id': 2,
                    'task_name': 'Lab Work Required for Discharge',
                    'duration_minutes': 45
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 102,
                            'job_name': 'Lab Technician',
                            'name': 'Lab Technician'
                        }
                    }
                ]
            },
            {
                'order': 2,  # Same order as Lab Work - can run in parallel
                'task': {
                    'task_id': 3,
                    'task_name': 'Consent Form Collection',
                    'duration_minutes': 20
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 103,
                            'job_name': 'Patient Services Coordinator',
                            'name': 'Patient Services Coordinator'
                        }
                    }
                ]
            },
            {
                'order': 2,  # Same order - can run in parallel
                'task': {
                    'task_id': 4,
                    'task_name': 'Pharmacy Medication Review',
                    'duration_minutes': 30
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 104,
                            'job_name': 'Clinical Pharmacist',
                            'name': 'Clinical Pharmacist'
                        }
                    }
                ]
            },
            {
                'order': 2,  # Same order - can run in parallel
                'task': {
                    'task_id': 5,
                    'task_name': 'Physical Therapy Clearance Required',
                    'duration_minutes': 40
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 105,
                            'job_name': 'Physical Therapist',
                            'name': 'Physical Therapist'
                        }
                    }
                ]
            },
            {
                'order': 3,
                'task': {
                    'task_id': 6,
                    'task_name': 'Discharge Documentation and Instructions',
                    'duration_minutes': 25
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 101,
                            'job_name': 'Discharge Nurse',
                            'name': 'Discharge Nurse'
                        }
                    }
                ]
            }
        ],
        'constraints': {
            'dependencies': [
                {'from': 1, 'to': 2},  # Lab Work depends on Assessment
                {'from': 1, 'to': 3},  # Consent depends on Assessment
                {'from': 1, 'to': 4},  # Pharmacy depends on Assessment
                {'from': 1, 'to': 5},  # PT depends on Assessment
                {'from': 2, 'to': 6},  # Documentation waits for Lab (if done)
                {'from': 3, 'to': 6},  # Documentation waits for Consent (if done)
                {'from': 4, 'to': 6},  # Documentation waits for Pharmacy (if done)
                {'from': 5, 'to': 6}   # Documentation waits for PT (if done)
            ]
        }
    }
    
    return process_data


def create_notification_process():
    """
    Create a multi-channel notification process (simpler example).
    
    This process has:
    - Task 10: Generate Invoice
    - Task 11: Send Email Notification (conditional)
    - Task 12: Send SMS Notification (conditional)
    - Task 13: Send Push Notification (conditional)
    - Task 14: Log Notification Status
    """
    process_data = {
        'process_id': 888,
        'name': 'Multi-Channel Customer Notification',
        'process_task': [
            {
                'order': 1,
                'task': {
                    'task_id': 10,
                    'task_name': 'Generate Invoice',
                    'duration_minutes': 5
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 201,
                            'job_name': 'Billing System',
                            'name': 'Billing System'
                        }
                    }
                ]
            },
            {
                'order': 2,
                'task': {
                    'task_id': 11,
                    'task_name': 'Send Email Notification',
                    'duration_minutes': 2
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 202,
                            'job_name': 'Email Service',
                            'name': 'Email Service'
                        }
                    }
                ]
            },
            {
                'order': 2,
                'task': {
                    'task_id': 12,
                    'task_name': 'Send SMS Notification',
                    'duration_minutes': 1
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 203,
                            'job_name': 'SMS Gateway',
                            'name': 'SMS Gateway'
                        }
                    }
                ]
            },
            {
                'order': 2,
                'task': {
                    'task_id': 13,
                    'task_name': 'Send Push Notification',
                    'duration_minutes': 1
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 204,
                            'job_name': 'Push Notification Service',
                            'name': 'Push Notification Service'
                        }
                    }
                ]
            },
            {
                'order': 3,
                'task': {
                    'task_id': 14,
                    'task_name': 'Log Notification Status',
                    'duration_minutes': 1
                },
                'process_task_job': [
                    {
                        'job': {
                            'job_id': 205,
                            'job_name': 'Logging System',
                            'name': 'Logging System'
                        }
                    }
                ]
            }
        ],
        'constraints': {
            'dependencies': [
                {'from': 10, 'to': 11},
                {'from': 10, 'to': 12},
                {'from': 10, 'to': 13},
                {'from': 11, 'to': 14},
                {'from': 12, 'to': 14},
                {'from': 13, 'to': 14}
            ]
        }
    }
    
    return process_data


def test_inclusive_or_detection():
    """
    Test the Inclusive OR gateway detector on custom processes.
    """
    print("=" * 80)
    print("INCLUSIVE OR GATEWAY DETECTION TEST")
    print("=" * 80)
    print()
    
    # Initialize detector
    detector = InclusiveGatewayDetector(min_confidence=0.65)
    
    # Test 1: Patient Discharge Process
    print("Test 1: Hospital Patient Discharge Process")
    print("-" * 80)
    discharge_process = create_patient_discharge_process()
    
    print(f"Process: {discharge_process['name']}")
    print(f"Tasks: {len(discharge_process['process_task'])}")
    print()
    
    print("Analyzing process for Inclusive OR gateway opportunities...")
    suggestions = detector.analyze_process(discharge_process)
    
    print(f"\n✅ Found {len(suggestions)} Inclusive OR gateway suggestion(s)\n")
    
    for idx, suggestion in enumerate(suggestions):
        print(f"Gateway #{idx + 1}:")
        print(f"  Type: {suggestion.gateway_type}")
        print(f"  After Task: {suggestion.after_task_name} (ID: {suggestion.after_task_id})")
        print(f"  Confidence: {suggestion.confidence_score:.2%}")
        print(f"  Branches: {len(suggestion.branches)}")
        
        for branch_idx, branch in enumerate(suggestion.branches):
            print(f"    Branch {branch_idx + 1}:")
            print(f"      Task: {branch.task_name}")
            print(f"      Condition: {branch.condition}")
            print(f"      Is Default: {branch.is_default}")
        
        print(f"\n  Justification:")
        print(f"    Pattern Type: {suggestion.justification.get('pattern_type')}")
        print(f"    Matched Patterns: {', '.join(suggestion.justification.get('matched_patterns', []))}")
        print(f"    Reasoning:")
        for reason in suggestion.justification.get('reasoning', []):
            print(f"      - {reason}")
        
        print(f"\n  Confidence Factors:")
        for factor, score in suggestion.justification.get('confidence_factors', {}).items():
            print(f"    {factor}: {score:.3f}")
        
        print()
    
    print("\n" + "=" * 80)
    print()
    
    # Test 2: Notification Process
    print("Test 2: Multi-Channel Customer Notification")
    print("-" * 80)
    notification_process = create_notification_process()
    
    print(f"Process: {notification_process['name']}")
    print(f"Tasks: {len(notification_process['process_task'])}")
    print()
    
    print("Analyzing process for Inclusive OR gateway opportunities...")
    suggestions = detector.analyze_process(notification_process)
    
    print(f"\n✅ Found {len(suggestions)} Inclusive OR gateway suggestion(s)\n")
    
    for idx, suggestion in enumerate(suggestions):
        print(f"Gateway #{idx + 1}:")
        print(f"  Type: {suggestion.gateway_type}")
        print(f"  After Task: {suggestion.after_task_name} (ID: {suggestion.after_task_id})")
        print(f"  Confidence: {suggestion.confidence_score:.2%}")
        print(f"  Branches: {len(suggestion.branches)}")
        
        for branch_idx, branch in enumerate(suggestion.branches):
            print(f"    Branch {branch_idx + 1}:")
            print(f"      Task: {branch.task_name}")
            print(f"      Condition: {branch.condition}")
            print(f"      Is Default: {branch.is_default}")
        
        print(f"\n  Justification:")
        print(f"    Pattern Type: {suggestion.justification.get('pattern_type')}")
        print(f"    Matched Patterns: {', '.join(suggestion.justification.get('matched_patterns', []))}")
        
        print(f"\n  Benefits:")
        for benefit_key, benefit_value in suggestion.benefits.items():
            print(f"    {benefit_key}: {benefit_value}")
        
        print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_inclusive_or_detection()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
