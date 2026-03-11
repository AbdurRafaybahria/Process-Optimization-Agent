"""
Comprehensive Test for All Gateway Types (Exclusive, Parallel, Inclusive OR)
Tests whether all three gateway types can coexist and be detected successfully.
"""

import json
from process_optimization_agent.Optimization.gateways import (
    ExclusiveGatewayDetector,
    ParallelGatewayDetector,
    InclusiveGatewayDetector
)


def load_test_process():
    """Load the testGatewaysProcess from JSON file"""
    with open('examples/test_gateways_process.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_gateway_details(gateway_type, suggestions):
    """Print detailed information about detected gateways"""
    if not suggestions:
        print(f"\n[X] No {gateway_type} gateways detected")
        return
    
    print(f"\n[OK] Detected {len(suggestions)} {gateway_type} gateway(s):")
    
    for idx, suggestion in enumerate(suggestions, 1):
        print(f"\n{gateway_type} Gateway #{idx}:")
        print(f"  After Task ID: {suggestion.after_task_id}")
        print(f"  After Task Name: {suggestion.after_task_name}")
        print(f"  Gateway Type: {suggestion.gateway_type}")
        print(f"  Confidence: {suggestion.confidence_score:.0%}")
        print(f"  Number of Branches: {len(suggestion.branches)}")
        
        print(f"\n  Branches:")
        for branch_idx, branch in enumerate(suggestion.branches, 1):
            branch_name = branch.branch_name if hasattr(branch, 'branch_name') else f"Branch {branch_idx}"
            condition = branch.condition if hasattr(branch, 'condition') else "No condition"
            is_default = branch.is_default if hasattr(branch, 'is_default') else False
            
            default_marker = " [DEFAULT]" if is_default else ""
            print(f"    {branch_name}{default_marker}:")
            print(f"      Condition: {condition}")
            
            if hasattr(branch, 'tasks') and branch.tasks:
                task_names = []
                # Handle both CMS format (process_task) and simple format (tasks)
                all_tasks = process_data.get('tasks', [])
                if not all_tasks and 'process_task' in process_data:
                    all_tasks = [pt['task'] for pt in process_data['process_task']]
                
                for task_id in branch.tasks:
                    for task in all_tasks:
                        t_id = task.get('id', task.get('task_id'))
                        if str(t_id) == str(task_id):
                            task_names.append(task.get('name', task.get('task_name')))
                            break
                print(f"      Tasks: {', '.join(task_names) if task_names else 'None'}")
        
        if suggestion.justification:
            print(f"\n  Justification:")
            for key, value in suggestion.justification.items():
                print(f"    {key}: {value}")
        
        if suggestion.benefits:
            print(f"\n  Benefits:")
            for key, value in suggestion.benefits.items():
                print(f"    {key}: {value}")


def analyze_process_structure():
    """Analyze and display the process structure"""
    print("\nProcess Structure:")
    print(f"  Process ID: {process_data.get('id', process_data.get('process_id'))}")
    print(f"  Process Name: {process_data.get('name', process_data.get('process_name'))}")
    
    # Handle both CMS format (process_task) and simple format (tasks)
    tasks = process_data.get('tasks', [])
    if not tasks and 'process_task' in process_data:
        tasks = [pt['task'] for pt in process_data['process_task']]
    
    print(f"  Total Tasks: {len(tasks)}")
    
    print("\n  Task Flow:")
    for idx, task in enumerate(tasks, 1):
        task_id = task.get('id', task.get('task_id'))
        task_name = task.get('name', task.get('task_name'))
        duration = task.get('duration', task.get('task_capacity_minutes', 0))
        dependencies = task.get('dependencies', [])
        deps_str = f" (depends on: {', '.join(map(str, dependencies))})" if dependencies else ""
        print(f"    {idx}. [{task_id}] {task_name} ({duration} min){deps_str}")


def test_exclusive_gateway():
    """Test Exclusive (XOR) Gateway Detection"""
    print_section_header("TEST 1: EXCLUSIVE (XOR) GATEWAY DETECTION")
    
    print("\nExpected Pattern:")
    print("  After 'Initial Assessment' (Task 1001)")
    print("  -> Choose ONE deployment type:")
    print("    - Cloud Deployment (Task 1002)")
    print("    - On-Premise Deployment (Task 1003)")
    print("    - Hybrid Deployment (Task 1004)")
    print("  Only ONE path should execute (mutually exclusive)")
    
    detector = ExclusiveGatewayDetector(min_confidence=0.70)
    suggestions = detector.analyze_process(process_data)
    
    print_gateway_details("EXCLUSIVE", suggestions)
    
    return suggestions


def test_parallel_gateway():
    """Test Parallel (AND) Gateway Detection"""
    print_section_header("TEST 2: PARALLEL (AND) GATEWAY DETECTION")
    
    print("\nExpected Pattern:")
    print("  After 'Pre-Release Preparation' (Task 1005)")
    print("  -> Execute ALL quality checks simultaneously:")
    print("    - Security Audit (Task 1006)")
    print("    - Performance Testing (Task 1007)")
    print("    - Compliance Validation (Task 1008)")
    print("  ALL paths must execute in parallel")
    
    detector = ParallelGatewayDetector(min_confidence=0.70)
    suggestions = detector.analyze_process(process_data)
    
    print_gateway_details("PARALLEL", suggestions)
    
    return suggestions


def test_inclusive_or_gateway():
    """Test Inclusive (OR) Gateway Detection"""
    print_section_header("TEST 3: INCLUSIVE (OR) GATEWAY DETECTION")
    
    print("\nExpected Pattern:")
    print("  After 'Quality Assurance Review' (Task 1009)")
    print("  -> Send notifications (one or more channels):")
    print("    - Send Email Notification (Task 1010)")
    print("    - Send SMS Alerts (Task 1011)")
    print("    - Post Slack Update (Task 1012)")
    print("    - Send Push Notification (Task 1013)")
    print("  ONE OR MORE paths can execute (conditional multi-channel)")
    
    detector = InclusiveGatewayDetector(min_confidence=0.65)
    suggestions = detector.analyze_process(process_data)
    
    print_gateway_details("INCLUSIVE OR", suggestions)
    
    return suggestions


def print_summary(xor_count, parallel_count, or_count):
    """Print test summary"""
    print_section_header("TEST SUMMARY")
    
    total_detected = xor_count + parallel_count + or_count
    
    print(f"\n  Exclusive (XOR) Gateways: {xor_count}")
    print(f"  Parallel (AND) Gateways: {parallel_count}")
    print(f"  Inclusive (OR) Gateways: {or_count}")
    print(f"  Total Gateways Detected: {total_detected}")
    
    print("\n  Expected Results:")
    print("    - At least 1 Exclusive (XOR) gateway (deployment choice)")
    print("    - At least 1 Parallel (AND) gateway (quality checks)")
    print("    - At least 1 Inclusive (OR) gateway (notifications)")
    
    print("\n  Validation:")
    xor_pass = "[PASS]" if xor_count >= 1 else "[FAIL]"
    parallel_pass = "[PASS]" if parallel_count >= 1 else "[FAIL]"
    or_pass = "[PASS]" if or_count >= 1 else "[FAIL]"
    
    print(f"    Exclusive Gateway Detection: {xor_pass}")
    print(f"    Parallel Gateway Detection: {parallel_pass}")
    print(f"    Inclusive OR Gateway Detection: {or_pass}")
    
    all_pass = xor_count >= 1 and parallel_count >= 1 and or_count >= 1
    
    print("\n" + "="*80)
    if all_pass:
        print("  [SUCCESS] TEST PASSED: All three gateway types coexist and detected successfully!")
    else:
        print("  [WARNING] TEST INCOMPLETE: Some gateway types were not detected")
    print("="*80 + "\n")
    
    return all_pass


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  COMPREHENSIVE GATEWAY DETECTION TEST")
    print("  Testing: testGatewaysProcess")
    print("="*80)
    
    # Load process data
    process_data = load_test_process()
    
    # Analyze structure
    analyze_process_structure()
    
    # Test each gateway type
    xor_suggestions = test_exclusive_gateway()
    parallel_suggestions = test_parallel_gateway()
    or_suggestions = test_inclusive_or_gateway()
    
    # Print summary
    success = print_summary(
        len(xor_suggestions),
        len(parallel_suggestions),
        len(or_suggestions)
    )
    
    print("\n[COMPLETE] TEST FINISHED")
    
    # Exit with appropriate code
    exit(0 if success else 1)
