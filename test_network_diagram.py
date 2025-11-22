"""
Test network diagram structure
"""
import requests
import json
import time

time.sleep(3)

try:
    # Test with process 504
    print("Testing Process 504 (Insurance Underwriting)...")
    response = requests.post("http://localhost:8000/cms/optimize/504/json", timeout=120)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract network diagram
        network = data.get('network_diagram', {})
        
        print("\n" + "="*80)
        print("NETWORK DIAGRAM STRUCTURE")
        print("="*80)
        
        # Start node
        start = network.get('start', {})
        print(f"\n📍 START: {start.get('label')} at t={start.get('time')}h")
        
        # Stages
        stages = network.get('stages', [])
        print(f"\n🔄 STAGES: {len(stages)} total\n")
        
        for stage in stages:
            exec_type = stage.get('execution_type', 'unknown')
            task_count = stage.get('active_task_count', 0)
            time_point = stage.get('time', 0)
            
            icon = "⚡" if exec_type == "parallel" else "➡️"
            print(f"{icon} Stage {stage.get('stage')}: {exec_type.upper()}")
            print(f"   Time: t={time_point:.2f}h")
            print(f"   Active Tasks: {task_count}")
            
            tasks = stage.get('tasks', [])
            for task in tasks:
                print(f"   • {task.get('task_name')}")
                print(f"     Resource: {task.get('resource_name')}")
                print(f"     Duration: {task.get('duration_hours'):.2f}h ({task.get('start_time'):.2f}h - {task.get('end_time'):.2f}h)")
                print(f"     Progress: {task.get('progress', 0):.1f}%")
                if task.get('has_dependencies'):
                    print(f"     ⚠️ Has Dependencies")
            print()
        
        # Finish node
        finish = network.get('finish', {})
        print(f"🏁 FINISH: {finish.get('label')} at t={finish.get('time')}h")
        
        # Execution pattern summary
        exec_pattern = data.get('parallel_execution', {}).get('execution_pattern', {})
        print("\n" + "="*80)
        print("EXECUTION PATTERN SUMMARY")
        print("="*80)
        print(f"Mode: {exec_pattern.get('execution_mode')}")
        print(f"Description: {exec_pattern.get('description')}")
        print(f"Parallel Tasks: {exec_pattern.get('parallel_tasks_count')}")
        print(f"Sequential Tasks: {exec_pattern.get('sequential_tasks_count')}")
        if exec_pattern.get('warning'):
            print(f"⚠️ Warning: {exec_pattern.get('warning')}")
        
        # Save full network diagram
        with open("network_diagram_504.json", "w") as f:
            json.dump(network, f, indent=2)
        print(f"\n✅ Network diagram saved to: network_diagram_504.json")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text[:500])
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
