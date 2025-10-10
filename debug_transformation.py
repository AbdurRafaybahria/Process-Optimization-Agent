#!/usr/bin/env python3
"""
Debug CMS data transformation to see what's happening
"""
import os
from dotenv import load_dotenv
from process_optimization_agent.cms_client import CMSClient
from process_optimization_agent.cms_transformer import CMSDataTransformer
import json

load_dotenv()

def debug_transformation():
    """Debug the CMS data transformation process"""
    
    # Get CMS data
    client = CMSClient()
    print("Fetching process 7 from CMS...")
    cms_data = client.get_process_with_relations(7)
    
    if not cms_data:
        print("Failed to fetch CMS data")
        return
    
    print("CMS data fetched successfully")
    print(f"CMS data keys: {list(cms_data.keys())}")
    
    # Save raw CMS data for inspection
    with open("debug_cms_raw.json", "w") as f:
        json.dump(cms_data, f, indent=2, default=str)
    print("Raw CMS data saved to debug_cms_raw.json")
    
    # Transform data
    transformer = CMSDataTransformer()
    print("\nTransforming CMS data...")
    
    try:
        agent_format = transformer.transform_process(cms_data)
        print("Transformation completed")
        
        # Save transformed data for inspection
        with open("debug_agent_format.json", "w") as f:
            json.dump(agent_format, f, indent=2, default=str)
        print("Transformed data saved to debug_agent_format.json")
        
        # Analyze transformed data
        print(f"\n=== TRANSFORMATION ANALYSIS ===")
        print(f"Process ID: {agent_format.get('id', 'MISSING')}")
        print(f"Process Name: {agent_format.get('name', 'MISSING')}")
        
        tasks = agent_format.get('tasks', [])
        resources = agent_format.get('resources', [])
        
        print(f"Tasks count: {len(tasks)}")
        print(f"Resources count: {len(resources)}")
        
        if tasks:
            print(f"\nFirst task example:")
            print(f"  ID: {tasks[0].get('id', 'MISSING')}")
            print(f"  Name: {tasks[0].get('name', 'MISSING')}")
            print(f"  Duration: {tasks[0].get('duration_hours', 'MISSING')}")
            print(f"  Dependencies: {tasks[0].get('dependencies', 'MISSING')}")
        else:
            print("NO TASKS FOUND!")
            
        if resources:
            print(f"\nFirst resource example:")
            print(f"  ID: {resources[0].get('id', 'MISSING')}")
            print(f"  Name: {resources[0].get('name', 'MISSING')}")
            print(f"  Role: {resources[0].get('role', 'MISSING')}")
            print(f"  Rate: {resources[0].get('hourly_rate', 'MISSING')}")
        else:
            print("NO RESOURCES FOUND!")
            
    except Exception as e:
        print(f"Transformation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_transformation()
