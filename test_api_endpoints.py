#!/usr/bin/env python3
"""
Test script for Process Optimization API endpoints
"""
import json
import requests
import time
from pathlib import Path

# API Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 300  # 5 minutes timeout

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"✅ Health Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def load_test_data():
    """Load the test data from the provided JSON"""
    test_data = {
        "id": "11",
        "name": "Outpatient Consultation",
        "process_name": "Outpatient Consultation",
        "process_id": 11,
        "company": "Maldova Hospital",
        "description": "Outpatient consultation is a fundamental component of modern healthcare systems...",
        "tasks": [
            {
                "id": "46",
                "name": "Schedule Appointment",
                "description": "Scheduling an appointment is a fundamental task...",
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
                "description": "Calling a patient for consultation...",
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
                "description": "The initial assessment is a crucial step...",
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
                "description": "The medical examination is the core component...",
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
                "description": "Documenting consultation notes is a critical step...",
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
    return test_data

def test_direct_optimization():
    """Test optimization with direct agent format data (bypassing CMS transformation)"""
    print("\n🔍 Testing Direct Optimization (Agent Format)...")
    
    # The current API only supports CMS endpoints, but the data is in agent format
    # This will help us identify the issue
    
    test_data = load_test_data()
    
    # Try to use the CMS payload endpoint with agent format data
    try:
        print("   Sending request to /cms/optimize/payload...")
        response = requests.post(
            f"{BASE_URL}/cms/optimize/payload",
            json=test_data,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Optimization successful!")
            print(f"   Process: {result.get('process_name', 'Unknown')}")
            print(f"   Company: {result.get('company', 'Unknown')}")
            
            # Check if charts were generated
            if 'optimization_results' in result:
                alloc_chart = result['optimization_results'].get('alloc_chart', {})
                summary_chart = result['optimization_results'].get('summary_chart', {})
                print(f"   Allocation Chart: {alloc_chart.get('filename', 'Not generated')}")
                print(f"   Summary Chart: {summary_chart.get('filename', 'Not generated')}")
            
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (optimization taking too long)")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_cms_format_conversion():
    """Test if we need to convert agent format to CMS format"""
    print("\n🔍 Testing CMS Format Conversion...")
    
    # Load the CMS format data for comparison
    cms_data_path = Path("debug_cms_raw.json")
    if cms_data_path.exists():
        with open(cms_data_path, 'r') as f:
            cms_data = json.load(f)
        
        print("   Found CMS format data, testing with that...")
        
        try:
            response = requests.post(
                f"{BASE_URL}/cms/optimize/payload",
                json=cms_data,
                timeout=TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ CMS format optimization successful!")
                print(f"   Process: {result.get('process_name', 'Unknown')}")
                return True
            else:
                print(f"❌ CMS format failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ CMS format test failed: {e}")
            return False
    else:
        print("   No CMS format data found to test with")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting API Endpoint Tests")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("❌ Server is not responding, please check if it's running")
        return
    
    # Test 2: Direct optimization with agent format
    test_direct_optimization()
    
    # Test 3: CMS format conversion test
    test_cms_format_conversion()
    
    print("\n" + "=" * 50)
    print("🏁 Tests completed")

if __name__ == "__main__":
    main()
