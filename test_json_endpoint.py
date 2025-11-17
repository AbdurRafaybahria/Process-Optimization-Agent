"""
Test script for the new JSON optimization endpoint
"""
import requests
import json

# Test the JSON endpoint
def test_json_endpoint(process_id=11):
    url = f"http://localhost:8000/cms/optimize/{process_id}/json"
    
    print(f"Testing JSON endpoint for process ID: {process_id}")
    print(f"URL: {url}\n")
    
    try:
        response = requests.post(url)
        
        print(f"Status Code: {response.status_code}\n")
        
        if response.status_code == 200:
            data = response.json()
            
            # Print formatted JSON
            print("="*80)
            print("OPTIMIZATION RESULTS (JSON)")
            print("="*80)
            print(json.dumps(data, indent=2))
            print("\n" + "="*80)
            
            # Print key metrics
            print("\nKEY METRICS:")
            print(f"Process: {data['process_name']}")
            print(f"Type: {data['process_type']['type']}")
            print(f"Suggestions: {data['optimization_summary']['total_suggestions']}")
            print(f"Time Saved: {data['optimization_summary']['potential_time_saved']}")
            print(f"Before: {data['current_state']['total_time_hours']} hours, ${data['current_state']['total_cost']}")
            print(f"After: {data['optimized_state']['total_time_hours']} hours, ${data['optimized_state']['total_cost']}")
            print(f"Parallel Tasks: {data['parallel_execution']['total_parallel_tasks']}")
            
            # Save to file
            with open("optimization_result.json", "w") as f:
                json.dump(data, f, indent=2)
            print("\n✓ Results saved to: optimization_result.json")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure the API is running on port 8000")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_json_endpoint(11)
