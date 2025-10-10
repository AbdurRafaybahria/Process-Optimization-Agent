"""
Debug script to test CMS API call
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from process_optimization_agent.cms_client import CMSClient
from process_optimization_agent.cms_transformer import CMSDataTransformer

# Test CMS API call
def test_cms_api():
    print("Testing CMS API call...")
    
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjEsImVtYWlsIjoic3VwZXJhZG1pbkBleGFtcGxlLmNvbSIsInJvbGUiOiJTVVBFUl9BRE1JTiIsIm5hbWUiOiJTdXBlciBBZG1pbiIsImlhdCI6MTc1NzMxNDc2OSwiZXhwIjoxNzU3OTE5NTY5fQ.OLdaZNroqLnbfub-0jRVwZUQZJIyMTegioFGtj2dsEk"
    
    client = CMSClient(bearer_token=token)
    
    # Try to fetch process 7
    cms_data = client.get_process_with_relations(7)
    
    if cms_data:
        print("CMS API call successful")
        print(f"Process name: {cms_data.get('name', 'N/A')}")
        
        # Test transformation
        transformer = CMSDataTransformer()
        agent_format = transformer.transform_process(cms_data)
        
        print(f"Transformation successful")
        print(f"Agent format has 'id': {'id' in agent_format}")
        print(f"Agent format has 'name': {'name' in agent_format}")
        print(f"Agent format has 'duration_hours' in first task: {'duration_hours' in agent_format['tasks'][0] if agent_format['tasks'] else False}")
        
        # Print the structure
        print("\nTransformed structure:")
        for key in agent_format.keys():
            print(f"  {key}: {type(agent_format[key])}")
            
        return agent_format
    else:
        print("CMS API call failed")
        return None

if __name__ == "__main__":
    test_cms_api()
