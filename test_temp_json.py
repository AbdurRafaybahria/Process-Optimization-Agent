"""
Test what's being written to temp JSON files
"""

import sys
import os
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from process_optimization_agent.cms_client import CMSClient
from process_optimization_agent.cms_transformer import CMSDataTransformer
from API.main import write_temp_process_json

def test_temp_json():
    print("Testing temp JSON file creation...")
    
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjEsImVtYWlsIjoic3VwZXJhZG1pbkBleGFtcGxlLmNvbSIsInJvbGUiOiJTVVBFUl9BRE1JTiIsIm5hbWUiOiJTdXBlciBBZG1pbiIsImlhdCI6MTc1NzMxNDc2OSwiZXhwIjoxNzU3OTE5NTY5fQ.OLdaZNroqLnbfub-0jRVwZUQZJIyMTegioFGtj2dsEk"
    
    client = CMSClient(bearer_token=token)
    transformer = CMSDataTransformer()
    
    # Get CMS data and transform it
    cms_data = client.get_process_with_relations(7)
    agent_format = transformer.transform_process(cms_data)
    
    print(f"Agent format has 'id': {'id' in agent_format}")
    print(f"Agent format has 'name': {'name' in agent_format}")
    
    # Write to temp file
    meta = write_temp_process_json(agent_format)
    
    # Read back the file
    with open(meta["path"], "r") as f:
        file_content = json.load(f)
    
    print(f"File content has 'id': {'id' in file_content}")
    print(f"File content has 'name': {'name' in file_content}")
    print(f"File content keys: {list(file_content.keys())}")
    
    # Clean up
    os.unlink(meta["path"])

if __name__ == "__main__":
    test_temp_json()
