import requests
import json

print("Testing process ID 11...")
try:
    r = requests.post('http://localhost:8000/cms/optimize/11', timeout=120)
    print(f'Status: {r.status_code}')
    
    if r.status_code == 200:
        print('✓ Success!')
        resp = r.json()
        print(f"Process: {resp.get('process_name', 'Unknown')}")
        print(f"Before Cost: ${resp.get('before_cost', 0):.2f}")
        print(f"After Cost: ${resp.get('after_cost', 0):.2f}")
    else:
        resp = r.json()
        print(f'✗ Error: {resp.get("detail", "Unknown")[:300]}')
except Exception as e:
    print(f'✗ Exception: {str(e)}')
