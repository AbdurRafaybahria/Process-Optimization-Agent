import json

json1 = """
{JSON content from first response would go here}
"""

json2 = """
{JSON content from second response would go here}
"""

def find_differences(obj1, obj2, path=""):
    """Recursively find differences between two objects"""
    differences = []
    
    if type(obj1) != type(obj2):
        differences.append(f"{path}: Type mismatch - {type(obj1).__name__} vs {type(obj2).__name__}")
        return differences
    
    if isinstance(obj1, dict):
        all_keys = set(obj1.keys()) | set(obj2.keys())
        for key in all_keys:
            new_path = f"{path}.{key}" if path else key
            if key not in obj1:
                differences.append(f"{new_path}: Missing in first JSON")
            elif key not in obj2:
                differences.append(f"{new_path}: Missing in second JSON")
            else:
                differences.extend(find_differences(obj1[key], obj2[key], new_path))
    
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            differences.append(f"{path}: Array length mismatch - {len(obj1)} vs {len(obj2)}")
        else:
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                differences.extend(find_differences(item1, item2, f"{path}[{i}]"))
    
    else:
        if obj1 != obj2:
            differences.append(f"{path}: Value mismatch - '{obj1}' vs '{obj2}'")
    
    return differences

# Parse and compare
try:
    data1 = json.loads(json1)
    data2 = json.loads(json2)
    
    diffs = find_differences(data1, data2)
    
    if not diffs:
        print("✅ JSONs are IDENTICAL")
    else:
        print(f"❌ Found {len(diffs)} difference(s):")
        for diff in diffs:
            print(f"  - {diff}")
            
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
