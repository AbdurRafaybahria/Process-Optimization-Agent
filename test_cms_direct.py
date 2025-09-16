#!/usr/bin/env python3
"""
Direct test of CMS API connection to debug the 404 error
"""
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_cms_connection():
    """Test direct connection to CMS API"""
    base_url = os.getenv("REACT_APP_BASE_URL", "http://localhost:3000")
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjEsImVtYWlsIjoic3VwZXJhZG1pbkBleGFtcGxlLmNvbSIsInJvbGUiOiJTVVBFUl9BRE1JTiIsIm5hbWUiOiJTdXBlciBBZG1pbiIsImlhdCI6MTc1NzMxNDc2OSwiZXhwIjoxNzU3OTE5NTY5fQ.OLdaZNroqLnbfub-0jRVwZUQZJIyMTegioFGtj2dsEk"
    
    # Check token expiration (manual decode)
    import base64
    import json
    import time
    try:
        # JWT has 3 parts separated by dots
        parts = token.split('.')
        if len(parts) >= 2:
            # Decode the payload (second part)
            payload = parts[1]
            # Add padding if needed
            payload += '=' * (4 - len(payload) % 4)
            decoded_bytes = base64.urlsafe_b64decode(payload)
            decoded = json.loads(decoded_bytes)
            
            exp_time = decoded.get('exp', 0)
            current_time = int(time.time())
            print(f"Token expires at: {exp_time} (current: {current_time})")
            if exp_time < current_time:
                print("⚠️  TOKEN IS EXPIRED!")
            else:
                print("✅ Token is still valid")
        else:
            print("Invalid token format")
    except Exception as e:
        print(f"Could not decode token: {e}")
    
    print(f"Testing CMS connection to: {base_url}")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test basic connectivity first
    try:
        print(f"Testing basic connectivity to: {base_url}")
        response = requests.get(base_url, headers=headers, timeout=10)
        print(f"Root endpoint status: {response.status_code}")
        if response.status_code == 200:
            print(f"Root response: {response.text[:200]}...")
        else:
            print(f"Root error: {response.text[:200]}...")
    except Exception as e:
        print(f"Basic connectivity failed: {e}")
        print("The CMS server appears to be down or the URL is incorrect.")
        return
    
    # Test different endpoint patterns based on CMS client structure
    endpoints_to_try = [
        "/",
        "/health",
        "/api/health", 
        "/process/7/with-relations",  # This is what the CMS client uses
        "/processes",
        "/api/processes",
        "/processes/7",
        "/api/processes/7"
    ]
    
    for endpoint in endpoints_to_try:
        try:
            url = f"{base_url}{endpoint}"
            print(f"Trying: {url}")
            response = requests.get(url, headers=headers, timeout=5)
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        print(f"  Found {len(data)} items")
                        if data:
                            print(f"  Sample: {data[0] if len(str(data[0])) < 100 else str(data[0])[:100]+'...'}")
                    else:
                        print(f"  Data: {str(data)[:200]+'...' if len(str(data)) > 200 else data}")
                except:
                    print(f"  Response: {response.text[:100]}...")
            elif response.status_code != 404:
                print(f"  Response: {response.text[:100]}...")
        except Exception as e:
            print(f"  Error: {str(e)[:100]}...")
        print()

if __name__ == "__main__":
    test_cms_connection()
