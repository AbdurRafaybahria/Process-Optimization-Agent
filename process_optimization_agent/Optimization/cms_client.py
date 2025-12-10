"""
CMS API Client for fetching process data from the enterprise digital twin system
Supports HttpOnly cookie-based authentication
"""

import requests
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class CMSClient:
    """Client for interacting with the CMS API using HttpOnly cookie authentication"""
    
    def __init__(self, base_url: str = None, 
                 bearer_token: str = None,
                 use_cookies: bool = True):
        """
        Initialize the CMS client
        
        Args:
            base_url: Base URL of the CMS API
            bearer_token: Bearer token for authentication (legacy support, optional)
            use_cookies: Whether to use HttpOnly cookie authentication (default: True)
        """
        import os
        self.base_url = base_url or os.getenv("REACT_APP_BASE_URL", "https://server-digitaltwin-enterprise-production.up.railway.app")
        self.use_cookies = use_cookies
        
        # Create a session to maintain cookies across requests
        self.session = requests.Session()
        
        # Legacy bearer token support (for backwards compatibility)
        self.bearer_token = bearer_token
        
        if bearer_token:
            # If bearer token provided, use legacy Authorization header
            self.session.headers.update({
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json"
            })
        elif use_cookies:
            # Use cookie-based authentication
            self.session.headers.update({
                "Content-Type": "application/json"
            })
            # Authenticate to get HttpOnly cookie
            self._authenticate_with_cookies()
        else:
            # No authentication
            self.session.headers.update({
                "Content-Type": "application/json"
            })
    
    def _authenticate_with_cookies(self) -> bool:
        """
        Authenticate with the CMS using HttpOnly cookie authentication
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            import os
            auth_url = f"{self.base_url}/auth/login"
            auth_data = {
                "email": os.getenv("CMS_AUTH_EMAIL", ""),
                "password": os.getenv("CMS_AUTH_PASSWORD", "")
            }
            
            if not auth_data["email"] or not auth_data["password"]:
                print("Warning: CMS_AUTH_EMAIL or CMS_AUTH_PASSWORD not set in environment variables")
                return False
            
            # Use session to maintain cookies
            response = self.session.post(auth_url, json=auth_data, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # With HttpOnly cookies, the token is NOT in the response body
            # It's automatically set as a cookie by the browser/session
            # The response contains user info and success message
            
            if data.get("message") == "Login successful" or data.get("user"):
                print("Successfully authenticated with CMS (HttpOnly cookie)")
                return True
            
            # Fallback: check for legacy token response (backwards compatibility)
            if data.get("access_token"):
                self.bearer_token = data.get("access_token")
                self.session.headers.update({
                    "Authorization": f"Bearer {self.bearer_token}"
                })
                print("Successfully authenticated with CMS (legacy token)")
                return True
            
            print("Authentication response indicates success")
            return True
                
        except requests.exceptions.RequestException as e:
            print(f"Authentication failed: {e}")
            return False
    
    def _authenticate(self) -> Optional[str]:
        """
        Legacy authentication method for backwards compatibility
        
        Returns:
            Access token string or None if authentication fails
        """
        try:
            import os
            auth_url = f"{self.base_url}/auth/login"
            auth_data = {
                "email": os.getenv("CMS_AUTH_EMAIL", ""),
                "password": os.getenv("CMS_AUTH_PASSWORD", "")
            }
            
            if not auth_data["email"] or not auth_data["password"]:
                print("Warning: CMS_AUTH_EMAIL or CMS_AUTH_PASSWORD not set in environment variables")
                return None
            
            response = self.session.post(auth_url, json=auth_data, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Try to get token from response (legacy)
            token = data.get("access_token")
            
            if token:
                print("Successfully authenticated with CMS (legacy)")
                return token
            
            # With HttpOnly cookies, authentication is via cookies, not token
            # If we get here without a token but request succeeded, cookies are set
            if response.status_code == 200:
                print("Successfully authenticated with CMS (HttpOnly cookies)")
                return "cookie_auth"  # Indicate cookie-based auth
            
            print("Authentication response missing access_token")
            return None
                
        except requests.exceptions.RequestException as e:
            print(f"Authentication failed: {e}")
            return None
    
    def get_process_with_relations(self, process_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a process with all its relations from the CMS
        
        Args:
            process_id: ID of the process to fetch
            
        Returns:
            Dict containing process data with relations, or None if error
        """
        try:
            url = f"{self.base_url}/process/{process_id}/with-relations"
            # Session automatically sends cookies
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching process {process_id}: {e}")
            # Try re-authenticating and retry once
            if "401" in str(e) or "Unauthorized" in str(e):
                print("Session expired, re-authenticating...")
                if self._authenticate_with_cookies():
                    try:
                        response = self.session.get(url)
                        response.raise_for_status()
                        return response.json()
                    except:
                        pass
            return None
    
    def get_all_processes(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch all processes from the CMS
        
        Returns:
            List of process dictionaries, or None if error
        """
        try:
            url = f"{self.base_url}/process"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching processes: {e}")
            # Try re-authenticating and retry once
            if "401" in str(e) or "Unauthorized" in str(e):
                print("Session expired, re-authenticating...")
                if self._authenticate_with_cookies():
                    try:
                        response = self.session.get(url)
                        response.raise_for_status()
                        return response.json()
                    except:
                        pass
            return None
    
    def update_bearer_token(self, new_token: str):
        """
        Update the bearer token for authentication (legacy support)
        
        Args:
            new_token: New bearer token
        """
        self.bearer_token = new_token
        self.session.headers.update({
            "Authorization": f"Bearer {new_token}"
        })
    
    def logout(self) -> bool:
        """
        Logout and clear the HttpOnly cookie
        
        Returns:
            True if logout successful, False otherwise
        """
        try:
            logout_url = f"{self.base_url}/auth/logout"
            response = self.session.post(logout_url, timeout=10)
            response.raise_for_status()
            
            # Clear session cookies
            self.session.cookies.clear()
            print("Successfully logged out from CMS")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Logout failed: {e}")
            return False
    
    def verify_session(self) -> Optional[Dict[str, Any]]:
        """
        Verify if the current session is still valid
        
        Returns:
            User info dict if session valid, None otherwise
        """
        try:
            url = f"{self.base_url}/auth/me"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Session verification failed: {e}")
            return None

    def get_all_jobs_with_relations(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch all jobs with their skills and relations from the CMS
        
        Returns:
            List of job dictionaries with skills, or None if error
        """
        try:
            url = f"{self.base_url}/job/with-relations"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching jobs with relations: {e}")
            # Try re-authenticating and retry once
            if "401" in str(e) or "Unauthorized" in str(e):
                print("Session expired, re-authenticating...")
                if self._authenticate_with_cookies():
                    try:
                        response = self.session.get(url)
                        response.raise_for_status()
                        return response.json()
                    except:
                        pass
            return None

    def get_jobs_for_process(self, process_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Get all jobs with their skills that are used in a specific process
        
        Args:
            process_data: Process data containing process_task array
            
        Returns:
            Dictionary mapping job_id to job data with skills
        """
        # Extract all job IDs from the process
        job_ids_in_process = set()
        for pt in process_data.get("process_task", []):
            task = pt.get("task", {})
            for jt in task.get("jobTasks", []):
                job_id = jt.get("job_id") or jt.get("job", {}).get("job_id")
                if job_id:
                    job_ids_in_process.add(int(job_id))
        
        if not job_ids_in_process:
            print("No jobs found in process")
            return {}
        
        # Fetch all jobs with relations
        all_jobs = self.get_all_jobs_with_relations()
        if not all_jobs:
            print("Failed to fetch jobs with relations")
            return {}
        
        # Filter to only jobs in the process
        jobs_map = {}
        for job in all_jobs:
            job_id = job.get("job_id")
            if job_id in job_ids_in_process:
                # Extract skills in a standardized format
                skills = []
                for job_skill in job.get("jobSkills", []):
                    skill_data = job_skill.get("skill", {})
                    skill_level = job_skill.get("skill_level", {})
                    skills.append({
                        "skill_id": skill_data.get("skill_id"),
                        "name": skill_data.get("name", ""),
                        "description": skill_data.get("description"),
                        "level_id": skill_level.get("id"),
                        "level_name": skill_level.get("level_name", "INTERMEDIATE"),
                        "level_rank": skill_level.get("level_rank", 3)
                    })
                
                jobs_map[job_id] = {
                    "job_id": job_id,
                    "name": job.get("name", ""),
                    "description": job.get("description", ""),
                    "jobCode": job.get("jobCode", ""),
                    "hourlyRate": job.get("hourlyRate", 0),
                    "maxHoursPerDay": job.get("maxHoursPerDay", 8),
                    "job_level": job.get("job_level", {}),
                    "skills": skills
                }
        
        print(f"Found {len(jobs_map)} jobs with skills for process")
        return jobs_map

