"""
CMS API Client for fetching process data from the enterprise digital twin system
Supports HttpOnly cookie-based authentication
"""

import requests
from typing import Dict, Any, Optional, List
import json
from datetime import datetime


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
            auth_url = f"{self.base_url}/auth/login"
            auth_data = {
                "email": "superadmin@example.com",
                "password": "ChangeMe123!"
            }
            
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
            auth_url = f"{self.base_url}/auth/login"
            auth_data = {
                "email": "superadmin@example.com",
                "password": "ChangeMe123!"
            }
            
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

