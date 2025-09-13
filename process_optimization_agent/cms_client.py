"""
CMS API Client for fetching process data from the enterprise digital twin system
"""

import requests
from typing import Dict, Any, Optional, List
import json
from datetime import datetime


class CMSClient:
    """Client for interacting with the CMS API"""
    
    def __init__(self, base_url: str = None, 
                 bearer_token: str = None):
        """
        Initialize the CMS client
        
        Args:
            base_url: Base URL of the CMS API
            bearer_token: Bearer token for authentication
        """
        import os
        self.base_url = base_url or os.getenv("REACT_APP_BASE_URL", "http://localhost:3000")
        self.bearer_token = bearer_token
        self.headers = {
            "Authorization": f"Bearer {bearer_token}" if bearer_token else "",
            "Content-Type": "application/json"
        }
    
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
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching process {process_id}: {e}")
            return None
    
    def get_all_processes(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch all processes from the CMS
        
        Returns:
            List of process dictionaries, or None if error
        """
        try:
            url = f"{self.base_url}/process"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching processes: {e}")
            return None
    
    def update_bearer_token(self, new_token: str):
        """
        Update the bearer token for authentication
        
        Args:
            new_token: New bearer token
        """
        self.bearer_token = new_token
        self.headers["Authorization"] = f"Bearer {new_token}"
