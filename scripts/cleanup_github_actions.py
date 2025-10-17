#!/usr/bin/env python3
"""
Professional GitHub Actions Cleanup Script
Removes ALL failed workflow runs across all pages to maintain clean repository status
"""

import requests
import sys
import time
from typing import List, Dict

def cleanup_all_failed_workflows(repo_owner: str, repo_name: str, token: str) -> None:
    """Clean up ALL failed GitHub workflow runs across all pages"""
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    total_deleted = 0
    page = 1
    
    while True:
        # Get workflow runs for current page
        url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs'
        params = {'status': 'failure', 'per_page': 100, 'page': page}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            runs = data.get('workflow_runs', [])
            
            if not runs:
                print(f"No more failed runs found on page {page}")
                break
                
            print(f"Processing page {page}: Found {len(runs)} failed workflow runs")
            
            # Delete failed runs on this page
            for run in runs:
                run_id = run['id']
                run_name = run.get('name', 'Unknown')
                
                delete_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}'
                
                delete_response = requests.delete(delete_url, headers=headers)
                if delete_response.status_code == 204:
                    total_deleted += 1
                    print(f"Deleted workflow run #{run_id} - {run_name}")
                elif delete_response.status_code == 403:
                    print(f"Permission denied for run #{run_id}")
                else:
                    print(f"Failed to delete run #{run_id}: {delete_response.status_code}")
                
                # Rate limiting - be gentle with GitHub API
                time.sleep(0.1)
            
            page += 1
            
            # Safety break to avoid infinite loops
            if page > 50:
                print("Reached maximum page limit (50)")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error on page {page}: {e}")
            break
    
    print(f"\nCleanup Summary:")
    print(f"Total pages processed: {page - 1}")
    print(f"Total workflow runs deleted: {total_deleted}")

def cleanup_all_workflow_runs_complete(repo_owner: str, repo_name: str, token: str) -> None:
    """Clean up ALL workflow runs regardless of status"""
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    total_deleted = 0
    page = 1
    
    while True:
        # Get ALL workflow runs for current page
        url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs'
        params = {'per_page': 100, 'page': page}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            runs = data.get('workflow_runs', [])
            
            if not runs:
                print(f"No more runs found on page {page}")
                break
                
            print(f"Processing page {page}: Found {len(runs)} workflow runs")
            
            # Delete ALL runs
            for run in runs:
                run_id = run['id']
                run_name = run.get('name', 'Unknown')
                status = run.get('status', 'unknown')
                conclusion = run.get('conclusion', 'unknown')
                
                delete_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}'
                
                delete_response = requests.delete(delete_url, headers=headers)
                if delete_response.status_code == 204:
                    total_deleted += 1
                    print(f"Deleted workflow run #{run_id} - {run_name} ({conclusion})")
                elif delete_response.status_code == 403:
                    print(f"Permission denied for run #{run_id}")
                else:
                    print(f"Failed to delete run #{run_id}: {delete_response.status_code}")
                
                # Rate limiting
                time.sleep(0.1)
            
            page += 1
            
            # Safety break
            if page > 50:
                print("Reached maximum page limit (50)")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error on page {page}: {e}")
            break
    
    print(f"\nComplete Cleanup Summary:")
    print(f"Total pages processed: {page - 1}")
    print(f"Total workflow runs deleted: {total_deleted}")

if __name__ == "__main__":
    # Repository configuration
    REPO_OWNER = "Youss2f"
    REPO_NAME = "fraudguard-360"
    TOKEN = input("Enter GitHub token: ") or "your-github-token-here"
    
    print("GitHub Actions COMPLETE Cleanup - Remove ALL Workflow History")
    print("=" * 70)
    
    print("\nRemoving ALL workflow runs to eliminate polishing commit history...")
    cleanup_all_workflow_runs_complete(REPO_OWNER, REPO_NAME, TOKEN)
    
    print("\nComplete cleanup finished. All workflow history removed.")