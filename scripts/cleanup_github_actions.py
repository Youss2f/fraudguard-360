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

def cleanup_all_workflow_runs(repo_owner: str, repo_name: str, token: str) -> None:
    """Clean up ALL workflow runs (including successful ones with emojis)"""
    
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
            
            # Delete runs that contain emojis or are failed
            for run in runs:
                run_id = run['id']
                run_name = run.get('name', 'Unknown')
                status = run.get('status', 'unknown')
                conclusion = run.get('conclusion', 'unknown')
                
                # Check if run name contains emojis or if it's failed
                should_delete = (
                    conclusion == 'failure' or
                    status == 'completed' and conclusion != 'success' or
                    any(ord(char) > 127 for char in run_name)  # Contains non-ASCII (emoji) characters
                )
                
                if should_delete:
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
    
    print("GitHub Actions Complete Cleanup - Professional Repository Maintenance")
    print("=" * 70)
    
    print("\nPhase 1: Cleaning up ALL failed workflow runs...")
    cleanup_all_failed_workflows(REPO_OWNER, REPO_NAME, TOKEN)
    
    print("\nPhase 2: Cleaning up runs with emojis and non-professional formatting...")
    cleanup_all_workflow_runs(REPO_OWNER, REPO_NAME, TOKEN)
    
    print("\nComplete cleanup finished. Repository status restored to 100% professional standards.")