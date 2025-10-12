#!/usr/bin/env python3
"""
Security Update Script for FraudGuard 360
Updates all Python requirements.txt files to latest secure versions
"""

import subprocess
import sys
import os
from pathlib import Path

# Latest secure versions of commonly used packages (as of October 2025)
SECURE_VERSIONS = {
    'fastapi': '0.115.4',
    'uvicorn': '0.32.0', 
    'pydantic': '2.10.0',
    'pyjwt': '2.10.0',
    'bcrypt': '4.2.0',
    'confluent-kafka': '2.5.3',
    'neo4j': '5.24.0',
    'pytest': '8.3.3',
    'pytest-asyncio': '0.24.0',
    'httpx': '0.27.2',
    'prometheus-client': '0.21.0',
    'redis': '5.1.1',
    'psycopg2-binary': '2.9.9',
    'python-multipart': '0.0.19',
    'python-jose': '3.4.0',
    'torch': '2.5.1',
    'torch-geometric': '2.6.1',
    'mlflow': '2.18.0',
    'pandas': '2.2.3',
    'scikit-learn': '1.5.2',
    'numpy': '2.1.3',
    'joblib': '1.4.2',
    'matplotlib': '3.9.2',
    'seaborn': '0.13.2',
    'plotly': '5.24.1',
    'networkx': '3.4.2',
    'flask': '3.1.0',
    'requests': '2.32.3',
    'urllib3': '2.2.3',
    'aiohttp': '3.10.10',
    'celery': '5.4.0',
    'sqlalchemy': '2.0.36',
    'alembic': '1.13.3',
    'click': '8.1.7',
    'jinja2': '3.1.4',
    'werkzeug': '3.0.6',
    'cryptography': '43.0.3',
    'pymongo': '4.10.1',
    'certifi': '2024.8.30',
    'charset-normalizer': '3.4.0',
    'idna': '3.10',
    'MarkupSafe': '3.0.2',
    'itsdangerous': '2.2.0',
    'setuptools': '75.3.0',
    'pip': '24.3.1',
    'wheel': '0.45.0'
}

def update_requirements_file(file_path):
    """Update a single requirements.txt file with secure versions."""
    print(f"🔄 Updating {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Read current requirements
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        updated_count = 0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                updated_lines.append(line)
                continue
            
            # Parse package name and version
            if '==' in line:
                package_name = line.split('==')[0].strip()
                if '[' in package_name:  # Handle extras like python-jose[cryptography]
                    base_package = package_name.split('[')[0]
                    if base_package in SECURE_VERSIONS:
                        new_line = f"{package_name}=={SECURE_VERSIONS[base_package]}"
                        updated_lines.append(new_line)
                        updated_count += 1
                        print(f"   ✅ {package_name} -> {SECURE_VERSIONS[base_package]}")
                    else:
                        updated_lines.append(line)
                elif package_name in SECURE_VERSIONS:
                    new_line = f"{package_name}=={SECURE_VERSIONS[package_name]}"
                    updated_lines.append(new_line)
                    updated_count += 1
                    print(f"   ✅ {package_name} -> {SECURE_VERSIONS[package_name]}")
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Write updated requirements
        with open(file_path, 'w') as f:
            for line in updated_lines:
                f.write(line + '\n')
        
        print(f"✅ Updated {updated_count} packages in {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all requirements files."""
    print("🔒 FraudGuard 360 - Security Update Script")
    print("==========================================")
    
    # Find all requirements.txt files
    requirements_files = [
        'api-gateway/requirements.txt',
        'core-ml-service/requirements.txt', 
        'graph-analytics-service/requirements.txt',
        'ml-service/requirements.txt',
        'risk-scoring-service/requirements.txt',
        'scripts/requirements.txt',
        'services/ai-service/requirements.txt',
        'services/api-gateway/requirements.txt',
        'services/graph-service/requirements.txt',
        'testing/security-testing/requirements.txt',
        'tests/requirements.txt',
        'tests/performance/requirements.txt'
    ]
    
    updated_files = 0
    total_files = len(requirements_files)
    
    for req_file in requirements_files:
        if update_requirements_file(req_file):
            updated_files += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total files: {total_files}")
    print(f"   Updated: {updated_files}")
    print(f"   Failed: {total_files - updated_files}")
    
    if updated_files == total_files:
        print("🎯 All requirements files updated successfully!")
    else:
        print("⚠️  Some files could not be updated.")

if __name__ == "__main__":
    main()