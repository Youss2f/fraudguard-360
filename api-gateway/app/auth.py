"""
Enhanced Authentication and Authorization Module
Implements secure JWT authentication with RBAC (Role-Based Access Control)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

import jwt
import bcrypt
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Security configuration from environment variables
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your-very-secure-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
REFRESH_TOKEN_EXPIRATION_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRATION_DAYS", "7"))

# Password hashing settings
BCRYPT_ROUNDS = 12

# Security headers middleware
security = HTTPBearer(auto_error=False)

class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    ANALYST = "analyst"
    INVESTIGATOR = "investigator"
    VIEWER = "viewer"

class Permission(str, Enum):
    """System permissions"""
    READ_ALERTS = "read:alerts"
    WRITE_ALERTS = "write:alerts"
    READ_CASES = "read:cases"
    WRITE_CASES = "write:cases"
    READ_USERS = "read:users"
    WRITE_USERS = "write:users"
    READ_SYSTEM = "read:system"
    WRITE_SYSTEM = "write:system"
    EXPORT_DATA = "export:data"
    ADMIN_ACCESS = "admin:access"

# Role to permissions mapping
ROLE_PERMISSIONS: Dict[UserRole, List[Permission]] = {
    UserRole.ADMIN: [
        Permission.READ_ALERTS, Permission.WRITE_ALERTS,
        Permission.READ_CASES, Permission.WRITE_CASES,
        Permission.READ_USERS, Permission.WRITE_USERS,
        Permission.READ_SYSTEM, Permission.WRITE_SYSTEM,
        Permission.EXPORT_DATA, Permission.ADMIN_ACCESS
    ],
    UserRole.ANALYST: [
        Permission.READ_ALERTS, Permission.WRITE_ALERTS,
        Permission.READ_CASES, Permission.WRITE_CASES,
        Permission.EXPORT_DATA
    ],
    UserRole.INVESTIGATOR: [
        Permission.READ_ALERTS, Permission.WRITE_ALERTS,
        Permission.READ_CASES, Permission.WRITE_CASES,
        Permission.EXPORT_DATA
    ],
    UserRole.VIEWER: [
        Permission.READ_ALERTS, Permission.READ_CASES
    ]
}

class TokenData(BaseModel):
    """Token payload structure"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    exp: datetime
    iat: datetime
    jti: str = Field(description="JWT ID for token revocation")

class User(BaseModel):
    """User model for authentication"""
    id: str
    username: str
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(user_data: Dict[str, Any]) -> str:
    """Create JWT access token with user data and permissions"""
    now = datetime.utcnow()
    exp = now + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    # Get user permissions based on role
    role = UserRole(user_data.get('role', UserRole.VIEWER))
    permissions = ROLE_PERMISSIONS.get(role, [])
    
    payload = {
        "user_id": user_data["id"],
        "username": user_data["username"],
        "email": user_data["email"],
        "role": role.value,
        "permissions": [p.value for p in permissions],
        "exp": exp,
        "iat": now,
        "jti": f"{user_data['id']}_{int(now.timestamp())}"  # JWT ID for revocation
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> TokenData:
    """Verify JWT token and return user data"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(
            credentials.credentials, 
            JWT_SECRET, 
            algorithms=[JWT_ALGORITHM]
        )
        
        # Validate required fields
        required_fields = ["user_id", "username", "email", "role", "permissions", "exp", "iat"]
        for field in required_fields:
            if field not in payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token: missing {field}"
                )
        
        # Check token expiration
        exp = datetime.fromtimestamp(payload["exp"])
        if datetime.utcnow() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        return TokenData(
            user_id=payload["user_id"],
            username=payload["username"],
            email=payload["email"],
            role=UserRole(payload["role"]),
            permissions=[Permission(p) for p in payload["permissions"]],
            exp=exp,
            iat=datetime.fromtimestamp(payload["iat"]),
            jti=payload.get("jti", "")
        )
        
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed"
        )

def require_permission(required_permission: Permission):
    """Decorator to require specific permission"""
    def permission_dependency(token_data: TokenData = Depends(verify_token)) -> TokenData:
        if required_permission not in token_data.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_permission.value}"
            )
        return token_data
    return permission_dependency

def require_role(required_role: UserRole):
    """Decorator to require specific role"""
    def role_dependency(token_data: TokenData = Depends(verify_token)) -> TokenData:
        if token_data.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient privileges. Required role: {required_role.value}"
            )
        return token_data
    return role_dependency

# Common permission dependencies
require_admin = require_role(UserRole.ADMIN)
require_read_alerts = require_permission(Permission.READ_ALERTS)
require_write_alerts = require_permission(Permission.WRITE_ALERTS)
require_read_cases = require_permission(Permission.READ_CASES)
require_write_cases = require_permission(Permission.WRITE_CASES)
require_export_data = require_permission(Permission.EXPORT_DATA)

def get_current_user(token_data: TokenData = Depends(verify_token)) -> TokenData:
    """Get current authenticated user"""
    return token_data

def validate_input(data: str, max_length: int = 255) -> str:
    """Validate and sanitize input data"""
    if not data or not isinstance(data, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input data"
        )
    
    # Remove potentially dangerous characters
    sanitized = data.strip()[:max_length]
    
    # Basic SQL injection protection (additional to parameterized queries)
    dangerous_patterns = [
        "DROP", "DELETE", "UPDATE", "INSERT", "SELECT", 
        "--", ";", "/*", "*/", "xp_", "sp_"
    ]
    
    for pattern in dangerous_patterns:
        if pattern.lower() in sanitized.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input: potentially dangerous content detected"
            )
    
    return sanitized
