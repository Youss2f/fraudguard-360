"""
FraudGuard 360° API Gateway
Central entry point for all microservices

This service provides:
- Authentication and authorization
- Request routing and load balancing
- API rate limiting and throttling
- Request/response transformation
- Centralized logging and monitoring
- Service discovery and health checking
"""

import os
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import httpx
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import redis.asyncio as redis
from passlib.context import CryptContext
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('gateway_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('gateway_request_duration_seconds', 'Request latency', ['method', 'endpoint'])
SERVICE_REQUESTS = Counter('gateway_service_requests_total', 'Requests to backend services', ['service', 'status'])

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "fraudguard-360-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))

# Service endpoints
SERVICES = {
    "ai-service": os.getenv("AI_SERVICE_URL", "http://ai-service:8001"),
    "graph-service": os.getenv("GRAPH_SERVICE_URL", "http://graph-service:8002"),
    "alerting-service": os.getenv("ALERTING_SERVICE_URL", "http://alerting-service:8003"),
}

# Global variables
redis_client: Optional[redis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Pydantic models
class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=6)
    role: str = Field(default="analyst", regex=r'^(admin|analyst|viewer)$')
    department: str = Field(default="fraud-detection")

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_info: Dict[str, Any]

class User(BaseModel):
    id: str
    username: str
    email: str
    role: str
    department: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

class ServiceHealth(BaseModel):
    service: str
    status: str
    response_time_ms: float
    last_check: datetime

class APIResponse(BaseModel):
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

async def get_redis_client():
    """Dependency to get Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(REDIS_URL)
    return redis_client

async def get_http_client():
    """Dependency to get HTTP client"""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=30.0)
    return http_client

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        # Check if token is blacklisted (optional - implement token blacklisting)
        redis_client = await get_redis_client()
        blacklisted = await redis_client.get(f"blacklist:{credentials.credentials}")
        if blacklisted:
            raise HTTPException(status_code=401, detail="Token has been revoked")
        
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

async def check_rate_limit(request: Request, redis_client: redis.Redis):
    """Check rate limit for the request"""
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    
    current_requests = await redis_client.get(key)
    if current_requests is None:
        await redis_client.setex(key, 60, 1)
    else:
        current_count = int(current_requests)
        if current_count >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        await redis_client.incr(key)

async def proxy_request(
    service: str,
    path: str,
    method: str,
    params: Dict[str, Any] = None,
    json_data: Dict[str, Any] = None,
    headers: Dict[str, str] = None
) -> Dict[str, Any]:
    """Proxy request to backend service"""
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service {service} not found")
    
    base_url = SERVICES[service]
    url = f"{base_url}/{path.lstrip('/')}"
    
    http_client = await get_http_client()
    
    try:
        start_time = time.time()
        
        response = await http_client.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            headers=headers
        )
        
        response_time = time.time() - start_time
        
        # Record metrics
        SERVICE_REQUESTS.labels(service=service, status=str(response.status_code)).inc()
        
        if response.status_code >= 400:
            logger.warning("Service request failed", 
                         service=service, 
                         url=url, 
                         status_code=response.status_code,
                         response_time=response_time)
        
        return {
            "status_code": response.status_code,
            "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
            "headers": dict(response.headers),
            "response_time": response_time
        }
        
    except httpx.RequestError as e:
        logger.error("Service request error", service=service, url=url, error=str(e))
        SERVICE_REQUESTS.labels(service=service, status="error").inc()
        raise HTTPException(status_code=503, detail=f"Service {service} unavailable")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting FraudGuard 360° API Gateway")
    
    # Initialize Redis connection
    global redis_client, http_client
    redis_client = redis.from_url(REDIS_URL)
    http_client = httpx.AsyncClient(timeout=30.0)
    
    # Health check all services
    for service_name, service_url in SERVICES.items():
        try:
            response = await http_client.get(f"{service_url}/health", timeout=5.0)
            logger.info("Service health check", service=service_name, status=response.status_code)
        except Exception as e:
            logger.warning("Service health check failed", service=service_name, error=str(e))
    
    yield
    
    # Shutdown
    if redis_client:
        await redis_client.close()
    if http_client:
        await http_client.aclose()
    logger.info("API Gateway shutdown complete")

# FastAPI app initialization
app = FastAPI(
    title="FraudGuard 360° API Gateway",
    description="Central entry point for all microservices",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Request middleware for logging and metrics"""
    start_time = time.time()
    
    # Check rate limiting
    redis_client = await get_redis_client()
    await check_rate_limit(request, redis_client)
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    process_time = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code)
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    # Add headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = str(id(request))
    
    return response

@app.get("/health")
async def health_check():
    """Gateway health check"""
    return {
        "status": "healthy",
        "services": list(SERVICES.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/auth/login", response_model=Token)
async def login(user_login: UserLogin, redis_client: redis.Redis = Depends(get_redis_client)):
    """User login endpoint"""
    try:
        # In production, verify against user database
        # For demo, using simple hardcoded users
        demo_users = {
            "admin": {"password": "admin123", "role": "admin", "email": "admin@fraudguard360.com"},
            "analyst": {"password": "analyst123", "role": "analyst", "email": "analyst@fraudguard360.com"},
            "viewer": {"password": "viewer123", "role": "viewer", "email": "viewer@fraudguard360.com"}
        }
        
        if user_login.username not in demo_users:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        user_info = demo_users[user_login.username]
        if not pwd_context.verify(user_login.password, pwd_context.hash(user_info["password"])):
            # For demo, simple password check
            if user_login.password != user_info["password"]:
                raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_login.username, "role": user_info["role"]},
            expires_delta=access_token_expires
        )
        
        # Store session in Redis
        session_key = f"session:{user_login.username}"
        session_data = {
            "user_id": user_login.username,
            "role": user_info["role"],
            "login_time": datetime.now().isoformat(),
            "token": access_token
        }
        await redis_client.setex(session_key, ACCESS_TOKEN_EXPIRE_MINUTES * 60, str(session_data))
        
        logger.info("User logged in", username=user_login.username, role=user_info["role"])
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_info={
                "username": user_login.username,
                "role": user_info["role"],
                "email": user_info["email"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", username=user_login.username, error=str(e))
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/auth/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """User logout endpoint"""
    try:
        # Add token to blacklist
        await redis_client.setex(
            f"blacklist:{credentials.credentials}",
            ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "revoked"
        )
        
        logger.info("User logged out")
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(status_code=500, detail="Logout failed")

# AI Service routes
@app.post("/api/ai/scores")
async def ai_fraud_scores(
    request: Request,
    user: dict = Depends(verify_token)
):
    """Proxy to AI service for fraud scoring"""
    request_data = await request.json()
    
    result = await proxy_request(
        service="ai-service",
        path="scores",
        method="POST",
        json_data=request_data
    )
    
    if result["status_code"] >= 400:
        raise HTTPException(status_code=result["status_code"], detail=result["data"])
    
    return result["data"]

# Graph Service routes
@app.get("/api/graph/subgraph/{user_id}")
async def get_user_subgraph(
    user_id: str,
    depth: int = 2,
    min_amount: float = 0,
    hours_back: int = 24,
    user: dict = Depends(verify_token)
):
    """Proxy to Graph service for subgraph extraction"""
    result = await proxy_request(
        service="graph-service",
        path=f"subgraph/{user_id}",
        method="GET",
        params={
            "depth": depth,
            "min_amount": min_amount,
            "hours_back": hours_back
        }
    )
    
    if result["status_code"] >= 400:
        raise HTTPException(status_code=result["status_code"], detail=result["data"])
    
    return result["data"]

@app.get("/api/graph/patterns/fraud")
async def get_fraud_patterns(
    confidence_threshold: float = 0.7,
    hours_back: int = 24,
    limit: int = 100,
    user: dict = Depends(verify_token)
):
    """Proxy to Graph service for fraud pattern detection"""
    result = await proxy_request(
        service="graph-service",
        path="patterns/fraud",
        method="GET",
        params={
            "confidence_threshold": confidence_threshold,
            "hours_back": hours_back,
            "limit": limit
        }
    )
    
    if result["status_code"] >= 400:
        raise HTTPException(status_code=result["status_code"], detail=result["data"])
    
    return result["data"]

@app.get("/api/system/services/health")
async def check_services_health(user: dict = Depends(verify_token)) -> List[ServiceHealth]:
    """Check health of all backend services"""
    health_checks = []
    http_client = await get_http_client()
    
    for service_name, service_url in SERVICES.items():
        try:
            start_time = time.time()
            response = await http_client.get(f"{service_url}/health", timeout=5.0)
            response_time = (time.time() - start_time) * 1000
            
            health_checks.append(ServiceHealth(
                service=service_name,
                status="healthy" if response.status_code == 200 else "unhealthy",
                response_time_ms=response_time,
                last_check=datetime.now()
            ))
        except Exception as e:
            health_checks.append(ServiceHealth(
                service=service_name,
                status="unhealthy",
                response_time_ms=0,
                last_check=datetime.now()
            ))
            logger.error("Service health check failed", service=service_name, error=str(e))
    
    return health_checks

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)