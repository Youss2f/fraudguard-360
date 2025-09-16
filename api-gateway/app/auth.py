from fastapi.security import HTTPAuthorizationCredentials
from fastapi import HTTPException, Depends
import jwt

JWT_SECRET = "secret_key"  # From env

def verify_token(credentials: HTTPAuthorizationCredentials = Depends()):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
