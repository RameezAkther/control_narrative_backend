import time
import jwt
from config import JWT_SECRET, JWT_ALGORITHM

def create_access_token(data: dict, expires_in: int = 3600):
    payload = data.copy()
    payload["exp"] = time.time() + expires_in
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except:
        return None
