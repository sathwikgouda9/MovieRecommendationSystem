import redis
import json
import os

_redis_client = None

def get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
    return _redis_client

def get_cached_recs(user_id: str):
    r    = get_redis()
    data = r.get(f"recs:{user_id}")
    return json.loads(data) if data else None

def set_cached_recs(user_id: str, recs: list, ttl: int = 300):
    r = get_redis()
    r.setex(f"recs:{user_id}", ttl, json.dumps(recs))
