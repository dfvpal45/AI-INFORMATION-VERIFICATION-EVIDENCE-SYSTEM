"""Best-effort Redis cache; the app continues normally when Redis is unavailable."""
import hashlib, json
from .config import settings

def _client():
    try:
        import redis
        c=redis.Redis.from_url(settings.redis_url, socket_connect_timeout=0.25, socket_timeout=0.25, decode_responses=True)
        c.ping(); return c
    except Exception: return None

def get_cached(text: str, kind: str):
    c=_client()
    if not c: return None
    key='aivs:v2:'+kind+':'+hashlib.sha256(text.encode('utf-8')).hexdigest()
    try:
        value=c.get(key); return json.loads(value) if value else None
    except Exception: return None

def set_cached(text: str, kind: str, result: dict, ttl: int=1800):
    c=_client()
    if not c: return
    key='aivs:v2:'+kind+':'+hashlib.sha256(text.encode('utf-8')).hexdigest()
    try: c.setex(key,ttl,json.dumps(result,ensure_ascii=False))
    except Exception: pass
