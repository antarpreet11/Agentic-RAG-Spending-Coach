from diskcache import Cache
from pathlib import Path
from core.config import CACHE_DIR

_cache = Cache(Path(CACHE_DIR) / "cache.db")

def get_cache(namespace: str):
    # Use the cache directly with namespace prefix
    return _cache

def cached(ns: str, key: str):
    full_key = f"{ns}:{key}"
    return _cache.get(full_key)

def set_cached(ns: str, key: str, value, expire=None):
    full_key = f"{ns}:{key}"
    _cache.set(full_key, value, expire=expire)
    return value
