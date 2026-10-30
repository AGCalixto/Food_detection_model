import json
import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, CACHE_TTL_SECONDS
from db import fetch_from_postgres, save_to_postgres
from usda_client import fetch_from_usda_api

# Redis Client Instance
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def get_food_nutrition_optimized(food_name: str, cache_ttl: int = CACHE_TTL_SECONDS) -> dict:
    """
    3-Tier Caching Strategy:
    1. Check Redis (In-Memory). On Hit -> Reset 24-hr TTL clock and return.
    2. Check PostgreSQL (Persistent DB). On Hit -> Backfill Redis and return.
    3. Call USDA API (External REST Call). On Hit -> Save to Postgres & Redis, then return.
    """
    clean_name = food_name.lower().strip()
    cache_key = f"nutrition:{clean_name}"

    # -------------------------------------------------------------
    # TIER 1: Check Redis
    # -------------------------------------------------------------
    cached_data = redis_client.get(cache_key)
    if cached_data:
        # Extend/Refresh TTL on cache hit
        redis_client.expire(cache_key, cache_ttl)
        return json.loads(cached_data)

    # -------------------------------------------------------------
    # TIER 2: Check PostgreSQL
    # -------------------------------------------------------------
    pg_data = fetch_from_postgres(clean_name)
    if pg_data:
        # Backfill Redis cache for subsequent calls
        redis_client.set(cache_key, json.dumps(pg_data), ex=cache_ttl)
        return pg_data

    # -------------------------------------------------------------
    # TIER 3: Call External USDA API
    # -------------------------------------------------------------
    api_data = fetch_from_usda_api(clean_name)

    # Store valid results across both storage tiers
    if "error" not in api_data:
        save_to_postgres(api_data)
        redis_client.set(cache_key, json.dumps(api_data), ex=cache_ttl)

    return api_data