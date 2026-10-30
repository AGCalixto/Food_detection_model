import json
import pytest
from unittest.mock import patch
from cache import get_food_nutrition_optimized

def test_redis_hit_resets_ttl(mocker):
    """Tier 1: Redis hit returns instantly and resets TTL without touching DB or API."""
    mock_redis_data = json.dumps({
        "food_name": "hamburger",
        "calories": 250.0,
        "protein": 12.0,
        "fat": 14.0,
        "carbs": 20.0
    })

    # Mock Redis client
    mock_redis = mocker.patch("cache.redis_client")
    mock_redis.get.return_value = mock_redis_data

    # UPDATED PATCH PATHS: Target db and usda_client modules
    mock_pg = mocker.patch("cache.fetch_from_postgres")
    mock_api = mocker.patch("cache.fetch_from_usda_api")

    result = get_food_nutrition_optimized("hamburger")

    assert result["food_name"] == "hamburger"
    assert result["calories"] == 250.0
    mock_redis.get.assert_called_once_with("nutrition:hamburger")
    mock_redis.expire.assert_called_once_with("nutrition:hamburger", 86400)
    mock_pg.assert_not_called()
    mock_api.assert_not_called()


def test_postgres_hit_populates_redis(mocker):
    """Tier 2: Redis miss checks Postgres, populates Redis, and skips external API."""
    mock_pg_data = {
        "food_name": "pizza",
        "calories": 280.0,
        "protein": 10.0,
        "fat": 12.0,
        "carbs": 32.0
    }

    mock_redis = mocker.patch("cache.redis_client")
    mock_redis.get.return_value = None

    # UPDATED PATCH PATH: Target fetch_from_postgres imported in cache
    mock_pg = mocker.patch("cache.fetch_from_postgres", return_value=mock_pg_data)
    mock_api = mocker.patch("cache.fetch_from_usda_api")

    result = get_food_nutrition_optimized("pizza")

    assert result["food_name"] == "pizza"
    mock_pg.assert_called_once_with("pizza")
    mock_redis.set.assert_called_once_with(
        "nutrition:pizza", json.dumps(mock_pg_data), ex=86400
    )
    mock_api.assert_not_called()


def test_api_fallback_and_persistence(mocker):
    """Tier 3: Miss on Redis & Postgres triggers USDA API, persists in PG, and caches in Redis."""
    mock_api_data = {
        "food_name": "apple",
        "calories": 52.0,
        "protein": 0.3,
        "fat": 0.2,
        "carbs": 14.0
    }

    mock_redis = mocker.patch("cache.redis_client")
    mock_redis.get.return_value = None
    mocker.patch("cache.fetch_from_postgres", return_value=None)
    mocker.patch("cache.fetch_from_usda_api", return_value=mock_api_data)
    mock_save_pg = mocker.patch("cache.save_to_postgres")

    result = get_food_nutrition_optimized("apple")

    assert result["calories"] == 52.0
    mock_save_pg.assert_called_once_with(mock_api_data)
    mock_redis.set.assert_called_once_with(
        "nutrition:apple", json.dumps(mock_api_data), ex=86400
    )