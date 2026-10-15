import json  # Used to serialize Python dicts to string format for Redis
import redis # Official Python client library for Redis
import requests # Used to perform external HTTP requests to USDA API

# Initialize Redis connection pointing to local instance running on port 6379
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# USDA API configurations
API_KEY = 'your_usda_api_key' # Replace with actual API key
BASE_URL = "https://api.nal.usda.gov/fdc/v1" # Base URL provided in README

def fetch_from_usda_api(food_name: str) -> dict:
    """Fallback function to fetch macro details directly from USDA API."""
    params = {"query": food_name, "api_key": API_KEY}
    response = requests.get(f"{BASE_URL}/foods/search", params=params)
    
    if response.status_code == 200 and response.json().get('foods'):
        # Extract first search match from USDA dataset response
        food_data = response.json()['foods'][0]
        
        # Structure unified response matching calories, protein, fat, carbs core metrics
        return {
            "food_name": food_name,
            "calories": next((n['value'] for n in food_data.get('foodNutrients', []) if 'Energy' in n['nutrientName']), 0),
            "protein": next((n['value'] for n in food_data.get('foodNutrients', []) if 'Protein' in n['nutrientName']), 0),
            "fat": next((n['value'] for n in food_data.get('foodNutrients', []) if 'Total lipid (fat)' in n['nutrientName']), 0),
            "carbs": next((n['value'] for n in food_data.get('foodNutrients', []) if 'Carbohydrate' in n['nutrientName']), 0)
        }
    return {"error": "Food item not found"}

def get_food_nutrition_cached(food_name: str) -> dict:
    """Checks Redis cache first; falls back to USDA API on cache miss."""
    cache_key = f"nutrition:{food_name.lower().strip()}" # Generate standard key string
    
    # Attempt to pull existing value from in-memory Redis store
    cached_data = redis_client.get(cache_key)
    
    if cached_data:
        print(f"CACHE HIT: Fetching '{food_name}' from Redis.")
        return json.loads(cached_data) # Deserialize JSON string back to Python dict
    
    print(f"CACHE MISS: Querying external USDA API for '{food_name}'.")
    # Fetch directly from API if not found in cache
    nutrition_info = fetch_from_usda_api(food_name)
    
    if "error" not in nutrition_info:
        # Cache result into Redis with Time-To-Live (ex=86400 seconds / 24 hours)
        redis_client.set(cache_key, json.dumps(nutrition_info), ex=86400)
        
    return nutrition_info

if __name__ == "__main__":
    # Test call 1: Should be a CACHE MISS (queries API)
    print(get_food_nutrition_cached("hamburger"))
    
    # Test call 2: Should be a CACHE HIT (retrieves instantly from Redis)
    print(get_food_nutrition_cached("hamburger"))