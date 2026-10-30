import requests
from config import USDA_API_KEY, USDA_BASE_URL


def fetch_from_usda_api(food_name: str) -> dict:
    """Tier 3 Lookup: Query external USDA REST API."""
    clean_name = food_name.lower().strip()
    params = {"query": clean_name, "api_key": USDA_API_KEY}

    try:
        response = requests.get(f"{USDA_BASE_URL}/foods/search", params=params, timeout=5)
        if response.status_code == 200 and response.json().get('foods'):
            food_data = response.json()['foods'][0]
            nutrients = food_data.get('foodNutrients', [])

            return {
                "food_name": clean_name,
                "calories": next((n['value'] for n in nutrients if 'Energy' in n['nutrientName']), 0.0),
                "protein": next((n['value'] for n in nutrients if 'Protein' in n['nutrientName']), 0.0),
                "fat": next((n['value'] for n in nutrients if 'Total lipid (fat)' in n['nutrientName']), 0.0),
                "carbs": next((n['value'] for n in nutrients if 'Carbohydrate' in n['nutrientName']), 0.0)
            }
    except Exception as e:
        print(f"Error fetching from USDA API: {e}")

    return {"error": "Food item not found"}