import psycopg2
from config import DB_CONFIG
# Responsibility: Handles schema creation, reads, and writes to PostgreSQL.

def get_db_connection():
    """Establishes and returns an active connection to PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)

def initialize_db():
    """Initializes schema tables for nutritional records and model detection logs."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Table 1: Cached nutritional records from USDA API
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS food_nutrition (
                    id SERIAL PRIMARY KEY,
                    food_name VARCHAR(100) UNIQUE NOT NULL,
                    calories FLOAT NOT NULL,
                    protein FLOAT NOT NULL,
                    fat FLOAT NOT NULL,
                    carbs FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Table 2: Model inference history logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id SERIAL PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    predicted_class VARCHAR(100) NOT NULL,
                    confidence_score FLOAT NOT NULL,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

def fetch_from_postgres(food_name: str) -> dict:
    """Tier 2 Lookup: Retrieve nutritional data from PostgreSQL."""
    clean_name = food_name.lower().strip()
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT food_name, calories, protein, fat, carbs FROM food_nutrition WHERE food_name = %s;",
                (clean_name,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "food_name": row[0],
                    "calories": row[1],
                    "protein": row[2],
                    "fat": row[3],
                    "carbs": row[4]
                }
    return None

def save_to_postgres(nutrition_data: dict):
    """Persist newly fetched USDA nutritional data into PostgreSQL."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO food_nutrition (food_name, calories, protein, fat, carbs)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (food_name) DO NOTHING;
                """,
                (
                    nutrition_data["food_name"],
                    nutrition_data["calories"],
                    nutrition_data["protein"],
                    nutrition_data["fat"],
                    nutrition_data["carbs"]
                )
            )
            conn.commit()

if __name__ == "__main__":
    initialize_db()
    print("PostgreSQL tables successfully initialized.")