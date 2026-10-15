import psycopg2  # Import PostgreSQL adapter for Python
from psycopg2.extras import RealDictCursor  # Allows fetching DB rows as native Python dictionaries

# Connection configuration details for the  PostgreSQL database instance
DB_CONFIG = {
    "dbname": "food_db",        # Name of the database created in Postgres
    "user": "postgres",         # Database master user name
    "password": "contrapass", # Database password
    "host": "localhost",        # Host server running PostgreSQL
    "port": "5432"              # Default PostgreSQL port
}

def get_connection():
    """Establishes and returns an active connection object to PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)

def initialize_db():
    """Creates the tables for storing cached nutrition and image predictions if they don't exist."""
    # Context manager automatically handles connection open/close
    with get_connection() as conn:
        # Cursor handles execution of SQL statements; RealDictCursor formats returns as dicts
        with conn.cursor() as cursor:
            # Create a table to permanently cache USDA nutritional data
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
            # Create a table to track model inference history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id SERIAL PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    predicted_class VARCHAR(100) NOT NULL,
                    confidence_score FLOAT NOT NULL,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Commit the schema changes to the database
            conn.commit()

def save_detection(image_path: str, predicted_class: str, confidence: float):
    """Inserts a new model detection record into the inference_logs table."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Parameterized query prevents SQL injection attacks
            cursor.execute(
                """
                INSERT INTO inference_logs (image_path, predicted_class, confidence_score)
                VALUES (%s, %s, %s);
                """,
                (image_path, predicted_class, confidence)
            )
            conn.commit() # Save transaction

if __name__ == "__main__":
    # Run database initialization setup when script executed directly
    initialize_db()
    print("PostgreSQL tables successfully initialized.")