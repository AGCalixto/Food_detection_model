import os
# Responsibility: Holds all database credentials, Redis connections, and API endpoints.

# PostgreSQL Configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "food_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "yourpassword"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# USDA API Configuration
USDA_API_KEY = os.getenv("USDA_API_KEY", "your_usda_api_key")
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# Cache TTL Configuration (Default 24 Hours)
CACHE_TTL_SECONDS = 86400