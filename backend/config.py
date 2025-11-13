import os
from dotenv import load_dotenv

# Load .env file (if present)
load_dotenv()

# === Basic server configuration ===
API_PROTOCOL = os.getenv("API_PROTOCOL", "http")      # or "https"
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")

# === Construct the standardized API URL ===
API_BASE_URL = f"{API_PROTOCOL}://{API_HOST}:{API_PORT}"

# === Optional: Print once for clarity when imported ===
print(f"✅ Loaded API base URL: {API_BASE_URL}")
