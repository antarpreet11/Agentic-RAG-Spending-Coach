import os
from dotenv import load_dotenv

load_dotenv()

# Google Gemini API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
