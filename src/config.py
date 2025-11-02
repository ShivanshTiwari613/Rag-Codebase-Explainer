# src/config.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Neon Database Configuration ---
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")


# --- Validation ---
# A simple check to ensure that the essential environment variables are loaded.
# The application will not start if these are missing.
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

if not NEON_DATABASE_URL:
    raise ValueError("NEON_DATABASE_URL is not set in the .env file.")