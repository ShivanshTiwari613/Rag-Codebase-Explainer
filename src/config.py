# src/config.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Chroma Cloud Configuration ---
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT_ID = os.getenv("CHROMA_TENANT_ID")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# --- Validation ---
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

if not CHROMA_API_KEY:
    raise ValueError("CHROMA_API_KEY is not set in the .env file.")

if not CHROMA_TENANT_ID:
    raise ValueError("CHROMA_TENANT_ID is not set in the .env file.")

if not CHROMA_DATABASE:
    raise ValueError("CHROMA_DATABASE is not set in the .env file.")
