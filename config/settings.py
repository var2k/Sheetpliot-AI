"""
SheetPilot AI - Configuration and Settings
Manages global constants, application parameters, file size limits, and environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = DATA_DIR / "samples"
TEMPLATE_DIR = DATA_DIR / "templates"
OUTPUT_DIR = DATA_DIR / "outputs"
WORKFLOW_DIR = DATA_DIR / "workflows"

# Ensure all critical folders exist
for folder in [DATA_DIR, UPLOAD_DIR, SAMPLE_DIR, TEMPLATE_DIR, OUTPUT_DIR, WORKFLOW_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Application Metadata
APP_NAME = "SheetPilot AI"
APP_SUBTITLE = "Intelligent Excel & Tabular Data Wrangling, Cleaning & Transformation Copilot"
APP_VERSION = "1.0.0"
APP_ICON = "⚡"

# File limits and formats
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_EXTENSIONS = [".xlsx", ".xls", ".csv", ".tsv"]

# Data Profiling & Display limits
MAX_PREVIEW_ROWS = int(os.getenv("MAX_PREVIEW_ROWS", "500"))
PROFILE_SAMPLE_SIZE = int(os.getenv("PROFILE_SAMPLE_SIZE", "10000"))
IQR_OUTLIER_FACTOR = float(os.getenv("IQR_OUTLIER_FACTOR", "1.5"))

# AI / LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_LOCAL_PARSER_FALLBACK = True  # Always available even without OpenAI key

# UI Theme Tokens
THEME = {
    "primary": "#4F46E5",       # Indigo
    "primary_light": "#6366F1",
    "secondary": "#06B6D4",     # Cyan
    "accent": "#10B981",        # Emerald
    "danger": "#EF4444",        # Rose / Red
    "warning": "#F59E0B",       # Amber
    "bg_dark": "#0F172A",       # Slate 900
    "card_dark": "#1E293B",     # Slate 800
    "text_light": "#F8FAFC",
    "text_muted": "#94A3B8"
}
