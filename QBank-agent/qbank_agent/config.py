import os
from pathlib import Path

# --- DIRECTORY CONFIGURATION ---
# PROJECT_ROOT can be overridden via env var so the package works as a subpackage.
# When imported from pydantic-ai-tutorial/src/, set QBANK_PROJECT_ROOT to point
# to the original QBank-agent directory (or any directory containing data/).
_env_root = os.getenv("QBANK_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).resolve()
else:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"

SLIDES_DIR = DATA_DIR / "input_slides"
JSONS_DIR = DATA_DIR / "parsed_jsons"
GEN_DIR = DATA_DIR / "generated_mcqs"
SYLLABUS_DIR = DATA_DIR / "syllabus"
EVAL_DIR = DATA_DIR / "eval"

def setup_directories():
    """Ensure all required data directories exist."""
    for directory in [DATA_DIR, SLIDES_DIR, JSONS_DIR, GEN_DIR, SYLLABUS_DIR, EVAL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# --- LLM CONFIGURATION ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.6"))

# Ensure you have your API key set in your environment
# Or you can hardcode it here (not recommended for production).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your api key")

# --- MCQ GENERATION CONFIGURATION ---
MAX_RETRIES = 3
REGEN_ATTEMPTS_PER_TOPIC = 2
QUESTIONS_PER_TOPIC = 5
DIFFICULTY_SCALE = "1-3"
MAX_LINES_PER_TOPIC = 500
