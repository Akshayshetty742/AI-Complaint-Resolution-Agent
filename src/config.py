import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
PROMPTS_DIR = BASE_DIR / "prompts"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

CATEGORIES = [
    "Refund",
    "Return",
    "Delivery",
    "Damaged Product",
    "Warranty",
    "Payment",
    "General Support",
]

SENTIMENTS = ["Positive", "Neutral", "Negative"]
PRIORITIES = ["Low", "Medium", "High"]
