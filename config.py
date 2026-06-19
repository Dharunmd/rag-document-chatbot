import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DB_PATH = BASE_DIR / "vector_db"
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "samples"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "10"))

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]
