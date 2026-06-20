import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

APP_NAME = "DocuQuery"
APP_TAGLINE = "Retrieval-Augmented Document Q&A"
GITHUB_REPO = "https://github.com/Dharunmd/rag-document-chatbot"

UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DB_PATH = BASE_DIR / "vector_db"
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "samples"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "12"))

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]

SUGGESTED_QUESTIONS = [
    "Summarize the main objective of this document.",
    "What technologies or tools are mentioned?",
    "List the key limitations discussed.",
    "What methodology or approach is described?",
]
