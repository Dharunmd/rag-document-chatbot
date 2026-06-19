from pathlib import Path

from docx import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(documents):
    """Split loaded pages into overlapping chunks for retrieval."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks")
    return chunks


def load_and_split_pdf(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    pages = PyPDFLoader(str(path)).load()
    print(f"Loaded {len(pages)} pages from {path.name}")
    return split_documents(pages)


def load_and_split_docx(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = Document(str(path))
    text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    documents = [
        LangchainDocument(
            page_content=text,
            metadata={"source": path.name},
        )
    ]

    return split_documents(documents)


def load_and_split_text(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")

    documents = [
        LangchainDocument(
            page_content=text,
            metadata={"source": path.name},
        )
    ]

    return split_documents(documents)


def load_document(file_path: str):
    """Load and chunk a supported document type."""

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return load_and_split_pdf(file_path)
    if extension == ".docx":
        return load_and_split_docx(file_path)
    if extension in {".txt", ".md"}:
        return load_and_split_text(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


def get_document_info(chunks: list):
    """Return basic stats shown in the UI after indexing."""

    if not chunks:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "avg_chunk_size": 0,
            "pages": 0,
            "source": "unknown",
        }

    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    pages = max((chunk.metadata.get("page", 0) for chunk in chunks), default=0) + 1
    source = chunks[0].metadata.get("source", "unknown")

    return {
        "total_chunks": len(chunks),
        "total_characters": total_chars,
        "avg_chunk_size": total_chars // len(chunks),
        "pages": pages,
        "source": source,
    }
