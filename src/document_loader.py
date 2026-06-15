# src/document_loader.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def load_and_split_pdf(file_path: str) -> list:
    """
    Load a PDF file and split it into overlapping chunks.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects (chunks)
    """
    
    # Validate file exists before processing
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")
    
    # Load PDF - each page becomes one Document object
    print(f"📄 Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"✅ Loaded {len(pages)} pages")
    
    # Split pages into overlapping chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Max characters per chunk
        chunk_overlap=200,      # Shared characters between chunks
        length_function=len,    # Measure size by character count
        separators=["\n\n", "\n", " ", ""]  # Split priority order
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"✅ Split into {len(chunks)} chunks")
    
    # Preview first chunk for verification
    if chunks:
        print(f"\n📌 Sample chunk preview:")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
    
    return chunks


def get_pdf_info(chunks: list) -> dict:
    """
    Get basic statistics about the loaded PDF chunks.
    
    Args:
        chunks: List of Document chunks
        
    Returns:
        Dictionary with PDF statistics
    """
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    
    return {
        "total_chunks": len(chunks),
        "total_characters": total_chars,
        "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
        "pages": max(chunk.metadata.get("page", 0) for chunk in chunks) + 1
    }