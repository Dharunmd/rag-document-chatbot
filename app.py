import os
import streamlit as st

from src.document_loader import (
    load_document,
    get_document_info
)
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.llm_chain import create_qa_chain

MAX_FILE_SIZE_MB = 10

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# ----------------------------
# Session State
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False


# ----------------------------
# Header
# ----------------------------

st.title("RAG Chatbot 🤖")

st.caption(
    "Document Analysis Assistant powered by Retrieval-Augmented Generation (RAG)"
)

st.info(
    """
    🚧 Current Project Scope

    This version is optimized for:

    • Research Papers  
    • Resumes  
    • Academic Documents  
    • Technical Documentation  
    • Reports

    Best performance is achieved with structured text-based documents.
    """
)

# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:

    st.header("Project Scope")

    st.markdown("""
### Supported Documents

- PDF
- DOCX
- TXT
- Markdown

### Features

- Semantic Search
- Vector Database Retrieval
- Source References
- Gemini-Powered Answers
- Document Statistics

### Tech Stack

- LangChain
- ChromaDB
- Hugging Face Embeddings
- Google Gemini
- Streamlit
""")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ----------------------------
# File Upload
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt", "md"]
)

if uploaded_file:

    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"File size exceeds {MAX_FILE_SIZE_MB} MB limit."
        )
        st.stop()

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(
        f"📂 {uploaded_file.name} "
        f"({file_size_mb:.2f} MB)"
    )

    if st.button("🚀 Process Document"):

        st.session_state.messages = []

        with st.spinner("Processing document..."):

            chunks = load_document(file_path)

            create_vector_store(chunks)

            info = get_document_info(chunks)

        st.session_state.document_processed = True

        st.success(
            "✅ Document processed successfully!"
        )

        st.write("## 📊 Document Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Pages",
                info["pages"]
            )

        with col2:
            st.metric(
                "Chunks",
                info["total_chunks"]
            )

        with col3:
            st.metric(
                "Characters",
                info["total_characters"]
            )

        with col4:
            st.metric(
                "Avg Chunk Size",
                info["avg_chunk_size"]
            )


# ----------------------------
# Chat History
# ----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ----------------------------
# User Input
# ----------------------------

question = st.chat_input(
    "Ask about skills, experience, summary, findings, conclusions..."
)

if question:

    if not st.session_state.document_processed:

        st.warning(
            "Please upload and process a document first."
        )

        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    retriever = get_retriever()

    qa_chain = create_qa_chain(retriever)

    with st.spinner("Generating response..."):

        response = qa_chain.invoke(
            {"query": question}
        )

    answer = response["result"]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.write(answer)

    if "source_documents" in response:

        st.write("## 📚 Source References")

        for i, doc in enumerate(
            response["source_documents"],
            start=1
        ):

            with st.expander(
                f"📄 Source Chunk {i}"
            ):

                st.write(
                    doc.page_content
                )

                if hasattr(
                    doc,
                    "metadata"
                ):
                    st.caption(
                        f"Metadata: {doc.metadata}"
                    )