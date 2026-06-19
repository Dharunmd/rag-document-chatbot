import streamlit as st

from config import (
    ALLOWED_EXTENSIONS,
    GOOGLE_API_KEY,
    MAX_FILE_SIZE_MB,
    SAMPLE_DOCS_DIR,
    SUPPORTED_TYPES,
    UPLOAD_DIR,
)
from src.document_loader import get_document_info, load_document
from src.llm_chain import create_qa_chain
from src.retriever import get_retriever
from src.vector_store import create_vector_store

st.set_page_config(
    page_title="DocuQuery | RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None


def reset_chat():
    st.session_state.messages = []


def process_document(file_path: str, display_name: str):
    reset_chat()

    with st.spinner("Reading document and building the search index..."):
        chunks = load_document(file_path)
        create_vector_store(chunks)
        info = get_document_info(chunks)

    st.session_state.document_processed = True
    st.session_state.document_name = display_name
    return info


def render_sidebar():
    with st.sidebar:
        st.title("DocuQuery")
        st.caption("RAG-based document Q&A system")
        st.markdown("---")

        st.subheader("1. Upload")
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=SUPPORTED_TYPES,
            help=f"Supported formats: {', '.join(SUPPORTED_TYPES)}. Max {MAX_FILE_SIZE_MB} MB.",
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"{uploaded_file.name} ({file_size_mb:.2f} MB)")

            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File exceeds the {MAX_FILE_SIZE_MB} MB limit.")
            elif st.button("Index document", use_container_width=True):
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                file_path = UPLOAD_DIR / uploaded_file.name
                file_path.write_bytes(uploaded_file.getbuffer())

                try:
                    info = process_document(str(file_path), uploaded_file.name)
                    st.success("Document indexed successfully.")
                    st.metric("Chunks", info["total_chunks"])
                    st.metric("Pages", info["pages"])
                except Exception as exc:
                    st.session_state.document_processed = False
                    st.error(f"Indexing failed: {exc}")

        st.markdown("---")
        st.subheader("2. Try a sample")

        SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        sample_files = sorted(SAMPLE_DOCS_DIR.glob("*"))
        sample_files = [
            path
            for path in sample_files
            if path.suffix.lower() in ALLOWED_EXTENSIONS
        ]

        if sample_files:
            sample_choice = st.selectbox(
                "Sample documents",
                options=[path.name for path in sample_files],
            )
            if st.button("Load sample", use_container_width=True):
                sample_path = SAMPLE_DOCS_DIR / sample_choice
                try:
                    info = process_document(str(sample_path), sample_choice)
                    st.success(f"Loaded {sample_choice}")
                    st.metric("Chunks", info["total_chunks"])
                except Exception as exc:
                    st.error(f"Could not load sample: {exc}")
        else:
            st.caption("Add files under data/samples/ to enable quick demos.")

        st.markdown("---")

        if st.session_state.document_processed:
            st.info(f"Active document: **{st.session_state.document_name}**")
        else:
            st.warning("No document indexed yet.")

        if not GOOGLE_API_KEY:
            st.error("GOOGLE_API_KEY is not set. Add it to `.env` before asking questions.")

        if st.button("Clear chat", use_container_width=True):
            reset_chat()
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            **Stack used**
            - Streamlit (UI)
            - LangChain (pipeline)
            - ChromaDB (vector store)
            - HuggingFace MiniLM (embeddings)
            - Google Gemini (answers)
            """
        )


def render_chat():
    st.title("Document Q&A Assistant")
    st.write(
        "Upload a PDF, DOCX, TXT, or Markdown file, index it, then ask questions "
        "about the content. Answers are grounded in retrieved document chunks."
    )

    if not st.session_state.document_processed:
        st.info("Start by uploading a document or loading a sample from the sidebar.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View source chunks"):
                    for index, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**Chunk {index}**")
                        st.write(source["content"])
                        if source.get("metadata"):
                            st.caption(str(source["metadata"]))

    question = st.chat_input("Ask something about the uploaded document...")

    if not question:
        return

    if not GOOGLE_API_KEY:
        st.error("Set GOOGLE_API_KEY in your environment before chatting.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            try:
                retriever = get_retriever()
                qa_chain = create_qa_chain(retriever)
                response = qa_chain.invoke({"query": question})
                answer = response["result"]
                sources = []

                for doc in response.get("source_documents", []):
                    sources.append(
                        {
                            "content": doc.page_content,
                            "metadata": getattr(doc, "metadata", {}),
                        }
                    )

                st.markdown(answer)

                if sources:
                    with st.expander("View source chunks"):
                        for index, source in enumerate(sources, start=1):
                            st.markdown(f"**Chunk {index}**")
                            st.write(source["content"])
                            if source.get("metadata"):
                                st.caption(str(source["metadata"]))

            except Exception as exc:
                answer = f"Something went wrong: {exc}"
                sources = []
                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


render_sidebar()
render_chat()
