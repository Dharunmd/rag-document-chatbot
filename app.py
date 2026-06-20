import streamlit as st

from config import (
    ALLOWED_EXTENSIONS,
    APP_NAME,
    APP_TAGLINE,
    GITHUB_REPO,
    GOOGLE_API_KEY,
    MAX_FILE_SIZE_MB,
    SAMPLE_DOCS_DIR,
    SUGGESTED_QUESTIONS,
    SUPPORTED_TYPES,
    UPLOAD_DIR,
)
from src.pipeline import ask_document, index_document

st.set_page_config(
    page_title=f"{APP_NAME} | RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stat-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    .hero-sub { color: #64748b; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_stats" not in st.session_state:
    st.session_state.document_stats = None

if "index_time" not in st.session_state:
    st.session_state.index_time = None


def reset_chat():
    st.session_state.messages = []


def handle_index(file_path: str, display_name: str):
    reset_chat()
    result = index_document(file_path)
    st.session_state.document_processed = True
    st.session_state.document_name = display_name
    st.session_state.document_stats = result.info
    st.session_state.index_time = round(result.elapsed_seconds, 2)
    return result


def render_source_chunks(sources: list[dict]):
    if not sources:
        return

    with st.expander("Source chunks used for this answer"):
        for source in sources:
            chunk_id = source.get("metadata", {}).get("chunk_id", "?")
            page = source.get("metadata", {}).get("page")
            page_label = f" | page {page + 1}" if page is not None else ""
            st.markdown(f"**Chunk {chunk_id}{page_label}**")
            st.write(source["content"])
            st.divider()


def render_sidebar():
    with st.sidebar:
        st.title(APP_NAME)
        st.caption(APP_TAGLINE)
        st.markdown(f"[GitHub repo]({GITHUB_REPO})")
        st.markdown("---")

        st.subheader("Upload document")
        uploaded_file = st.file_uploader(
            "PDF, DOCX, TXT, or Markdown",
            type=SUPPORTED_TYPES,
            help=f"Max file size: {MAX_FILE_SIZE_MB} MB",
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"{uploaded_file.name} ({file_size_mb:.2f} MB)")

            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File exceeds the {MAX_FILE_SIZE_MB} MB limit.")
            elif st.button("Index document", use_container_width=True, type="primary"):
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                file_path = UPLOAD_DIR / uploaded_file.name
                file_path.write_bytes(uploaded_file.getbuffer())

                try:
                    with st.spinner("Indexing document..."):
                        result = handle_index(str(file_path), uploaded_file.name)
                    st.success(f"Indexed in {result.elapsed_seconds:.1f}s")
                except Exception as exc:
                    st.session_state.document_processed = False
                    st.error(f"Indexing failed: {exc}")

        st.markdown("---")
        st.subheader("Demo sample")

        SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        sample_files = sorted(
            path for path in SAMPLE_DOCS_DIR.glob("*")
            if path.suffix.lower() in ALLOWED_EXTENSIONS
        )

        if sample_files:
            sample_choice = st.selectbox(
                "Pick a sample",
                options=[path.name for path in sample_files],
            )
            if st.button("Load sample", use_container_width=True):
                sample_path = SAMPLE_DOCS_DIR / sample_choice
                try:
                    with st.spinner("Loading sample..."):
                        result = handle_index(str(sample_path), sample_choice)
                    st.success(f"Loaded in {result.elapsed_seconds:.1f}s")
                except Exception as exc:
                    st.error(f"Could not load sample: {exc}")

        st.markdown("---")

        if st.session_state.document_processed:
            st.success(f"Active: {st.session_state.document_name}")
            if st.session_state.index_time:
                st.caption(f"Indexed in {st.session_state.index_time}s")
        else:
            st.warning("No document indexed yet.")

        if not GOOGLE_API_KEY:
            st.error("Set GOOGLE_API_KEY in `.env` or Streamlit secrets.")

        if st.button("Clear chat", use_container_width=True):
            reset_chat()
            st.rerun()


def render_stats():
    stats = st.session_state.document_stats
    if not stats:
        return

    cols = st.columns(4)
    cols[0].metric("Pages", stats["pages"])
    cols[1].metric("Chunks", stats["total_chunks"])
    cols[2].metric("Characters", f"{stats['total_characters']:,}")
    cols[3].metric("Avg chunk", stats["avg_chunk_size"])


def submit_question(question: str):
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY is not configured.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    sources = []
    elapsed = None

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = ask_document(question)
                answer = result.answer
                sources = result.sources
                elapsed = result.elapsed_seconds
                st.markdown(answer)
                st.caption(f"Response time: {elapsed:.2f}s")
                render_source_chunks(sources)
            except Exception as exc:
                answer = f"Error: {exc}"
                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "elapsed": elapsed,
        }
    )


def render_chat():
    st.title(APP_NAME)
    st.markdown(
        f'<p class="hero-sub">{APP_TAGLINE} — upload a document, ask questions, '
        f"and inspect the exact chunks used to generate each answer.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.document_processed:
        st.info("Upload a document or load the demo sample from the sidebar to begin.")
        return

    render_stats()
    st.markdown("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("elapsed"):
                    st.caption(f"Response time: {message['elapsed']:.2f}s")
                render_source_chunks(message.get("sources", []))

    st.markdown("**Try asking:**")
    question_cols = st.columns(2)
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
        if question_cols[index % 2].button(suggestion, key=f"suggest_{index}"):
            submit_question(suggestion)
            st.rerun()

    question = st.chat_input("Ask a question about the indexed document...")
    if question:
        submit_question(question)
        st.rerun()


render_sidebar()
render_chat()
