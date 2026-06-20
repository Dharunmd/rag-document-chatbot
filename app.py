import streamlit as st

from config import (
    APP_NAME,
    APP_TAGLINE,
    GROQ_API_KEY,
    MAX_FILE_SIZE_MB,
    SUPPORTED_TYPES,
    UPLOAD_DIR,
)

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📄",
    layout="wide",
)

# ----------------------------
# Startup Check
# ----------------------------

st.title("📄 Document RAG Chatbot")

st.success("Application started successfully")

# Create upload directory
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------
# Lazy Load Pipeline
# ----------------------------

def load_pipeline():
    from src.pipeline import (
        ask_document,
        index_document,
    )

    return ask_document, index_document


try:
    ask_document, index_document = load_pipeline()

except Exception as e:

    st.error(
        f"Pipeline failed to load:\n{e}"
    )

    st.stop()

# ----------------------------
# Session State
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_stats" not in st.session_state:
    st.session_state.document_stats = None

# ----------------------------
# Helper Functions
# ----------------------------

def clear_chat():
    st.session_state.messages = []


def process_document(
    file_path,
    file_name,
):
    result = index_document(file_path)

    st.session_state.document_processed = True
    st.session_state.document_name = file_name
    st.session_state.document_stats = result.info

    return result

# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:

    st.title(APP_NAME)

    st.caption(APP_TAGLINE)

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=SUPPORTED_TYPES
    )

    if uploaded_file:

        file_size_mb = (
            uploaded_file.size
            / (1024 * 1024)
        )

        st.caption(
            f"{uploaded_file.name} "
            f"({file_size_mb:.2f} MB)"
        )

        if file_size_mb > MAX_FILE_SIZE_MB:

            st.error(
                f"File exceeds "
                f"{MAX_FILE_SIZE_MB} MB limit."
            )

        elif st.button(
            "Process Document",
            use_container_width=True,
        ):

            file_path = (
                UPLOAD_DIR
                / uploaded_file.name
            )

            file_path.write_bytes(
                uploaded_file.getbuffer()
            )

            try:

                with st.spinner(
                    "Indexing document..."
                ):

                    result = process_document(
                        str(file_path),
                        uploaded_file.name,
                    )

                st.success(
                    f"Indexed in "
                    f"{result.elapsed_seconds:.2f}s"
                )

            except Exception as e:

                st.error(
                    f"Indexing failed:\n{e}"
                )

    st.divider()

    if not GROQ_API_KEY:

        st.error(
            "GROQ_API_KEY is missing"
        )

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):

        clear_chat()
        st.rerun()

# ----------------------------
# Main Content
# ----------------------------

st.markdown(
    """
Upload a document and ask questions about:

- 📑 Research Papers
- 📄 Resumes
- 📚 Reports
- 📝 Documentation

Powered by:
LangChain • ChromaDB • HuggingFace • Groq
"""
)

# ----------------------------
# Stats
# ----------------------------

if st.session_state.document_stats:

    stats = st.session_state.document_stats

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Pages",
        stats["pages"]
    )

    c2.metric(
        "Chunks",
        stats["total_chunks"]
    )

    c3.metric(
        "Characters",
        f"{stats['total_characters']:,}"
    )

    c4.metric(
        "Avg Chunk",
        stats["avg_chunk_size"]
    )

# ----------------------------
# Chat History
# ----------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# ----------------------------
# Chat Input
# ----------------------------

question = st.chat_input(
    "Ask a question about the document..."
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
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:

        with st.spinner("Thinking..."):

            result = ask_document(question)

            answer = result.answer

    except Exception as e:

        import traceback

        answer = (
            f"Error:\n\n"
            f"{traceback.format_exc()}"
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)