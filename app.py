import os
import streamlit as st

from src.document_loader import (
    load_and_split_pdf,
    get_pdf_info
)
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.llm_chain import create_qa_chain

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

st.title("📄 RAG Chatbot")
st.markdown(
    "Upload a PDF document and ask questions about its contents."
)

# Sidebar
with st.sidebar:

    st.header("About")

    st.markdown("""
    This application allows users to upload PDF documents
    and ask questions based on the document content.

    **Tech Stack**
    - LangChain
    - ChromaDB
    - Hugging Face Embeddings
    - Google Gemini
    - Streamlit
    """)

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Upload Section
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"Uploaded: {uploaded_file.name}")

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            chunks = load_and_split_pdf(file_path)

            create_vector_store(chunks)

            info = get_pdf_info(chunks)

        st.session_state.document_processed = True

        st.success("Document processed successfully!")

        st.write("## Document Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Pages", info["pages"])
            st.metric("Chunks", info["total_chunks"])

        with col2:
            st.metric("Characters", info["total_characters"])
            st.metric("Average Chunk Size", info["avg_chunk_size"])

# Display Previous Messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat Input
question = st.chat_input(
    "Ask a question about the document..."
)

if question:

    if not st.session_state.document_processed:
        st.warning("Please process a document first.")
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

        st.write("## Retrieved Context")

        for i, doc in enumerate(
            response["source_documents"],
            start=1
        ):

            with st.expander(f"Chunk {i}"):

                st.write(doc.page_content)

                if hasattr(doc, "metadata"):
                    st.caption(
                        f"Metadata: {doc.metadata}"
                    )