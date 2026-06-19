# DocuQuery

**Retrieval-Augmented Generation (RAG) chatbot for document question answering**

Final-year project built to explore how vector search and LLMs can be combined for
grounded Q&A over PDFs, resumes, reports, and notes.

Live demo: _Add your Streamlit Cloud URL after deployment_

GitHub: https://github.com/Dharunmd/rag-document-chatbot

---

## Problem

Manually searching long documents for specific details is slow. Generic chatbots can
also invent facts when they are not tied to a source document. This project implements
a small end-to-end RAG pipeline where every answer is generated from retrieved chunks
from the uploaded file.

## What it does

1. Upload a document (PDF / DOCX / TXT / Markdown)
2. Split text into chunks and store embeddings in ChromaDB
3. Ask questions in a chat interface
4. View the answer plus the source chunks used to generate it

## Architecture

```text
User Upload
    |
    v
Document Loader  -->  Text Splitter  -->  Embeddings  -->  ChromaDB
                                                                  |
User Question  -->  Query Embedding  -->  Retriever (MMR)  -------+
                                                                  |
                                                                  v
                                                         Gemini LLM + Prompt
                                                                  |
                                                                  v
                                                         Answer + Source Chunks
```

## Tech stack

| Layer | Tool |
|------|------|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Vector DB | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | Google Gemini |
| Language | Python 3.11 |

## Project structure

```text
rag-document-chatbot/
├── app.py                 # Streamlit UI
├── config.py              # App settings and env vars
├── src/
│   ├── document_loader.py # Parsing + chunking
│   ├── embeddings.py      # HuggingFace embeddings
│   ├── vector_store.py    # Chroma indexing
│   ├── retriever.py       # MMR retrieval
│   └── llm_chain.py       # Gemini QA chain
├── data/samples/          # Demo documents
├── tests/                 # Basic sanity checks
├── Dockerfile
└── docker-compose.yml
```

## Setup (local)

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Dharunmd/rag-document-chatbot.git
cd rag-document-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
```

Add your Gemini API key:

```env
GOOGLE_API_KEY=your_key_here
```

Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Quick demo without uploading

A sample report is included at `data/samples/project_report_sample.txt`.
Use **Load sample** in the sidebar and try questions like:

- What embedding model is used?
- What chunk size was chosen?
- What are the project limitations?

## Deployment

### Option A: Streamlit Community Cloud (recommended for resume link)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repository
4. Set main file path to `app.py`
5. Add `GOOGLE_API_KEY` under **Secrets**

Example `secrets.toml`:

```toml
GOOGLE_API_KEY = "your_key_here"
```

For local `.env` loading to work on Streamlit Cloud, the app already reads env vars via
`python-dotenv`. Streamlit secrets are also exposed as environment variables.

### Option B: Docker

```bash
cp .env.example .env
# edit .env with your API key

docker compose up --build
```

App runs at `http://localhost:8501`.

## Resume bullet points (you can adapt)

- Built an end-to-end RAG application using LangChain, ChromaDB, HuggingFace embeddings,
  and Google Gemini for grounded document Q&A
- Implemented document ingestion for PDF/DOCX/TXT, chunking, vector indexing, and MMR
  retrieval with source attribution in the UI
- Deployed a Streamlit web app with Docker support for reproducible demos

## Challenges faced

- **Model download time:** first run downloads the embedding model (~90 MB)
- **PDF quality:** scanned PDFs without text layers give poor results
- **Context limits:** very large documents need careful chunk sizing
- **API dependency:** chat answers require a valid Gemini API key

## Future improvements

- OCR for scanned PDFs
- Chat history persistence
- Multi-document collections
- Evaluation metrics (precision@k, answer faithfulness)
- FastAPI backend + React frontend split

## Author

**Dharun M**  
Computer Science and Engineering  
GitHub: [@Dharunmd](https://github.com/Dharunmd)

## License

MIT
