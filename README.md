# DocuQuery

**Retrieval-Augmented Generation (RAG) chatbot for grounded document Q&A**

RAG project — upload PDFs, resumes, or reports and ask questions with
source-backed answers powered by vector search + Google Gemini.

[![CI](https://github.com/Dharunmd/rag-document-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Dharunmd/rag-document-chatbot/actions/workflows/ci.yml)
**Live demo:** _Deploy on [Streamlit Cloud]([https://share.streamlit.io[](https://rag-document-chatbot-6t9bpuumgoektqfsmjx4zd.streamlit.app)]
**Repository:** https://github.com/Dharunmd/rag-document-chatbot

---

## Highlights (for recruiters)

| Feature | Detail |
|---------|--------|
| Document ingestion | PDF, DOCX, TXT, Markdown |
| Retrieval | ChromaDB + MMR (Maximal Marginal Relevance) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (384-dim, cached in memory) |
| LLM | Google Gemini 2.0 Flash |
| Explainability | Source chunks shown with every answer |
| Deployment | Streamlit UI, Docker, GitHub Actions CI |

## Problem statement

Searching long documents manually is slow, and generic chatbots hallucinate when they
are not tied to a source. DocuQuery implements a full RAG pipeline: chunk the document,
embed and store vectors, retrieve relevant passages, and generate answers strictly
from retrieved context.

## Architecture

```text
Upload (PDF/DOCX/TXT/MD)
        |
        v
  Document Loader --> Text Splitter (1000/200 overlap)
        |
        v
  HuggingFace Embeddings --> ChromaDB (persisted index)
        |
User Question --> Query Embedding --> MMR Retriever (k=4)
        |
        v
  Gemini 2.0 Flash + grounded prompt --> Answer + source chunks
```

## Tech stack

- **Python 3.11** · **Streamlit** · **LangChain**
- **ChromaDB** · **sentence-transformers** · **Google Gemini API**
- **Docker** · **GitHub Actions**

## Project structure

```text
rag-document-chatbot/
├── app.py                  # Streamlit frontend
├── config.py               # Environment-driven settings
├── src/
│   ├── document_loader.py  # Parsing + chunking
│   ├── embeddings.py       # Cached HuggingFace model
│   ├── vector_store.py     # Chroma indexing
│   ├── retriever.py        # MMR retrieval
│   ├── llm_chain.py        # Gemini QA chain
│   └── pipeline.py         # Index + query orchestration
├── tests/                  # Pipeline unit tests
├── data/samples/           # Demo document
├── Dockerfile
└── docker-compose.yml
```

## Quick start

```bash
git clone https://github.com/Dharunmd/rag-document-chatbot.git
cd rag-document-chatbot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add GOOGLE_API_KEY from https://aistudio.google.com/app/apikey

streamlit run app.py
```

Open http://localhost:8501 → sidebar → **Load sample** → ask a question.

## Deployment

### Streamlit Cloud (free, best for resume link)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Repository: `Dharunmd/rag-document-chatbot`, branch: `main`, file: `app.py`
4. Under **Secrets**, add:

```toml
GOOGLE_API_KEY = "your_key_here"
```

5. Deploy and add the public URL to your resume/README.

### Docker

```bash
cp .env.example .env   # add your API key
docker compose up --build
```

## Resume bullet points

Copy-paste and tweak:

> **DocuQuery — RAG Document Q&A System** | Python, LangChain, ChromaDB, Gemini  
> Built an end-to-end retrieval-augmented generation app that indexes PDF/DOCX/TXT
> documents into ChromaDB using HuggingFace embeddings and answers user queries via
> Google Gemini with MMR retrieval and inline source attribution.  
> Deployed with Streamlit and Docker; added CI pipeline with pytest on GitHub Actions.

## Design decisions

- **MMR retrieval** — reduces redundant chunks in the context window
- **Chunk overlap (200 chars)** — keeps sentences split across boundaries retrievable
- **`lru_cache` on embeddings** — model loads once per process, not per query
- **Grounded prompt** — model instructed to refuse when context lacks the answer

## Limitations & future work

- Scanned PDFs without a text layer need OCR (not implemented)
- Single-document mode only (multi-doc collections planned)
- No persistent chat history across sessions

## Author

**Dharun M** · [GitHub](https://github.com/Dharunmd)

## License

MIT — see [LICENSE](LICENSE)
