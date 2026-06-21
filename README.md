# DocuQuery

**Retrieval-Augmented Generation (RAG) chatbot for grounded document Q&A**

RAG project — upload PDFs, resumes, or reports and ask questions with
source-backed answers powered by vector search + Google Gemini.

[![CI](https://github.com/Dharunmd/rag-document-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Dharunmd/rag-document-chatbot/actions/workflows/ci.yml)
**Live demo:** _Deploy on [(https://rag-document-chatbot-mtkgf2ug7veksbd8mcsf2s.streamlit.app)]
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








# 📄 DocuQuery — Chat With Your Documents, Backed by Real Sources

**Ask questions about any PDF, DOCX, or TXT file and get answers with the exact source chunks that prove them — no hallucinations, no guessing.**

[![CI](https://github.com/Dharunmd/rag-document-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Dharunmd/rag-document-chatbot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/Dharunmd/rag-document-chatbot)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit App](https://img.shields.io/badge/demo-live-brightgreen)](https://rag-document-chatbot-mtkgf2ug7veksbd8mcsf2s.streamlit.app)
[![GitHub stars](https://img.shields.io/github/stars/Dharunmd/rag-document-chatbot?style=social)](https://github.com/Dharunmd/rag-document-chatbot/stargazers)

**[🚀 Try the live demo](https://rag-document-chatbot-mtkgf2ug7veksbd8mcsf2s.streamlit.app)** · **[⭐ Star this repo](#)** if it's useful to you — it helps others discover it too.

---

### 🎬 See it in action

> _Suggested: 10–15s GIF here showing: upload a PDF → ask a question → answer appears with highlighted source chunk._
> `<img src="docs/demo.gif" width="700">`

![screenshot placeholder](docs/screenshot-main.png)

---

## Why DocuQuery?

Generic chatbots (and copy-pasting into ChatGPT) make things up when they don't know the answer. DocuQuery only answers from **your** document, and shows you the exact passage it used — so you can verify every answer in one click.

## ✨ Features

- 📂 **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- 🔍 **MMR retrieval** — diverse, non-redundant context chunks (ChromaDB)
- 🧠 **Free local embeddings** — HuggingFace `all-MiniLM-L6-v2`, no embedding API cost
- ⚡ **Gemini 2.0 Flash** — fast, cheap, accurate generation
- 📌 **Source-cited answers** — every response shows the chunk it came from
- 🚫 **Grounded refusal** — says "I don't know" instead of hallucinating
- 🐳 **One-command Docker deploy**
- ✅ **CI-tested** with pytest + GitHub Actions

## 🏗 Architecture

```text
Upload (PDF/DOCX/TXT/MD)
        │
        ▼
  Document Loader ──▶ Text Splitter (1000 chars / 200 overlap)
        │
        ▼
  HuggingFace Embeddings ──▶ ChromaDB (persisted vector index)
        │
User Question ──▶ Query Embedding ──▶ MMR Retriever (k=4)
        │
        ▼
  Gemini 2.0 Flash + grounded prompt ──▶ Answer + cited source chunks
```

## 🚀 Quick Start (5 minutes)

```bash
git clone https://github.com/Dharunmd/rag-document-chatbot.git
cd rag-document-chatbot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your free GOOGLE_API_KEY from https://aistudio.google.com/app/apikey

streamlit run app.py
```

Open **http://localhost:8501** → sidebar → **Load sample** → ask: *"What is this document about?"*

### 🐳 Or run with Docker

```bash
cp .env.example .env   # add your API key
docker compose up --build
```

### ☁️ Or skip setup entirely

**[Try the hosted demo →](https://rag-document-chatbot-mtkgf2ug7veksbd8mcsf2s.streamlit.app)**

## 💬 Usage Example

```python
from src.pipeline import RAGPipeline

rag = RAGPipeline()
rag.index_document("data/samples/sample.pdf")
result = rag.query("What were the key findings?")

print(result["answer"])
print(result["sources"])  # exact chunks used
```

## 🧰 Tech Stack

Python 3.11 · Streamlit · LangChain · ChromaDB · sentence-transformers · Google Gemini API · Docker · GitHub Actions

## 📂 Project Structure

```text
rag-document-chatbot/
├── app.py                  # Streamlit frontend
├── config.py                # Environment-driven settings
├── src/
│   ├── document_loader.py   # Parsing + chunking
│   ├── embeddings.py        # Cached HuggingFace model
│   ├── vector_store.py      # Chroma indexing
│   ├── retriever.py         # MMR retrieval
│   ├── llm_chain.py         # Gemini QA chain
│   └── pipeline.py          # Index + query orchestration
├── tests/                   # Pipeline unit tests
├── data/samples/             # Demo document
├── Dockerfile
└── docker-compose.yml
```

## ❓ FAQ

**Does it work with scanned PDFs?**
Not yet — scanned (image-only) PDFs need OCR, which isn't implemented. Text-layer PDFs work fine.

**Can I use a different LLM instead of Gemini?**
The `llm_chain.py` module is the only place tied to Gemini — swapping in OpenAI/Claude/Ollama is a small change. PRs welcome.

**Is my document data stored anywhere?**
No — ChromaDB is persisted locally/in your deployment only. Nothing is sent anywhere except the retrieved chunks to the Gemini API for answer generation.

**Can I query multiple documents at once?**
Not yet — currently single-document mode. Multi-doc collections are on the roadmap.

## 🗺 Roadmap

- [ ] Multi-document collections
- [ ] OCR support for scanned PDFs
- [ ] Persistent chat history
- [ ] Pluggable LLM backend (OpenAI / Claude / local models)
- [ ] Citation highlighting in the UI

## 🤝 Contributing

Contributions are welcome! Good first issues are tagged [`good first issue`](https://github.com/Dharunmd/rag-document-chatbot/labels/good%20first%20issue).

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit and push
4. Open a PR

## 📋 Resume Bullet (for students using this as a portfolio project)

> **DocuQuery — RAG Document Q&A System** | Python, LangChain, ChromaDB, Gemini
> Built an end-to-end retrieval-augmented generation app indexing PDF/DOCX/TXT documents into ChromaDB with HuggingFace embeddings; answers queries via Google Gemini using MMR retrieval and inline source attribution. Deployed with Streamlit and Docker, with CI via pytest + GitHub Actions.

## ⭐ Found this useful?

If DocuQuery helped you, learn from it, or saved you time — **please star the repo**. It genuinely helps other students and developers find it.

## Author

**Dharun M** · [GitHub](https://github.com/Dharunmd)

## License

MIT — see [LICENSE](LICENSE)
