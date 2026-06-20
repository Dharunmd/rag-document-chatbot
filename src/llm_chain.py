from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

PROMPT_TEMPLATE = """
You are a helpful document Q&A assistant.

Rules:
- Answer based only on the provided context.
- If the user asks for a summary or general description, summarize the information available in the context.
- Keep answers concise and factual.
- If the context does not contain enough information to answer a specific question, say:
  "I couldn't find that in the uploaded document."
- Do not make up information or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


def create_qa_chain(retriever):
    """Create RetrievalQA chain using Groq."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    print("Importing ChatGroq...")

    # Lazy import
    from langchain_groq import ChatGroq

    print("Creating LLM...")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    print("LLM created successfully")

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=[
            "context",
            "question",
        ],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
        },
    )

    print("QA Chain created")

    return qa_chain