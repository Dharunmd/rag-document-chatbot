from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE
)

PROMPT_TEMPLATE = """
You are a document Q&A assistant.

Rules:
- Answer only using the provided context.
- Keep answers concise and factual.
- If the answer is not found in the context, say:
  "I couldn't find that in the uploaded document."
- Do not make up information.
- Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


def create_qa_chain(retriever):

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing in .env"
        )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=[
            "context",
            "question"
        ]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )

    return qa_chain