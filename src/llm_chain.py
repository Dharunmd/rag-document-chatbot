from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE


PROMPT_TEMPLATE = """
You are a document Q&A assistant for a college project demo.

Rules:
- Answer only from the context below.
- Keep answers concise and factual.
- If the context does not contain the answer, say:
  "I couldn't find that in the uploaded document."
- Do not invent names, dates, or numbers.

Context:
{context}

Question:
{question}

Answer:
"""


def create_qa_chain(retriever):
    """Create a retrieval-augmented QA chain backed by Gemini."""

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your .env file before chatting."
        )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
