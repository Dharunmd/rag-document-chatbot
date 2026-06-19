from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def create_qa_chain(retriever):
    """
    Create the Retrieval-Augmented QA chain.
    """

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        input_variables=[
            "context",
            "question"
        ],
        template="""
You are a helpful document assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
respond with:

"I could not find the answer in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
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