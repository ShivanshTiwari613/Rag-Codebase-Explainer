# src/rag_pipeline/qa_chain.py

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY


def format_docs(docs: List[Document]) -> str:
    """
    A helper function to format the retrieved documents into a single string.
    This string will be injected into the prompt's context.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever) -> Runnable:
    """
    Creates and returns a RAG chain that takes a question, retrieves relevant
    documents, constructs a prompt, and generates an answer from the LLM.
    The chain also returns the source documents used for the answer.

    Args:
        retriever: A configured LangChain retriever object from our vector store.

    Returns:
        A LangChain runnable object that expects a question string and returns
        a dictionary with "answer" and "source_documents".
    """
    prompt_template = """
You are an expert programming assistant. Your task is to answer questions about a codebase.
Answer the user's question based ONLY on the following context of source code snippets.
If the answer is not found in the context, explicitly state that.
Do not make up any information.
Be concise and clear in your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
        convert_system_message_to_human=True,
    )

    output_parser = StrOutputParser()

    def run_chain(question: str):
        docs = retriever.invoke(question)
        formatted_context = format_docs(docs)
        prompt_value = prompt.format(context=formatted_context, question=question)
        llm_result = llm.invoke(prompt_value)
        answer = output_parser.invoke(llm_result)
        return {"answer": answer, "source_documents": docs}

    return RunnableLambda(run_chain)
