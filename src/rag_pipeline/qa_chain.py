# src/rag_pipeline/qa_chain.py

from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, Runnable
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
    # 1. Define the Prompt Template
    # This template is crucial. It instructs the LLM on how to behave, telling it
    # to base its answer strictly on the provided context.
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

    # 2. Initialize the LLM
    # We use Gemini Pro, setting a low temperature for more factual, less creative answers.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GEMINI_API_KEY, 
        temperature=0.1,
        convert_system_message_to_human=True # Helps with some models
    )

    # 3. Build the RAG Chain using LCEL (LangChain Expression Language)
    
    # This part of the chain is responsible for getting the documents
    # and passing the question through.
    retrieval_and_context_chain = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    
    # This is the main chain that combines retrieval, prompt, and LLM.
    # The final output is a dictionary containing the generated answer
    # and the source documents that were used as context.
    rag_chain = (
        {
            "source_documents": retriever, 
            "question": RunnablePassthrough()
        }
        | {
            "answer": retrieval_and_context_chain | prompt | llm | StrOutputParser(),
            "source_documents": itemgetter("source_documents"),
        }
    )

    return rag_chain