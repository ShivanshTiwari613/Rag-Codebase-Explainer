# src/rag_pipeline/qa_chain.py

from typing import List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY


def format_docs(docs: List[Document]) -> str:
    """
    A helper function to format the retrieved documents into a single string.
    This string will be injected into the prompt's context.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _format_chat_history(history: Sequence[Tuple[str, str]], limit: int = 5) -> str:
    """
    Convert a list of (question, answer) tuples into a concise conversation string
    suitable for the prompt.
    """
    if not history:
        return "None."

    trimmed = history[-limit:]
    lines = []
    for question, answer in trimmed:
        question = question.strip()
        answer = (answer or "").strip()
        if question:
            lines.append(f"User: {question}")
        if answer:
            lines.append(f"Assistant: {answer}")
    return "\n".join(lines) if lines else "None."


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
Use the recent chat history to maintain context and avoid repeating the full answer when not necessary.
Answer the user's question based ONLY on the following context of source code snippets.
If the answer is not found in the context, explicitly state that.
Do not make up any information.
Be concise and clear in your answer.

CHAT HISTORY:
{chat_history}

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

    def run_chain(inputs):
        if isinstance(inputs, str):
            question = inputs
            history: Sequence[Tuple[str, str]] = []
        else:
            question = inputs.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string.")
            history = inputs.get("chat_history") or []
        docs = retriever.invoke(question)
        formatted_context = format_docs(docs)
        history_text = _format_chat_history(history)
        prompt_value = prompt.format(
            context=formatted_context,
            question=question,
            chat_history=history_text,
        )
        llm_result = llm.invoke(prompt_value)
        answer = output_parser.invoke(llm_result)
        return {"answer": answer, "source_documents": docs}

    return RunnableLambda(run_chain)
