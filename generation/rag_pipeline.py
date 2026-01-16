from generation.llm import get_llm
from generation.prompt import SYSTEM_PROMPT
from retrieval.vector_store import get_vector_store


def format_context(documents):
    return "\n\n".join(
        f"Source {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )


def answer_question(question: str, k: int = 3):
    vector_store = get_vector_store()
    llm = get_llm()

    docs = vector_store.similarity_search(question, k=k)
    context = format_context(docs)

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response
