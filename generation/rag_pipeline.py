import json
import re

from generation.llm import get_llm
from generation.prompt import SYSTEM_PROMPT
from generation.schema import RAGResponse, Source
from retrieval.vector_store import get_vector_store


def extract_json(text: str) -> dict:
    """
    Safely extract JSON object from LLM output.
    """
    # Remove code fences if present
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try regex extraction
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError("LLM output is not valid JSON")


def format_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)


def answer_question(question: str, k: int = 3) -> RAGResponse:
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
"""

    raw_response = llm.invoke(prompt)

    # 🔍 DEBUG ONCE (comment later if you want)
    print("\n--- RAW LLM OUTPUT ---\n")
    print(raw_response)
    print("\n----------------------\n")

    data = extract_json(raw_response)

    sources = [Source(content=doc.page_content) for doc in docs]

    return RAGResponse(
        answer=data.get("answer", ""),
        confidence=float(data.get("confidence", 0.0)),
        sources=sources,
        found=bool(data.get("found", False)),
    )
