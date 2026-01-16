import json
import re
from typing import List

from retrieval.vector_store import get_vector_store
from generation.llm import get_llm
from generation.schema import RAGResponse, Source
from prompts.registry import load_prompt


def extract_json(text: str) -> dict:
    """
    Safely extract JSON from LLM output.
    Handles:
    - Extra text before/after JSON
    - ```json code fences
    - Minor formatting noise
    """
    if not text:
        raise ValueError("Empty response from LLM")

    # Clean common markdown fences
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Attempt direct JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object via regex
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError("Failed to extract valid JSON from LLM output")


def format_context(documents: List) -> str:
    """
    Combine retrieved documents into a single context string.
    """
    return "\n\n".join(doc.page_content for doc in documents)


def answer_question(
    question: str,
    k: int = 3,
    prompt_version: str = "v2_strict_grounding",
) -> RAGResponse:
    """
    Core RAG pipeline:
    - Retrieve relevant chunks
    - Apply versioned system prompt
    - Generate grounded answer
    - Return structured response
    """

    # Load components
    vector_store = get_vector_store()
    llm = get_llm()
    system_prompt = load_prompt(prompt_version)

    # Retrieve context
    docs = vector_store.similarity_search(question, k=k)
    context = format_context(docs)

    # Build final prompt
    prompt = f"""
{system_prompt}

Context:
{context}

Question:
{question}
"""

    # Invoke LLM
    raw_response = llm.invoke(prompt)

    # 🔍 Debug (keep for development; remove in prod)
    print("\n--- RAW LLM OUTPUT ---\n")
    print(raw_response)
    print("\n----------------------\n")

    # Parse JSON safely
    data = extract_json(raw_response)

    # Attach sources explicitly (LLM never controls sources)
    sources = [Source(content=doc.page_content) for doc in docs]

    return RAGResponse(
        answer=data.get("answer", ""),
        confidence=float(data.get("confidence", 0.0)),
        sources=sources,
        found=bool(data.get("found", False)),
    )
