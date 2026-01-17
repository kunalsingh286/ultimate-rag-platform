from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from retrieval.vector_store import get_vector_store
from retrieval.create_collection import COLLECTION_NAME

from generation.rag_pipeline import answer_question
from evaluation.metrics import (
    faithfulness_score,
    hallucination_flag,
)

# -----------------------------
# Prompt versions to test
# -----------------------------
PROMPT_VERSIONS = [
    "v1_basic",
    "v2_strict_grounding",
]

# -----------------------------
# Fixed regression test cases
# -----------------------------
TEST_CASES = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "expected_found": True,
    },
    {
        "question": "Explain quantum entanglement",
        "expected_found": False,
    },
]

# -----------------------------
# Quality thresholds
# -----------------------------
MIN_FAITHFULNESS = 0.5
MAX_HALLUCINATIONS = 0


def recreate_test_collection():
    """
    Always recreate the Qdrant collection before tests.
    This avoids validation errors and guarantees clean state.
    """
    print("\n🔧 Recreating test vector collection...")

    client = QdrantClient(url="http://localhost:6333")

    # MiniLM embedding dimension = 384
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    print(f"✅ Collection '{COLLECTION_NAME}' recreated")


def setup_test_data():
    """
    Index test documents into a fresh collection.
    """
    docs = load_document("docs/sample.txt")
    chunks = chunk_documents(docs)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    print(f"✅ Indexed {len(chunks)} chunks into '{COLLECTION_NAME}'")


def run_prompt(prompt_version: str):
    """
    Run all test cases for a single prompt version.
    """
    print(f"\n🔹 Testing prompt: {prompt_version}")
    results = []

    for case in TEST_CASES:
        response = answer_question(
            question=case["question"],
            prompt_version=prompt_version,
        )

        faith = faithfulness_score(response)
        hallucinated = hallucination_flag(response)

        result = {
            "question": case["question"],
            "expected_found": case["expected_found"],
            "found": response.found,
            "faithfulness": faith,
            "hallucinated": hallucinated,
        }

        results.append(result)

        print(f"\nQuestion: {case['question']}")
        print(f"Expected Found: {case['expected_found']}")
        print(f"Actual Found: {response.found}")
        print(f"Faithfulness: {faith:.2f}")
        print(f"Hallucinated: {hallucinated}")

    return results


def evaluate_results(results):
    """
    Aggregate evaluation and enforce regression thresholds.
    """
    hallucinations = sum(1 for r in results if r["hallucinated"])
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)

    print("\n📊 Aggregate Metrics")
    print(f"Average faithfulness: {avg_faithfulness:.2f}")
    print(f"Total hallucinations: {hallucinations}")

    if avg_faithfulness < MIN_FAITHFULNESS:
        raise AssertionError("❌ Faithfulness regression detected")

    if hallucinations > MAX_HALLUCINATIONS:
        raise AssertionError("❌ Hallucination regression detected")


if __name__ == "__main__":
    # 🔥 CRITICAL ORDER (DO NOT CHANGE)
    recreate_test_collection()
    setup_test_data()

    for prompt in PROMPT_VERSIONS:
        results = run_prompt(prompt)
        evaluate_results(results)

    print("\n✅ Prompt regression tests passed successfully")
