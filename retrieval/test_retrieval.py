from retrieval.vector_store import get_vector_store


if __name__ == "__main__":
    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query="What is Retrieval-Augmented Generation?",
        k=3,
    )

    print(f"Retrieved {len(results)} documents\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:300])
        print()
