from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from retrieval.vector_store import get_vector_store


if __name__ == "__main__":
    docs = load_document("docs/sample.txt")
    chunks = chunk_documents(docs)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    print(f"Indexed {len(chunks)} chunks into Qdrant")
