from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from retrieval.embeddings import get_embedding_model

COLLECTION_NAME = "documents"


def get_qdrant_client():
    return QdrantClient(url="http://localhost:6333")


def get_vector_store():
    client = get_qdrant_client()
    embeddings = get_embedding_model()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    return vector_store
