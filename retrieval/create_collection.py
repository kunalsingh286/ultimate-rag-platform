from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

COLLECTION_NAME = "documents"


if __name__ == "__main__":
    client = QdrantClient(url="http://localhost:6333")

    # MiniLM embedding size = 384
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    print(f"Collection '{COLLECTION_NAME}' created successfully")
