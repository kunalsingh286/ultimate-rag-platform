from fastapi import FastAPI
from backend.routes import router

app = FastAPI(
    title="Ultimate RAG Platform",
    description="Production-grade Retrieval-Augmented Generation system",
    version="1.0.0",
)

app.include_router(router)
