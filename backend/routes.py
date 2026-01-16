from fastapi import APIRouter, UploadFile, File
import shutil
import os

from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from retrieval.vector_store import get_vector_store
from generation.rag_pipeline import answer_question

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = load_document(file_path)
    chunks = chunk_documents(docs)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return {
        "filename": file.filename,
        "chunks_indexed": len(chunks),
    }


@router.post("/query")
def query_rag(question: str):
    response = answer_question(question)
    return response
