import hashlib
import os
from pathlib import Path

from fastapi import FastAPI,UploadFile,File,HTTPException,BackgroundTasks
from backend.services.file_service import save_files
from backend.services.pdf_service import validate_pdf, InvalidPDFError
from backend.services.log_service import register_document, create_session, add_message, get_document, get_document_by_hash, list_documents, mark_stale_processing_failed, backfill_missing_hashes, get_recent_messages
from backend.rag.rag_service import RagService
from backend.rag.llm_service import LLMService, LLMTimeoutError
from backend.rag.retrieval_service import RetrievalService
from backend.rag.embedding_service import EmbeddingService
from backend.rag.vector_store import VectorStore
from backend.rag.ingestion_service import IngestionService
from backend.rag.keyword_store import KeywordStore, backfill_keyword_index

app = FastAPI()

embedding_service = EmbeddingService()
vector_store = VectorStore()
keyword_store = KeywordStore()
llm_service = LLMService()
retrieval_service = RetrievalService(embedding_service=embedding_service,vector_store=vector_store,keyword_store=keyword_store,mode=os.environ.get("RETRIEVAL_MODE","hybrid"))
rag_service = RagService(retrieval_service=retrieval_service,llm_service=llm_service)
ingestion_service = IngestionService(embedding_service=embedding_service,vector_store=vector_store,keyword_store=keyword_store)

mark_stale_processing_failed()
backfill_missing_hashes()
backfill_keyword_index(vector_store, keyword_store)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/documents")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Expected application/pdf"
        )
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail = "Empty file"
        )
    try:
        validate_pdf(content)
    except InvalidPDFError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    file_hash = hashlib.sha256(content).hexdigest()
    existing = get_document_by_hash(file_hash)
    if existing is not None:
        return {
            "duplicate": True,
            "document_id": existing["id"],
            "filename": existing["filename"],
            "status": existing["status"],
        }

    path = save_files(file.filename,content)
    document_id = Path(path).stem
    register_document(document_id=document_id, filename=file.filename, file_path=path, status="processing", file_hash=file_hash)

    background_tasks.add_task(_run_ingestion, document_id, path)

    return {
        "filename":file.filename,
        "path" : path,
        "document_id" : document_id,
        "status" : "processing"
    }

def _run_ingestion(document_id:str,file_path:str):
    try:
        ingestion_service.ingest(document_id=document_id, file_path=file_path)
    except Exception as exc:
        print(f"Ingestion failed for {document_id}: {exc}")

@app.get("/documents")
def list_all_documents():
    return list_documents()

@app.get("/documents/{document_id}")
def get_document_status(document_id:str):
    document = get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

HISTORY_LIMIT = 10  # last N=5 user/assistant pairs

@app.post("/chat")
async def post_question(question:str, session_id:int | None = None, document_ids: list[str] | None = None):
    if session_id is None:
        session_id = create_session()

    history = get_recent_messages(session_id, limit=HISTORY_LIMIT)

    add_message(session_id=session_id, role="user", content=question)

    try:
        response = rag_service.answer(question=question, history=history, document_ids=document_ids)
    except LLMTimeoutError:
        raise HTTPException(
            status_code=504,
            detail="The request took too long to process. Please try again."
        )

    add_message(session_id=session_id, role="assistant", content=response["answer"])

    return {
        "session_id": session_id,
        "answer": response["answer"],
        "sources": response["sources"]
    }
