import sys
from pathlib import Path

from backend.rag.embedding_service import EmbeddingService
from backend.rag.vector_store import VectorStore
from backend.rag.ingestion_service import IngestionService
from backend.services.file_service import save_files
from backend.services.log_service import register_document, get_document

embedding_service = EmbeddingService()
vector_store = VectorStore()
ingestion_service = IngestionService(embedding_service=embedding_service,vector_store=vector_store)

for arg in sys.argv[1:]:
    source = Path(arg)
    if not source.exists():
        print(f"Skipping '{arg}' (file not found)")
        continue

    path = save_files(source.name, source.read_bytes())
    document_id = Path(path).stem
    register_document(document_id=document_id, filename=source.name, file_path=path)

    num_chunks = ingestion_service.ingest(document_id=document_id, file_path=path)
    print(f"Ingested '{source.name}' as {document_id} ({num_chunks} chunks)")
    print(get_document(document_id))

print("ingestion Completed")
