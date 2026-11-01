from backend.services.pdf_service import extract_pages
from backend.services.text_cleaner import clean_text
from backend.services.log_service import update_document_status
from backend.rag.chunker import chunk_text
from backend.rag.models import DocumentChunk


class IngestionService:
    def __init__(self,embedding_service,vector_store,keyword_store=None):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.keyword_store = keyword_store

    def ingest(self,document_id,file_path):
        try:
            pages=extract_pages(file_path)

            all_chunks = []

            for page in pages:
                text = clean_text(page["text"])
                chunks = chunk_text(text)

                for index,chunk in enumerate(chunks):
                    all_chunks.append(
                        DocumentChunk(
                            chunk_id=f"{document_id}_{page['page_number']}_{index}",
                            document_id=document_id,
                            page_number=page['page_number'],
                            text=chunk
                        )
                    )

            embeddings = self.embedding_service.embed(
                    [chunk.text for chunk in all_chunks]
                )
            self.vector_store.add_chunks(
                    all_chunks,
                    embeddings
                )
            if self.keyword_store is not None:
                self.keyword_store.add_chunks(all_chunks)
        except Exception:
            self.vector_store.delete_by_document_id(document_id)
            if self.keyword_store is not None:
                self.keyword_store.delete_by_document_id(document_id)
            update_document_status(document_id, "failed")
            raise

        update_document_status(document_id, "ready", num_chunks=len(all_chunks))
        return len(all_chunks)