import chromadb

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="data/chroma"
        )

        self.collection = self.client.get_or_create_collection(
            name="documents"
        )

    def add_chunks(self,chunks,embeddings):
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents = [chunk.text for chunk in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{
                "document_id":chunk.document_id,
                "page_number":chunk.page_number
            }
            for chunk in chunks]
                            )

    def delete_by_document_id(self,document_id):
        self.collection.delete(where={"document_id": document_id})

    def search(self,query_embedding,top_k=5,document_ids=None):
        return self.collection.query(
            query_embeddings = [
                query_embedding.tolist()
            ],
            n_results = top_k,
            where = {"document_id": {"$in": document_ids}} if document_ids else None
        )

    def get_embeddings(self,chunk_ids):
        result = self.collection.get(ids=chunk_ids,include=["embeddings"])
        return dict(zip(result["ids"],result["embeddings"]))

    def get_all_chunks(self):
        result = self.collection.get(include=["documents","metadatas"])
        return [{
            "chunk_id":chunk_id,
            "text":text,
            "document_id":metadata.get("document_id"),
            "page_number":metadata.get("page_number")
        }
        for chunk_id,text,metadata in zip(result["ids"],result["documents"],result["metadatas"])]
