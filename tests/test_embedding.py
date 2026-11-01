from backend.rag.embedding_service import EmbeddingService

service = EmbeddingService()

vectors = service.embed(["What is a process?","Explain process execution"])

print(vectors.shape)
print(vectors)