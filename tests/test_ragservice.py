from backend.rag.rag_service import RagService
from backend.rag.llm_service import LLMService
from backend.rag.retrieval_service import RetrievalService
from backend.rag.embedding_service import EmbeddingService
from backend.rag.vector_store import VectorStore


question = "what is pandas?"

embedding_service = EmbeddingService()
vector_service = VectorStore()
llm_service = LLMService()
retrieval_service = RetrievalService(embedding_service=embedding_service,vector_store=vector_service)
rag_service = RagService(retrieval_service=retrieval_service,llm_service=llm_service)

result = rag_service.answer(question=question)
print(result["answer"])
print(result["sources"])