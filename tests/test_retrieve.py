from backend.rag.retrieval_service import RetrievalService
from backend.rag.embedding_service import EmbeddingService
from backend.rag.vector_store import VectorStore

vector_store = VectorStore()
embedding_service = EmbeddingService()

retrieval_service = RetrievalService(vector_store=vector_store,embedding_service=embedding_service)

result=retrieval_service.retrieve("What is Pandas")



def print_retrieval_results(results):
    """
    Print retrieval results in a readable format for inspection.
    """
    print("=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)
    
    # Get the first (and only) query results
    ids = results['ids'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    print(f"Number of chunks retrieved: {len(documents)}")
    print("-" * 80)
    
    for i, (chunk_id, doc, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances), 1):
        print(f"\n📄 CHUNK {i}")
        print(f"  ID: {chunk_id}")
        print(f"  Document ID: {metadata.get('document_id')}")
        print(f"  Page Number: {metadata.get('page_number')}")
        print(f"  Similarity Distance: {distance:.4f}")
        print(f"  Text Preview: {doc[:200]}..." if len(doc) > 200 else f"  Text: {doc}")
        print("-" * 80)
    
    # Summary statistics
    print("\n📊 SUMMARY")
    print(f"  Total chunks: {len(documents)}")
    print(f"  Unique documents: {len(set(m['document_id'] for m in metadatas))}")
    print(f"  Average distance: {sum(distances)/len(distances):.4f}")
    print("=" * 80)

# Usage example:
# results = retrieval_service.retrieve("your question")
print_retrieval_results(result)