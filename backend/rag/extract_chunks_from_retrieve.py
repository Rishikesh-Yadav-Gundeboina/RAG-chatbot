def extract_chunks(results):
    chunks = []
    chunk_ids = results['ids'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    coverages = results.get('keyword_coverage', [[0.0] * len(chunk_ids)])[0]

    for chunk_id, doc_text, metadata, distance, coverage in zip(chunk_ids, documents, metadatas, distances, coverages):
        chunks.append({
            "text": doc_text,
            "chunk_id": chunk_id,
            "document_id": metadata.get("document_id"),
            "page_number": metadata.get("page_number"),
            "distance": distance,
            "keyword_coverage": coverage
        })
    
    return chunks