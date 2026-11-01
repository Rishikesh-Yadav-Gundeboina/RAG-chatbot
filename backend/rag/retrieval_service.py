import numpy as np

from backend.rag.fusion import reciprocal_rank_fusion
from backend.rag.keyword_store import tokenize

# How many candidates each retriever contributes before fusion.
CANDIDATES_PER_RETRIEVER = 20


def _normalize(token):
    # Cheap plural folding so "files" matches "file", like FTS5's porter stemmer does.
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def keyword_coverage(question, text):
    """Fraction of the question's (non-stopword) terms that appear in text."""
    terms = {_normalize(t) for t in tokenize(question)}
    if not terms:
        return 0.0
    text_tokens = {_normalize(t) for t in tokenize(text)}
    return sum(term in text_tokens for term in terms) / len(terms)


class RetrievalService:
    def __init__(self,embedding_service,vector_store,keyword_store=None,mode="hybrid"):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.mode = mode

    def retrieve(self,question,top_k=5,document_ids=None):
        embedding = self.embedding_service.embed(
            [question]
        )[0]
        if self.mode == "dense" or self.keyword_store is None:
            return self.vector_store.search(embedding,top_k,document_ids=document_ids)
        return self._hybrid_retrieve(question,embedding,top_k,document_ids)

    def _hybrid_retrieve(self,question,embedding,top_k,document_ids):
        dense = self.vector_store.search(embedding,CANDIDATES_PER_RETRIEVER,document_ids=document_ids)
        keyword_hits = self.keyword_store.search(question,top_k=CANDIDATES_PER_RETRIEVER,document_ids=document_ids)

        # chunk_id -> (text, metadata, distance or None)
        candidates = {}
        for chunk_id, text, metadata, distance in zip(
            dense["ids"][0], dense["documents"][0], dense["metadatas"][0], dense["distances"][0]
        ):
            candidates[chunk_id] = (text, metadata, distance)
        for hit in keyword_hits:
            if hit["chunk_id"] not in candidates:
                metadata = {"document_id": hit["document_id"], "page_number": hit["page_number"]}
                candidates[hit["chunk_id"]] = (hit["text"], metadata, None)

        fused = reciprocal_rank_fusion([dense["ids"][0], [hit["chunk_id"] for hit in keyword_hits]])
        top_ids = [chunk_id for chunk_id, _ in fused[:top_k]]

        # Keyword-only chunks have no dense distance yet; compute it from their stored
        # embeddings so the relevance threshold applies to every chunk on the same scale.
        missing = [chunk_id for chunk_id in top_ids if candidates[chunk_id][2] is None]
        if missing:
            query = np.asarray(embedding)
            for chunk_id, stored in self.vector_store.get_embeddings(missing).items():
                text, metadata, _ = candidates[chunk_id]
                distance = float(np.sum((query - np.asarray(stored)) ** 2))
                candidates[chunk_id] = (text, metadata, distance)
        # A keyword hit missing from Chroma would have no distance; drop it.
        top_ids = [chunk_id for chunk_id in top_ids if candidates[chunk_id][2] is not None]

        return {
            "ids": [top_ids],
            "documents": [[candidates[i][0] for i in top_ids]],
            "metadatas": [[candidates[i][1] for i in top_ids]],
            "distances": [[candidates[i][2] for i in top_ids]],
            "keyword_coverage": [[keyword_coverage(question, candidates[i][0]) for i in top_ids]],
        }
