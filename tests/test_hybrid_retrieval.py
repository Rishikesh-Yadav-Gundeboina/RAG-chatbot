import numpy as np

from backend.rag.retrieval_service import RetrievalService, keyword_coverage


class FakeEmbeddingService:
    def embed(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


class FakeVectorStore:
    def __init__(self, dense_chunks, embeddings):
        # dense_chunks: list of (id, text, document_id, page_number, distance), best first
        self._dense = dense_chunks
        self._embeddings = embeddings
        self.search_calls = []

    def search(self, query_embedding, top_k=5, document_ids=None):
        self.search_calls.append((top_k, document_ids))
        chunks = self._dense[:top_k]
        return {
            "ids": [[c[0] for c in chunks]],
            "documents": [[c[1] for c in chunks]],
            "metadatas": [[{"document_id": c[2], "page_number": c[3]} for c in chunks]],
            "distances": [[c[4] for c in chunks]],
        }

    def get_embeddings(self, chunk_ids):
        return {i: self._embeddings[i] for i in chunk_ids if i in self._embeddings}


class FakeKeywordStore:
    def __init__(self, hits):
        self._hits = hits
        self.search_calls = []

    def search(self, question, top_k=20, document_ids=None):
        self.search_calls.append((top_k, document_ids))
        return self._hits[:top_k]


def _hit(chunk_id, text, document_id="doc1", page_number=1):
    return {"chunk_id": chunk_id, "text": text, "document_id": document_id, "page_number": page_number, "bm25_score": -1.0}


DENSE = [("c1", "dense only text", "doc1", 1, 0.4), ("c2", "shared groupby text", "doc1", 2, 0.6)]


def test_hybrid_returns_chroma_shape_with_shared_chunk_first():
    vector_store = FakeVectorStore(DENSE, {})
    keyword_store = FakeKeywordStore([_hit("c2", "shared groupby text", page_number=2)])
    service = RetrievalService(FakeEmbeddingService(), vector_store, keyword_store)

    result = service.retrieve("groupby", top_k=5)

    assert result["ids"] == [["c2", "c1"]]
    assert result["metadatas"][0][0] == {"document_id": "doc1", "page_number": 2}
    assert result["distances"] == [[0.6, 0.4]]
    assert result["keyword_coverage"] == [[1.0, 0.0]]


def test_keyword_only_chunk_gets_distance_from_stored_embedding():
    vector_store = FakeVectorStore(DENSE, {"k1": [0.0, 1.0]})
    keyword_store = FakeKeywordStore([_hit("k1", "error E1042 refused")])
    service = RetrievalService(FakeEmbeddingService(), vector_store, keyword_store)

    result = service.retrieve("E1042", top_k=5)

    index = result["ids"][0].index("k1")
    # squared L2 between [1, 0] and [0, 1]
    assert result["distances"][0][index] == 2.0


def test_keyword_hit_missing_from_vector_store_is_dropped():
    vector_store = FakeVectorStore(DENSE, {})
    keyword_store = FakeKeywordStore([_hit("ghost", "orphan text")])
    service = RetrievalService(FakeEmbeddingService(), vector_store, keyword_store)

    result = service.retrieve("orphan", top_k=5)

    assert "ghost" not in result["ids"][0]
    assert len(result["ids"][0]) == len(result["distances"][0]) == 2


def test_top_k_and_document_ids_are_respected():
    vector_store = FakeVectorStore(DENSE, {})
    keyword_store = FakeKeywordStore([])
    service = RetrievalService(FakeEmbeddingService(), vector_store, keyword_store)

    result = service.retrieve("anything", top_k=1, document_ids=["doc1"])

    assert result["ids"] == [["c1"]]
    assert vector_store.search_calls[0][1] == ["doc1"]
    assert keyword_store.search_calls[0][1] == ["doc1"]


def test_dense_mode_matches_plain_vector_search():
    vector_store = FakeVectorStore(DENSE, {})
    keyword_store = FakeKeywordStore([_hit("k1", "ignored")])
    service = RetrievalService(FakeEmbeddingService(), vector_store, keyword_store, mode="dense")

    result = service.retrieve("groupby", top_k=5)

    assert result == vector_store.search(None, 5)
    assert keyword_store.search_calls == []


def test_keyword_coverage_handles_plurals_and_underscores():
    assert keyword_coverage("how do I read csv files", "use pandas.read_csv on the file") == 1.0
    assert keyword_coverage("capital of France", "capitalize the string") == 0.0
    assert keyword_coverage("what is the", "anything") == 0.0
