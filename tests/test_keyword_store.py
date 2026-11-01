from backend.rag.keyword_store import KeywordStore, tokenize, backfill_keyword_index
from backend.rag.models import DocumentChunk


def _chunk(chunk_id, document_id, text, page_number=1):
    return DocumentChunk(chunk_id=chunk_id, document_id=document_id, page_number=page_number, text=text)


def _store(tmp_path):
    store = KeywordStore(db_path=tmp_path / "kw.db")
    store.add_chunks([
        _chunk("d1_1_0", "d1", "Use pandas.read_csv to load a CSV file into a DataFrame."),
        _chunk("d1_2_0", "d1", "NumPy arrays support vectorized arithmetic."),
        _chunk("d2_1_0", "d2", "Error E1042 means the connection was refused."),
    ])
    return store


def test_tokenize_drops_stopwords_and_splits_underscores():
    assert tokenize("What is read_csv in the Pandas library?") == ["read", "csv", "pandas", "library"]


def test_search_ranks_exact_term_match_first(tmp_path):
    results = _store(tmp_path).search("what does error E1042 mean")
    assert results[0]["chunk_id"] == "d2_1_0"
    assert results[0]["document_id"] == "d2"
    assert results[0]["page_number"] == 1


def test_search_filters_by_document_ids(tmp_path):
    store = _store(tmp_path)
    assert store.search("E1042", document_ids=["d1"]) == []
    assert [r["chunk_id"] for r in store.search("E1042", document_ids=["d2"])] == ["d2_1_0"]


def test_add_chunks_is_an_upsert(tmp_path):
    store = _store(tmp_path)
    store.add_chunks([_chunk("d2_1_0", "d2", "Error E1042 now means timeout.")])
    assert store.count() == 3
    assert "timeout" in store.search("E1042")[0]["text"]


def test_delete_by_document_id(tmp_path):
    store = _store(tmp_path)
    store.delete_by_document_id("d1")
    assert store.count() == 1
    assert not store.has_document("d1")
    assert store.has_document("d2")


def test_stopword_or_punctuation_only_query_returns_empty(tmp_path):
    store = _store(tmp_path)
    assert store.search("what is the") == []
    assert store.search('"-:*()') == []


def test_fts_syntax_in_query_does_not_raise(tmp_path):
    store = _store(tmp_path)
    assert store.search('NOT pandas AND "csv OR -numpy*') != []


class FakeVectorStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def get_all_chunks(self):
        return self._chunks


def test_backfill_only_indexes_missing_documents(tmp_path):
    store = _store(tmp_path)
    vector_store = FakeVectorStore([
        {"chunk_id": "d1_1_0", "document_id": "d1", "page_number": 1, "text": "already indexed"},
        {"chunk_id": "d3_1_0", "document_id": "d3", "page_number": 4, "text": "brand new chunk"},
    ])
    assert backfill_keyword_index(vector_store, store) == 1
    assert store.count() == 4
    assert store.search("brand")[0]["page_number"] == 4
