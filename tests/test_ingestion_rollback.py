import pytest

import backend.rag.ingestion_service as ingestion_service_module
from backend.rag.ingestion_service import IngestionService


class FakeEmbeddings:
    def __init__(self, texts):
        self._texts = texts

    def tolist(self):
        return [[0.0] for _ in self._texts]


class FakeEmbeddingService:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def embed(self, texts):
        if self.should_fail:
            raise RuntimeError("embedding failed")
        return FakeEmbeddings(texts)


class FakeVectorStore:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.added = []
        self.deleted_ids = []

    def add_chunks(self, chunks, embeddings):
        if self.should_fail:
            raise RuntimeError("vector store failed")
        self.added.append(chunks)

    def delete_by_document_id(self, document_id):
        self.deleted_ids.append(document_id)


class FakeKeywordStore:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.added = []
        self.deleted_ids = []

    def add_chunks(self, chunks):
        if self.should_fail:
            raise RuntimeError("keyword store failed")
        self.added.append(chunks)

    def delete_by_document_id(self, document_id):
        self.deleted_ids.append(document_id)


def _patch_pipeline(monkeypatch, pages):
    monkeypatch.setattr(ingestion_service_module, "extract_pages", lambda file_path: pages)
    monkeypatch.setattr(ingestion_service_module, "clean_text", lambda text: text)
    monkeypatch.setattr(ingestion_service_module, "chunk_text", lambda text: [text])


def _patch_status_recorder(monkeypatch):
    statuses = []
    monkeypatch.setattr(
        ingestion_service_module,
        "update_document_status",
        lambda document_id, status, num_chunks=None: statuses.append((document_id, status, num_chunks)),
    )
    return statuses


def test_ingest_success_upserts_chunks_and_marks_ready(monkeypatch):
    _patch_pipeline(monkeypatch, [{"page_number": 1, "text": "hello world"}])
    statuses = _patch_status_recorder(monkeypatch)

    vector_store = FakeVectorStore()
    service = IngestionService(embedding_service=FakeEmbeddingService(), vector_store=vector_store)

    num_chunks = service.ingest(document_id="doc1", file_path="fake.pdf")

    assert num_chunks == 1
    assert len(vector_store.added) == 1
    assert vector_store.deleted_ids == []
    assert statuses == [("doc1", "ready", 1)]


def test_ingest_failure_during_embedding_deletes_chunks_and_marks_failed(monkeypatch):
    _patch_pipeline(monkeypatch, [{"page_number": 1, "text": "hello world"}])
    statuses = _patch_status_recorder(monkeypatch)

    vector_store = FakeVectorStore()
    service = IngestionService(embedding_service=FakeEmbeddingService(should_fail=True), vector_store=vector_store)

    with pytest.raises(RuntimeError):
        service.ingest(document_id="doc1", file_path="fake.pdf")

    assert vector_store.deleted_ids == ["doc1"]
    assert vector_store.added == []
    assert statuses == [("doc1", "failed", None)]


def test_ingest_failure_during_vector_store_write_deletes_chunks(monkeypatch):
    _patch_pipeline(monkeypatch, [{"page_number": 1, "text": "hello world"}])
    statuses = _patch_status_recorder(monkeypatch)

    vector_store = FakeVectorStore(should_fail=True)
    service = IngestionService(embedding_service=FakeEmbeddingService(), vector_store=vector_store)

    with pytest.raises(RuntimeError):
        service.ingest(document_id="doc1", file_path="fake.pdf")

    assert vector_store.deleted_ids == ["doc1"]
    assert statuses == [("doc1", "failed", None)]


def test_ingest_failure_before_any_chunks_still_cleans_up(monkeypatch):
    def raise_extract(file_path):
        raise RuntimeError("pdf extraction failed")

    monkeypatch.setattr(ingestion_service_module, "extract_pages", raise_extract)
    statuses = _patch_status_recorder(monkeypatch)

    vector_store = FakeVectorStore()
    service = IngestionService(embedding_service=FakeEmbeddingService(), vector_store=vector_store)

    with pytest.raises(RuntimeError):
        service.ingest(document_id="doc1", file_path="fake.pdf")

    assert vector_store.deleted_ids == ["doc1"]
    assert statuses == [("doc1", "failed", None)]


def test_ingest_success_writes_keyword_store(monkeypatch):
    _patch_pipeline(monkeypatch, [{"page_number": 1, "text": "hello world"}])
    statuses = _patch_status_recorder(monkeypatch)

    keyword_store = FakeKeywordStore()
    service = IngestionService(embedding_service=FakeEmbeddingService(), vector_store=FakeVectorStore(), keyword_store=keyword_store)

    service.ingest(document_id="doc1", file_path="fake.pdf")

    assert [c.chunk_id for c in keyword_store.added[0]] == ["doc1_1_0"]
    assert keyword_store.deleted_ids == []
    assert statuses == [("doc1", "ready", 1)]


def test_ingest_failure_in_keyword_store_rolls_back_both_stores(monkeypatch):
    _patch_pipeline(monkeypatch, [{"page_number": 1, "text": "hello world"}])
    statuses = _patch_status_recorder(monkeypatch)

    vector_store = FakeVectorStore()
    keyword_store = FakeKeywordStore(should_fail=True)
    service = IngestionService(embedding_service=FakeEmbeddingService(), vector_store=vector_store, keyword_store=keyword_store)

    with pytest.raises(RuntimeError):
        service.ingest(document_id="doc1", file_path="fake.pdf")

    assert vector_store.deleted_ids == ["doc1"]
    assert keyword_store.deleted_ids == ["doc1"]
    assert statuses == [("doc1", "failed", None)]
