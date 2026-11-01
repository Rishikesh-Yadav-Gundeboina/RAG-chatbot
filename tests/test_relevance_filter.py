from backend.rag.relevance import filter_relevant_chunks


def test_filter_relevant_chunks_keeps_chunks_at_or_below_threshold():
    chunks = [
        {"text": "a", "distance": 0.3},
        {"text": "b", "distance": 1.9},
    ]
    result = filter_relevant_chunks(chunks, max_distance=1.0)
    assert result == [chunks[0]]


def test_filter_relevant_chunks_empty_input_returns_empty():
    assert filter_relevant_chunks([]) == []


def test_filter_relevant_chunks_boundary_is_inclusive():
    chunks = [{"text": "a", "distance": 1.0}]
    assert filter_relevant_chunks(chunks, max_distance=1.0) == chunks


def test_filter_relevant_chunks_uses_default_max_distance():
    chunks = [{"text": "a", "distance": 0.5}, {"text": "b", "distance": 1.5}]
    assert filter_relevant_chunks(chunks) == [chunks[0]]


def test_filter_keeps_distant_chunk_with_full_keyword_coverage():
    chunks = [
        {"text": "exact match", "distance": 1.4, "keyword_coverage": 1.0},
        {"text": "partial match", "distance": 1.4, "keyword_coverage": 0.5},
    ]
    assert filter_relevant_chunks(chunks) == [chunks[0]]


def test_filter_respects_custom_min_keyword_coverage():
    chunks = [{"text": "partial", "distance": 1.4, "keyword_coverage": 0.5}]
    assert filter_relevant_chunks(chunks, min_keyword_coverage=0.5) == chunks
