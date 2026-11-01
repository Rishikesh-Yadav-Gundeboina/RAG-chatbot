# Maximum Chroma distance (lower = more similar) for a chunk to be treated
# as relevant. Verified against this project's ingested documents
# (all-MiniLM-L6-v2 embeddings): on-topic questions' closest chunk landed at
# 0.38-0.80, off-topic questions' closest chunk landed at 1.17-1.57 — 1.0
# sits cleanly in the gap. Re-check with tests/test_retrieve.py (prints
# per-chunk distance) if the document set changes significantly.
MAX_DISTANCE = 1.0

# Minimum fraction of the question's non-stopword terms a chunk must contain
# to be kept despite a distance above MAX_DISTANCE. This lets exact-term hits
# (names, codes, identifiers) that embeddings score as distant through, while
# off-topic questions — which rarely match most of their terms — still fail.
MIN_KEYWORD_COVERAGE = 1.0


def filter_relevant_chunks(chunks, max_distance=MAX_DISTANCE, min_keyword_coverage=MIN_KEYWORD_COVERAGE):
    """Keep chunks whose distance is <= max_distance, or whose keyword_coverage
    is >= min_keyword_coverage.

    `chunks` is the list of dicts produced by extract_chunks() (each has a
    'distance' key, and 'keyword_coverage' in hybrid mode). Returns a new
    list, order preserved.
    """
    return [
        c for c in chunks
        if c["distance"] <= max_distance or c.get("keyword_coverage", 0.0) >= min_keyword_coverage
    ]
