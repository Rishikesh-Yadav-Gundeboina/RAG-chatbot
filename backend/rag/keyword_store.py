import re
import sqlite3
from pathlib import Path

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "its", "me", "of", "on", "or",
    "tell", "that", "the", "this", "to", "was", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "about", "explain", "please",
}


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens with stopwords removed, order preserved, no duplicates.

    Splits on underscores too, matching FTS5's unicode61 tokenizer (read_csv -> read, csv).
    """
    seen = []
    for token in re.findall(r"[^\W_]+", text.lower()):
        if token not in STOPWORDS and token not in seen:
            seen.append(token)
    return seen


class KeywordStore:
    def __init__(self, db_path="data/keyword_index.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "text, chunk_id UNINDEXED, document_id UNINDEXED, page_number UNINDEXED, "
            "tokenize='porter unicode61')"
        )
        conn.commit()
        conn.close()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def add_chunks(self, chunks):
        conn = self._connect()
        conn.executemany(
            "DELETE FROM chunks_fts WHERE chunk_id = ?",
            [(chunk.chunk_id,) for chunk in chunks],
        )
        conn.executemany(
            "INSERT INTO chunks_fts (text, chunk_id, document_id, page_number) VALUES (?, ?, ?, ?)",
            [(chunk.text, chunk.chunk_id, chunk.document_id, chunk.page_number) for chunk in chunks],
        )
        conn.commit()
        conn.close()

    def delete_by_document_id(self, document_id):
        conn = self._connect()
        conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
        conn.commit()
        conn.close()

    def has_document(self, document_id) -> bool:
        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM chunks_fts WHERE document_id = ? LIMIT 1", (document_id,)
        ).fetchone()
        conn.close()
        return row is not None

    def count(self) -> int:
        conn = self._connect()
        (total,) = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
        conn.close()
        return total

    def search(self, question, top_k=20, document_ids=None):
        terms = tokenize(question)
        if not terms:
            return []
        # Quoting each term escapes FTS5 query syntax (e.g. '-', ':', 'NOT').
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)

        sql = (
            "SELECT chunk_id, text, document_id, page_number, bm25(chunks_fts) AS score "
            "FROM chunks_fts WHERE chunks_fts MATCH ?"
        )
        params = [match]
        if document_ids:
            sql += f" AND document_id IN ({','.join('?' for _ in document_ids)})"
            params.extend(document_ids)
        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)

        conn = self._connect()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [
            {
                "chunk_id": row[0],
                "text": row[1],
                "document_id": row[2],
                "page_number": int(row[3]) if row[3] is not None else None,
                "bm25_score": row[4],
            }
            for row in rows
        ]


def backfill_keyword_index(vector_store, keyword_store) -> int:
    """Index chunks already in Chroma whose document is missing from the keyword index."""
    from backend.rag.models import DocumentChunk

    chunks = [
        DocumentChunk(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            page_number=chunk["page_number"],
            text=chunk["text"],
        )
        for chunk in vector_store.get_all_chunks()
    ]
    missing_docs = {c.document_id for c in chunks if not keyword_store.has_document(c.document_id)}
    to_add = [c for c in chunks if c.document_id in missing_docs]
    if to_add:
        keyword_store.add_chunks(to_add)
    return len(to_add)
