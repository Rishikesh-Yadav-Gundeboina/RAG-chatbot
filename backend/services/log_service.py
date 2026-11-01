import hashlib
import sqlite3
from pathlib import Path

DB_PATH = Path("data/logs/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT,
    file_path TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    status TEXT
);
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES chat_sessions(id),
    role TEXT,
    content TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    if "num_chunks" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN num_chunks INTEGER")
        conn.commit()
    if "file_hash" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT")
        conn.commit()
    return conn


def register_document(document_id: str, filename: str, file_path: str, status: str = "uploaded", file_hash: str | None = None):
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO documents (id, filename, file_path, status, file_hash) VALUES (?, ?, ?, ?, ?)",
        (document_id, filename, file_path, status, file_hash),
    )
    conn.commit()
    conn.close()


def get_document_by_hash(file_hash: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT id, filename, file_path, created_at, status, num_chunks FROM documents WHERE file_hash = ?",
        (file_hash,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "filename": row[1],
        "file_path": row[2],
        "created_at": row[3],
        "status": row[4],
        "num_chunks": row[5],
    }


def backfill_missing_hashes() -> int:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, file_path FROM documents WHERE file_hash IS NULL"
    ).fetchall()
    updated = 0
    for doc_id, file_path in rows:
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            continue
        conn.execute("UPDATE documents SET file_hash = ? WHERE id = ?", (file_hash, doc_id))
        updated += 1
    conn.commit()
    conn.close()
    return updated


def update_document_status(document_id: str, status: str, num_chunks: int | None = None):
    conn = _connect()
    conn.execute(
        "UPDATE documents SET status = ?, num_chunks = COALESCE(?, num_chunks) WHERE id = ?",
        (status, num_chunks, document_id),
    )
    conn.commit()
    conn.close()


def get_document(document_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT id, filename, file_path, created_at, status, num_chunks FROM documents WHERE id = ?",
        (document_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "filename": row[1],
        "file_path": row[2],
        "created_at": row[3],
        "status": row[4],
        "num_chunks": row[5],
    }


def list_documents() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, filename, file_path, created_at, status, num_chunks FROM documents ORDER BY created_at DESC",
    ).fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "filename": row[1],
            "file_path": row[2],
            "created_at": row[3],
            "status": row[4],
            "num_chunks": row[5],
        }
        for row in rows
    ]


def mark_stale_processing_failed() -> int:
    conn = _connect()
    cursor = conn.execute("UPDATE documents SET status = 'failed' WHERE status = 'processing'")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def create_session() -> int:
    conn = _connect()
    cursor = conn.execute("INSERT INTO chat_sessions DEFAULT VALUES")
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def add_message(session_id: int, role: str, content: str) -> int:
    conn = _connect()
    cursor = conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id


def get_recent_messages(session_id: int, limit: int = 10) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]
