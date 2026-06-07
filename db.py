"""
db.py - SQLite database management for LDRM
Handles initialization, insertion, querying, and searching of documents.
"""

import sqlite3
import os
import sys
from datetime import datetime

from nlp import normalize_text

APP_NAME = "LDRM"


def get_data_dir() -> str:
    """Return the local application data directory for the app."""
    if getattr(sys, "frozen", False):
        # When bundled, use a persistent user data directory instead of temp extraction path.
        if sys.platform == "darwin":
            return os.path.join(os.path.expanduser("~"), "Library", "Application Support", APP_NAME)
        if sys.platform.startswith("win"):
            return os.path.join(os.getenv("APPDATA", os.path.expanduser("~\\AppData\\Roaming")), APP_NAME)
        return os.path.join(os.path.expanduser("~"), ".local", "share", APP_NAME)

    # During normal development, keep data alongside the project.
    return os.path.join(os.path.dirname(__file__), "data")


DATA_DIR = get_data_dir()
DB_PATH = os.path.join(DATA_DIR, "app.db")


def get_connection():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-like access
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            file_path   TEXT NOT NULL UNIQUE,
            content     TEXT,
            category    TEXT DEFAULT 'Uncategorized',
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_document(title: str, file_path: str, content: str) -> int:
    """
    Insert a new document record into the database.
    Returns the new row id, or -1 if the file already exists.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO documents (title, file_path, content, category, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (title, file_path, content, "Uncategorized", datetime.now().isoformat()),
        )
        conn.commit()
        row_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        # file_path is UNIQUE – document already imported
        row_id = -1
    finally:
        conn.close()
    return row_id


def get_all_documents() -> list[dict]:
    """Return all documents as a list of dicts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, file_path, content, category, created_at FROM documents ORDER BY id DESC"
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_document_by_id(doc_id: int) -> dict | None:
    """Return a single document by id, or None if not found."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, file_path, content, category, created_at FROM documents WHERE id = ?",
        (doc_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def search_documents(keyword: str) -> list[dict]:
    """Search documents by keyword in title or content.

    Uses light normalization and a simple relevance score so Chinese phrase
    searches behave better than a raw LIKE query.
    """
    query = normalize_text(keyword)
    if not query:
        return []

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, title, file_path, content, category, created_at
        FROM documents
        ORDER BY id DESC
        """
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    compact_query = query.replace(" ", "")
    scored_rows: list[tuple[int, int, dict]] = []

    for row in rows:
        title = normalize_text(row.get("title", ""))
        content = normalize_text(row.get("content", "") or "")
        title_compact = title.replace(" ", "")
        content_compact = content.replace(" ", "")

        score = 0
        if compact_query in title_compact:
            score += 30
            score += max(0, 10 - title_compact.find(compact_query))
        if compact_query in content_compact:
            score += 12

        query_terms = [term for term in query.split(" ") if term]
        if len(query_terms) > 1:
            score += sum(3 for term in query_terms if term in title_compact)
            score += sum(1 for term in query_terms if term in content_compact)

        if score > 0:
            scored_rows.append((score, row["id"], row))

    scored_rows.sort(key=lambda item: (-item[0], -item[1]))
    return [row for _, _, row in scored_rows]


def update_document_category(doc_id: int, category: str):
    """Update the category field for a document."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE documents SET category = ? WHERE id = ?",
        (category, doc_id),
    )
    conn.commit()
    conn.close()


def get_all_contents() -> list[tuple[int, str]]:
    """Return (id, content) tuples for all documents (used by NLP modules)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM documents ORDER BY id")
    rows = [(row["id"], row["content"] or "") for row in cursor.fetchall()]
    conn.close()
    return rows


def delete_document(doc_id: int):
    """Delete a document record from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
