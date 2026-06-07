"""
doc_parser.py - DOCX file parsing and local storage management.
Reads .docx files, extracts plain text, and copies files into data/docs/.
"""

import os
import shutil

from docx import Document
from db import DATA_DIR

DOCS_DIR = os.path.join(DATA_DIR, "docs")


def ensure_docs_dir():
    """Create the docs storage directory if it doesn't exist."""
    """
    Copy a .docx file into data/docs/.
    If a file with the same name already exists, a counter suffix is added.
    Returns the destination path.
    """
    ensure_docs_dir()
    filename = os.path.basename(src_path)
    dest_path = os.path.join(DOCS_DIR, filename)

    # Handle filename conflicts by appending a counter
    if os.path.exists(dest_path) and not os.path.samefile(src_path, dest_path):
        name, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(DOCS_DIR, f"{name}_{counter}{ext}")
            counter += 1

    if not os.path.exists(dest_path):
        shutil.copy2(src_path, dest_path)

    return dest_path


def extract_text(docx_path: str) -> str:
    """
    Extract all paragraph text from a .docx file.
    Returns the full text as a single string, paragraphs joined by newlines.
    Returns an empty string on failure.
    """
    try:
        doc = Document(docx_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"[doc_parser] Failed to parse {docx_path}: {e}")
        return ""


def get_title_from_path(docx_path: str) -> str:
    """
    Derive a human-readable title from the file name.
    Strips the extension and replaces underscores/hyphens with spaces.
    """
    filename = os.path.basename(docx_path)
    name, _ = os.path.splitext(filename)
    return name.replace("_", " ").replace("-", " ").strip()


def import_docx(src_path: str) -> tuple[str, str, str]:
    """
    Full import pipeline for a single .docx file:
      1. Copy to local data/docs/
      2. Extract text content
      3. Derive title

    Returns (title, local_file_path, content).
    """
    local_path = copy_to_local(src_path)
    content = extract_text(local_path)
    title = get_title_from_path(local_path)
    return title, local_path, content
