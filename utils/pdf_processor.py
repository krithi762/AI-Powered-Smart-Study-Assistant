"""
PDF processing module for AI-Powered Smart Study Assistant.
Provides document validation, metadata extraction, and page-by-page text extraction
using PyMuPDF. Detects corrupted files, password protection, and scanned documents.
"""

import contextlib
from pathlib import Path
from typing import Any

import pymupdf


class PDFProcessingError(Exception):
    """Custom exception raised for unrecoverable PDF processing errors."""


def validate_pdf(file_path: str | Path) -> dict[str, Any]:
    """
    Performs multi-stage secure validation on a candidate PDF document.
    
    Checks:
    1. File existence on the filesystem.
    2. File size (rejects empty 0-byte files).
    3. Binary magic bytes header (%PDF-).
    4. PyMuPDF structure parsing (detects corrupted and malformed PDFs).
    5. Password protection (detects encrypted documents).
    6. Page count (rejects documents containing 0 pages).
    
    Returns:
        dict: {
            "valid": bool,
            "error": Optional[str],
            "page_count": int,
            "is_encrypted": bool,
            "file_size": int
        }
    """
    path = Path(file_path)

    if not path.is_file():
        return {
            "valid": False,
            "error": "File does not exist or is not a valid regular file.",
            "page_count": 0,
            "is_encrypted": False,
            "file_size": 0,
        }

    try:
        file_size = path.stat().st_size
    except OSError as e:
        return {
            "valid": False,
            "error": f"Unable to access file: {e!s}",
            "page_count": 0,
            "is_encrypted": False,
            "file_size": 0,
        }

    # Reject empty files
    if file_size == 0:
        return {
            "valid": False,
            "error": "The uploaded PDF file is empty (0 bytes).",
            "page_count": 0,
            "is_encrypted": False,
            "file_size": 0,
        }

    # Verify binary magic bytes (%PDF-)
    try:
        with open(path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                return {
                    "valid": False,
                    "error": "Invalid PDF format: Missing standard '%PDF-' binary header.",
                    "page_count": 0,
                    "is_encrypted": False,
                    "file_size": file_size,
                }
    except OSError as e:
        return {
            "valid": False,
            "error": f"Failed reading file header: {e!s}",
            "page_count": 0,
            "is_encrypted": False,
            "file_size": file_size,
        }

    # Parse document structure with PyMuPDF
    doc = None
    try:
        doc = pymupdf.open(str(path))

        # Check for password encryption
        if getattr(doc, "is_encrypted", False) and getattr(doc, "needs_pass", 0):
            return {
                "valid": False,
                "error": "The PDF is password-protected. Please upload an unprotected document.",
                "page_count": 0,
                "is_encrypted": True,
                "file_size": file_size,
            }

        page_count = doc.page_count
        if page_count <= 0:
            return {
                "valid": False,
                "error": "The PDF contains no pages.",
                "page_count": 0,
                "is_encrypted": False,
                "file_size": file_size,
            }

        return {
            "valid": True,
            "error": None,
            "page_count": page_count,
            "is_encrypted": False,
            "file_size": file_size,
        }

    except (pymupdf.EmptyFileError, pymupdf.FileDataError) as e:
        return {
            "valid": False,
            "error": f"Corrupted or invalid PDF document: {e!s}",
            "page_count": 0,
            "is_encrypted": False,
            "file_size": file_size,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "valid": False,
            "error": f"Unable to parse PDF document: {e!s}",
            "page_count": 0,
            "is_encrypted": False,
            "file_size": file_size,
        }
    finally:
        if doc is not None:
            with contextlib.suppress(Exception):
                doc.close()


def get_pdf_metadata(file_path: str | Path) -> dict[str, Any]:
    """
    Extracts metadata from a PDF document including author, title, dates, and size.
    
    Returns:
        dict: Metadata dictionary or fallback dictionary on failure.
    """
    path = Path(file_path)
    file_size = path.stat().st_size if path.exists() else 0

    metadata = {
        "title": None,
        "author": None,
        "subject": None,
        "keywords": None,
        "creator": None,
        "producer": None,
        "creation_date": None,
        "modification_date": None,
        "page_count": 0,
        "file_size": file_size,
    }

    doc = None
    try:
        doc = pymupdf.open(str(path))
        doc_meta = doc.metadata or {}
        metadata.update({
            "title": doc_meta.get("title") or path.name,
            "author": doc_meta.get("author") or "Unknown",
            "subject": doc_meta.get("subject"),
            "keywords": doc_meta.get("keywords"),
            "creator": doc_meta.get("creator"),
            "producer": doc_meta.get("producer"),
            "creation_date": doc_meta.get("creationDate"),
            "modification_date": doc_meta.get("modDate"),
            "page_count": doc.page_count,
        })
    except Exception:  # noqa: S110, BLE001
        pass
    finally:
        if doc is not None:
            with contextlib.suppress(Exception):
                doc.close()

    return metadata


def extract_text_from_pdf(file_path: str | Path) -> dict[str, Any]:
    """
    Extracts textual content page-by-page from a PDF document using PyMuPDF.
    
    Preserves page numbers (1-indexed), calculates word count per page and total word count.
    Detects documents with zero extractable text (e.g. scanned image PDFs) and flags them.
    
    Returns:
        dict: {
            "success": bool,
            "total_pages": int,
            "total_words": int,
            "has_text": bool,
            "is_scanned": bool,
            "pages": List[{"page_number": int, "text": str, "word_count": int}],
            "metadata": Dict[str, Any],
            "error": Optional[str]
        }
    """
    path = Path(file_path)

    # First validate the document
    validation = validate_pdf(path)
    if not validation["valid"]:
        return {
            "success": False,
            "total_pages": 0,
            "total_words": 0,
            "has_text": False,
            "is_scanned": False,
            "pages": [],
            "metadata": {},
            "error": validation["error"],
        }

    doc = None
    pages: list[dict[str, Any]] = []
    total_words = 0

    try:
        doc = pymupdf.open(str(path))
        total_pages = doc.page_count

        for page_idx in range(total_pages):
            page_number = page_idx + 1
            page = doc.load_page(page_idx)
            raw_text = page.get_text("text") or ""
            clean_text = raw_text.strip() if isinstance(raw_text, str) else str(raw_text).strip()

            # Calculate word count for this page
            words_on_page = len(clean_text.split()) if clean_text else 0
            total_words += words_on_page

            pages.append({
                "page_number": page_number,
                "text": clean_text,
                "word_count": words_on_page,
            })

        # Check if the document has extractable text
        has_text = total_words > 0
        is_scanned = not has_text

        metadata = get_pdf_metadata(path)

        return {
            "success": True,
            "total_pages": total_pages,
            "total_words": total_words,
            "has_text": has_text,
            "is_scanned": is_scanned,
            "pages": pages,
            "metadata": metadata,
            "error": None,
        }

    except Exception as e:  # noqa: BLE001
        return {
            "success": False,
            "total_pages": 0,
            "total_words": 0,
            "has_text": False,
            "is_scanned": False,
            "pages": [],
            "metadata": {},
            "error": f"Failed to extract text from PDF: {e!s}",
        }
    finally:
        if doc is not None:
            with contextlib.suppress(Exception):
                doc.close()
