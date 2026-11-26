"""
Document Chunking Module for AI-Powered Smart Study Assistant (Phase 4 - RAG).
Provides sentence-aware, overlapping text chunking preserving paragraph boundaries,
page numbers, and document metadata for vector indexing and semantic retrieval.
"""

import re
import unicodedata
from typing import Any


def clean_chunk_text(text: str) -> str:
    """
    Normalizes extracted PDF text for chunking:
    - Normalizes unicode (NFKC)
    - Fixes hyphenated words broken across line wraps ('con-\\ncept' -> 'concept')
    - Strips page markers and extra whitespace
    - Preserves sentence and paragraph boundaries
    """
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphens across line breaks
    cleaned = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", cleaned)

    # Remove artificial page markers
    cleaned = re.sub(r"---+\s*Page\s*\d+\s*---+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*Page\s+\d+\s+(?:of\s+\d+)?\s*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)

    # Standardize whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_sentences_for_chunking(text: str) -> list[str]:
    """
    Splits text into sentences preserving abbreviations (e.g., Dr., Prof., Fig., e.g., i.e.).
    """
    if not text or not text.strip():
        return []

    abbreviations = [
        "e.g.", "i.e.", "etc.", "vs.", "al.", "approx.",
        "dr.", "mr.", "mrs.", "ms.", "prof.", "fig.", "eq.",
    ]
    abbr_map: dict[str, str] = {}

    def _replace_abbr(match: re.Match) -> str:
        matched_str = match.group(0)
        placeholder = f"__ABBR_{len(abbr_map)}__"
        abbr_map[placeholder] = matched_str
        return placeholder

    protected = text
    for abbr in abbreviations:
        escaped = re.escape(abbr)
        pattern = re.compile(rf"\b{escaped}", re.IGNORECASE)
        protected = pattern.sub(_replace_abbr, protected)

    # Split on sentence terminals followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", protected)

    restored: list[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        for placeholder, original in abbr_map.items():
            s_clean = s_clean.replace(placeholder, original)
        restored.append(s_clean)

    return restored


def chunk_page_text(
    text: str,
    document_id: int,
    original_filename: str,
    page_number: int,
    start_chunk_index: int = 0,
    chunk_size_words: int = 120,
    overlap_words: int = 30,
) -> list[dict[str, Any]]:
    """
    Chunks text from a single document page into overlapping, sentence-aware segments.
    
    Args:
        text: Raw or pre-cleaned text of the page.
        document_id: Database ID of the parent document.
        original_filename: Original PDF filename for citation references.
        page_number: 1-based page number where the text originates.
        start_chunk_index: Starting sequential chunk index on this page.
        chunk_size_words: Maximum target word count per chunk.
        overlap_words: Target word overlap between successive chunks.

    Returns:
        List of chunk dictionaries with metadata.
    """
    cleaned = clean_chunk_text(text)
    if not cleaned:
        return []

    sentences = split_sentences_for_chunking(cleaned)
    if not sentences:
        words = cleaned.split()
        if not words:
            return []
        sentences = [cleaned]

    chunks: list[dict[str, Any]] = []
    current_sentences: list[str] = []
    current_word_count = 0
    chunk_idx = start_chunk_index

    def _flush_chunk(sents: list[str]) -> None:
        nonlocal chunk_idx
        chunk_text = " ".join(sents).strip()
        if not chunk_text:
            return
        w_count = len(chunk_text.split())
        chunks.append({
            "document_id": document_id,
            "original_filename": original_filename,
            "page_number": page_number,
            "chunk_index": chunk_idx,
            "chunk_id": f"doc_{document_id}_p{page_number}_c{chunk_idx}",
            "chunk_text": chunk_text,
            "word_count": w_count,
        })
        chunk_idx += 1

    for sentence in sentences:
        s_words = sentence.split()
        s_len = len(s_words)

        # Handle very long sentences that exceed the target size by themselves
        if s_len > chunk_size_words:
            if current_sentences:
                _flush_chunk(current_sentences)
                current_sentences = []
                current_word_count = 0

            # Break long sentence into word slices
            for i in range(0, s_len, max(1, chunk_size_words - overlap_words)):
                slice_words = s_words[i:i + chunk_size_words]
                if slice_words:
                    _flush_chunk([" ".join(slice_words)])
            continue

        if current_word_count + s_len > chunk_size_words and current_sentences:
            _flush_chunk(current_sentences)

            # Build overlap from the tail of current_sentences
            overlap_sents: list[str] = []
            accumulated_overlap = 0
            for sent in reversed(current_sentences):
                cnt = len(sent.split())
                if accumulated_overlap + cnt <= overlap_words or not overlap_sents:
                    overlap_sents.insert(0, sent)
                    accumulated_overlap += cnt
                else:
                    break

            current_sentences = overlap_sents + [sentence]
            current_word_count = sum(len(s.split()) for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_word_count += s_len

    if current_sentences:
        _flush_chunk(current_sentences)

    return chunks


def chunk_document_pages(
    document_id: int,
    original_filename: str,
    pages: list[dict[str, Any]],
    chunk_size_words: int = 120,
    overlap_words: int = 30,
) -> list[dict[str, Any]]:
    """
    Processes all extracted pages for a document and returns a list of chunk dicts.

    Args:
        document_id: Database ID of the document.
        original_filename: Original file name.
        pages: List of page dicts (must contain 'page_number' and 'extracted_text' or 'text').
        chunk_size_words: Target maximum words per chunk.
        overlap_words: Overlapping word count between consecutive chunks.

    Returns:
        Consolidated list of chunk dicts across all pages.
    """
    all_chunks: list[dict[str, Any]] = []

    for page_data in pages:
        p_num = page_data.get("page_number", 1)
        raw_text = page_data.get("extracted_text") or page_data.get("text", "")
        if not raw_text or not raw_text.strip():
            continue

        page_chunks = chunk_page_text(
            text=raw_text,
            document_id=document_id,
            original_filename=original_filename,
            page_number=p_num,
            start_chunk_index=len(all_chunks),
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
        all_chunks.extend(page_chunks)

    return all_chunks
