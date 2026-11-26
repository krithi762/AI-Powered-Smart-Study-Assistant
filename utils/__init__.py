"""
Utility helper modules for AI-Powered Smart Study Assistant.
Includes PDF parsing, summarization helpers, and text chunking.
"""

from utils.chatbot import (
    ChatbotError,
    QAModelEngine,
    ask_document_question,
    format_qa_prompt,
    generate_suggested_questions,
    get_qa_engine,
    index_document_for_rag,
)
from utils.chunker import (
    chunk_document_pages,
    chunk_page_text,
    clean_chunk_text,
    split_sentences_for_chunking,
)
from utils.embeddings import (
    EmbeddingEngine,
    compute_chunks_embeddings,
    compute_text_embedding,
    get_embedding_engine,
)
from utils.pdf_processor import (
    PDFProcessingError,
    extract_text_from_pdf,
    get_pdf_metadata,
    validate_pdf,
)
from utils.summarizer import (
    SummarizationError,
    SummarizerEngine,
    chunk_text,
    clean_text,
    extract_exam_points,
    extract_key_concepts,
    extract_keywords,
    extract_topics_and_subtopics,
    generate_detailed_summary,
    generate_quick_summary,
    generate_standard_summary,
    generate_study_summary,
    split_into_sentences,
)
from utils.vector_store import (
    FAISSVectorStore,
    VectorStoreError,
    get_vector_store,
)

__all__ = [
    "ChatbotError",
    "EmbeddingEngine",
    "FAISSVectorStore",
    "PDFProcessingError",
    "QAModelEngine",
    "SummarizationError",
    "SummarizerEngine",
    "VectorStoreError",
    "ask_document_question",
    "chunk_document_pages",
    "chunk_page_text",
    "chunk_text",
    "clean_chunk_text",
    "clean_text",
    "compute_chunks_embeddings",
    "compute_text_embedding",
    "extract_exam_points",
    "extract_key_concepts",
    "extract_keywords",
    "extract_text_from_pdf",
    "extract_topics_and_subtopics",
    "format_qa_prompt",
    "generate_detailed_summary",
    "generate_quick_summary",
    "generate_standard_summary",
    "generate_study_summary",
    "generate_suggested_questions",
    "get_embedding_engine",
    "get_pdf_metadata",
    "get_qa_engine",
    "get_vector_store",
    "index_document_for_rag",
    "split_into_sentences",
    "split_sentences_for_chunking",
    "validate_pdf",
]


