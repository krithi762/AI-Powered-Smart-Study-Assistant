"""
AI Chatbot and RAG Question-Answering Engine for AI-Powered Smart Study Assistant (Phase 4).
Orchestrates document indexing, semantic chunk retrieval via FAISS, grounded answer generation
using Hugging Face Flan-T5, source citation generation, and conversational history tracking.
"""

import os
import re
import threading
from typing import Any, Optional

# Configure HuggingFace cache directory on D: drive to protect C: drive space
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_hf_cache = os.path.join(_base_dir, "models", "cache")
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Lazy import flag for PyTorch and Transformers
_torch_available = False
_transformers_available = False

try:
    import torch
    _torch_available = True
except ImportError:
    torch = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _transformers_available = True
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

from database.db import (
    get_chat_history,
    get_document_by_id,
    get_document_chunks,
    get_document_pages,
    get_or_create_conversation,
    save_chat_message,
    save_document_chunks,
)
from utils.chunker import chunk_document_pages
from utils.vector_store import get_vector_store


class ChatbotError(Exception):
    """Custom exception raised for chatbot QA errors."""


class QAModelEngine:
    """
    Thread-safe Singleton wrapper for Hugging Face Seq2Seq Question Answering model.
    Defaults to 'google/flan-t5-small' for lightweight, instruction-tuned CPU execution.
    """

    _instance: Optional["QAModelEngine"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str = "google/flan-t5-small") -> "QAModelEngine":
        """Returns the shared singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Checks if the QA model is loaded into memory."""
        return self.model is not None and self.tokenizer is not None

    def load_model(self) -> None:
        """Loads the tokenizer and model into CPU memory."""
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return

            if not _transformers_available or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
                raise RuntimeError(
                    "transformers is not installed in the active environment. "
                    "Run 'pip install transformers torch'."
                )

            # Load tokenizer and model weights with low memory allocation
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

    def generate_answer(
        self,
        prompt: str,
        max_length: int = 160,
        min_length: int = 15,
        num_beams: int = 2,
    ) -> str:
        """
        Executes forward inference on prompt with torch.inference_mode().
        """
        self.load_model()
        if self.tokenizer is None or self.model is None or torch is None:
            raise ChatbotError("QA model, tokenizer, or torch runtime is not available.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()


def get_qa_engine(model_name: str = "google/flan-t5-small") -> QAModelEngine:
    """Convenience getter for singleton QA engine."""
    return QAModelEngine.get_instance(model_name=model_name)


# ------------------------------------------------------------------------------
# RAG Coordination & Indexing Pipeline (Task 1 & Task 4)
# ------------------------------------------------------------------------------

def index_document_for_rag(
    document_id: int,
    chunk_size_words: int = 120,
    overlap_words: int = 30,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """
    Extracts text from database pages, chunks it with sentence boundary preservation,
    computes embeddings, and persists the FAISS vector index.

    Returns:
        Dict containing index status, chunk count, and document metadata.
    """
    doc = get_document_by_id(document_id)
    if not doc:
        raise ChatbotError(f"Document with ID {document_id} was not found.")

    if doc.get("processing_status") == "scanned_no_text" or doc.get("total_words", 0) == 0:
        raise ChatbotError("Cannot index a scanned PDF or document with no extractable text.")

    vs = get_vector_store()
    is_indexed = vs.is_document_indexed(document_id)

    # Return cached index if not forcing reindex
    if is_indexed and not force_reindex:
        existing_chunks = get_document_chunks(document_id)
        return {
            "status": "already_indexed",
            "document_id": document_id,
            "filename": doc["original_filename"],
            "total_chunks": len(existing_chunks) if existing_chunks else 1,
            "message": "Document vector index is already built and ready for chat.",
        }

    # Retrieve extracted pages from Phase 2
    pages = get_document_pages(document_id)
    if not pages:
        raise ChatbotError("No page text found in database for this document.")

    # Chunk pages with metadata retention (Task 2)
    chunks = chunk_document_pages(
        document_id=document_id,
        original_filename=doc["original_filename"],
        pages=pages,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    if not chunks:
        raise ChatbotError("Text chunking produced 0 valid chunks. Document may be empty.")

    # Save chunks to SQLite for fast retrieval and persistence
    save_document_chunks(chunks)

    # Generate embeddings and build FAISS index (Task 3 & 4)
    n_vectors = vs.index_document(
        document_id=document_id,
        original_filename=doc["original_filename"],
        chunks=chunks,
    )

    return {
        "status": "indexed",
        "document_id": document_id,
        "filename": doc["original_filename"],
        "total_chunks": n_vectors,
        "message": f"Successfully indexed {n_vectors} chunks in vector database for study chat.",
    }


# ------------------------------------------------------------------------------
# Grounded Question Answering Engine (Task 5 & Task 6)
# ------------------------------------------------------------------------------

def generate_suggested_questions(
    question: str, matched_chunks: list[dict[str, Any]]
) -> list[str]:
    """
    Generates intelligent follow-up suggestions based on the user's question and matched context.
    """
    q_lower = question.lower()
    suggestions: list[str] = []

    if any(k in q_lower for k in ["what is", "define", "concept", "meaning"]):
        suggestions.append("Can you explain this in simpler words with an analogy?")
        suggestions.append("What are the key exam takeaways for this concept?")
        suggestions.append("Can you give a practical real-world example?")
    elif any(k in q_lower for k in ["difference", "compare", "versus", "vs"]):
        suggestions.append("What are the main advantages and disadvantages of each?")
        suggestions.append("Which approach is preferred in modern systems and why?")
    elif any(k in q_lower for k in ["how", "process", "step", "algorithm"]):
        suggestions.append("Can you summarize this process in 3 key steps?")
        suggestions.append("What are the common failure modes or edge cases?")
    else:
        suggestions.append("Can you explain this in simpler terms?")
        suggestions.append("What are the important points for examinations?")
        suggestions.append("Can you provide an example illustrating this?")

    return suggestions[:3]


def format_qa_prompt(
    question: str,
    context_chunks: list[dict[str, Any]],
    recent_history: list[dict[str, Any]] | None = None,
) -> str:
    """
    Builds a strictly grounded instruction prompt for Seq2Seq generation.
    """
    # Concatenate context excerpts with page markers
    context_lines = []
    for c in context_chunks:
        p_num = c.get("page_number", 1)
        text = c.get("chunk_text") or c.get("text", "")
        # Truncate overly long single chunk snippets
        snippet = text[:400]
        context_lines.append(f"[Page {p_num}]: {snippet}")

    context_str = "\n".join(context_lines)

    # Conversational context resolution for follow-ups
    conv_prefix = ""
    if recent_history:
        last_turns = recent_history[-2:]
        history_snippets = []
        for turn in last_turns:
            role = "Student" if turn.get("role") == "user" else "Assistant"
            content = turn.get("content", "")[:100]
            history_snippets.append(f"{role}: {content}")
        if history_snippets:
            conv_prefix = "Recent chat:\n" + "\n".join(history_snippets) + "\n\n"

    prompt = (
        f"Answer the question based strictly on the provided study context. "
        f"If the answer cannot be found in the context, say 'I could not find information about this in the uploaded document.'\n\n"
        f"{conv_prefix}"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt


def ask_document_question(
    document_id: int,
    question: str,
    top_k: int = 4,
    threshold: float = 0.28,
    conversation_id: int | None = None,
    save_to_history: bool = True,
) -> dict[str, Any]:
    """
    Executes the full RAG pipeline for a student query:
    1. Validates document and query.
    2. Ensures document vector index is ready.
    3. Runs FAISS similarity search.
    4. Enforces grounding: if no chunks exceed threshold, gracefully returns not found.
    5. Feeds retrieved evidence into Seq2Seq QA model.
    6. Stores user and assistant turns in conversation history.
    7. Returns answer with source page references and follow-up suggestions.
    """
    cleaned_query = (question or "").strip()
    if not cleaned_query:
        raise ChatbotError("Please enter a question about your study material.")

    doc = get_document_by_id(document_id)
    if not doc:
        raise ChatbotError(f"Document with ID {document_id} does not exist.")

    if doc.get("processing_status") == "scanned_no_text":
        raise ChatbotError("This PDF appears to be a scanned image with no extractable text.")

    # Ensure document is indexed
    vs = get_vector_store()
    if not vs.is_document_indexed(document_id):
        index_document_for_rag(document_id)

    # Retrieve or create conversation record
    conv = get_or_create_conversation(document_id)
    active_conv_id = conversation_id or conv["id"]

    # Retrieve recent chat history for context continuity
    chat_history = get_chat_history(active_conv_id) if save_to_history else []

    # Conversational query resolution for follow-up questions (e.g., 'How does it work?')
    search_query = cleaned_query
    if chat_history:
        has_pronoun = bool(
            re.search(
                r"\b(it|its|they|them|their|this|that|these|those|the process|such|above|said)\b",
                cleaned_query,
                re.IGNORECASE,
            )
        )
        if has_pronoun or len(cleaned_query.split()) <= 4:
            last_user_turn = next(
                (t["content"] for t in reversed(chat_history) if t.get("role") == "user"),
                None,
            )
            if last_user_turn:
                search_query = f"{last_user_turn} {cleaned_query}"

    # 1. Similarity search in FAISS (Task 4)
    matched_chunks = vs.search_document(
        document_id=document_id,
        query=search_query,
        top_k=top_k,
        threshold=threshold,
    )

    # 2. Accuracy Rule: If no relevant chunks match threshold (Task 5)
    if not matched_chunks:
        fallback_msg = (
            "I could not find information about this topic in the uploaded document. "
            "Please check your question phrasing or refer to the full document text."
        )

        if save_to_history:
            save_chat_message(active_conv_id, document_id, "user", cleaned_query)
            save_chat_message(active_conv_id, document_id, "assistant", fallback_msg, sources=[])

        return {
            "conversation_id": active_conv_id,
            "document_id": document_id,
            "filename": doc["original_filename"],
            "question": cleaned_query,
            "answer": fallback_msg,
            "sources": [],
            "suggested_questions": [
                "What is the main topic of this document?",
                "Can you summarize this document for me?",
            ],
            "is_grounded": False,
            "model_name": "google/flan-t5-small",
        }

    # 3. Format grounded prompt and run QA generation (Task 5)
    prompt = format_qa_prompt(cleaned_query, matched_chunks, recent_history=chat_history)
    qa_engine = get_qa_engine()

    try:
        raw_answer = qa_engine.generate_answer(prompt, max_length=160, min_length=12)
    except (RuntimeError, ValueError, TypeError, ChatbotError):
        # Fallback to direct extractive synthesis if model generation encounters an issue
        top_chunk = matched_chunks[0]
        raw_answer = (
            f"Based on page {top_chunk.get('page_number', 1)}: "
            f"{top_chunk.get('chunk_text', '')[:300]}..."
        )

    # Check for ungrounded or not-found response from the model
    not_found_phrases = [
        "could not find",
        "cannot find",
        "not found in the",
        "not mentioned in the",
        "unsupported",
        "no information",
    ]
    is_grounded = True
    if any(phrase in raw_answer.lower() for phrase in not_found_phrases) or not raw_answer.strip():
        is_grounded = False
        final_answer = (
            "I could not find information about this topic in the uploaded document. "
            "Please check your question phrasing or refer to the full document text."
        )
        sources: list[dict[str, Any]] = []
    else:
        # Clean and ensure capitalized answer
        final_answer = raw_answer.strip()
        if final_answer and not final_answer[0].isupper():
            final_answer = final_answer[0].upper() + final_answer[1:]
        if final_answer and not final_answer.endswith((".", "!", "?")):
            final_answer += "."

        # Format source references for student inspection
        sources = []
        for c in matched_chunks:
            sources.append({
                "chunk_id": c.get("chunk_id", ""),
                "page_number": c.get("page_number", 1),
                "original_filename": doc["original_filename"],
                "similarity_score": c.get("similarity_score", 0.0),
                "excerpt": (c.get("chunk_text") or c.get("text", ""))[:280] + "...",
                "word_count": c.get("word_count", 0),
            })

    # 5. Suggested follow-up questions
    suggested = generate_suggested_questions(cleaned_query, matched_chunks)

    # 6. Save conversation history
    if save_to_history:
        save_chat_message(active_conv_id, document_id, "user", cleaned_query)
        save_chat_message(
            active_conv_id,
            document_id,
            "assistant",
            final_answer,
            sources=sources,
        )

    return {
        "conversation_id": active_conv_id,
        "document_id": document_id,
        "filename": doc["original_filename"],
        "question": cleaned_query,
        "answer": final_answer,
        "sources": sources,
        "suggested_questions": suggested,
        "is_grounded": is_grounded,
        "model_name": qa_engine.model_name,
    }
