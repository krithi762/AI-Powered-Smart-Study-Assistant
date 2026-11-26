"""
App-level facade for AI Chatbot and RAG Question Answering (Phase 4).
Re-exports ask_document_question, index_document_for_rag, and QAModelEngine from utils.chatbot.
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

__all__ = [
    "ChatbotError",
    "QAModelEngine",
    "ask_document_question",
    "format_qa_prompt",
    "generate_suggested_questions",
    "get_qa_engine",
    "index_document_for_rag",
]
