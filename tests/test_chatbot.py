"""
Automated Test Suite for Phase 4 – AI Chatbot using Retrieval-Augmented Generation (RAG).
Tests:
1. Document chunking, text cleaning, sentence splitting with abbreviation protection, and overlap.
2. SentenceTransformers embedding generation, vector normalization, and batch encoding.
3. FAISS vector store indexing, similarity search, multi-document isolation, and index deletion.
4. RAG question-answering pipeline, grounded response generation, citation verification, and follow-up queries.
5. Conversational history tracking, history retrieval, and clear chat.
6. Flask API endpoints for chat, ask, history, clear, indexing, and health check diagnostics.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app import create_app
from database.db import (
    init_db,
    insert_document,
    insert_document_pages,
)
from utils.chatbot import (
    ask_document_question,
    format_qa_prompt,
    generate_suggested_questions,
    index_document_for_rag,
)
from utils.chunker import (
    chunk_document_pages,
    chunk_page_text,
    clean_chunk_text,
    split_sentences_for_chunking,
)
from utils.embeddings import (
    compute_text_embedding,
    get_embedding_engine,
)
from utils.vector_store import (
    FAISSVectorStore,
    reset_vector_store,
)


@pytest.fixture
def chat_test_env():
    """Sets up an isolated temporary workspace and database for testing Phase 4."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    upload_folder = temp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)
    vector_dir = temp_path / "vector_store"
    vector_dir.mkdir(parents=True, exist_ok=True)
    test_db_path = temp_path / "test_chat.db"

    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = str(upload_folder)
    app.config["DATABASE_PATH"] = test_db_path
    app.config["VECTOR_STORE_DIR"] = str(vector_dir)
    app.config["TESTING"] = True
    app.config["SUMMARIZATION_MODEL"] = "mock"

    init_db(test_db_path)

    # Reset the module-level singleton so this test gets its own isolated instance
    # rooted at vector_dir (the new `get_vector_store` is config-aware, but resetting
    # guarantees a clean slate even when tests run sequentially in the same process).
    reset_vector_store()

    # Initialize isolated vector store for this test
    vs = FAISSVectorStore(storage_dir=vector_dir, dimension=384)

    client = app.test_client()

    yield {
        "app": app,
        "client": client,
        "temp_path": temp_path,
        "upload_folder": upload_folder,
        "vector_dir": vector_dir,
        "db_path": test_db_path,
        "vector_store": vs,
    }

    # Cleanup after test: reset singleton so subsequent tests start fresh
    reset_vector_store()
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 1. Document Chunking & Text Cleaning Tests (Task A)
# ==============================================================================

def test_clean_chunk_text_normalizes_and_fixes_hyphens():
    """Verifies that line-break hyphens and extraneous formatting are cleaned."""
    raw_text = "This is a demon-\nstration of text clean-\nup. --- Page 1 ---\nExtra   spaces."
    cleaned = clean_chunk_text(raw_text)
    assert "demonstration" in cleaned
    assert "cleanup" in cleaned
    assert "--- Page 1 ---" not in cleaned
    assert "   " not in cleaned


def test_split_sentences_preserves_abbreviations():
    """Verifies sentence splitting preserves abbreviations such as Dr., Prof., e.g., and i.e."""
    text = "Dr. Smith met with Prof. Davis at approx. 9:00 a.m. E.g., they discussed biology. It went well."
    sentences = split_sentences_for_chunking(text)
    assert len(sentences) == 2
    assert "Dr. Smith met with Prof. Davis at approx. 9:00 a.m. E.g., they discussed biology." in sentences[0]
    assert sentences[1] == "It went well."


def test_chunk_page_text_preserves_metadata_and_overlap():
    """Verifies page-level chunking produces overlapping chunks with page metadata."""
    sample_text = (
        "Cellular biology is the study of cell structure and function. "
        "Cells are the fundamental units of all living organisms. "
        "They carry genetic information in DNA and synthesize proteins. "
        "Mitochondria generate adenosine triphosphate through cellular respiration. "
        "Plant cells also contain chloroplasts for photosynthesis."
    )
    chunks = chunk_page_text(
        text=sample_text,
        document_id=42,
        original_filename="biology_chapter1.pdf",
        page_number=3,
        start_chunk_index=0,
        chunk_size_words=25,
        overlap_words=10,
    )

    assert len(chunks) >= 2
    for c in chunks:
        assert c["document_id"] == 42
        assert c["original_filename"] == "biology_chapter1.pdf"
        assert c["page_number"] == 3
        assert "chunk_text" in c
        assert c["word_count"] > 0
        assert c["chunk_id"].startswith("doc_42_p3_c")


def test_chunk_document_pages_multiple_pages():
    """Verifies chunking across a multi-page document maintains sequential indices."""
    pages = [
        {"page_number": 1, "text": "Page one text describes classical mechanics and Newton laws."},
        {"page_number": 2, "text": "Page two text discusses kinetic energy and gravitational potential."},
    ]
    all_chunks = chunk_document_pages(
        document_id=7,
        original_filename="physics.pdf",
        pages=pages,
        chunk_size_words=50,
        overlap_words=10,
    )
    assert len(all_chunks) == 2
    assert all_chunks[0]["page_number"] == 1
    assert all_chunks[1]["page_number"] == 2
    assert all_chunks[0]["chunk_index"] == 0
    assert all_chunks[1]["chunk_index"] == 1


# ==============================================================================
# 2. Embedding Generation Tests (Task B)
# ==============================================================================

def test_embedding_engine_singleton_and_properties():
    """Verifies that the EmbeddingEngine behaves as a singleton and reports 384 dimensions."""
    engine1 = get_embedding_engine()
    engine2 = get_embedding_engine()
    assert engine1 is engine2
    assert engine1.dimension == 384


def test_encode_text_normalized_l2():
    """Verifies single text encoding returns an L2-normalized 1D vector."""
    vec = compute_text_embedding("Artificial neural networks learn representations from data.")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Verify unit length (L2 norm is approximately 1.0)
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4


def test_encode_batch_shape():
    """Verifies batch encoding returns correct 2D float32 array shape."""
    texts = [
        "First concept snippet.",
        "Second biological principle.",
        "Third mathematical theorem.",
    ]
    engine = get_embedding_engine()
    batch_embs = engine.encode_batch(texts, normalize=True)
    assert batch_embs.shape == (3, 384)
    assert batch_embs.dtype == np.float32


# ==============================================================================
# 3. Vector Database (FAISS) Tests (Task C)
# ==============================================================================

def test_vector_store_indexing_and_similarity_search(chat_test_env):
    """Verifies FAISS index creation, persistence, and semantic search."""
    vs = chat_test_env["vector_store"]
    doc_id = 101

    chunks = [
        {
            "document_id": doc_id,
            "original_filename": "botany.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "chunk_id": "doc_101_p1_c0",
            "chunk_text": "Photosynthesis is the biological process used by plants to convert solar energy into chemical sugars.",
            "word_count": 15,
        },
        {
            "document_id": doc_id,
            "original_filename": "botany.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "chunk_id": "doc_101_p2_c1",
            "chunk_text": "Xylem vessels transport water and dissolved mineral nutrients upward from roots to plant leaves.",
            "word_count": 14,
        },
    ]

    indexed_count = vs.index_document(doc_id, "botany.pdf", chunks)
    assert indexed_count == 2
    assert vs.is_document_indexed(doc_id) is True

    # Search for photosynthesis
    results = vs.search_document(
        document_id=doc_id,
        query="How do plants convert sunlight into sugar?",
        top_k=2,
        threshold=0.25,
    )
    assert len(results) >= 1
    top_hit = results[0]
    assert top_hit["page_number"] == 1
    assert "Photosynthesis" in top_hit["chunk_text"]
    assert top_hit["similarity_score"] > 0.35


def test_vector_store_multi_document_isolation(chat_test_env):
    """Verifies that search on Document A NEVER returns chunks from Document B."""
    vs = chat_test_env["vector_store"]

    chunks_a = [
        {
            "document_id": 1,
            "original_filename": "astronomy.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "chunk_id": "doc_1_p1_c0",
            "chunk_text": "Supernovae are catastrophic stellar explosions that occur during the last evolutionary stages of massive stars.",
            "word_count": 14,
        }
    ]
    chunks_b = [
        {
            "document_id": 2,
            "original_filename": "economics.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "chunk_id": "doc_2_p1_c0",
            "chunk_text": "Gross domestic product measures the monetary value of final goods and services produced within a nation.",
            "word_count": 16,
        }
    ]

    vs.index_document(1, "astronomy.pdf", chunks_a)
    vs.index_document(2, "economics.pdf", chunks_b)

    # Search for economic terms inside document 1 (astronomy)
    econ_results_in_doc1 = vs.search_document(
        document_id=1,
        query="What is gross domestic product?",
        top_k=2,
        threshold=0.28,
    )
    # Astronomy doc must not contain economic concepts exceeding threshold
    for r in econ_results_in_doc1:
        assert r["document_id"] == 1
        assert "Gross domestic product" not in r["chunk_text"]

    # Search inside document 2 returns the correct economic chunk
    econ_results_in_doc2 = vs.search_document(
        document_id=2,
        query="What is gross domestic product?",
        top_k=2,
        threshold=0.28,
    )
    assert len(econ_results_in_doc2) == 1
    assert econ_results_in_doc2[0]["document_id"] == 2
    assert "Gross domestic product" in econ_results_in_doc2[0]["chunk_text"]


def test_vector_store_delete_document_index(chat_test_env):
    """Verifies that delete_document_index removes index and chunk files from disk."""
    vs = chat_test_env["vector_store"]
    doc_id = 99
    chunks = [
        {
            "document_id": doc_id,
            "original_filename": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "chunk_id": "doc_99_p1_c0",
            "chunk_text": "Sample text content for index deletion testing.",
            "word_count": 7,
        }
    ]
    vs.index_document(doc_id, "sample.pdf", chunks)
    assert vs.is_document_indexed(doc_id) is True

    deleted = vs.delete_document_index(doc_id)
    assert deleted is True
    assert vs.is_document_indexed(doc_id) is False


# ==============================================================================
# 4. RAG Question Answering & Grounding Tests (Task D)
# ==============================================================================

def test_format_qa_prompt_structure():
    """Verifies prompt construction contains study context, page tags, and strict instruction."""
    chunks = [
        {"page_number": 2, "chunk_text": "The Krebs cycle produces NADH and FADH2 in cellular respiration."},
    ]
    prompt = format_qa_prompt("What does the Krebs cycle produce?", chunks)
    assert "[Page 2]" in prompt
    assert "Krebs cycle" in prompt
    assert "Answer the question based strictly on the provided study context." in prompt


def test_generate_suggested_questions_heuristics():
    """Verifies intelligent contextual follow-up suggestions."""
    matched = [{"page_number": 1, "chunk_text": "Mitochondria are organelles."}]
    suggestions = generate_suggested_questions("What is mitochondria?", matched)
    assert len(suggestions) > 0
    assert any("simpler" in s.lower() or "exam" in s.lower() for s in suggestions)


def test_ask_document_question_grounded_answer(chat_test_env):
    """
    Verifies full end-to-end RAG question answering:
    - Inserts a document with biology text
    - Asks a grounded question
    - Returns answer with page number, filename citation, and is_grounded=True.
    """
    app = chat_test_env["app"]
    db_path = chat_test_env["db_path"]

    with app.app_context():
        doc_id = insert_document(
            original_filename="biology_cell.pdf",
            stored_filename="bio_hash_123.pdf",
            file_path=str(chat_test_env["upload_folder"] / "bio.pdf"),
            total_pages=2,
            total_words=60,
            db_path=db_path,
        )
        insert_document_pages(
            doc_id,
            [
                {
                    "page_number": 1,
                    "text": "The cell membrane consists of a phospholipid bilayer with embedded transport proteins.",
                },
                {
                    "page_number": 2,
                    "text": "Mitochondria generate adenosine triphosphate through aerobic cellular respiration.",
                },
            ],
            db_path=db_path,
        )

        response = ask_document_question(
            document_id=doc_id,
            question="What is the structure of the cell membrane?",
            top_k=2,
        )

        assert response["document_id"] == doc_id
        assert response["filename"] == "biology_cell.pdf"
        assert response["is_grounded"] is True
        assert len(response["sources"]) >= 1
        assert response["sources"][0]["page_number"] == 1
        assert "biology_cell.pdf" in response["sources"][0]["original_filename"]
        assert len(response["answer"]) > 10


def test_ask_document_question_unsupported_not_found(chat_test_env):
    """
    Verifies that asking an out-of-domain question (not present in PDF)
    returns an honest 'could not find' message without fabricating answers.
    """
    app = chat_test_env["app"]
    db_path = chat_test_env["db_path"]

    with app.app_context():
        doc_id = insert_document(
            original_filename="calculus.pdf",
            stored_filename="calc_123.pdf",
            file_path=str(chat_test_env["upload_folder"] / "calc.pdf"),
            total_pages=1,
            total_words=30,
            db_path=db_path,
        )
        insert_document_pages(
            doc_id,
            [
                {
                    "page_number": 1,
                    "text": "Derivatives represent the instantaneous rate of change of a mathematical function.",
                }
            ],
            db_path=db_path,
        )

        # Unrelated question about culinary recipes
        response = ask_document_question(
            document_id=doc_id,
            question="What is the best recipe for baking chocolate chip cookies?",
            top_k=2,
            threshold=0.32,
        )

        assert response["is_grounded"] is False
        assert "could not find" in response["answer"].lower()
        assert len(response["sources"]) == 0


def test_ask_document_question_follow_up_with_pronoun(chat_test_env):
    """Verifies that follow-up questions with pronouns (e.g., 'How does it work?') leverage chat history."""
    app = chat_test_env["app"]
    db_path = chat_test_env["db_path"]

    with app.app_context():
        doc_id = insert_document(
            original_filename="thermodynamics.pdf",
            stored_filename="thermo_123.pdf",
            file_path=str(chat_test_env["upload_folder"] / "thermo.pdf"),
            total_pages=1,
            total_words=45,
            db_path=db_path,
        )
        insert_document_pages(
            doc_id,
            [
                {
                    "page_number": 1,
                    "text": "The Carnot heat engine is an idealized thermodynamic cycle operating between two temperatures to achieve maximum efficiency.",
                }
            ],
            db_path=db_path,
        )

        # First question establishes the subject
        q1_res = ask_document_question(
            document_id=doc_id,
            question="What is the Carnot engine?",
            save_to_history=True,
        )
        assert q1_res["is_grounded"] is True

        # Follow-up question with pronoun 'it'
        q2_res = ask_document_question(
            document_id=doc_id,
            question="What does it operate between?",
            save_to_history=True,
        )
        assert q2_res["is_grounded"] is True
        assert len(q2_res["sources"]) >= 1
        assert q2_res["sources"][0]["page_number"] == 1


# ==============================================================================
# 5. Flask API Endpoint Tests (Task E & F)
# ==============================================================================

def test_chat_html_page_loads(chat_test_env):
    """Verifies GET /document/<id>/chat loads successfully and renders the chatbot template."""
    client = chat_test_env["client"]
    db_path = chat_test_env["db_path"]

    doc_id = insert_document(
        original_filename="chemistry.pdf",
        stored_filename="chem_123.pdf",
        file_path=str(chat_test_env["upload_folder"] / "chem.pdf"),
        total_pages=1,
        total_words=25,
        db_path=db_path,
    )
    insert_document_pages(
        doc_id,
        [{"page_number": 1, "text": "Covalent bonds involve the sharing of electron pairs between atoms."}],
        db_path=db_path,
    )

    response = client.get(f"/document/{doc_id}/chat")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "AI Study Chatbot" in html
    assert "chemistry.pdf" in html
    assert "chat-window-card" in html
    assert "chat-question-input" in html


def test_chat_api_ask_valid_question(chat_test_env):
    """Verifies POST /document/<id>/chat/ask returns JSON with answer and source citations."""
    client = chat_test_env["client"]
    db_path = chat_test_env["db_path"]

    doc_id = insert_document(
        original_filename="geology.pdf",
        stored_filename="geo_123.pdf",
        file_path=str(chat_test_env["upload_folder"] / "geo.pdf"),
        total_pages=1,
        total_words=30,
        db_path=db_path,
    )
    insert_document_pages(
        doc_id,
        [{"page_number": 1, "text": "Igneous rocks form through the cooling and solidification of magma or lava."}],
        db_path=db_path,
    )

    response = client.post(
        f"/document/{doc_id}/chat/ask",
        json={"question": "How do igneous rocks form?"},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "answer" in data
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["page_number"] == 1
    assert "suggested_questions" in data


def test_chat_api_empty_question_rejected(chat_test_env):
    """Verifies POST /document/<id>/chat/ask rejects empty or whitespace queries with 400."""
    client = chat_test_env["client"]
    db_path = chat_test_env["db_path"]

    doc_id = insert_document(
        original_filename="sample.pdf",
        stored_filename="sample_123.pdf",
        file_path=str(chat_test_env["upload_folder"] / "sample.pdf"),
        total_pages=1,
        total_words=20,
        db_path=db_path,
    )

    response = client.post(
        f"/document/{doc_id}/chat/ask",
        json={"question": "   "},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "valid question" in data["error"].lower()


def test_chat_api_clear_history(chat_test_env):
    """Verifies POST /document/<id>/chat/clear deletes messages and resets conversation."""
    client = chat_test_env["client"]
    db_path = chat_test_env["db_path"]

    doc_id = insert_document(
        original_filename="history.pdf",
        stored_filename="hist_123.pdf",
        file_path=str(chat_test_env["upload_folder"] / "hist.pdf"),
        total_pages=1,
        total_words=20,
        db_path=db_path,
    )
    insert_document_pages(
        doc_id,
        [{"page_number": 1, "text": "Historical text for conversation test."}],
        db_path=db_path,
    )

    # Ask a question to generate chat records
    client.post(
        f"/document/{doc_id}/chat/ask",
        json={"question": "What is this text?"},
        headers={"Accept": "application/json"},
    )

    # Check history has messages
    hist_res = client.get(f"/document/{doc_id}/chat/history")
    assert hist_res.status_code == 200
    assert len(hist_res.get_json()["messages"]) >= 2

    # Clear history
    clear_res = client.post(f"/document/{doc_id}/chat/clear")
    assert clear_res.status_code == 200
    assert clear_res.get_json()["success"] is True

    # Check history is now empty
    hist_after = client.get(f"/document/{doc_id}/chat/history")
    assert len(hist_after.get_json()["messages"]) == 0


def test_delete_document_cleans_vector_index(chat_test_env):
    """Verifies that deleting a document cleans up its physical vector index and chunk files."""
    app = chat_test_env["app"]
    client = chat_test_env["client"]
    db_path = chat_test_env["db_path"]
    vs = chat_test_env["vector_store"]

    pdf_path = chat_test_env["upload_folder"] / "delete_me.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy content")

    with app.app_context():
        doc_id = insert_document(
            original_filename="delete_me.pdf",
            stored_filename="delete_me_123.pdf",
            file_path=str(pdf_path),
            total_pages=1,
            total_words=20,
            db_path=db_path,
        )
        insert_document_pages(
            doc_id,
            [{"page_number": 1, "text": "Content to be indexed and deleted."}],
            db_path=db_path,
        )

        # Index document inside app context so get_vector_store() picks up VECTOR_STORE_DIR
        index_document_for_rag(doc_id)

    # Confirm the isolated VS shows the doc as indexed
    assert vs.is_document_indexed(doc_id) is True

    # Call delete route — it should remove both DB records and vector index files
    del_res = client.post(f"/document/{doc_id}/delete", headers={"Accept": "application/json"})
    assert del_res.status_code == 200

    # Verify vector index file was removed from the isolated directory
    assert vs.is_document_indexed(doc_id) is False


def test_health_check_shows_phase4_fields(chat_test_env):
    """Verifies that /health returns Phase 4 RAG details, models, and vector store ready status."""
    client = chat_test_env["client"]
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "Phase 4" in data["phase"]
    assert "rag_embedding_model" in data
    assert "rag_qa_model" in data
    assert "vector_store" in data

