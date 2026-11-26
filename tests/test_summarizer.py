"""
Automated test suite for Phase 3 – AI Summarization.
Tests text preprocessing, token-safe chunking, multi-mode summarization (Quick, Standard, Detailed),
concept and definition extraction, TF-IDF keyword extraction, exam points extraction,
database persistence, and Flask summarization endpoints.
"""

import shutil
import tempfile
from pathlib import Path

import pymupdf
import pytest

from app import create_app
from database.db import (
    delete_document,
    delete_summaries_for_document,
    get_all_summaries_for_document,
    get_summary,
    init_db,
    insert_document,
    insert_document_pages,
    save_summary,
)
from utils.summarizer import (
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


@pytest.fixture
def test_env():
    """Sets up an isolated temporary workspace and database for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    upload_folder = temp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)
    test_db_path = temp_path / "test_summarizer.db"

    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = str(upload_folder)
    app.config["DATABASE_PATH"] = test_db_path
    app.config["TESTING"] = True
    # Default to mock model in automated tests for instant execution
    app.config["SUMMARIZATION_MODEL"] = "mock"

    init_db(test_db_path)
    client = app.test_client()

    yield {
        "app": app,
        "client": client,
        "temp_path": temp_path,
        "upload_folder": upload_folder,
        "db_path": test_db_path,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def create_study_pdf(pages_text: list) -> bytes:
    """Helper to generate in-memory PDF with specified page text."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ------------------------------------------------------------------------------
# 1. Text Preprocessing & Cleaning Unit Tests (Task 1 & Task 2)
# ------------------------------------------------------------------------------

def test_clean_text_removes_hyphenated_breaks_and_page_markers():
    """Verifies that line-break hyphens and artificial page markers are cleaned."""
    dirty_text = (
        "--- Page 1 ---\n"
        "Trans-\nformer architecture is widely used in deep learn-\ning.\n\n\n\n"
        "Page 1 of 10\n"
        "It   enables   efficient   parallel   processing."
    )
    cleaned = clean_text(dirty_text)
    assert "Transformer" in cleaned
    assert "learning" in cleaned
    assert "--- Page 1 ---" not in cleaned
    assert "Page 1 of 10" not in cleaned
    assert "   " not in cleaned


def test_split_into_sentences_preserves_abbreviations():
    """Verifies sentence splitting handles common abbreviations without fragmentation."""
    text = (
        "Dr. Smith explained the theorem (e.g. Newton's laws) in Fig. 4. "
        "The reaction proceeded at approx. 3.14 mol/s! "
        "Is this consistent with the hypothesis?"
    )
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert "Dr. Smith" in sentences[0]
    assert "e.g. Newton's laws" in sentences[0]
    assert "3.14 mol/s!" in sentences[1]
    assert sentences[2].endswith("?")


def test_chunk_text_preserves_sentence_boundaries():
    """Verifies chunk_text does not truncate in the middle of sentences."""
    long_text = (
        "Photosynthesis is essential for plant life on Earth. "
        "It converts solar radiation into stored chemical energy. "
        "The light reactions produce ATP and NADPH in the thylakoid membranes. "
        "The Calvin cycle subsequently fixes carbon dioxide into carbohydrates. "
        "This biological pathway maintains global atmospheric oxygen balance."
    )
    # Target small chunks to force splitting
    chunks = chunk_text(long_text, max_chunk_words=18, overlap_words=4)
    assert len(chunks) >= 2
    for chunk in chunks:
        # Every chunk must end with valid sentence punctuation
        assert chunk[-1] in (".", "!", "?")


# ------------------------------------------------------------------------------
# 2. Concept, Keyword, and Exam Points Extraction Tests (Task 4)
# ------------------------------------------------------------------------------

def test_extract_key_concepts():
    """Verifies extraction of formal definitions and concepts."""
    text = (
        "Photosynthesis is defined as a biological process converting light into chemical energy. "
        "Mitochondria refers to the powerhouse organelle of eukaryotic cells. "
        "Cellular respiration is a metabolic pathway that breaks down glucose."
    )
    concepts = extract_key_concepts(text)
    assert len(concepts) >= 2
    terms = [c["term"].lower() for c in concepts]
    assert "photosynthesis" in terms or "cellular respiration" in terms


def test_extract_keywords_tfidf():
    """Verifies TF-IDF keyword extraction returns domain-specific terms."""
    study_text = (
        "Machine learning models require training data and loss functions. "
        "Supervised learning algorithms optimize weight parameters using gradient descent. "
        "Neural networks utilize backpropagation to update layers."
    )
    keywords = extract_keywords(study_text, top_n=10)
    assert len(keywords) > 0
    kw_lower = [k.lower() for k in keywords]
    assert any("learning" in k or "models" in k or "neural" in k or "gradient" in k for k in kw_lower)


def test_extract_exam_points():
    """Verifies high-yield exam preparation sentence extraction."""
    text = (
        "Welcome to chapter one. Today we begin our study. "
        "It is crucial to remember that energy cannot be created or destroyed. "
        "A fundamental principle in thermodynamics is the conservation of energy. "
        "Let us consider another simple introductory comment."
    )
    points = extract_exam_points(text, max_points=3)
    assert len(points) >= 1
    assert any("crucial to remember" in p.lower() or "fundamental principle" in p.lower() for p in points)


def test_extract_topics_and_subtopics():
    """Verifies extraction of topic outlines from structured headings."""
    text = (
        "Chapter 1: Principles of Genetics\n"
        "- Mendelian inheritance laws\n"
        "- Punnett square applications\n\n"
        "Chapter 2: Molecular Biology\n"
        "- DNA replication mechanisms\n"
        "- RNA transcription processes"
    )
    topics = extract_topics_and_subtopics(text)
    assert len(topics) >= 2
    assert "Genetics" in topics[0]["topic"]
    assert len(topics[0]["subtopics"]) >= 1


# ------------------------------------------------------------------------------
# 3. AI Summarization Modes (Task 3)
# ------------------------------------------------------------------------------

def test_quick_summary_generates_sentences():
    """Verifies Quick Summary produces 5 to 7 key sentences."""
    study_text = (
        "Artificial intelligence is transforming higher education and academic study methods. "
        "Machine learning algorithms automate repetitive analysis and summarize extensive textbooks. "
        "Natural language processing enables computers to extract core semantic concepts. "
        "Deep learning architectures process unstructured educational documents efficiently. "
        "Students benefit from targeted review notes that emphasize exam preparation. "
        "Automated study tools reduce reading fatigue while improving conceptual retention. "
        "Future learning platforms will integrate conversational tutoring with study aids."
    )
    engine = SummarizerEngine.get_instance(model_name="mock")
    summary = generate_quick_summary(study_text, engine=engine)
    sentences = split_into_sentences(summary)
    assert 3 <= len(sentences) <= 7


def test_standard_and_detailed_summaries():
    """Verifies Standard Summary and Detailed Summary with chapter breakdown."""
    pages = [
        {"page_number": 1, "text": "Page one introduces cell structure, organelles, and plasma membranes."},
        {"page_number": 2, "text": "Page two explains metabolic pathways, ATP synthesis, and cellular respiration."},
    ]
    engine = SummarizerEngine.get_instance(model_name="mock")

    # Standard summary
    std_summary = generate_standard_summary(" ".join([p["text"] for p in pages]), engine=engine)
    assert len(std_summary) > 0

    # Detailed summary
    det_summary, chapter_sums = generate_detailed_summary(" ".join([p["text"] for p in pages]), pages=pages, engine=engine)
    assert "Executive Overview" in det_summary
    assert len(chapter_sums) == 2
    assert chapter_sums[0]["section"] == "Page 1"
    assert chapter_sums[1]["section"] == "Page 2"


def test_generate_study_summary_empty_text():
    """Verifies graceful handling of documents with zero extractable text."""
    res = generate_study_summary("", summary_type="standard")
    assert res["success"] is False
    assert "no extractable text" in res["error"].lower()


# ------------------------------------------------------------------------------
# 4. Database Summaries CRUD Tests
# ------------------------------------------------------------------------------

def test_database_summary_persistence(test_env):
    """Verifies saving, retrieving, updating, and deleting summaries in SQLite."""
    db_path = test_env["db_path"]

    # 1. Insert parent document
    doc_id = insert_document(
        original_filename="biology_ch1.pdf",
        stored_filename="hash_biology_ch1.pdf",
        file_path=str(test_env["upload_folder"] / "bio.pdf"),
        total_pages=5,
        total_words=1200,
        db_path=db_path,
    )

    # 2. Save Quick Summary
    sum_id = save_summary(
        document_id=doc_id,
        summary_type="quick",
        summary_text="Photosynthesis produces glucose and oxygen from carbon dioxide and water.",
        key_concepts=[{"term": "Photosynthesis", "definition": "Process of light conversion."}],
        keywords=["Photosynthesis", "Chlorophyll"],
        exam_points=["Light reaction occurs in thylakoid."],
        topics=[{"topic": "Plant Biology", "subtopics": ["Light reactions"]}],
        model_name="mock",
        db_path=db_path,
    )
    assert sum_id is not None

    # 3. Retrieve summary
    fetched = get_summary(doc_id, summary_type="quick", db_path=db_path)
    assert fetched is not None
    assert fetched["summary_type"] == "quick"
    assert "Photosynthesis" in fetched["summary_text"]
    assert isinstance(fetched["key_concepts"], list)
    assert fetched["key_concepts"][0]["term"] == "Photosynthesis"

    # 4. Save Standard Summary (second summary for same document)
    save_summary(
        document_id=doc_id,
        summary_type="standard",
        summary_text="A more comprehensive overview of cellular biology and plant reactions.",
        model_name="mock",
        db_path=db_path,
    )

    all_sums = get_all_summaries_for_document(doc_id, db_path=db_path)
    assert len(all_sums) == 2

    # 5. Delete single summary type
    deleted = delete_summaries_for_document(doc_id, summary_type="quick", db_path=db_path)
    assert deleted == 1
    assert get_summary(doc_id, summary_type="quick", db_path=db_path) is None
    assert get_summary(doc_id, summary_type="standard", db_path=db_path) is not None

    # 6. Cascade delete on document removal
    delete_document(doc_id, db_path=db_path)
    assert len(get_all_summaries_for_document(doc_id, db_path=db_path)) == 0


# ------------------------------------------------------------------------------
# 5. Flask Endpoints Integration Tests (Task 5)
# ------------------------------------------------------------------------------

def test_summarize_endpoint_success(test_env):
    """Verifies POST /document/<id>/summarize returns JSON summary and persists to DB."""
    client = test_env["client"]
    db_path = test_env["db_path"]

    # Insert document and pages
    doc_id = insert_document(
        original_filename="study_notes.pdf",
        stored_filename="hash_study_notes.pdf",
        file_path=str(test_env["upload_folder"] / "notes.pdf"),
        total_pages=2,
        total_words=300,
        db_path=db_path,
    )
    insert_document_pages(
        document_id=doc_id,
        pages_data=[
            {
                "page_number": 1,
                "text": "Cellular respiration is a metabolic pathway that breaks down glucose to produce ATP.",
            },
            {
                "page_number": 2,
                "text": "Glycolysis occurs in the cytoplasm and is an essential preparatory step for the Krebs cycle.",
            },
        ],
        db_path=db_path,
    )

    response = client.post(
        f"/document/{doc_id}/summarize",
        json={"summary_type": "standard"},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["summary_type"] == "standard"
    assert len(data["summary_text"]) > 0
    assert "stats" in data
    assert data["stats"]["original_words"] > 0

    # Verify summary is retrievable via GET /document/<id>/summary
    get_res = client.get(f"/document/{doc_id}/summary")
    assert get_res.status_code == 200
    get_data = get_res.get_json()
    assert get_data["success"] is True
    assert get_data["summary"]["summary_type"] == "standard"


def test_summarize_endpoint_scanned_pdf_rejected(test_env):
    """Verifies that attempting to summarize a scanned PDF returns 400 with helpful notice."""
    client = test_env["client"]
    db_path = test_env["db_path"]

    doc_id = insert_document(
        original_filename="scanned_diagram.pdf",
        stored_filename="hash_scanned.pdf",
        file_path=str(test_env["upload_folder"] / "scanned.pdf"),
        total_pages=1,
        total_words=0,
        processing_status="scanned_no_text",
        db_path=db_path,
    )

    response = client.post(
        f"/document/{doc_id}/summarize",
        json={"summary_type": "quick"},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "scanned" in data["error"].lower() or "no extractable text" in data["error"].lower()


def test_summarize_endpoint_nonexistent_document(test_env):
    """Verifies that requesting summary for non-existent document ID returns 404."""
    client = test_env["client"]
    response = client.post(
        "/document/9999/summarize",
        json={"summary_type": "quick"},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 404


def test_delete_summary_endpoint(test_env):
    """Verifies POST /document/<id>/summary/delete removes stored summary."""
    client = test_env["client"]
    db_path = test_env["db_path"]

    doc_id = insert_document(
        original_filename="test.pdf",
        stored_filename="hash_test.pdf",
        file_path=str(test_env["upload_folder"] / "test.pdf"),
        total_pages=1,
        total_words=50,
        db_path=db_path,
    )
    save_summary(
        document_id=doc_id,
        summary_type="standard",
        summary_text="Sample summary text.",
        db_path=db_path,
    )

    del_res = client.post(
        f"/document/{doc_id}/summary/delete",
        headers={"Accept": "application/json"},
    )
    assert del_res.status_code == 200
    assert del_res.get_json()["success"] is True

    # Subsequent GET returns 404
    get_res = client.get(f"/document/{doc_id}/summary")
    assert get_res.status_code == 404


def test_health_check_shows_phase3(test_env):
    """Verifies /health endpoint returns Phase 3 status and AI model readiness."""
    client = test_env["client"]
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "Phase 3" in data["phase"]
    assert "summarization_model" in data
    assert "model_loaded" in data


# ------------------------------------------------------------------------------
# 6. Real Model Smoke Test (Cached sshleifer/distilbart-cnn-12-6)
# ------------------------------------------------------------------------------

def test_real_distilbart_model_summarization():
    """
    Executes actual inference with the cached local HuggingFace model
    on a realistic biology textbook paragraph.
    """
    real_text = (
        "Cellular respiration is a set of metabolic reactions and processes that take place "
        "in the cells of organisms to convert biochemical energy from nutrients into adenosine "
        "triphosphate (ATP), and then release waste products. The reactions involved in respiration "
        "are catabolic reactions, which break large molecules into smaller ones, releasing energy. "
        "Nutrients that are commonly used by animal and plant cells in respiration include carbohydrates, "
        "amino acids and fatty acids, and the most common oxidizing agent is molecular oxygen."
    )
    # Use real cached model
    engine = SummarizerEngine.get_instance(model_name="sshleifer/distilbart-cnn-12-6")
    summary = engine.generate_summary_for_chunk(
        real_text,
        max_length=60,
        min_length=20,
        num_beams=2,
    )
    assert len(summary) > 15
    assert "energy" in summary.lower() or "respiration" in summary.lower() or "atp" in summary.lower()
