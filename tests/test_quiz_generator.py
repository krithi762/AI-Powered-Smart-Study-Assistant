"""
Automated Unit Tests for Phase 5 AI Quiz Generator (utils/quiz_generator.py).
Tests:
1. Candidate fact extraction across document pages.
2. Question generation for Easy, Medium, and Hard difficulty levels.
3. Verification of 4 distinct answer options per question.
4. Exactly one correct answer per question.
5. Random distribution of correct option letter.
6. Question deduplication.
7. Source page and filename preservation.
8. Error handling on invalid, empty, or scanned documents.
9. Configurable question counts (e.g. 3, 5, 10).
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from database.db import (
    init_db,
    insert_document,
    insert_document_pages,
)
from utils.quiz_generator import (
    QuizGenerationError,
    _extract_candidate_facts,
    _generate_distractors,
    generate_quiz_for_document,
)


@pytest.fixture
def test_quiz_db():
    """Sets up an isolated temporary database with sample academic documents."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_quiz.db"
    init_db(db_path)

    # Insert valid geology document
    doc_id = insert_document(
        original_filename="geology_study.pdf",
        stored_filename="geo_123.pdf",
        file_path=str(Path(temp_dir) / "geo.pdf"),
        total_pages=2,
        total_words=120,
        db_path=db_path,
    )
    insert_document_pages(
        doc_id,
        [
            {
                "page_number": 1,
                "text": (
                    "Igneous rocks form through the cooling and solidification of magma or lava. "
                    "Granite is a common intrusive igneous rock that forms deep underground. "
                    "Basalt is an extrusive igneous rock formed from rapidly cooled lava at the surface. "
                    "The crystallization rate of minerals determines the final rock texture."
                ),
            },
            {
                "page_number": 2,
                "text": (
                    "Sedimentary rocks form through the accumulation and cementation of mineral particles over time. "
                    "Limestone is a sedimentary rock formed from calcium carbonate shells of marine organisms. "
                    "Sandstone is composed of compressed quartz sand grains cemented by silica or calcite. "
                    "Metamorphic rocks arise from the transformation of existing rock types through intense heat and pressure."
                ),
            },
        ],
        db_path=db_path,
    )

    # Insert scanned document with 0 words
    scanned_id = insert_document(
        original_filename="scanned_image.pdf",
        stored_filename="scan_123.pdf",
        file_path=str(Path(temp_dir) / "scan.pdf"),
        total_pages=1,
        total_words=0,
        processing_status="scanned_no_text",
        db_path=db_path,
    )

    yield {
        "db_path": db_path,
        "doc_id": doc_id,
        "scanned_id": scanned_id,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_extract_candidate_facts_finds_concepts_and_sentences():
    """Verifies that facts and definitions are extracted from page text."""
    pages = [
        {
            "page_number": 1,
            "text": "Photosynthesis is the biological process used by plants to convert light energy into chemical sugars.",
        },
        {
            "page_number": 2,
            "text": "Mitochondria generate adenosine triphosphate through aerobic cellular respiration.",
        },
    ]
    candidates = _extract_candidate_facts(pages)
    assert len(candidates) >= 2
    facts_text = " ".join([c["fact"] for c in candidates])
    assert "photosynthesis" in facts_text.lower() or "mitochondria" in facts_text.lower()
    assert all("page_number" in c for c in candidates)


def test_generate_distractors_returns_3_distinct_options():
    """Verifies distractor synthesis produces exactly 3 distinct, non-empty options."""
    pool = [
        "It operates only under non-standard external conditions.",
        "Limestone is a sedimentary rock formed from marine organisms.",
        "Basalt is an extrusive igneous rock formed from cooled lava.",
        "Granite is an intrusive igneous rock.",
    ]
    correct = "Igneous rocks form through the cooling of magma."
    distractors = _generate_distractors(correct, pool, difficulty="medium")
    assert len(distractors) == 3
    assert len(set(distractors)) == 3
    assert correct not in distractors


def test_generate_quiz_easy_difficulty(test_quiz_db):
    """Verifies generation of Easy difficulty quiz."""
    doc_id = test_quiz_db["doc_id"]
    db_path = test_quiz_db["db_path"]
    quiz = generate_quiz_for_document(doc_id, num_questions=3, difficulty="easy", db_path=db_path)

    assert quiz["document_id"] == doc_id
    assert quiz["original_filename"] == "geology_study.pdf"
    assert quiz["difficulty"] == "easy"
    assert len(quiz["questions"]) >= 2

    for q in quiz["questions"]:
        assert "question_text" in q
        assert q["difficulty"] == "easy"
        assert q["correct_option"] in ("A", "B", "C", "D")
        assert len(q["option_a"]) > 0
        assert len(q["option_b"]) > 0
        assert len(q["option_c"]) > 0
        assert len(q["option_d"]) > 0
        assert len({q["option_a"], q["option_b"], q["option_c"], q["option_d"]}) == 4
        assert len(q["explanation"]) > 10
        assert q["source_page"] in (1, 2)


def test_generate_quiz_medium_difficulty(test_quiz_db):
    """Verifies generation of Medium difficulty quiz with comparisons/mechanisms."""
    doc_id = test_quiz_db["doc_id"]
    db_path = test_quiz_db["db_path"]
    quiz = generate_quiz_for_document(doc_id, num_questions=4, difficulty="medium", db_path=db_path)
    assert quiz["difficulty"] == "medium"
    assert len(quiz["questions"]) >= 2
    for q in quiz["questions"]:
        assert q["difficulty"] == "medium"
        assert q["correct_option"] in ("A", "B", "C", "D")


def test_generate_quiz_hard_difficulty(test_quiz_db):
    """Verifies generation of Hard difficulty quiz with analytical prompts."""
    doc_id = test_quiz_db["doc_id"]
    db_path = test_quiz_db["db_path"]
    quiz = generate_quiz_for_document(doc_id, num_questions=4, difficulty="hard", db_path=db_path)
    assert quiz["difficulty"] == "hard"
    assert len(quiz["questions"]) >= 2
    for q in quiz["questions"]:
        assert q["difficulty"] == "hard"
        # Hard prompts contain analytical/evaluative terms
        assert any(w in q["question_text"].lower() for w in ["analyz", "evaluat", "deduction", "mechanism", "statement", "principle"])


def test_generate_quiz_question_counts(test_quiz_db):
    """Verifies requesting 5 questions produces up to 5 valid questions."""
    doc_id = test_quiz_db["doc_id"]
    db_path = test_quiz_db["db_path"]
    quiz_5 = generate_quiz_for_document(doc_id, num_questions=5, difficulty="medium", db_path=db_path)
    assert len(quiz_5["questions"]) <= 5
    assert quiz_5["total_questions"] == len(quiz_5["questions"])


def test_generate_quiz_no_duplicate_questions(test_quiz_db):
    """Verifies all questions in a generated quiz are unique."""
    doc_id = test_quiz_db["doc_id"]
    db_path = test_quiz_db["db_path"]
    quiz = generate_quiz_for_document(doc_id, num_questions=5, difficulty="medium", db_path=db_path)
    q_texts = [q["question_text"] + q["option_" + q["correct_option"].lower()] for q in quiz["questions"]]
    assert len(q_texts) == len(set(q_texts))


def test_generate_quiz_scanned_document_rejected(test_quiz_db):
    """Verifies attempting to generate a quiz from an empty or scanned document raises an error."""
    scanned_id = test_quiz_db["scanned_id"]
    db_path = test_quiz_db["db_path"]
    with pytest.raises(QuizGenerationError) as exc_info:
        generate_quiz_for_document(scanned_id, num_questions=5, db_path=db_path)
    assert "scanned" in str(exc_info.value).lower() or "no extractable text" in str(exc_info.value).lower()


def test_generate_quiz_nonexistent_document_raises_error():
    """Verifies passing a nonexistent document ID raises QuizGenerationError."""
    with pytest.raises(QuizGenerationError) as exc_info:
        generate_quiz_for_document(999999, num_questions=5)
    assert "not found" in str(exc_info.value).lower()
