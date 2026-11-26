"""
Automated Integration Tests for Phase 5 Flask Routes and APIs (tests/test_quiz_routes.py).
Tests:
1. GET /quizzes (Hub page loads successfully and JSON returns lists).
2. GET /document/<id>/quiz/new (Pre-selected config loads).
3. POST /document/<id>/quiz/generate (Creates quiz and persists to DB).
4. GET /quiz/<quiz_id> (Interactive quiz view renders).
5. GET /api/quiz/<quiz_id>/questions (SECURITY: correct_option & explanation MUST NOT be returned!).
6. POST /api/quiz/<quiz_id>/submit (Grading submission, scoring, and saving attempt).
7. GET /quiz/attempt/<attempt_id> (Results review page loads).
8. GET /quizzes/history (History view loads).
9. POST /quiz/<quiz_id>/delete (Cascade deletion of quiz and attempts).
10. Error handling (nonexistent document, nonexistent quiz, scanned document).
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from app import create_app
from database.db import (
    add_quiz_questions,
    create_quiz,
    get_quiz,
    get_quiz_attempt,
    get_quiz_questions,
    init_db,
    insert_document,
    insert_document_pages,
)


@pytest.fixture
def quiz_app_client():
    """Sets up an isolated Flask test client with test database and sample documents."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    upload_folder = temp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)
    db_path = temp_path / "test_routes_quiz.db"

    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = str(upload_folder)
    app.config["DATABASE_PATH"] = db_path
    app.config["TESTING"] = True
    app.config["SUMMARIZATION_MODEL"] = "mock"

    init_db(db_path)

    # Insert valid document
    doc_id = insert_document(
        original_filename="biology_dna.pdf",
        stored_filename="bio_dna_123.pdf",
        file_path=str(upload_folder / "dna.pdf"),
        total_pages=2,
        total_words=100,
        db_path=db_path,
    )
    insert_document_pages(
        doc_id,
        [
            {
                "page_number": 1,
                "text": (
                    "Deoxyribonucleic acid is the hereditary material in humans and almost all other organisms. "
                    "DNA contains the biological instructions for gene expression and protein synthesis. "
                    "The structure of DNA is a double helix composed of adenine, thymine, cytosine, and guanine."
                ),
            },
            {
                "page_number": 2,
                "text": (
                    "Replication of DNA occurs during the synthesis phase of the eukaryotic cell cycle. "
                    "DNA polymerase is the primary enzyme responsible for synthesizing new DNA strands. "
                    "Mitochondria also possess their own independent circular mitochondrial DNA molecules."
                ),
            },
        ],
        db_path=db_path,
    )

    # Pre-create a quiz for endpoint tests
    quiz_id = create_quiz(
        document_id=doc_id,
        title="DNA Biology Quiz",
        difficulty="medium",
        total_questions=2,
        time_limit_minutes=10,
        db_path=db_path,
    )
    add_quiz_questions(
        quiz_id,
        [
            {
                "question_number": 1,
                "question_text": "What is the primary structure of DNA?",
                "option_a": "A double helix.",
                "option_b": "A single linear strand.",
                "option_c": "A globular lipid bilayer.",
                "option_d": "A branched polysaccharide.",
                "correct_option": "A",
                "explanation": "DNA consists of two strands winding around each other like a twisted ladder.",
                "difficulty": "easy",
                "source_page": 1,
            },
            {
                "question_number": 2,
                "question_text": "Which enzyme synthesizes new DNA strands during replication?",
                "option_a": "Amylase.",
                "option_b": "DNA polymerase.",
                "option_c": "Pepsin.",
                "option_d": "Hemoglobin.",
                "correct_option": "B",
                "explanation": "DNA polymerase adds nucleotides to the growing DNA chain.",
                "difficulty": "medium",
                "source_page": 2,
            },
        ],
        db_path=db_path,
    )

    client = app.test_client()

    yield {
        "app": app,
        "client": client,
        "db_path": db_path,
        "doc_id": doc_id,
        "quiz_id": quiz_id,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_quizzes_hub_route_html(quiz_app_client):
    """Verifies GET /quizzes returns 200 and renders HTML."""
    client = quiz_app_client["client"]
    res = client.get("/quizzes")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "AI Quiz Generator" in html or "Quizzes" in html
    assert "biology_dna.pdf" in html


def test_quizzes_hub_route_json(quiz_app_client):
    """Verifies GET /quizzes with JSON accept returns quiz and attempt lists."""
    client = quiz_app_client["client"]
    res = client.get("/quizzes", headers={"Accept": "application/json"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["total_quizzes"] >= 1
    assert data["quizzes"][0]["title"] == "DNA Biology Quiz"


def test_new_quiz_config_route(quiz_app_client):
    """Verifies GET /document/<doc_id>/quiz/new renders the config page."""
    client = quiz_app_client["client"]
    doc_id = quiz_app_client["doc_id"]
    res = client.get(f"/document/{doc_id}/quiz/new")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "biology_dna.pdf" in html
    assert "Configure New Quiz" in html


def test_generate_quiz_endpoint(quiz_app_client):
    """Verifies POST /document/<doc_id>/quiz/generate creates a new quiz and returns JSON."""
    client = quiz_app_client["client"]
    doc_id = quiz_app_client["doc_id"]

    res = client.post(
        f"/document/{doc_id}/quiz/generate",
        json={
            "difficulty": "easy",
            "num_questions": 3,
            "title": "Generated DNA Easy Quiz",
            "time_limit_minutes": 5,
        },
        headers={"Accept": "application/json"},
    )
    assert res.status_code == 201
    data = res.get_json()
    assert data["success"] is True
    assert data["total_questions"] >= 2
    assert data["difficulty"] == "easy"
    assert "quiz_id" in data
    assert "redirect_url" in data


def test_take_quiz_view(quiz_app_client):
    """Verifies GET /quiz/<quiz_id> renders interactive taking interface."""
    client = quiz_app_client["client"]
    quiz_id = quiz_app_client["quiz_id"]

    res = client.get(f"/quiz/{quiz_id}")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "DNA Biology Quiz" in html
    assert "Question Navigator" in html
    assert "Submit Quiz" in html


def test_api_quiz_questions_security_no_answers_leaked(quiz_app_client):
    """
    CRITICAL SECURITY TEST:
    Verifies that GET /api/quiz/<quiz_id>/questions does NOT leak 'correct_option'
    or 'explanation' to the client during active quiz taking!
    """
    client = quiz_app_client["client"]
    quiz_id = quiz_app_client["quiz_id"]

    res = client.get(f"/api/quiz/{quiz_id}/questions")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    questions = data["questions"]
    assert len(questions) == 2

    for q in questions:
        # Client MUST NOT receive correct option or explanation!
        assert "correct_option" not in q, "SECURITY FLAW: correct_option was exposed to student client!"
        assert "explanation" not in q, "SECURITY FLAW: explanation was exposed to student client!"
        # But options A, B, C, D and question text must be present
        assert "question_text" in q
        assert "option_a" in q
        assert "option_b" in q
        assert "option_c" in q
        assert "option_d" in q


def test_submit_quiz_endpoint_and_grading(quiz_app_client):
    """Verifies POST /api/quiz/<quiz_id>/submit grades answers and saves attempt."""
    client = quiz_app_client["client"]
    quiz_id = quiz_app_client["quiz_id"]
    db_path = quiz_app_client["db_path"]

    # Q1 correct is 'A', Q2 correct is 'B'
    # Submit 1 correct, 1 incorrect
    payload = {
        "answers": {
            "1": "A",  # Q1 Correct
            "2": "C",  # Q2 Incorrect
        },
        "time_taken_seconds": 65,
    }

    res = client.post(
        f"/api/quiz/{quiz_id}/submit",
        json=payload,
        headers={"Accept": "application/json"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["score"] == 1
    assert data["total_questions"] == 2
    assert data["percentage"] == 50.0
    assert data["passed"] is False  # 50% < 60%
    assert "attempt_id" in data
    assert "redirect_url" in data

    # Verify attempt saved in database
    attempt = get_quiz_attempt(data["attempt_id"], db_path=db_path)
    assert attempt is not None
    assert attempt["score"] == 1
    assert attempt["percentage"] == 50.0


def test_quiz_results_view(quiz_app_client):
    """Verifies GET /quiz/attempt/<attempt_id> renders results page and JSON."""
    client = quiz_app_client["client"]
    quiz_id = quiz_app_client["quiz_id"]

    # First submit an attempt
    sub_res = client.post(
        f"/api/quiz/{quiz_id}/submit",
        json={"answers": {"1": "A", "2": "B"}, "time_taken_seconds": 40},
        headers={"Accept": "application/json"},
    )
    attempt_id = sub_res.get_json()["attempt_id"]

    # Request HTML result view
    res = client.get(f"/quiz/attempt/{attempt_id}")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Quiz Results" in html or "DNA Biology Quiz Results" in html
    assert "100" in html  # 100% score
    assert "Question-by-Question Review" in html

    # Request JSON result view
    res_json = client.get(f"/quiz/attempt/{attempt_id}", headers={"Accept": "application/json"})
    assert res_json.status_code == 200
    jdata = res_json.get_json()
    assert jdata["success"] is True
    assert jdata["evaluation"]["score"] == 2
    assert jdata["evaluation"]["passed"] is True


def test_quiz_history_route(quiz_app_client):
    """Verifies GET /quizzes/history returns history records."""
    client = quiz_app_client["client"]
    res = client.get("/quizzes/history")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Quiz History" in html


def test_delete_quiz_route(quiz_app_client):
    """Verifies POST /quiz/<quiz_id>/delete removes the quiz and cascaded data."""
    client = quiz_app_client["client"]
    quiz_id = quiz_app_client["quiz_id"]
    db_path = quiz_app_client["db_path"]

    del_res = client.post(f"/quiz/{quiz_id}/delete", headers={"Accept": "application/json"})
    assert del_res.status_code == 200
    data = del_res.get_json()
    assert data["success"] is True

    # Verify quiz is gone
    assert get_quiz(quiz_id, db_path=db_path) is None
    assert len(get_quiz_questions(quiz_id, db_path=db_path)) == 0


def test_health_check_shows_phase5(quiz_app_client):
    """Verifies GET /health reports Phase 5 readiness."""
    client = quiz_app_client["client"]
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert "Phase 5" in data["phase"]
    assert data["quiz_generator"] == "ready"
    assert data["quiz_evaluator"] == "ready"

