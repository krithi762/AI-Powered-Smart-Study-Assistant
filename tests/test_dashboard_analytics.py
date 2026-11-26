"""
Automated Unit and Integration Tests for Phase 6:
Dashboard, Progress Tracking, Performance Analytics & Study Goals.
Tests:
1. Authoritative dashboard statistics calculation and zero states.
2. Performance analytics (weekly Mon-Sun activity, quiz trends, difficulty distribution).
3. Study goals CRUD, dynamic progress calculation, and auto-completion.
4. Study activities logging, filtering by type/date, and total counts.
5. Dashboard, Analytics, Goals, and Activity routes (HTML and JSON APIs).
6. Operational readiness health check for Phase 6.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from app import create_app
from database.db import (
    add_quiz_questions,
    create_quiz,
    create_study_goal,
    delete_study_goal,
    get_dashboard_statistics,
    get_performance_analytics,
    get_study_activities,
    get_study_goal,
    get_total_activities_count,
    init_db,
    insert_document,
    insert_document_pages,
    log_activity,
    save_quiz_attempt,
    save_summary,
    update_study_goal,
)


@pytest.fixture
def test_p6_db():
    """Sets up an isolated temporary database for Phase 6 unit tests."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_phase6.db"
    init_db(db_path)

    # 1. Insert a document
    doc_id = insert_document(
        original_filename="ai_principles.pdf",
        stored_filename="ai_123.pdf",
        file_path=str(Path(temp_dir) / "ai.pdf"),
        total_pages=3,
        total_words=250,
        db_path=db_path,
    )
    insert_document_pages(
        doc_id,
        [
            {"page_number": 1, "text": "Artificial intelligence algorithms learn patterns from empirical training datasets."},
            {"page_number": 2, "text": "Supervised learning utilizes labeled input-output pairs to minimize regression loss."},
            {"page_number": 3, "text": "Neural networks utilize backpropagation to update internal weights via gradient descent."},
        ],
        db_path=db_path,
    )

    # 2. Insert summary
    save_summary(
        document_id=doc_id,
        summary_type="standard",
        summary_text="AI systems learn predictive representations through supervised neural network optimization.",
        db_path=db_path,
    )

    # 3. Insert quiz and attempt
    quiz_id = create_quiz(
        document_id=doc_id,
        title="AI Foundations Quiz",
        difficulty="medium",
        total_questions=3,
        db_path=db_path,
    )
    add_quiz_questions(
        quiz_id,
        [
            {
                "question_number": 1,
                "question_text": "What do supervised learning models utilize?",
                "option_a": "Labeled input-output pairs",
                "option_b": "Unlabeled noise",
                "option_c": "Random weights only",
                "option_d": "Static memory registers",
                "correct_option": "A",
                "explanation": "Supervised learning explicitly pairs inputs with target labels.",
                "difficulty": "medium",
                "source_page": 2,
            }
        ],
        db_path=db_path,
    )
    attempt_id = save_quiz_attempt(
        quiz_id=quiz_id,
        document_id=doc_id,
        total_questions=3,
        correct_answers=2,
        incorrect_answers=1,
        unanswered_count=0,
        score=2,
        percentage=66.7,
        passed=True,
        time_taken_seconds=95,
        answers={"1": "A"},
        db_path=db_path,
    )

    yield {
        "db_path": db_path,
        "doc_id": doc_id,
        "quiz_id": quiz_id,
        "attempt_id": attempt_id,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_dashboard_statistics_calculation(test_p6_db):
    """Verifies that dashboard statistics accurately compute from SQLite records."""
    db_path = test_p6_db["db_path"]
    stats = get_dashboard_statistics(db_path=db_path)

    assert stats["total_documents"] == 1
    assert stats["total_summaries"] == 1
    assert stats["total_quizzes_generated"] == 1
    assert stats["total_quizzes_completed"] == 1
    assert stats["average_quiz_score"] == 66.7
    assert stats["total_study_sessions"] >= 1
    assert stats["overall_progress"] > 0
    assert "progress_breakdown" in stats
    assert len(stats["recent_documents"]) == 1
    assert len(stats["recent_summaries"]) == 1
    assert len(stats["recent_attempts"]) == 1


def test_dashboard_statistics_empty_state():
    """Verifies that an empty database produces zeroed statistics without exceptions."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "empty.db"
    init_db(db_path)

    stats = get_dashboard_statistics(db_path=db_path)
    assert stats["total_documents"] == 0
    assert stats["total_summaries"] == 0
    assert stats["total_chat_questions"] == 0
    assert stats["total_quizzes_generated"] == 0
    assert stats["total_quizzes_completed"] == 0
    assert stats["average_quiz_score"] == 0.0
    assert stats["overall_progress"] == 0.0
    assert stats["total_study_sessions"] == 0
    assert len(stats["recent_documents"]) == 0

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_performance_analytics_structure(test_p6_db):
    """Verifies analytics computations (weekly activity, quiz timeline, difficulty breakdown)."""
    db_path = test_p6_db["db_path"]
    analytics = get_performance_analytics(db_path=db_path)

    # Weekly activity
    weekly = analytics["weekly_activity"]
    assert len(weekly["labels"]) == 7
    assert len(weekly["counts"]) == 7
    assert weekly["labels"] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Quiz performance timeline
    quiz_hist = analytics["quiz_performance_history"]
    assert len(quiz_hist) == 1
    assert quiz_hist[0]["quiz_title"] == "AI Foundations Quiz"
    assert quiz_hist[0]["percentage"] == 66.7
    assert quiz_hist[0]["passed"] is True

    # Difficulty breakdown
    diff = analytics["difficulty_breakdown"]
    assert "easy" in diff and "medium" in diff and "hard" in diff
    assert diff["medium"]["attempts"] == 1
    assert diff["medium"]["average_percentage"] == 66.7
    assert diff["medium"]["pass_rate"] == 100.0

    # Learning stats
    l_stats = analytics["learning_stats"]
    assert l_stats["total_quizzes_completed"] == 1
    assert l_stats["average_score"] == 66.7
    assert l_stats["total_correct_answers"] == 2
    assert l_stats["total_incorrect_answers"] == 1


def test_study_goals_lifecycle(test_p6_db):
    """Verifies creating, syncing, updating, and deleting study goals."""
    db_path = test_p6_db["db_path"]

    # Create goal with target 1 document (already satisfied by test_p6_db)
    goal_id = create_study_goal(
        title="Upload First PDF",
        goal_type="upload_documents",
        target_value=1.0,
        target_date="2026-12-31",
        db_path=db_path,
    )
    assert goal_id > 0

    # Retrieve goal - should automatically evaluate and mark completed
    goal = get_study_goal(goal_id, db_path=db_path)
    assert goal is not None
    assert goal["title"] == "Upload First PDF"
    assert goal["goal_type"] == "upload_documents"
    assert goal["current_value"] >= 1.0
    assert goal["status"] == "completed"
    assert goal["progress_percentage"] == 100.0

    # Create in-progress goal
    g2_id = create_study_goal(
        title="Complete 10 Quizzes",
        goal_type="complete_quizzes",
        target_value=10.0,
        target_date="2026-12-31",
        db_path=db_path,
    )
    g2 = get_study_goal(g2_id, db_path=db_path)
    assert g2 is not None
    assert g2["status"] == "in_progress"
    assert g2["current_value"] == 1.0
    assert g2["progress_percentage"] == 10.0

    # Update goal
    updated = update_study_goal(g2_id, target_value=5.0, db_path=db_path)
    assert updated is True
    g2_updated = get_study_goal(g2_id, db_path=db_path)
    assert g2_updated is not None
    assert g2_updated["target_value"] == 5.0

    # Delete goal
    deleted = delete_study_goal(g2_id, db_path=db_path)
    assert deleted is True
    assert get_study_goal(g2_id, db_path=db_path) is None


def test_study_activities_logging_and_filtering(test_p6_db):
    """Verifies activity logging, pagination, and filter queries."""
    db_path = test_p6_db["db_path"]
    doc_id = test_p6_db["doc_id"]

    # Log activities
    act_id1 = log_activity("upload", "Uploaded Document", "Page extraction complete", doc_id, db_path=db_path)
    act_id2 = log_activity("summarize", "Generated Summary", "Quick summary generated", doc_id, db_path=db_path)
    act_id3 = log_activity("chat", "Chat Question", "What is neural network?", doc_id, db_path=db_path)

    assert act_id1 > 0 and act_id2 > 0 and act_id3 > 0

    # Filter by activity type
    uploads = get_study_activities(activity_type="upload", db_path=db_path)
    assert any(a["id"] == act_id1 for a in uploads)
    assert all(a["activity_type"] == "upload" for a in uploads)

    # Total count
    c_all = get_total_activities_count(db_path=db_path)
    assert c_all >= 3


# ------------------------------------------------------------------------------
# Flask Integration Route Tests
# ------------------------------------------------------------------------------

@pytest.fixture
def app_client(test_p6_db):
    """Flask test client connected to the isolated Phase 6 database."""
    app = create_app("testing")
    app.config["DATABASE_PATH"] = str(test_p6_db["db_path"])
    with app.test_client() as client:
        yield client


def test_dashboard_route_html_and_json(app_client):
    """Verifies that GET / and /dashboard load properly in HTML and JSON modes."""
    # 1. HTML Dashboard View
    resp_html = app_client.get("/")
    assert resp_html.status_code == 200
    assert b"SmartStudy" in resp_html.data
    assert b"AI-Powered Smart Study Assistant" in resp_html.data
    assert b"Overview Statistics" in resp_html.data
    assert b"Weekly Study Activity" in resp_html.data

    # 2. JSON Dashboard API
    resp_json = app_client.get("/dashboard", headers={"Accept": "application/json"})
    assert resp_json.status_code == 200
    data = resp_json.get_json()
    assert data["success"] is True
    assert "statistics" in data
    assert data["statistics"]["total_documents"] >= 1

    # 3. Direct Stats API
    resp_stats = app_client.get("/api/dashboard/stats")
    assert resp_stats.status_code == 200
    assert resp_stats.get_json()["success"] is True


def test_analytics_route_html_and_json(app_client):
    """Verifies that /analytics and /api/analytics/data return calculated metrics."""
    # 1. HTML View
    resp = app_client.get("/analytics")
    assert resp.status_code == 200
    assert b"Performance Analytics" in resp.data
    assert b"Overall Learning Progress" in resp.data
    assert b"Quiz Difficulty Analysis" in resp.data

    # 2. JSON API
    resp_json = app_client.get("/api/analytics/data")
    assert resp_json.status_code == 200
    data = resp_json.get_json()
    assert data["success"] is True
    assert "weekly_activity" in data
    assert "difficulty_breakdown" in data
    assert "learning_stats" in data


def test_study_goals_routes_api(app_client):
    """Verifies goals page and API endpoints (create, update, delete)."""
    # 1. HTML View
    resp = app_client.get("/goals")
    assert resp.status_code == 200
    assert b"Study Goals" in resp.data
    assert b"Create a Study Goal" in resp.data

    # 2. API Create Goal
    create_payload = {
        "title": "Master 5 Quizzes",
        "goal_type": "complete_quizzes",
        "target_value": 5,
        "target_date": "2026-12-31",
    }
    resp_create = app_client.post("/api/goals/create", json=create_payload)
    assert resp_create.status_code == 201
    g_data = resp_create.get_json()
    assert g_data["success"] is True
    goal_id = g_data["goal_id"]

    # 3. API Get Goals
    resp_get = app_client.get("/api/goals")
    assert resp_get.status_code == 200
    goals = resp_get.get_json()["goals"]
    assert any(g["id"] == goal_id for g in goals)

    # 4. API Update Goal
    resp_up = app_client.post(f"/api/goals/{goal_id}/update", json={"status": "completed"})
    assert resp_up.status_code == 200

    # 5. API Delete Goal
    resp_del = app_client.post(f"/api/goals/{goal_id}/delete")
    assert resp_del.status_code == 200


def test_activity_history_routes(app_client):
    """Verifies /activity page and /api/activity/history API with filters."""
    # 1. HTML View
    resp = app_client.get("/activity")
    assert resp.status_code == 200
    assert b"Activity History" in resp.data
    assert b"Activity Records" in resp.data

    # 2. JSON API
    resp_json = app_client.get("/api/activity/history?type=upload")
    assert resp_json.status_code == 200
    data = resp_json.get_json()
    assert data["success"] is True
    assert "activities" in data
    assert "total_count" in data


def test_health_check_shows_phase6(app_client):
    """Verifies that /health confirms Phase 6 active status and subsystems."""
    resp = app_client.get("/health")
    assert resp.status_code == 200
    health = resp.get_json()
    assert "Phase 6" in health["phase"]
    assert health["dashboard_analytics"] == "ready"
    assert health["study_goals"] == "ready"
    assert health["activity_tracker"] == "ready"

