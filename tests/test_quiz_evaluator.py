"""
Automated Unit Tests for Phase 5 Quiz Evaluator (utils/quiz_evaluator.py).
Tests:
1. Exact score and percentage calculation.
2. Full correct submission (100% score).
3. Partial score calculation (e.g. 3 of 5 correct = 60%).
4. Unanswered questions handling.
5. All unanswered questions (0% score).
6. Pass/Fail threshold determination (default 60%).
7. Formatting of elapsed time (e.g. '1m 20s', '45s').
8. Question-by-question review builder with source citations and explanations.
9. Empty question list rejection.
"""

import pytest

from utils.quiz_evaluator import (
    QuizEvaluationError,
    evaluate_quiz_submission,
)


@pytest.fixture
def sample_quiz_questions():
    """Returns sample full question records as stored in the backend database."""
    return [
        {
            "id": 101,
            "question_number": 1,
            "question_text": "What type of rock is Granite?",
            "option_a": "Intrusive igneous rock.",
            "option_b": "Extrusive igneous rock.",
            "option_c": "Metamorphic rock.",
            "option_d": "Sedimentary rock.",
            "correct_option": "A",
            "explanation": "Granite forms deep underground as an intrusive igneous rock.",
            "difficulty": "easy",
            "source_page": 1,
        },
        {
            "id": 102,
            "question_number": 2,
            "question_text": "How does Basalt form?",
            "option_a": "From compressed marine shells.",
            "option_b": "Through rapid cooling of surface lava.",
            "option_c": "Under intense heat and tectonic pressure.",
            "option_d": "From cemented quartz grains.",
            "correct_option": "B",
            "explanation": "Basalt is formed from rapidly cooled lava at the surface.",
            "difficulty": "medium",
            "source_page": 1,
        },
        {
            "id": 103,
            "question_number": 3,
            "question_text": "What is the primary composition of Limestone?",
            "option_a": "Foliated mica sheets.",
            "option_b": "Basaltic volcanic glass.",
            "option_c": "Calcium carbonate from marine shells.",
            "option_d": "Pure elemental iron deposits.",
            "correct_option": "C",
            "explanation": "Limestone forms from the accumulation of calcium carbonate shells.",
            "difficulty": "easy",
            "source_page": 2,
        },
        {
            "id": 104,
            "question_number": 4,
            "question_text": "Which rock arises through intense heat and pressure transformation?",
            "option_a": "Sedimentary rock.",
            "option_b": "Extrusive rock.",
            "option_c": "Intrusive rock.",
            "option_d": "Metamorphic rock.",
            "correct_option": "D",
            "explanation": "Metamorphic rocks arise from the alteration of existing rocks under heat and pressure.",
            "difficulty": "medium",
            "source_page": 2,
        },
        {
            "id": 105,
            "question_number": 5,
            "question_text": "What determines the crystal texture of an igneous rock?",
            "option_a": "Rate of magma cooling.",
            "option_b": "Atmospheric humidity.",
            "option_c": "Total water depth.",
            "option_d": "Ocean wave frequency.",
            "correct_option": "A",
            "explanation": "Slow cooling creates large crystals, while fast cooling creates fine crystals.",
            "difficulty": "hard",
            "source_page": 1,
        },
    ]


def test_evaluate_quiz_perfect_score(sample_quiz_questions):
    """Verifies that answering all questions correctly gives 100% and PASSED status."""
    answers = {
        "101": "A",
        "102": "B",
        "103": "C",
        "104": "D",
        "105": "A",
    }
    result = evaluate_quiz_submission(
        quiz_questions=sample_quiz_questions,
        submitted_answers=answers,
        time_taken_seconds=95,
        original_filename="geology.pdf",
    )

    assert result["total_questions"] == 5
    assert result["correct_answers"] == 5
    assert result["incorrect_answers"] == 0
    assert result["unanswered_count"] == 0
    assert result["score"] == 5
    assert result["percentage"] == 100.0
    assert result["passed"] is True
    assert result["time_taken_formatted"] == "1m 35s"
    assert len(result["review"]) == 5
    assert all(r["is_correct"] is True for r in result["review"])


def test_evaluate_quiz_partial_score_pass(sample_quiz_questions):
    """Verifies that 4 of 5 correct yields 80% and PASSED status."""
    answers = {
        "101": "A",  # Correct
        "102": "B",  # Correct
        "103": "C",  # Correct
        "104": "D",  # Correct
        "105": "C",  # Incorrect (correct is A)
    }
    result = evaluate_quiz_submission(
        quiz_questions=sample_quiz_questions,
        submitted_answers=answers,
        time_taken_seconds=45,
    )

    assert result["score"] == 4
    assert result["correct_answers"] == 4
    assert result["incorrect_answers"] == 1
    assert result["unanswered_count"] == 0
    assert result["percentage"] == 80.0
    assert result["passed"] is True
    assert result["time_taken_formatted"] == "45s"


def test_evaluate_quiz_partial_score_fail(sample_quiz_questions):
    """Verifies that 2 of 5 correct yields 40% and FAILED status (< 60%)."""
    answers = {
        "101": "A",  # Correct
        "102": "C",  # Incorrect
        "103": "A",  # Incorrect
        "104": "D",  # Correct
        "105": "B",  # Incorrect
    }
    result = evaluate_quiz_submission(
        quiz_questions=sample_quiz_questions,
        submitted_answers=answers,
        time_taken_seconds=120,
    )

    assert result["score"] == 2
    assert result["correct_answers"] == 2
    assert result["incorrect_answers"] == 3
    assert result["percentage"] == 40.0
    assert result["passed"] is False


def test_evaluate_quiz_with_unanswered_questions(sample_quiz_questions):
    """Verifies that unanswered questions are properly flagged and counted."""
    # Only answer Q1 and Q2
    answers = {
        "101": "A",  # Correct
        "102": "C",  # Incorrect
        # 103, 104, 105 left blank
    }
    result = evaluate_quiz_submission(
        quiz_questions=sample_quiz_questions,
        submitted_answers=answers,
        time_taken_seconds=60,
    )

    assert result["correct_answers"] == 1
    assert result["incorrect_answers"] == 1
    assert result["unanswered_count"] == 3
    assert result["score"] == 1
    assert result["percentage"] == 20.0
    assert result["passed"] is False

    # Check review list
    r3 = result["review"][2]
    assert r3["question_id"] == 103
    assert r3["is_unanswered"] is True
    assert r3["is_correct"] is False
    assert r3["selected_option"] is None


def test_evaluate_quiz_all_unanswered(sample_quiz_questions):
    """Verifies submitting empty answers results in 0 score and 100% unanswered."""
    result = evaluate_quiz_submission(
        quiz_questions=sample_quiz_questions,
        submitted_answers={},
        time_taken_seconds=10,
    )

    assert result["score"] == 0
    assert result["correct_answers"] == 0
    assert result["unanswered_count"] == 5
    assert result["percentage"] == 0.0
    assert result["passed"] is False


def test_evaluate_quiz_review_contains_citations_and_explanations(sample_quiz_questions):
    """Verifies each review item includes explanation, source page, and correct answer."""
    answers = {"101": "A"}
    result = evaluate_quiz_submission(
        quiz_questions=sample_quiz_questions,
        submitted_answers=answers,
        original_filename="geology_text.pdf",
    )

    first_rev = result["review"][0]
    assert first_rev["source_page"] == 1
    assert first_rev["source_document"] == "geology_text.pdf"
    assert "intrusive igneous rock" in first_rev["explanation"].lower()
    assert first_rev["correct_option"] == "A"


def test_evaluate_quiz_empty_questions_raises_error():
    """Verifies evaluating an empty question list raises QuizEvaluationError."""
    with pytest.raises(QuizEvaluationError):
        evaluate_quiz_submission(quiz_questions=[], submitted_answers={})

