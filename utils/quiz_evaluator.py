"""
Quiz Evaluator Module for AI-Powered Smart Study Assistant (Phase 5).
Performs secure, server-side grading of student quiz submissions, score calculation,
pass/fail determination, and builds detailed question-by-question reviews.
"""

from typing import Any

PASSING_THRESHOLD_PERCENTAGE = 60.0


class QuizEvaluationError(Exception):
    """Custom exception raised for quiz evaluation errors."""


def evaluate_quiz_submission(
    quiz_questions: list[dict[str, Any]],
    submitted_answers: dict[str, Any],
    time_taken_seconds: int = 0,
    passing_threshold: float = PASSING_THRESHOLD_PERCENTAGE,
    original_filename: str = "",
) -> dict[str, Any]:
    """
    Grades student answers against the authoritative backend question key.

    Args:
        quiz_questions: Full question dicts from database (must include 'correct_option', 'explanation', etc.).
        submitted_answers: Dict mapping question identifier (str or int id / question_number) to selected option ('A', 'B', 'C', 'D').
        time_taken_seconds: Elapsed time in seconds.
        passing_threshold: Minimum percentage required to pass (default 60.0%).
        original_filename: Source document filename for citations.

    Returns:
        Structured evaluation dict containing score, percentage, counts, and question-by-question review.
    """
    if not quiz_questions:
        raise QuizEvaluationError("Cannot evaluate an empty quiz with no questions.")

    total_questions = len(quiz_questions)
    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0

    review_list: list[dict[str, Any]] = []

    # Normalize submitted answers keys to string
    # We support lookup by question ID (e.g. "q_12" or "12") or by question_number ("1", "2")
    norm_answers: dict[str, str] = {}
    for k, v in (submitted_answers or {}).items():
        if v:
            clean_val = str(v).strip().upper()
            norm_answers[str(k).strip()] = clean_val

    for q in quiz_questions:
        q_id = q.get("id")
        q_num = q.get("question_number", 1)
        correct_opt = str(q.get("correct_option", "")).strip().upper()
        explanation = q.get("explanation", "").strip()
        source_page = q.get("source_page")

        # Check submitted answer by question id or question number
        selected_opt = None
        if str(q_id) in norm_answers:
            selected_opt = norm_answers[str(q_id)]
        elif f"q_{q_id}" in norm_answers:
            selected_opt = norm_answers[f"q_{q_id}"]
        elif str(q_num) in norm_answers:
            selected_opt = norm_answers[str(q_num)]
        elif f"q_{q_num}" in norm_answers:
            selected_opt = norm_answers[f"q_{q_num}"]

        if selected_opt in ("A", "B", "C", "D"):
            if selected_opt == correct_opt:
                correct_count += 1
                is_correct = True
                is_unanswered = False
            else:
                incorrect_count += 1
                is_correct = False
                is_unanswered = False
        else:
            selected_opt = None
            unanswered_count += 1
            is_correct = False
            is_unanswered = True

        # Build detailed question review entry
        review_entry = {
            "question_id": q_id,
            "question_number": q_num,
            "question_text": q.get("question_text", ""),
            "option_a": q.get("option_a", ""),
            "option_b": q.get("option_b", ""),
            "option_c": q.get("option_c", ""),
            "option_d": q.get("option_d", ""),
            "selected_option": selected_opt,
            "correct_option": correct_opt,
            "is_correct": is_correct,
            "is_unanswered": is_unanswered,
            "explanation": explanation,
            "source_page": source_page,
            "source_document": original_filename,
            "difficulty": q.get("difficulty", "medium"),
        }
        review_list.append(review_entry)

    # Calculate metrics
    score = correct_count
    percentage = round((correct_count / total_questions) * 100.0, 2)
    passed = percentage >= passing_threshold

    # Format human-readable time taken
    seconds = max(0, int(time_taken_seconds or 0))
    minutes = seconds // 60
    rem_seconds = seconds % 60
    if minutes > 0:
        time_taken_formatted = f"{minutes}m {rem_seconds}s"
    else:
        time_taken_formatted = f"{seconds}s"

    return {
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "incorrect_answers": incorrect_count,
        "unanswered_count": unanswered_count,
        "score": score,
        "percentage": percentage,
        "passed": passed,
        "passing_threshold": passing_threshold,
        "time_taken_seconds": seconds,
        "time_taken_formatted": time_taken_formatted,
        "review": review_list,
    }

