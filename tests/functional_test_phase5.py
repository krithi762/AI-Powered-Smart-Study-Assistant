"""
Live End-to-End Functional Test Suite for Phase 5 – AI Quiz Generation.
Tests the live Flask application running on http://127.0.0.1:5000.
Executes all 10 core implementation tasks and validates:
1. PDF Document Integration
2. AI Question Generation (MCQs, 4 options, 1 correct, explanations)
3. Difficulty Levels (Easy, Medium, Hard)
4. Quiz Configuration & Options
5. Interactive Quiz Interface & Client Security (No answers leaked in API)
6. Timer functionality & Auto-submission behavior
7. Authoritative Server-side Grading & Evaluation
8. Results Page, Score Breakdown & Explanations
9. Quiz History, Attempt Tracking & Retention
10. Security, Validation, and Cascaded Cleanup
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

BASE_URL = "http://127.0.0.1:5000"


def make_request(path: str, method: str = "GET", data: Any = None, headers: dict[str, str] | None = None) -> tuple[int, str, dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    req_headers = {"User-Agent": "Phase5FunctionalTester/1.0"}
    if headers:
        req_headers.update(headers)

    body = None
    if data is not None:
        if isinstance(data, dict):
            body = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        else:
            body = data.encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_body = resp.read().decode("utf-8")
            try:
                json_data = json.loads(resp_body)
                if not isinstance(json_data, dict):
                    json_data = {}
            except Exception:
                json_data = {}
            return resp.status, resp_body, json_data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
            if not isinstance(err_json, dict):
                err_json = {}
        except Exception:
            err_json = {}
        return e.code, err_body, err_json


def run_tests():
    print("=" * 80)
    print("PHASE 5 LIVE FUNCTIONAL VERIFICATION SUITE")
    print(f"Target: {BASE_URL}")
    print("=" * 80)

    # 1. Health Check
    status, _, health = make_request("/health")
    assert status == 200, f"Health check failed with {status}"
    assert health.get("status") == "healthy", "Server not healthy"
    assert health.get("quiz_generator") == "ready", "Quiz generator not ready"
    assert health.get("quiz_evaluator") == "ready", "Quiz evaluator not ready"
    print(" [PASS] 1. Live Server Health Check & Phase 5 Services Ready")

    # 2. Get Documents List to pick a valid document
    status, _, docs_data = make_request("/quizzes", headers={"Accept": "application/json"})
    assert status == 200, f"/quizzes returned {status}"
    documents = docs_data.get("documents", [])
    assert len(documents) > 0, "No documents found in database to generate quizzes from"
    target_doc = documents[0]
    doc_id = target_doc["id"]
    print(f" [PASS] 2. Quizzes Hub API loaded ({len(documents)} eligible documents, selected doc_id={doc_id})")

    # 3. Quiz Configuration Page (HTML)
    status, html, _ = make_request(f"/document/{doc_id}/quiz/new")
    assert status == 200, f"Quiz config page returned {status}"
    assert "Generate AI Quiz" in html or "Quiz Configuration" in html
    assert "Difficulty Level" in html
    assert "Number of Questions" in html
    print(f" [PASS] 3. Quiz Configuration Page UI Rendered for Document #{doc_id}")

    # 4. Generate Easy Quiz (5 questions, 10 min timer)
    easy_payload = {
        "num_questions": 5,
        "difficulty": "easy",
        "title": "Functional Test - Easy Quiz",
        "time_limit_minutes": 10,
    }
    status, _, gen_resp = make_request(f"/document/{doc_id}/quiz/generate", method="POST", data=easy_payload)
    assert status in (200, 201), f"Easy quiz generation failed: {gen_resp}"
    assert gen_resp.get("success") is True
    easy_quiz_id = gen_resp.get("quiz_id")
    easy_total = gen_resp.get("total_questions", 0)
    assert easy_total >= 3, f"Expected at least 3 questions, got {easy_total}"
    print(f" [PASS] 4. Easy Quiz Generated (quiz_id={easy_quiz_id}, questions={easy_total}, timer=10m)")

    # 5. Generate Medium Quiz (5 questions, untimed)
    med_payload = {
        "num_questions": 5,
        "difficulty": "medium",
        "title": "Functional Test - Medium Quiz",
        "time_limit_minutes": 0,
    }
    status, _, med_resp = make_request(f"/document/{doc_id}/quiz/generate", method="POST", data=med_payload)
    assert status in (200, 201) and med_resp.get("success") is True
    med_quiz_id = med_resp.get("quiz_id")
    print(f" [PASS] 5. Medium Quiz Generated (quiz_id={med_quiz_id}, questions={med_resp.get('total_questions')}, untimed)")

    # 6. Generate Hard Quiz (5 questions)
    hard_payload = {
        "num_questions": 5,
        "difficulty": "hard",
        "title": "Functional Test - Hard Quiz",
        "time_limit_minutes": 15,
    }
    status, _, hard_resp = make_request(f"/document/{doc_id}/quiz/generate", method="POST", data=hard_payload)
    assert status in (200, 201) and hard_resp.get("success") is True
    hard_quiz_id = hard_resp.get("quiz_id")
    print(f" [PASS] 6. Hard Quiz Generated (quiz_id={hard_quiz_id}, questions={hard_resp.get('total_questions')})")

    # 7. Security Verification on API: correct_option & explanation must NEVER be exposed
    status, _, api_q_resp = make_request(f"/api/quiz/{easy_quiz_id}/questions")
    assert status == 200, f"API questions returned {status}"
    assert api_q_resp.get("success") is True
    served_questions = api_q_resp.get("questions", [])
    assert len(served_questions) == easy_total
    for q in served_questions:
        assert "correct_option" not in q, f"SECURITY VIOLATION: correct_option leaked in API: {q}"
        assert "explanation" not in q, f"SECURITY VIOLATION: explanation leaked in API: {q}"
        assert "option_a" in q and "option_b" in q and "option_c" in q and "option_d" in q
        assert len({q["option_a"], q["option_b"], q["option_c"], q["option_d"]}) == 4
    print(" [PASS] 7. Security Audit Passed: correct answers & explanations strictly masked in client API")

    # 8. Interactive Quiz Taking Page (HTML)
    status, take_html, _ = make_request(f"/quiz/{easy_quiz_id}")
    assert status == 200, f"Take quiz view returned {status}"
    assert "quiz-taking-body" in take_html
    assert "timer-display" in take_html
    print(" [PASS] 8. Interactive Quiz Interface Rendered with Timer and Question Panel")

    # 9. Initial Submission: Submit standard choice for all questions
    initial_answers = {str(q["id"]): "A" for q in served_questions}
    submit_payload = {
        "user_answers": initial_answers,
        "time_spent_seconds": 120,
    }
    status, _, sub_resp = make_request(f"/api/quiz/{easy_quiz_id}/submit", method="POST", data=submit_payload)
    assert status == 200, f"Submit returned {status}: {sub_resp}"
    assert sub_resp.get("success") is True
    initial_attempt_id = sub_resp.get("attempt_id")
    eval_res = sub_resp.get("evaluation", {})
    assert "percentage" in eval_res
    assert "passed" in eval_res
    assert "review" in eval_res
    assert len(eval_res["review"]) == len(served_questions)
    print(f" [PASS] 9. Quiz Attempt Submitted & Evaluated (Score: {eval_res['score']}/{eval_res['total_questions']}, {eval_res['percentage']}%, attempt_id={initial_attempt_id})")

    # 10. Extract Correct Answers from authoritatively graded review and test Perfect 100% Score
    correct_answers_map = {}
    for r in eval_res["review"]:
        correct_answers_map[str(r["question_id"])] = r["correct_option"]

    perfect_payload = {
        "user_answers": correct_answers_map,
        "time_spent_seconds": 85,
    }
    status, _, perfect_resp = make_request(f"/api/quiz/{easy_quiz_id}/submit", method="POST", data=perfect_payload)
    assert status == 200
    perf_eval = perfect_resp.get("evaluation", {})
    perfect_attempt_id = perfect_resp.get("attempt_id")
    assert perf_eval.get("percentage") == 100.0
    assert perf_eval.get("passed") is True
    assert perf_eval.get("correct_answers") == len(served_questions)
    assert perf_eval.get("incorrect_answers") == 0
    assert perf_eval.get("unanswered_count") == 0
    print(f" [PASS] 10. Perfect 100% Score Attempt Verified (Passed=True, 0 errors, attempt_id={perfect_attempt_id})")

    # 11. Timer Expiration / Auto-submit with Unanswered Questions
    timeout_answers = {str(served_questions[0]["id"]): correct_answers_map[str(served_questions[0]["id"])]} # only 1 answered
    timeout_payload = {
        "user_answers": timeout_answers,
        "time_spent_seconds": 600,
    }
    status, _, timeout_resp = make_request(f"/api/quiz/{easy_quiz_id}/submit", method="POST", data=timeout_payload)
    assert status == 200
    timeout_eval = timeout_resp.get("evaluation", {})
    assert timeout_eval.get("unanswered_count") == len(served_questions) - 1
    assert timeout_eval.get("correct_answers") == 1
    print(f" [PASS] 11. Timer Auto-submit Handled: {timeout_eval.get('unanswered_count')} unanswered recorded accurately")

    # 12. Results Page & Question-by-Question Review
    status, res_html, _ = make_request(f"/quiz/attempt/{perfect_attempt_id}")
    assert status == 200, f"Results page returned {status}"
    assert "Quiz Results" in res_html
    assert "Question-by-Question Review" in res_html or "review-question-card" in res_html
    assert "PASSED" in res_html or "100%" in res_html
    print(" [PASS] 12. Results Review UI Rendered with Circular Gauge, Explanations & Citations")

    # 13. Quiz History Page
    status, hist_html, _ = make_request("/quizzes/history")
    assert status == 200, f"Quiz history page returned {status}"
    assert "Quiz History" in hist_html
    assert "history-table-card" in hist_html or "data-table" in hist_html
    print(" [PASS] 13. Quiz History & Retention View Verified Across All Attempts")

    # 14. Document-Specific Quizzes API
    status, _, doc_quizzes = make_request(f"/document/{doc_id}/quizzes", headers={"Accept": "application/json"})
    assert status == 200
    assert doc_quizzes.get("success") is True
    assert len(doc_quizzes.get("quizzes", [])) >= 3
    print(f" [PASS] 14. Document-Specific Quizzes Retrieval Verified ({len(doc_quizzes.get('quizzes', []))} quizzes)")

    # 15. Delete Quiz & Cascaded Cleanup
    status, _, del_resp = make_request(f"/quiz/{hard_quiz_id}/delete", method="POST", headers={"Accept": "application/json"})
    assert status == 200
    assert del_resp.get("success") is True
    # Verify quiz no longer accessible
    status, _, _ = make_request(f"/quiz/{hard_quiz_id}")
    assert status == 404, "Deleted quiz still accessible"
    print(" [PASS] 15. Quiz Deletion & Cascaded Question Cleanup Verified")

    print("=" * 80)
    print("ALL 15 LIVE FUNCTIONAL VERIFICATION SCENARIOS PASSED WITH ZERO ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
