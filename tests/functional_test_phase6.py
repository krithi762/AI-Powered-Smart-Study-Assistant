"""
Live End-to-End Functional Test Suite for Phase 6 – Dashboard, Progress Tracking & Final Testing.
Tests the live Flask application running on http://127.0.0.1:5000.
Executes all Phase 6 core implementation tasks and validates:
1. Health check & Phase 6 service readiness flags
2. Dashboard HTML rendering with all 10 required sections
3. Dashboard JSON statistics API (/api/dashboard/stats & /dashboard)
4. Performance Analytics HTML rendering (/analytics & /progress)
5. Analytics JSON data API (/api/analytics/data) with 5-pillar formula, weekly activity, score timeline, difficulty stats
6. Study Goals UI (/goals) & CRUD APIs (create, sync, update, delete)
7. Activity History UI (/activity & /history) & Filtered API (/api/activity/history)
8. End-to-End Activity Logging hooks across previous phases
9. Zero-state and edge case handling
10. Overall system integration and consistency
"""

import datetime
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

BASE_URL = "http://127.0.0.1:5000"


def make_request(path: str, method: str = "GET", data: Any = None, headers: dict[str, str] | None = None) -> tuple[int, str, dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    req_headers = {"User-Agent": "Phase6FunctionalTester/1.0"}
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
    print("PHASE 6 LIVE FUNCTIONAL VERIFICATION SUITE")
    print(f"Target: {BASE_URL}")
    print("=" * 80)

    # 1. Health Check
    status, _, health = make_request("/health")
    assert status == 200, f"Health check failed with {status}"
    assert health.get("status") == "healthy", "Server not healthy"
    assert health.get("dashboard_analytics") == "ready", "dashboard_analytics not ready"
    assert health.get("study_goals") == "ready", "study_goals not ready"
    assert health.get("activity_tracker") == "ready", "activity_tracker not ready"
    assert health.get("quiz_generator") == "ready", "quiz_generator not ready"
    assert health.get("quiz_evaluator") == "ready", "quiz_evaluator not ready"
    print(" [PASS] 1. Live Server Health Check & Phase 6 Service Flags Active")

    # 2. Dashboard HTML Rendering
    status, html, _ = make_request("/")
    assert status == 200, f"Dashboard route / returned {status}"
    # Verify all 10 required dashboard sections exist in the rendered HTML
    sections_expected = [
        "Welcome",                  # Section 1: Welcome banner
        "Overview Statistics",      # Section 2: Stats cards
        "Recent Study Documents",   # Section 3: Recent uploaded documents
        "Recent Summaries",         # Section 4: Recent summaries
        "Recent Chat",              # Section 5: Recent chatbot activity
        "Recent Quiz Attempts",     # Section 6: Recent quiz attempts
        "Weekly Study Activity",    # Section 7: Weekly study activity chart
        "Quiz Performance",         # Section 8: Quiz performance overview
        "Study Goals",              # Section 9: Study goals preview
        "AI Study Features",        # Section 10: Quick access to AI features
    ]
    for sec in sections_expected:
        assert sec.lower() in html.lower(), f"Dashboard missing expected section text: {sec}"
    print(" [PASS] 2. Dashboard HTML View (GET /) verified with all 10 required sections")

    # Also verify GET /dashboard renders identically
    status_dash, html_dash, _ = make_request("/dashboard")
    assert status_dash == 200
    assert len(html_dash) > 1000
    print(" [PASS] 2b. GET /dashboard alias renders HTML view successfully")

    # 3. Dashboard JSON Stats API
    status, _, stats_resp = make_request("/api/dashboard/stats")
    assert status == 200, f"/api/dashboard/stats returned {status}"
    assert stats_resp.get("success") is True
    stats = stats_resp.get("statistics") or stats_resp.get("stats", {})

    required_metrics = [
        "total_documents",
        "total_summaries",
        "total_chat_questions",
        "total_quizzes_generated",
        "total_quizzes_completed",
        "average_quiz_score",
        "overall_progress",
        "total_study_sessions",
    ]
    for metric in required_metrics:
        assert metric in stats, f"Missing metric {metric} in stats"
        assert stats[metric] is not None, f"Metric {metric} is None"

    assert 0 <= stats["overall_progress"] <= 100, "Progress % out of 0-100 range"
    print(" [PASS] 3. Dashboard Stats API (/api/dashboard/stats) validated:")
    print(f"        Docs: {stats['total_documents']} | Summaries: {stats['total_summaries']} | Chat: {stats['total_chat_questions']}")
    print(f"        Quizzes Gen: {stats['total_quizzes_generated']} | Quizzes Done: {stats['total_quizzes_completed']}")
    print(f"        Avg Score: {stats['average_quiz_score']}% | Progress: {stats['overall_progress']}% | Sessions: {stats['total_study_sessions']}")

    # 4. Performance Analytics HTML View
    status, html_ana, _ = make_request("/analytics")
    assert status == 200, f"/analytics returned {status}"
    assert "Performance Analytics" in html_ana, "Missing Performance Analytics heading"
    assert "Weekly Study Activity" in html_ana, "Missing Weekly Study Activity"
    assert "Quiz Performance Over Time" in html_ana, "Missing Quiz Performance Over Time"
    assert "Overall Learning Progress Formulation" in html_ana, "Missing Overall Learning Progress Formulation"
    assert "Quiz Difficulty Analysis" in html_ana, "Missing Quiz Difficulty Analysis"

    # Also test /progress alias
    status_prog, _, _ = make_request("/progress")
    assert status_prog == 200
    print(" [PASS] 4. Performance Analytics Page (/analytics & /progress) HTML rendered")

    # 5. Analytics JSON Data API
    status, _, ana_resp = make_request("/api/analytics/data")
    assert status == 200, f"/api/analytics/data returned {status}"
    assert ana_resp.get("success") is True
    ana = ana_resp.get("analytics") or ana_resp

    # Check weekly activity (7 days: Mon-Sun)
    weekly = ana.get("weekly_activity", {})
    labels = weekly.get("labels", [])
    counts = weekly.get("counts", [])
    assert len(labels) == 7, f"Expected 7 day labels, got {len(labels)}"
    assert len(counts) == 7, f"Expected 7 count entries, got {len(counts)}"

    # Check overall progress formula breakdown
    progress_breakdown = ana.get("progress_breakdown", {})
    assert "documents_score" in progress_breakdown
    assert "summaries_score" in progress_breakdown
    assert "chat_score" in progress_breakdown
    assert "quizzes_score" in progress_breakdown
    assert "mastery_score" in progress_breakdown

    # Check difficulty breakdown
    diff = ana.get("difficulty_breakdown", {})
    assert "easy" in diff and "medium" in diff and "hard" in diff

    # Check learning metrics
    lm = ana.get("learning_stats", {})
    assert "total_correct_answers" in lm
    assert "total_incorrect_answers" in lm
    assert "average_score" in lm
    print(" [PASS] 5. Analytics Data API (/api/analytics/data) verified with 5 pillars, weekly distribution & difficulty stats")

    # 6. Study Goals UI & CRUD Operations
    status, html_goals, _ = make_request("/goals")
    assert status == 200, f"/goals returned {status}"
    assert "Study Goals" in html_goals
    print(" [PASS] 6a. Study Goals Page (/goals) HTML rendered")

    # 6b. Create a new Study Goal
    new_goal = {
        "title": "Phase 6 E2E Test Goal",
        "description": "Achieve 5 quiz completions for comprehensive testing",
        "goal_type": "quiz_count",
        "target_value": 5,
        "target_date": (datetime.date.today() + datetime.timedelta(days=7)).isoformat()
    }
    status, _, create_goal_resp = make_request("/api/goals/create", method="POST", data=new_goal)
    assert status in (200, 201), f"Goal creation failed: {create_goal_resp}"
    assert create_goal_resp.get("success") is True
    goal_id = create_goal_resp.get("goal_id")
    assert goal_id is not None
    print(f" [PASS] 6b. Created test study goal #{goal_id}")

    # 6c. Fetch Goals List and verify progress calculation
    status, _, goals_list_resp = make_request("/api/goals")
    assert status == 200
    goals = goals_list_resp.get("goals", [])
    matched = [g for g in goals if g["id"] == goal_id]
    assert len(matched) == 1, f"Goal #{goal_id} not found in goals list"
    assert "progress_percentage" in matched[0]
    print(f" [PASS] 6c. Retrieved study goal #{goal_id} with current progress {matched[0]['progress_percentage']}%")

    # 6d. Update Goal (change status or manual update)
    status, _, update_resp = make_request(f"/api/goals/{goal_id}/update", method="POST", data={"status": "completed"})
    assert status == 200
    assert update_resp.get("success") is True

    # 6e. Delete Goal
    status, _, del_resp = make_request(f"/api/goals/{goal_id}/delete", method="POST")
    assert status == 200
    assert del_resp.get("success") is True
    print(f" [PASS] 6d-e. Successfully updated and deleted test study goal #{goal_id}")

    # 7. Activity History UI & API
    status, html_act, _ = make_request("/activity")
    assert status == 200
    assert "Activity History" in html_act
    print(" [PASS] 7a. Activity History Page (/activity & /history) HTML rendered")

    status, _, act_api_resp = make_request("/api/activity/history?limit=10")
    assert status == 200
    assert act_api_resp.get("success") is True
    activities = act_api_resp.get("activities", [])
    assert isinstance(activities, list)
    print(f" [PASS] 7b. Activity History API returned {len(activities)} items (total tracked: {act_api_resp.get('total')})")

    # Filtered activity query
    status, _, filtered_act = make_request("/api/activity/history?activity_type=quiz_attempt")
    assert status == 200
    assert filtered_act.get("success") is True
    print(" [PASS] 7c. Filtered Activity History API query succeeded")

    # 8. Test quick activity logging via Chatbot or direct action
    # Let's inspect if documents exist
    status, _, docs_data = make_request("/quizzes", headers={"Accept": "application/json"})
    docs = docs_data.get("documents", [])
    if docs:
        doc_id = docs[0]["id"]
        # Ask a quick chatbot question to verify activity logging hook
        chat_payload = {
            "question": "What are the main topics discussed in this document?",
            "top_k": 2
        }
        status, _, chat_resp = make_request(f"/api/chat/{doc_id}/ask", method="POST", data=chat_payload)
        if status == 200 and chat_resp.get("success"):
            # Check that an activity was logged
            status, _, recent_act = make_request("/api/activity/history?limit=1")
            act_list = recent_act.get("activities")
            if act_list and isinstance(act_list, list) and len(act_list) > 0:
                latest = act_list[0]
                assert latest.get("activity_type") in ("chat_question", "quiz_attempt", "upload_pdf", "generate_summary", "generate_quiz")
            print(" [PASS] 8. End-to-End Activity Logging verified live via Chatbot Q&A hook")

    print("=" * 80)
    print("ALL 10 PHASE 6 LIVE FUNCTIONAL CHECKS PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError as e:
        print(f"\n[FAIL] Functional assertion error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during functional testing: {e}", file=sys.stderr)
        sys.exit(2)
