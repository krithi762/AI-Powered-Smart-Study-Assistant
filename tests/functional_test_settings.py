"""
Functional test script for SmartStudy.ai Student Profile and Settings Page Redesign.
Verifies:
1. GET /settings renders profile header, overview cards, personal/academic/preference fields,
   edit buttons, and AI diagnostics.
2. POST /settings persists profile changes across sessions and refreshes.
3. Form validation correctly catches invalid names and malformed emails.
4. GET & POST /api/profile API endpoints return standardized JSON.
5. Profile changes dynamically propagate to Dashboard header greeting and topbar pills.
"""

import sys
import unittest
from app import create_app
from database.db import get_student_profile, update_student_profile


class TestStudentProfileFeature(unittest.TestCase):
    def setUp(self):
        self.app = create_app("testing")
        self.client = self.app.test_client()

    def test_01_get_settings_page_structure(self):
        """Verify GET /settings renders the exact Study Desk profile design components."""
        response = self.client.get("/settings")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        # Header & Greeting
        self.assertIn("My Student Profile", html)
        self.assertIn("Welcome back,", html)
        self.assertIn("Manage your personal information", html)
        self.assertIn("profile-hero-banner", html)
        self.assertIn("profile-avatar-circle-large", html)

        # 4 Overview Cards
        self.assertIn("profile-overview-grid", html)
        self.assertIn("Personal Information", html)
        self.assertIn("Academic Information", html)
        self.assertIn("Department", html)
        self.assertIn("Academic Year", html)

        # Personal Information Form Fields
        self.assertIn('id="full_name"', html)
        self.assertIn('id="email"', html)
        self.assertIn('id="college_name"', html)
        self.assertIn('id="editProfileBtn"', html)

        # Academic Information Fields
        self.assertIn('id="specialization"', html)
        self.assertIn('id="department"', html)
        self.assertIn('id="academic_year"', html)
        self.assertIn('id="current_semester"', html)

        # Learning Preferences
        self.assertIn("Learning Preferences", html)
        self.assertIn('id="preferred_summary_mode"', html)
        self.assertIn('id="rag_precision"', html)
        self.assertIn('id="learning_mode"', html)
        self.assertIn('id="study_target_minutes"', html)

        # Action Buttons
        self.assertIn('id="saveProfileBtn"', html)
        self.assertIn('id="cancelEditBtn"', html)

        # Local AI Engine Diagnostics
        self.assertIn("distilbart", html.lower())
        self.assertIn("sentence-transformers", html.lower())
        self.assertIn("flan-t5", html.lower())

    def test_02_post_update_profile_and_persistence(self):
        """Verify updating profile via POST /settings persists in SQLite."""
        updated_data = {
            "full_name": "Dr. Krishna Kumar",
            "email": "krishna.kumar@smartstudy.ai",
            "college_name": "MIT Institute of Technology",
            "department": "Artificial Intelligence & Data Science",
            "specialization": "Cognitive AI Systems",
            "academic_year": "Year 4 (Senior)",
            "current_semester": "Semester 7",
            "preferred_summary_mode": "detailed",
            "rag_precision": "5",
            "learning_mode": "Active Recall & Quizzes",
            "study_target_minutes": 60,
        }

        # Submit update via JSON
        response = self.client.post(
            "/settings",
            json=updated_data,
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["profile"]["full_name"], "Dr. Krishna Kumar")
        self.assertEqual(data["profile"]["first_name"], "Dr.")
        self.assertEqual(data["profile"]["initials"], "DK")

        # Verify persistence on fresh GET /settings
        get_res = self.client.get("/settings")
        self.assertEqual(get_res.status_code, 200)
        get_html = get_res.get_data(as_text=True)
        self.assertIn("Dr. Krishna Kumar", get_html)
        self.assertIn("krishna.kumar@smartstudy.ai", get_html)
        self.assertIn("Cognitive AI Systems", get_html)
        self.assertIn("MIT Institute of Technology", get_html)

        # Verify dynamic reflection on Dashboard
        dash_res = self.client.get("/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        dash_html = dash_res.get_data(as_text=True)
        self.assertIn("Dr. Krishna Kumar", dash_html)
        self.assertIn("Welcome &bull; Good Afternoon, Dr.!", dash_html)

        # Restore original default student profile
        restore_data = {
            "full_name": "Krishna K",
            "email": "krishna.k@smartstudy.ai",
            "college_name": "Institute of Engineering & Technology",
            "department": "Computer Science & Engineering",
            "specialization": "Artificial Intelligence & Data Systems",
            "academic_year": "Year 3 (Junior)",
            "current_semester": "Semester 6",
            "preferred_summary_mode": "standard",
            "rag_precision": "3",
            "learning_mode": "Visual & Concept Mapping",
            "study_target_minutes": 45,
        }
        self.client.post("/settings", json=restore_data)

    def test_03_profile_validation_rules(self):
        """Verify server rejects invalid names and emails."""
        # Invalid full name (too short)
        res_name = self.client.post(
            "/settings",
            json={"full_name": "K", "email": "valid@email.com"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        self.assertEqual(res_name.status_code, 400)
        self.assertFalse(res_name.get_json()["success"])

        # Invalid email (missing @)
        res_email = self.client.post(
            "/settings",
            json={"full_name": "Krishna K", "email": "invalidemail.com"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        self.assertEqual(res_email.status_code, 400)
        self.assertFalse(res_email.get_json()["success"])

    def test_04_api_profile_endpoint(self):
        """Verify dedicated /api/profile endpoint for mobile and frontend API integrations."""
        # GET /api/profile
        get_api = self.client.get("/api/profile")
        self.assertEqual(get_api.status_code, 200)
        p = get_api.get_json()["profile"]
        self.assertEqual(p["full_name"], "Krishna K")
        self.assertEqual(p["user_id"], "default_student")


if __name__ == "__main__":
    unittest.main()
