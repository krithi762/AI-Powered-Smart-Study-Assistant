"""
DOM and Layout verification script for SmartStudy.ai Student Profile.
Checks all 9 design requirements specified by the user.
"""

import urllib.request
import re

url = "http://127.0.0.1:5000/settings"
req = urllib.request.urlopen(url)
content = req.read().decode("utf-8")

checks = [
    # 1. Design Reference & Theme
    ("CSS link present", '<link rel="stylesheet" href="/static/css/style.css">' in content),
    ("App wrapper present", 'class="app-wrapper"' in content),
    ("Deep forest sidebar present", 'class="app-sidebar"' in content),
    ("Botanical leaf accent present", 'class="sidebar-botanical-accent"' in content),
    ("Academic dotted canvas present", 'academic-dotted-canvas' in content),
    ("Sidebar coffee footer present", '☕' in content),

    # 2. Profile Page Header
    ("Welcome back header greeting", 'Welcome back, Krishna!' in content),
    ("My Student Profile title", 'My Student Profile' in content),
    ("Description text present", 'Manage your personal information, academic details and study preferences in one place.' in content),
    ("Hero avatar with initials", 'profile-avatar-circle-large' in content and 'KK' in content),
    ("Status pip present", 'profile-avatar-status-pip' in content),

    # 3. 4 Profile Overview Cards
    ("Overview grid present", 'profile-overview-grid' in content),
    ("Card 1: Personal Information", 'Personal Information' in content and 'cardFullName' in content),
    ("Card 2: Academic Information", 'Academic Information' in content and 'cardSpecialization' in content),
    ("Card 3: Department", 'Department' in content and 'cardDepartment' in content),
    ("Card 4: Academic Year", 'Academic Year' in content and 'cardYear' in content),

    # 4. Personal Information Section
    ("Full Name field", 'id="full_name"' in content and 'value="Krishna K"' in content),
    ("Email Address field", 'id="email"' in content and 'value="krishna.k@smartstudy.ai"' in content),
    ("College Name field", 'id="college_name"' in content and 'Institute of Engineering' in content),
    ("Edit Profile button", 'id="editProfileBtn"' in content),

    # 5. Academic Information Section
    ("Specialization field", 'id="specialization"' in content and 'Artificial Intelligence' in content),
    ("Department field", 'id="department"' in content and 'Computer Science' in content),
    ("Academic Year select", 'id="academic_year"' in content and 'Year 3 (Junior)' in content),
    ("Current Semester select", 'id="current_semester"' in content and 'Semester 6' in content),

    # 6. Learning Preferences
    ("Learning Preferences section", 'Learning Preferences' in content),
    ("Default Summarization Mode", 'id="preferred_summary_mode"' in content and 'standard' in content),
    ("RAG Retrieval Precision", 'id="rag_precision"' in content and 'Top 3 Excerpts' in content),
    ("Preferred Learning Mode", 'id="learning_mode"' in content and 'Visual & Concept Mapping' in content),
    ("Study Target Minutes", 'id="study_target_minutes"' in content and 'value="45"' in content),
    ("Local Diagnostics Box", 'distilbart-cnn-12-6' in content and 'all-MiniLM-L6-v2' in content and 'flan-t5-small' in content),

    # 7. Profile Actions
    ("Save Changes button", 'id="saveProfileBtn"' in content and 'Save Changes' in content),
    ("Cancel Edit button", 'id="cancelEditBtn"' in content and 'Cancel' in content),
    ("Alert banner present", 'id="profileAlertBanner"' in content),

    # 8. Responsive Layout
    ("Main content offset", 'app-main app-main-content' in content),
    ("Profile page wrapper", 'profile-page-wrapper' in content),
]

all_passed = True
print("=" * 70)
print("DOM & DESIGN VERIFICATION REPORT FOR STUDENT PROFILE PAGE")
print("=" * 70)
for label, passed in checks:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {label}")
    if not passed:
        all_passed = False

print("=" * 70)
if all_passed:
    print(f"ALL {len(checks)} CHECKS PASSED PERFECTLY!")
else:
    print("SOME CHECKS FAILED.")
print("=" * 70)

if not all_passed:
    exit(1)
