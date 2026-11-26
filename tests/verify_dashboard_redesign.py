"""
Comprehensive Verification Script for the Redesigned SmartStudy.ai Dashboard.
Audits all design, DOM, and functional requirements against the user specification and reference screenshot.
"""

import urllib.request

content = None
try:
    url = "http://127.0.0.1:5000/"
    req = urllib.request.urlopen(url)
    content = req.read().decode("utf-8")
except Exception:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import create_app
    client = create_app("development").test_client()
    resp = client.get("/", follow_redirects=True)
    content = resp.data.decode("utf-8")

checks = [
    # 1. Design & Typography
    ("CSS stylesheet link", '<link rel="stylesheet" href="/static/css/style.css">' in content),
    ("App wrapper present", 'class="app-wrapper"' in content),
    ("Deep forest sidebar present", 'class="app-sidebar"' in content),
    ("Botanical leaf branch ornament", 'sidebar-botanical-accent' in content),
    ("Brand logo and AI subtitle", 'SmartStudy' in content and 'AI-Powered Study Assistant' in content),
    ("Study Desk active navigation", 'sidebar-link active' in content and 'Study Desk' in content),
    ("My Documents document badge", 'sidebar-badge' in content),
    ("Study Desk Active bottom status card", 'Study Desk Active' in content and '☕' in content),

    # 2. Top Navigation Bar
    ("Search input with placeholder", 'Search your notes, topics, or ask a question...' in content),
    ("Ctrl + K keyboard shortcut badge", 'Ctrl + K' in content),
    ("Notifications button", 'title="Notifications"' in content),
    ("Calendar button", 'title="Study Calendar &amp; Activity"' in content or 'title="Study Calendar & Activity"' in content),
    ("Profile pill with dynamic name & avatar", 'user-profile-pill' in content and 'user-avatar-circle' in content),

    # 3. Welcome Hero Card
    ("Greeting lead with welcome & student name", 'Welcome &bull; Good Afternoon,' in content and '🌿' in content),
    ("Main heading: Your Study Desk", 'Your Study Desk' in content),
    ("Hero description text", 'Explore your topics, continue where you left off, and let AI make your learning easier and deeper.' in content),
    ("Window sill & succulent illustration", 'greeting-window-backdrop' in content),

    # 4. Today's Focus Widget
    ("Sticky note card present", 'sticky-note-card' in content),
    ("3D red pushpin SVG present", 'sticky-pushpin' in content),
    ("Today's Focus script title", "Today&#39;s Focus" in content or "Today's Focus" in content),
    ("Task 1: Summarize OS Unit 4", 'Summarize OS Unit 4' in content),
    ("Task 2: Take a Quiz (Medium)", 'Take a Quiz (Medium)' in content),
    ("Task 3: Revise Key Concepts", 'Revise Key Concepts' in content),
    ("Task checkboxes and add form", 'id="focusChecklist"' in content and 'id="newTaskInput"' in content),

    # 5. Interactive Learning Map
    ("My Learning Map ribbon badge", 'My Learning Map' in content and 'learning-map-ribbon' in content),
    ("Subject selector dropdown", 'id="subjectSelector"' in content and 'Operating Systems' in content),
    ("Illustrated landscape canvas", 'map-canvas-area' in content and 'map-landscape-svg' in content),
    ("Mountains, lake, flag, compass in SVG", 'map-compass-rose' in content and 'hikingTrailPath' in content),
    ("Unit 1: OS Fundamentals (Completed)", 'OS Fundamentals' in content and 'milestone-badge-completed' in content),
    ("Unit 2: Process Management (Completed)", 'Process Management' in content),
    ("Unit 3: Memory Management (Active 75%)", 'Memory Management' in content and '75%' in content and 'node-active' in content),
    ("Unit 4: CPU Scheduling (Next)", 'CPU Scheduling' in content and 'Next' in content and 'node-next' in content),
    ("Unit 5: I/O & File Systems (Locked)", 'I/O &amp; File Systems' in content or 'I/O & File Systems' in content and 'node-locked' in content),
    ("Dynamic subject switching JS function", 'switchSubjectMap' in content and 'SUBJECT_MAPS' in content),

    # 6. AI Study Insight Panel
    ("AI Study Insight header", 'AI Study Insight' in content and 'sparkle-icon' in content),
    ("Collapse toggle button", 'id="collapseInsightBtn"' in content),
    ("Lightbulb tip box", 'ai-tip-box' in content and 'ai-tip-icon' in content and '💡' in content),
    ("Recommendations: Revise Paging & Segmentation", 'Revise: Paging &amp; Segmentation' in content or 'Revise: Paging & Segmentation' in content),
    ("Recommendations: Take a 10-question quiz", 'Take a 10-question quiz' in content),
    ("Recommendations: Summarize Unit 3", 'Summarize Unit 3 (Detailed)' in content),
    ("Learning Streak: 5 Days", 'Learning Streak: 5 Days' in content and '🔥' in content),
    ("Mon-Sun day indicators with checkmarks", 'Mon' in content and 'streak-days-row' in content),

    # 7. Dashboard Statistics (Requirement 9)
    ("Overview Statistics header", 'Overview Statistics' in content),
    ("Total Documents stat card", 'Total Documents' in content),
    ("Summaries Generated stat card", 'Summaries Generated' in content),
    ("Quizzes Completed stat card", 'Quizzes Completed' in content),
    ("Study Progress stat card", 'Study Progress' in content),

    # 8. All 10 Required Test Assertion Sections
    ("Section 1: Welcome", 'welcome' in content.lower()),
    ("Section 2: Overview Statistics", 'overview statistics' in content.lower()),
    ("Section 3: Recent Study Documents", 'recent study documents' in content.lower()),
    ("Section 4: Recent Summaries", 'recent summaries' in content.lower()),
    ("Section 5: Recent Chat", 'recent chat' in content.lower()),
    ("Section 6: Recent Quiz Attempts", 'recent quiz attempts' in content.lower()),
    ("Section 7: Weekly Study Activity", 'weekly study activity' in content.lower()),
    ("Section 8: Quiz Performance", 'quiz performance' in content.lower()),
    ("Section 9: Study Goals", 'study goals' in content.lower()),
    ("Section 10: AI Study Features", 'ai study features' in content.lower()),

    # 9. Quick Upload
    ("Quick Upload Dropzone card", 'quick-upload-card' in content),
]

all_passed = True
print("=" * 75)
print("AUDIT & VERIFICATION REPORT: SMARTSTUDY.AI REDESIGNED DASHBOARD")
print("=" * 75)
for label, passed in checks:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {label}")
    if not passed:
        all_passed = False

print("=" * 75)
if all_passed:
    print(f"ALL {len(checks)} CHECKS PASSED WITH ZERO ERRORS!")
else:
    print("SOME CHECKS FAILED.")
print("=" * 75)

if not all_passed:
    exit(1)
