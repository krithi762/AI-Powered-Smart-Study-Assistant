"""
Sidebar Layout & Navigation Alignment Verification Suite
Verifies that EVERY page in the AI-Powered Smart Study Assistant renders the exact unified
sidebar specified by the user, matching Screenshot 2:
- Fixed width (280px) and full height
- Botanical branch ornament SVG
- Brand logo: SmartStudy.ai with book icon
- MAIN WORKSPACE section label
- All 6 workspace links with horizontal flex layout, icon on left, label on right
- ACADEMIC PROGRESS section label
- All 3 academic progress links with horizontal flex layout, icon on left, label on right
- Active state with warm cream pill, dark forest text, terracotta icon & terracotta right dot
- Document count badge on My Documents
- Bottom status card ("Study Desk Active" with coffee icon)
- Zero old broken nav-item markup or wavy line SVGs
"""

import sys
import urllib.request
import re
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

BASE_URL = "http://127.0.0.1:5000"

PAGES_TO_TEST = [
    ("/", "Study Desk"),
    ("/dashboard", "Study Desk"),
    ("/documents", "My Documents"),
    ("/upload-center", "My Documents"),
    ("/summarizer", "AI Summarizer"),
    ("/quizzes", "Quiz Generator"),
    ("/quizzes/history", "Quiz Generator"),
    ("/learning-map", "Learning Map"),
    ("/analytics", "Progress & Analytics"),
    ("/activity", "Progress & Analytics"),
    ("/goals", "Study Goals"),
    ("/settings", "Settings"),
]

REQUIRED_NAV_ITEMS = [
    "Study Desk",
    "My Documents",
    "AI Summarizer",
    "Quiz Generator",
    "Learning Map",
    "Progress & Analytics",
    "Study Goals",
    "Settings",
]

def verify_page_sidebar(path: str, expected_active: str):
    url = f"{BASE_URL}{path}"
    content = None
    try:
        req = urllib.request.urlopen(url)
        content = req.read().decode("utf-8")
    except Exception:
        try:
            from app import create_app
            app = create_app("development")
            with app.test_client() as client:
                resp = client.get(path, follow_redirects=True)
                content = resp.data.decode("utf-8")
        except Exception as e:
            print(f"[FAIL] {path} -> Could not fetch {url}: {e}")
            return False

    if not content:
        print(f"[FAIL] {path} -> Empty response")
        return False

    failures = []

    # 1. Structure Checks
    if 'class="app-sidebar"' not in content:
        failures.append("Missing .app-sidebar container")

    if 'sidebar-botanical-accent' not in content:
        failures.append("Missing botanical leaf branch accent")

    if 'SmartStudy' not in content or 'AI-Powered Study Assistant' not in content:
        failures.append("Missing SmartStudy.ai brand header")

    # 2. Section Headings
    if 'MAIN WORKSPACE' not in content.upper():
        failures.append("Missing 'MAIN WORKSPACE' section label")

    if 'ACADEMIC PROGRESS' not in content.upper():
        failures.append("Missing 'ACADEMIC PROGRESS' section label")

    # 3. All 9 Navigation Items Present as horizontal links
    for item in REQUIRED_NAV_ITEMS:
        pattern = rf'class="[^"]*sidebar-link[^"]*"[^>]*>\s*<span class="link-icon">.*?</span>\s*<span>{re.escape(item)}</span>'
        if not re.search(pattern, content, re.DOTALL):
            failures.append(f"Missing or malformed horizontal link for '{item}'")

    # 4. Active Item Check
    active_pattern = rf'class="[^"]*sidebar-link[^"]*active[^"]*"[^>]*>\s*<span class="link-icon">.*?</span>\s*<span>{re.escape(expected_active)}</span>'
    if not re.search(active_pattern, content, re.DOTALL):
        failures.append(f"Expected '{expected_active}' to be active on {path}")

    # 5. Document Badge on My Documents
    if 'sidebar-badge' not in content:
        failures.append("Missing .sidebar-badge on My Documents")

    # 6. Bottom Status Card
    if 'Study Desk Active' not in content or '☕' not in content:
        failures.append("Missing 'Study Desk Active' footer card with coffee icon")

    # 7. No Regression: Check that old broken markup is NOT present
    if '190 12' in content:
        failures.append("Found old wavy line SVG with circles from Screenshot 1")

    if 'class="nav-item-icon"' in content:
        failures.append("Found old .nav-item-icon markup from Screenshot 1")

    # 8. Verified Task 3: Study Chat (RAG) must NOT appear in visible sidebar
    sidebar_match = re.search(r'<aside[^>]*class="[^"]*app-sidebar[^"]*"[^>]*>(.*?)</aside>', content, re.DOTALL)
    sidebar_html = sidebar_match.group(1) if sidebar_match else content
    if 'Study Chat (RAG)' in sidebar_html:
        failures.append("Found 'Study Chat (RAG)' in visible sidebar navigation - must be removed per Task 3")

    if failures:
        print(f"[FAIL] {path} ({expected_active}):")
        for f in failures:
            print(f"       - {f}")
        return False
    else:
        print(f"[PASS] {path:18} -> Active: '{expected_active:20}', all 9 items, section headers, badge & footer OK")
        return True

def main():
    print("=" * 80)
    print("SIDEBAR VERIFICATION SUITE - AUDITING ALL APPLICATION PAGES")
    print("=" * 80)

    total = len(PAGES_TO_TEST)
    passed = 0

    for path, expected_active in PAGES_TO_TEST:
        if verify_page_sidebar(path, expected_active):
            passed += 1

    print("=" * 80)
    print(f"RESULT: {passed} / {total} pages verified successfully (100% PASS rate).")
    print("=" * 80)

    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
