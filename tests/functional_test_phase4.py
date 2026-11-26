"""
Phase 4 Complete Functional Test Suite (Tests 1 - 15)
Executes against the running Flask application at http://127.0.0.1:5000.
Verifies all 15 functional requirements specified in STEP 5 of the Phase 4 specification.
"""

import sys
from pathlib import Path

import requests

BASE = "http://127.0.0.1:5000"
results = []


def log_test(num: int, name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    tag = f"TEST {num}: {name}"
    msg = f"[{status}] {tag}"
    if detail:
        msg += f" | {detail}"
    print(msg)
    results.append((num, name, status, detail))


def ensure_test_pdf(path: Path, pages: list) -> Path:
    """Creates a sample academic PDF if not already present."""
    if path.exists() and path.stat().st_size > 0:
        return path
    import pymupdf
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


def run_all_tests():
    print("=" * 70)
    print("  PHASE 4 COMPLETE FUNCTIONAL VERIFICATION (TESTS 1 - 15)")
    print("  AI-Powered Smart Study Assistant — RAG AI Chatbot")
    print("=" * 70)

    # Pre-test Health Check
    try:
        hr = requests.get(f"{BASE}/health", timeout=10)
        hdata = hr.json()
        print(f"Server health: {hdata.get('status')} | Phase: {hdata.get('phase')}")
        print(f"RAG QA Model: {hdata.get('rag_qa_model')} | Embedding: {hdata.get('rag_embedding_model')}")
        print("-" * 70)
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {BASE}: {e}")
        sys.exit(1)

    # Setup test file paths
    tests_dir = Path("tests")
    pdf_geo = ensure_test_pdf(tests_dir / "test_geology_func.pdf", [
        (
            "Igneous rocks form through the cooling and solidification of magma or lava. "
            "Granite is a common intrusive igneous rock that forms deep underground. "
            "Basalt is an extrusive igneous rock formed from rapidly cooled lava. "
            "The texture of an igneous rock depends on the cooling rate of the magma. "
            "Slow cooling produces large crystals while rapid cooling produces fine-grained rocks."
        ),
        (
            "Sedimentary rocks form through the accumulation and cementation of mineral particles. "
            "Limestone is a sedimentary rock formed from marine organisms and their shells. "
            "Sandstone is composed of compressed sand particles cemented together by minerals."
        )
    ])

    pdf_thermo = ensure_test_pdf(tests_dir / "test_thermo_func.pdf", [
        (
            "The Carnot heat engine is an idealized thermodynamic cycle operating between two temperatures. "
            "It establishes the theoretical maximum efficiency for converting thermal energy into work. "
            "The Carnot efficiency depends strictly on the absolute temperatures of the hot and cold reservoirs. "
            "Entropy remains constant during the reversible adiabatic processes of the Carnot cycle."
        )
    ])

    # -------------------------------------------------------------
    # TEST 1: Upload a valid PDF
    # -------------------------------------------------------------
    doc1_id = None
    try:
        with open(pdf_geo, "rb") as f:
            res = requests.post(
                f"{BASE}/upload",
                files={"file": ("academic_geology.pdf", f, "application/pdf")},
                headers={"Accept": "application/json"},
                timeout=30,
            )
        data = res.json()
        doc1_id = data.get("document_id")
        passed = (res.status_code in (200, 201) and data.get("success") is True and doc1_id is not None)
        log_test(1, "Upload a valid PDF", passed, f"doc_id={doc1_id}, pages={data.get('total_pages')}, words={data.get('total_words')}")
    except Exception as e:
        log_test(1, "Upload a valid PDF", False, str(e))
        return

    # -------------------------------------------------------------
    # TEST 2: Extract PDF text
    # -------------------------------------------------------------
    try:
        res = requests.get(f"{BASE}/document/{doc1_id}/text", headers={"Accept": "application/json"}, timeout=15)
        data = res.json()
        extracted_pages = data.get("pages", [])
        total_text = data.get("full_text") or " ".join([p.get("extracted_text", "") for p in extracted_pages])
        passed = (res.status_code == 200 and len(extracted_pages) >= 2 and "igneous rocks" in total_text.lower())
        log_test(2, "Extract PDF text", passed, f"Extracted {len(extracted_pages)} pages, {len(total_text)} chars")
    except Exception as e:
        log_test(2, "Extract PDF text", False, str(e))

    # -------------------------------------------------------------
    # TEST 3: Generate embeddings and index the document
    # -------------------------------------------------------------
    try:
        res = requests.post(f"{BASE}/document/{doc1_id}/index", headers={"Accept": "application/json"}, timeout=60)
        data = res.json()
        chunks_count = data.get("total_chunks") or data.get("result", {}).get("total_chunks", 0)
        status_val = data.get("status") or data.get("result", {}).get("status")
        passed = (res.status_code == 200 and data.get("success") is True and chunks_count > 0)
        log_test(3, "Generate embeddings and index document", passed, f"Chunks indexed: {chunks_count}, status: {status_val}")
    except Exception as e:
        log_test(3, "Generate embeddings and index document", False, str(e))

    # -------------------------------------------------------------
    # TEST 4: Ask a question whose answer is clearly present in the PDF
    # -------------------------------------------------------------
    qa_data = {}
    try:
        res = requests.post(
            f"{BASE}/document/{doc1_id}/chat/ask",
            json={"question": "How do igneous rocks form?"},
            headers={"Accept": "application/json"},
            timeout=90,
        )
        qa_data = res.json()
        passed = (res.status_code == 200 and qa_data.get("success") is True and len(qa_data.get("answer", "")) > 10)
        log_test(4, "Ask a question present in PDF", passed, f"HTTP {res.status_code}, response length: {len(qa_data.get('answer', ''))}")
    except Exception as e:
        log_test(4, "Ask a question present in PDF", False, str(e))

    # -------------------------------------------------------------
    # TEST 5: Verify that the chatbot returns the correct answer
    # -------------------------------------------------------------
    try:
        answer = qa_data.get("answer", "")
        is_grounded = qa_data.get("is_grounded", False)
        # Check answer reflects magma, lava, cooling, or solidification
        answer_lower = answer.lower()
        has_keywords = any(w in answer_lower for w in ["magma", "lava", "cooling", "solidification"])
        passed = (is_grounded and has_keywords)
        log_test(5, "Verify correct grounded answer", passed, f"is_grounded={is_grounded}, snippet='{answer[:75]}...'")
    except Exception as e:
        log_test(5, "Verify correct grounded answer", False, str(e))

    # -------------------------------------------------------------
    # TEST 6: Verify the correct PDF filename and page number
    # -------------------------------------------------------------
    try:
        sources = qa_data.get("sources", [])
        passed = False
        detail = "No sources found"
        if sources:
            top_src = sources[0]
            p_num = top_src.get("page_number")
            fn = top_src.get("original_filename", "")
            passed = (p_num == 1 and "academic_geology.pdf" in fn)
            detail = f"filename='{fn}', page={p_num}, score={top_src.get('similarity_score')}"
        log_test(6, "Verify PDF filename and page citation", passed, detail)
    except Exception as e:
        log_test(6, "Verify PDF filename and page citation", False, str(e))

    # -------------------------------------------------------------
    # TEST 7: Ask a follow-up question
    # -------------------------------------------------------------
    try:
        res = requests.post(
            f"{BASE}/document/{doc1_id}/chat/ask",
            json={"question": "What does it produce when cooled slowly?"},
            headers={"Accept": "application/json"},
            timeout=90,
        )
        followup_data = res.json()
        ans = followup_data.get("answer", "")
        passed = (res.status_code == 200 and followup_data.get("success") is True and followup_data.get("is_grounded") is True)
        log_test(7, "Ask follow-up question (pronoun resolution)", passed, f"answer snippet='{ans[:75]}...'")
    except Exception as e:
        log_test(7, "Ask follow-up question", False, str(e))

    # -------------------------------------------------------------
    # TEST 8: Ask a question unrelated to the PDF (No hallucination)
    # -------------------------------------------------------------
    try:
        res = requests.post(
            f"{BASE}/document/{doc1_id}/chat/ask",
            json={"question": "What is the capital city of Australia and what is its population?"},
            headers={"Accept": "application/json"},
            timeout=90,
        )
        offtopic_data = res.json()
        is_grounded = offtopic_data.get("is_grounded", True)
        ans = offtopic_data.get("answer", "")
        # Should gracefully decline without fabricating document facts
        passed = (res.status_code == 200 and not is_grounded and "could not find" in ans.lower())
        log_test(8, "Unrelated question receives 'could not find' (no hallucination)", passed, f"is_grounded={is_grounded}, answer='{ans[:70]}...'")
    except Exception as e:
        log_test(8, "Unrelated question handling", False, str(e))

    # -------------------------------------------------------------
    # TEST 9: Upload two different PDFs and verify document-specific retrieval
    # -------------------------------------------------------------
    doc2_id = None
    try:
        with open(pdf_thermo, "rb") as f:
            res = requests.post(
                f"{BASE}/upload",
                files={"file": ("thermodynamics_cycle.pdf", f, "application/pdf")},
                headers={"Accept": "application/json"},
                timeout=30,
            )
        data = res.json()
        doc2_id = data.get("document_id")

        # Query Document 2 for Carnot efficiency
        res2 = requests.post(
            f"{BASE}/document/{doc2_id}/chat/ask",
            json={"question": "What does the Carnot engine establish?"},
            headers={"Accept": "application/json"},
            timeout=90,
        )
        d2 = res2.json()
        ans2 = d2.get("answer", "")
        # Document 2 must NOT have geology content
        no_leakage = "igneous" not in ans2.lower() and "magma" not in ans2.lower()
        has_thermo = "efficiency" in ans2.lower() or "maximum" in ans2.lower() or "carnot" in ans2.lower()
        passed = (res2.status_code in (200, 201) and d2.get("is_grounded") is True and no_leakage and has_thermo)
        log_test(9, "Multi-PDF document-specific retrieval & strict isolation", passed, f"doc2_id={doc2_id}, answer='{ans2[:75]}...'")
    except Exception as e:
        log_test(9, "Multi-PDF document-specific retrieval", False, str(e))

    # -------------------------------------------------------------
    # TEST 10: Clear conversation history
    # -------------------------------------------------------------
    try:
        # Check history exists first
        h_before = requests.get(f"{BASE}/document/{doc1_id}/chat/history", timeout=10).json()
        count_before = len(h_before.get("messages", []))
        # Clear
        clr_res = requests.post(f"{BASE}/document/{doc1_id}/chat/clear", timeout=10)
        clr_data = clr_res.json()
        # Verify 0 messages
        h_after = requests.get(f"{BASE}/document/{doc1_id}/chat/history", timeout=10).json()
        count_after = len(h_after.get("messages", []))
        passed = (clr_res.status_code == 200 and clr_data.get("success") is True and count_before > 0 and count_after == 0)
        log_test(10, "Clear conversation history", passed, f"Before: {count_before} messages -> After: {count_after} messages")
    except Exception as e:
        log_test(10, "Clear conversation history", False, str(e))

    # -------------------------------------------------------------
    # TEST 11: Upload an invalid or empty PDF
    # -------------------------------------------------------------
    try:
        # Empty file upload
        res_empty = requests.post(
            f"{BASE}/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        # Non-pdf invalid extension
        res_txt = requests.post(
            f"{BASE}/upload",
            files={"file": ("notes.txt", b"plain text content", "text/plain")},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        passed = (res_empty.status_code == 400 and res_txt.status_code == 400)
        log_test(11, "Upload invalid and empty PDF rejected gracefully", passed, f"empty_pdf_code={res_empty.status_code}, txt_code={res_txt.status_code}")
    except Exception as e:
        log_test(11, "Upload invalid/empty PDF", False, str(e))

    # -------------------------------------------------------------
    # TEST 12: Test large document processing
    # -------------------------------------------------------------
    try:
        # Create a multi-page larger document (5 pages, ~2000 words)
        pdf_large = tests_dir / "test_large_doc.pdf"
        import pymupdf
        ldoc = pymupdf.open()
        for i in range(1, 6):
            lpage = ldoc.new_page()
            rect = pymupdf.Rect(50, 50, 550, 750)
            page_content = (
                f"Chapter {i}: Advanced Neural Architectures and Deep Learning Systems.\n\n" +
                ("Recurrent and transformer networks model sequential and contextual patterns across large datasets. "
                 "Self-attention mechanisms calculate relative importance weights between input tokens dynamically. "
                 "Multi-head attention facilitates diverse subspace representations for Natural Language Processing.\n\n" * 10)
            )
            lpage.insert_textbox(rect, page_content)
        ldoc.save(str(pdf_large))
        ldoc.close()

        with open(pdf_large, "rb") as f:
            res = requests.post(
                f"{BASE}/upload",
                files={"file": ("large_study_guide.pdf", f, "application/pdf")},
                headers={"Accept": "application/json"},
                timeout=45,
            )
        data = res.json()
        large_doc_id = data.get("document_id")
        passed = (res.status_code in (200, 201) and data.get("total_pages") == 5 and data.get("total_words", 0) > 400)
        log_test(12, "Large multi-page document processing", passed, f"doc_id={large_doc_id}, pages={data.get('total_pages')}, words={data.get('total_words')}")
    except Exception as e:
        log_test(12, "Large multi-page document processing", False, str(e))

    # -------------------------------------------------------------
    # TEST 13: Verify that Phase 2 PDF processing still works
    # -------------------------------------------------------------
    try:
        res_list = requests.get(f"{BASE}/documents", timeout=15)
        res_det = requests.get(f"{BASE}/document/{doc1_id}", timeout=15)
        has_table = "academic_geology.pdf" in res_det.text or "academic_geology.pdf" in res_list.text
        passed = (res_list.status_code == 200 and res_det.status_code == 200 and has_table)
        log_test(13, "Phase 2 PDF processing & document routes still work", passed, f"Documents list HTTP {res_list.status_code}, details HTTP {res_det.status_code}")
    except Exception as e:
        log_test(13, "Phase 2 PDF processing", False, str(e))

    # -------------------------------------------------------------
    # TEST 14: Verify that Phase 3 AI summarization still works
    # -------------------------------------------------------------
    try:
        res_sum = requests.post(
            f"{BASE}/document/{doc1_id}/summarize",
            json={"summary_type": "quick"},
            headers={"Accept": "application/json"},
            timeout=90,
        )
        data_sum = res_sum.json()
        sum_text = data_sum.get("summary_text") or data_sum.get("summary", {}).get("summary_text", "")
        key_concepts = data_sum.get("key_concepts") or data_sum.get("summary", {}).get("key_concepts", [])
        passed = (res_sum.status_code == 200 and data_sum.get("success") is True and len(sum_text) > 10)
        log_test(14, "Phase 3 AI summarization still works", passed, f"Summary generated: {len(sum_text)} chars, concepts: {len(key_concepts)}")
    except Exception as e:
        log_test(14, "Phase 3 AI summarization", False, str(e))

    # -------------------------------------------------------------
    # TEST 15: Verify that the frontend works without JavaScript errors
    # -------------------------------------------------------------
    try:
        # Verify chatbot HTML contains all required DOM containers and script tags
        chat_html = requests.get(f"{BASE}/document/{doc1_id}/chat", timeout=15).text
        has_chat_window = "chat-window-card" in chat_html
        has_messages_div = "chat-messages-container" in chat_html
        has_input_form = "chat-form" in chat_html and "chat-question-input" in chat_html
        has_suggested = "suggested-chips-container" in chat_html
        has_script = "script.js" in chat_html

        # Verify static JS and CSS files are served with status 200
        res_js = requests.get(f"{BASE}/static/js/script.js", timeout=10)
        res_css = requests.get(f"{BASE}/static/css/style.css", timeout=10)

        # Check JS contains chatbot handlers
        js_text = res_js.text
        has_js_handlers = (
            "sendQuestion" in js_text and
            "appendUserMessage" in js_text and
            "appendAssistantMessage" in js_text and
            "clearChat" in js_text
        )

        passed = (
            has_chat_window and has_messages_div and has_input_form and
            has_suggested and has_script and res_js.status_code == 200 and
            res_css.status_code == 200 and has_js_handlers
        )
        log_test(15, "Frontend UI & JavaScript client controller verified", passed, f"HTML elements valid, script.js={res_js.status_code}, style.css={res_css.status_code}, handlers present")
    except Exception as e:
        log_test(15, "Frontend UI verification", False, str(e))

    # -------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------
    print()
    print("=" * 70)
    passed_count = sum(1 for _, _, s, _ in results if s == "PASS")
    failed_count = sum(1 for _, _, s, _ in results if s == "FAIL")
    total_count = len(results)
    print(f"  RESULTS: {passed_count}/{total_count} PASSED  |  {failed_count} FAILED")
    print("=" * 70)

    if failed_count > 0:
        print("\nFailed Tests:")
        for num, name, s, detail in results:
            if s == "FAIL":
                print(f"  - TEST {num}: {name} ({detail})")
        sys.exit(1)
    else:
        print("\nALL 15 PHASE 4 FUNCTIONAL TESTS COMPLETED SUCCESSFULLY!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
