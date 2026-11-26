import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from database.db import (
    add_quiz_questions,
    clear_chat_history,
    create_quiz,
    create_study_goal,
    delete_chat_for_document,
    delete_document,
    delete_document_chunks,
    delete_quiz,
    delete_quizzes_for_document,
    delete_study_goal,
    delete_summaries_for_document,
    get_all_documents,
    get_all_quizzes,
    get_all_summaries_for_document,
    get_chat_history,
    get_dashboard_statistics,
    get_document,
    get_document_chunks,
    get_document_pages,
    get_or_create_conversation,
    get_performance_analytics,
    get_quiz,
    get_quiz_attempt,
    get_quiz_attempts,
    get_quiz_questions,
    get_recent_documents,
    get_student_profile,
    get_study_activities,
    get_study_goal,
    get_study_goals,
    get_summary,
    get_total_activities_count,
    insert_document,
    insert_document_pages,
    log_activity,
    save_quiz_attempt,
    save_summary,
    update_document_ocr_metadata,
    update_student_profile,
    update_study_goal,
)
from utils.chatbot import (
    ChatbotError,
    ask_document_question,
    get_qa_engine,
    index_document_for_rag,
)
from utils.pdf_processor import (
    extract_text_from_pdf,
    validate_pdf,
)
from utils.ocr_processor import (
    check_tesseract_available,
    extract_text_via_ocr,
    should_use_ocr,
)
from utils.quiz_evaluator import (
    QuizEvaluationError,
    evaluate_quiz_submission,
)
from utils.quiz_generator import (
    QuizGenerationError,
    generate_quiz_for_document,
)
from utils.summarizer import (
    SummarizationError,
    SummarizerEngine,
    generate_study_summary,
)
from utils.vector_store import (
    get_vector_store,
)

main_bp = Blueprint("main", __name__)


def is_json_request() -> bool:
    """Helper to detect if incoming client request expects JSON response."""
    return (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.is_json
        or "application/json" in request.headers.get("Accept", "")
    )


@main_bp.context_processor
def inject_student_profile():
    """Injects current student profile into all template contexts."""
    profile = get_student_profile()
    return {"current_student": profile, "profile": profile}


@main_bp.route("/")
@main_bp.route("/dashboard")
def index():
    """
    Renders the student dashboard with real-time statistics, active goals,
    weekly activity, quiz performance, recent items, and quick AI feature access.
    """
    stats = get_dashboard_statistics()
    active_goals = get_study_goals(status="in_progress")
    completed_goals = get_study_goals(status="completed")
    analytics = get_performance_analytics()
    recent_activities = get_study_activities(limit=6)
    documents = get_all_documents(limit=6)
    recent_documents = stats.get("recent_documents", [])

    if is_json_request():
        return jsonify({
            "success": True,
            "statistics": stats,
            "active_goals": active_goals,
            "completed_goals": completed_goals,
            "weekly_activity": analytics.get("weekly_activity", {}),
            "learning_stats": analytics.get("learning_stats", {}),
            "recent_activities": recent_activities,
            "recent_documents": recent_documents,
        }), 200

    return render_template(
        "dashboard.html",
        stats=stats,
        goals=active_goals,
        completed_goals=completed_goals,
        weekly=analytics.get("weekly_activity", {}),
        difficulty=analytics.get("difficulty_breakdown", {}),
        learning_stats=analytics.get("learning_stats", {}),
        recent_activities=recent_activities,
        documents=documents,
        recent_documents=recent_documents,
        recent_summaries=stats.get("recent_summaries", []),
        recent_chats=stats.get("recent_chats", []),
        recent_attempts=stats.get("recent_attempts", []),
        progress_explanation=analytics.get("progress_explanation", ""),
    )


# Route alias for dashboard
dashboard = index


@main_bp.route("/upload-center")
def upload_center():
    """Renders standalone PDF upload & processing center."""
    recent_documents = get_recent_documents(limit=5)
    return render_template("index.html", recent_documents=recent_documents)


@main_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Handles PDF upload, secure validation, page-by-page text extraction,
    and SQLite database persistence.
    """
    # 1. Verify file part exists in request
    if "file" not in request.files:
        error_msg = "No file selected. Please select a PDF document to upload."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("main.index")), 400

    file = request.files["file"]

    # 2. Verify filename is not empty
    if not file or not file.filename or file.filename.strip() == "":
        error_msg = "No file selected. Please choose a PDF file."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("main.index")), 400

    original_filename = file.filename.strip()

    # 3. Validate file extension
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
    if ext != "pdf":
        error_msg = "Unsupported file format. Only PDF files are supported."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("main.index")), 400

    # 4. Sanitize and generate unique server-side filename
    safe_basename = secure_filename(original_filename)
    if not safe_basename or safe_basename.endswith("."):
        safe_basename = "document.pdf"
    elif not safe_basename.lower().endswith(".pdf"):
        safe_basename = f"{safe_basename}.pdf"

    unique_stored_name = f"{uuid.uuid4().hex}_{safe_basename}"

    upload_folder = Path(current_app.config["UPLOAD_FOLDER"]).resolve()
    upload_folder.mkdir(parents=True, exist_ok=True)
    destination_path = (upload_folder / unique_stored_name).resolve()

    # Ensure path stays inside designated upload directory (prevent directory traversal)
    if not destination_path.is_relative_to(upload_folder):
        error_msg = "Invalid file destination path detected."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("main.index")), 400

    # 5. Save file to disk
    try:
        file.save(str(destination_path))
    except Exception as e:
        error_msg = f"Failed to save uploaded file: {e!s}"
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 500
        flash(error_msg, "error")
        return redirect(url_for("main.index")), 500

    # 6. Deep PDF content validation (PyMuPDF, size, magic bytes, corruption, password)
    validation_result = validate_pdf(destination_path)
    if not validation_result["valid"]:
        # Clean up invalid file from disk immediately
        if destination_path.exists():
            try:
                destination_path.unlink()
            except OSError:
                pass
        error_msg = validation_result.get("error") or "Uploaded document is not a valid PDF."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("main.index")), 400

    # 7. Extract text page by page
    extraction_result = extract_text_from_pdf(destination_path)
    if not extraction_result["success"]:
        if destination_path.exists():
            try:
                destination_path.unlink()
            except OSError:
                pass
        error_msg = extraction_result.get("error") or "Failed to extract text from PDF."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("main.index")), 400

    total_pages = extraction_result["total_pages"]
    total_words = extraction_result["total_words"]
    is_scanned = extraction_result.get("is_scanned", False)

    # ── OCR FALLBACK ──────────────────────────────────────────────────────────
    # Attempt OCR when PyMuPDF found no text (scanned / image-only PDF).
    ocr_applied: bool = False
    ocr_engine: str = ""
    ocr_confidence: float | None = None
    extraction_method: str = "pymupdf"

    if should_use_ocr(extraction_result):
        tess_info = check_tesseract_available()
        if tess_info["available"]:
            current_app.logger.info(
                "Scanned PDF detected – applying Tesseract OCR for '%s'",
                original_filename,
            )
            ocr_result = extract_text_via_ocr(
                destination_path,
                existing_pages=extraction_result.get("pages", []),
            )
            if ocr_result["success"] and ocr_result["total_words"] > 0:
                # OCR succeeded — replace extraction result
                extraction_result = ocr_result
                total_pages = ocr_result["total_pages"]
                total_words = ocr_result["total_words"]
                is_scanned = ocr_result.get("is_scanned", False)
                ocr_applied = ocr_result.get("ocr_applied", True)
                ocr_engine = ocr_result.get("ocr_engine", "tesseract")
                ocr_confidence = ocr_result.get("ocr_confidence")
                extraction_method = ocr_result.get("extraction_method", "ocr")
            else:
                current_app.logger.warning(
                    "OCR found no readable text in '%s': %s",
                    original_filename,
                    ocr_result.get("error") or "zero words extracted",
                )
        else:
            current_app.logger.info(
                "Scanned PDF '%s' – Tesseract not installed, skipping OCR. %s",
                original_filename,
                tess_info.get("error", ""),
            )
    else:
        extraction_method = "pymupdf"

    # Determine processing status
    if is_scanned and total_words == 0:
        if should_use_ocr(extraction_result) and not check_tesseract_available()["available"]:
            processing_status = "scanned_no_text"
            scanned_notice = (
                "Notice: This PDF contains scanned images. "
                "Install Tesseract OCR to enable automatic text recognition. "
                "See Settings or the documentation for installation instructions."
            )
        elif total_words == 0:
            processing_status = "scanned_no_text"
            scanned_notice = (
                "Notice: This PDF appears to be blank, heavily damaged, or uses "
                "unsupported image encoding. No readable text could be extracted even with OCR."
            )
        else:
            processing_status = "processed"
            scanned_notice = None
    elif extraction_method in ("ocr", "mixed"):
        processing_status = "ocr_completed"
        scanned_notice = None
    else:
        processing_status = "processed"
        scanned_notice = None

    # 8. Persist to SQLite Database
    upload_timestamp = datetime.now(UTC).isoformat()
    try:
        doc_id = insert_document(
            original_filename=original_filename,
            stored_filename=unique_stored_name,
            file_path=str(destination_path),
            total_pages=total_pages,
            total_words=total_words,
            processing_status=processing_status,
            upload_date=upload_timestamp,
            ocr_applied=ocr_applied,
            ocr_engine=ocr_engine,
            ocr_confidence=ocr_confidence,
            extraction_method=extraction_method,
        )

        insert_document_pages(
            document_id=doc_id,
            pages_data=extraction_result["pages"],
        )

        # Phase 6: Log student activity
        try:
            log_activity(
                activity_type="upload",
                title=f"Uploaded '{original_filename}'",
                description=f"Processed {total_pages} pages, {total_words:,} words.",
                document_id=doc_id,
                metadata={"total_pages": total_pages, "total_words": total_words},
                created_at=upload_timestamp,
            )
        except Exception as log_err:
            current_app.logger.warning(f"Failed to log upload activity: {log_err}")
    except Exception as e:
        # Roll back stored file on database failure
        if destination_path.exists():
            try:
                destination_path.unlink()
            except OSError:
                pass
        error_msg = f"Database error while saving document: {e!s}"
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 500
        flash(error_msg, "error")
        return redirect(url_for("main.index")), 500

    # 9. Return success response
    if extraction_method in ("ocr", "mixed"):
        success_msg = (
            f"'{original_filename}' uploaded and OCR-processed successfully! "
            f"({total_pages} pages, {total_words:,} words extracted via Tesseract OCR)"
        )
    elif processing_status == "scanned_no_text":
        success_msg = (
            f"'{original_filename}' uploaded. ({total_pages} pages – "
            "scanned/image-only PDF, no extractable text)"
        )
    else:
        success_msg = (
            f"'{original_filename}' uploaded and processed successfully! "
            f"({total_pages} pages, {total_words:,} words)"
        )

    if is_json_request():
        return jsonify({
            "success": True,
            "message": success_msg,
            "document_id": doc_id,
            "redirect_url": url_for("main.document_details", doc_id=doc_id),
            "total_pages": total_pages,
            "total_words": total_words,
            "is_scanned": is_scanned,
            "ocr_applied": ocr_applied,
            "extraction_method": extraction_method,
            "notice": scanned_notice,
        }), 201

    if scanned_notice:
        flash(f"{success_msg} {scanned_notice}", "warning")
    else:
        flash(success_msg, "success")
    return redirect(url_for("main.document_details", doc_id=doc_id))


@main_bp.route("/documents")
def documents_list():
    """
    Displays the documents dashboard listing all uploaded PDF study materials.
    """
    documents = get_all_documents()
    total_docs = len(documents)
    total_pages = sum(d["total_pages"] for d in documents)
    total_words = sum(d["total_words"] for d in documents)

    return render_template(
        "documents.html",
        documents=documents,
        total_docs=total_docs,
        total_pages=total_pages,
        total_words=total_words,
    )


@main_bp.route("/document/<int:doc_id>/reprocess-ocr", methods=["POST"])
def reprocess_document_ocr(doc_id: int):
    """
    Re-applies Tesseract OCR to an existing scanned document.

    Useful for documents that were uploaded before OCR was available,
    or when OCR previously failed due to Tesseract not being installed.

    Returns JSON: { success, message, total_words, extraction_method }
    """
    document = get_document(doc_id)
    if not document:
        return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404

    file_path = document.get("file_path", "")
    if not file_path or not Path(file_path).is_file():
        return jsonify({
            "success": False,
            "error": "PDF file not found on disk. The document may have been deleted.",
        }), 400

    # Check Tesseract availability first
    tess_info = check_tesseract_available()
    if not tess_info["available"]:
        return jsonify({
            "success": False,
            "error": "Tesseract OCR is not installed on this server.",
            "install_instructions": tess_info.get("install_instructions"),
        }), 503

    # Get existing pages (for mixed-mode: preserve normal pages, OCR the rest)
    existing_pages = get_document_pages(doc_id)
    # Normalize key: get_document_pages returns "extracted_text" key
    normalized_pages = [
        {
            "page_number": p["page_number"],
            "text": p.get("extracted_text", "") or p.get("text", ""),
        }
        for p in existing_pages
    ]

    ocr_result = extract_text_via_ocr(
        file_path,
        existing_pages=normalized_pages,
    )

    if not ocr_result["success"]:
        return jsonify({
            "success": False,
            "error": ocr_result.get("error") or "OCR processing failed.",
        }), 500

    total_words = ocr_result["total_words"]
    extraction_method = ocr_result.get("extraction_method", "ocr")
    ocr_applied = ocr_result.get("ocr_applied", True)
    ocr_engine = ocr_result.get("ocr_engine", "tesseract")
    ocr_confidence = ocr_result.get("ocr_confidence")

    if total_words == 0:
        new_status = "scanned_no_text"
        message = "OCR completed but no readable text was found. The document may be blank or damaged."
    else:
        new_status = "ocr_completed"
        message = f"OCR completed successfully. Extracted {total_words:,} words."

    # Replace page texts in DB with OCR results
    insert_document_pages(
        document_id=doc_id,
        pages_data=ocr_result["pages"],
        replace_existing=True,
    )

    # Update document metadata
    update_document_ocr_metadata(
        document_id=doc_id,
        total_words=total_words,
        processing_status=new_status,
        ocr_applied=ocr_applied,
        ocr_engine=ocr_engine,
        ocr_confidence=ocr_confidence,
        extraction_method=extraction_method,
    )

    # Invalidate stale RAG chunks and vector index so new OCR text is indexed on next chat
    if total_words > 0:
        delete_document_chunks(doc_id)
        try:
            vs = get_vector_store()
            vs.delete_document_index(doc_id)
        except Exception as vs_err:
            current_app.logger.debug("Vector store reset after OCR: %s", vs_err)

    # Log activity
    try:
        log_activity(
            activity_type="ocr_reprocess",
            title=f"OCR Reprocessed '{document['original_filename']}'",
            description=f"Extracted {total_words:,} words via {ocr_engine} OCR.",
            document_id=doc_id,
            metadata={
                "total_words": total_words,
                "extraction_method": extraction_method,
                "ocr_confidence": ocr_confidence,
            },
        )
    except Exception as log_err:
        current_app.logger.warning("Failed to log OCR activity: %s", log_err)

    return jsonify({
        "success": True,
        "message": message,
        "total_words": total_words,
        "extraction_method": extraction_method,
        "processing_status": new_status,
        "ocr_confidence": ocr_confidence,
    }), 200


@main_bp.route("/document/<int:doc_id>")
def document_details(doc_id: int):
    """
    Displays metadata, page-by-page extracted text, and AI summarization for a specific document.
    """
    document = get_document(doc_id)
    if not document:
        if is_json_request():
            return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404
        flash(f"Document with ID {doc_id} not found.", "error")
        return redirect(url_for("main.documents_list")), 404

    pages = get_document_pages(doc_id)
    requested_type = request.args.get("summary_type")
    current_summary = get_summary(doc_id, summary_type=requested_type)
    all_summaries = get_all_summaries_for_document(doc_id)

    return render_template(
        "document_details.html",
        document=document,
        pages=pages,
        current_summary=current_summary,
        all_summaries=all_summaries,
    )


@main_bp.route("/document/<int:doc_id>/text")
def document_text(doc_id: int):
    """
    Returns the extracted text for a document.
    Renders a text viewer template or outputs JSON if requested.
    """
    document = get_document(doc_id)
    if not document:
        if is_json_request():
            return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404
        abort(404)

    pages = get_document_pages(doc_id)
    full_text = "\n\n".join(
        [f"--- Page {p['page_number']} ---\n{p['extracted_text']}" for p in pages]
    )

    # Format parameter or JSON header
    req_format = request.args.get("format", "").lower()
    if is_json_request() or req_format == "json":
        return jsonify({
            "success": True,
            "document": {
                "id": document["id"],
                "original_filename": document["original_filename"],
                "total_pages": document["total_pages"],
                "total_words": document["total_words"],
                "processing_status": document["processing_status"],
            },
            "pages": pages,
            "full_text": full_text,
        })

    # Render dedicated reading text view
    return render_template(
        "document_text.html",
        document=document,
        pages=pages,
        full_text=full_text,
    )


@main_bp.route("/document/<int:doc_id>/summarize", methods=["POST"])
def summarize_document(doc_id: int):
    """
    Generates AI-powered summaries, extracted concepts, keywords, and exam points for a document.
    Supports Quick, Standard, and Detailed summary types.
    """
    # 1. Verify document exists
    document = get_document(doc_id)
    if not document:
        error_msg = f"Document ID {doc_id} not found."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 404
        flash(error_msg, "error")
        return redirect(url_for("main.documents_list")), 404

    # 2. Check for scanned or empty documents
    if document.get("processing_status") == "scanned_no_text" or document.get("total_words", 0) == 0:
        error_msg = (
            "This document contains scanned images or non-extractable text. "
            "AI summarization cannot be performed on empty or image-based PDFs without OCR. "
            "Please run OCR text extraction on this document first."
        )
        if is_json_request():
            return jsonify({
                "success": False,
                "error": error_msg,
                "is_scanned": True,
                "requires_ocr": True,
            }), 400
        flash(error_msg, "warning")
        return redirect(url_for("main.document_details", doc_id=doc_id)), 400

    # 3. Parse requested summary type
    summary_type = "standard"
    if request.is_json:
        data = request.get_json(silent=True) or {}
        summary_type = data.get("summary_type", "standard")
    elif request.form:
        summary_type = request.form.get("summary_type", "standard")

    summary_type = (summary_type or "standard").lower().strip()
    if summary_type not in ("quick", "standard", "detailed"):
        summary_type = "standard"

    # 4. Fetch extracted pages
    pages = get_document_pages(doc_id)
    if not pages and document.get("file_path"):
        try:
            from utils.pdf_processor import extract_text_from_pdf
            ext_res = extract_text_from_pdf(document["file_path"])
            if ext_res.get("success") and ext_res.get("pages"):
                insert_document_pages(doc_id, ext_res["pages"])
                pages = get_document_pages(doc_id)
        except Exception as ext_err:
            current_app.logger.warning(f"Could not re-extract pages for doc {doc_id}: {ext_err}")

    if not pages:
        error_msg = "No extracted text found for this document to summarize."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("main.document_details", doc_id=doc_id)), 400

    # 5. Execute AI Summarization Engine
    model_override = current_app.config.get("SUMMARIZATION_MODEL")
    try:
        summary_result = generate_study_summary(
            content=pages,
            summary_type=summary_type,
            document_metadata=document,
            model_name=model_override,
        )
    except SummarizationError as e:
        error_msg = f"Summarization failed: {e!s}"
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 500
        flash(error_msg, "error")
        return redirect(url_for("main.document_details", doc_id=doc_id)), 500
    except Exception as e:
        error_msg = f"Unexpected error during summarization: {e!s}"
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 500
        flash(error_msg, "error")
        return redirect(url_for("main.document_details", doc_id=doc_id)), 500

    if not summary_result.get("success"):
        error_msg = summary_result.get("error") or "Failed to generate summary."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "warning")
        return redirect(url_for("main.document_details", doc_id=doc_id)), 400

    # 6. Save generated summary to database
    summary_id = None
    try:
        summary_id = save_summary(
            document_id=doc_id,
            summary_type=summary_type,
            summary_text=summary_result["summary_text"],
            chapter_summaries=summary_result.get("chapter_summaries"),
            key_concepts=summary_result.get("key_concepts"),
            keywords=summary_result.get("keywords"),
            exam_points=summary_result.get("exam_points"),
            topics=summary_result.get("topics"),
            model_name=summary_result.get("model_name", "unknown"),
        )
    except Exception as e:
        current_app.logger.error(f"Failed to save summary to database: {e!s}")

    # Phase 6: Log student activity
    try:
        w_count = summary_result.get("stats", {}).get("word_count") or len((summary_result.get("summary_text") or "").split())
        log_activity(
            activity_type="summarize",
            title=f"Generated {summary_type.capitalize()} Summary",
            description=f"Created {summary_type} study summary for '{document['original_filename']}' ({w_count} words).",
            document_id=doc_id,
            metadata={"summary_type": summary_type, "word_count": w_count, "summary_id": summary_id},
        )
    except Exception as log_err:
        current_app.logger.warning(f"Failed to log summary activity: {log_err}")

    success_msg = f"{summary_type.capitalize()} summary generated successfully!"

    if is_json_request():
        return jsonify({
            "success": True,
            "message": success_msg,
            "document_id": doc_id,
            "summary_id": summary_id,
            "summary_type": summary_type,
            "summary_text": summary_result["summary_text"],
            "chapter_summaries": summary_result.get("chapter_summaries", []),
            "key_concepts": summary_result.get("key_concepts", []),
            "keywords": summary_result.get("keywords", []),
            "exam_points": summary_result.get("exam_points", []),
            "topics": summary_result.get("topics", []),
            "stats": summary_result.get("stats", {}),
            "model_name": summary_result.get("model_name"),
        }), 200

    flash(success_msg, "success")
    return redirect(url_for("main.document_details", doc_id=doc_id, summary_type=summary_type))


@main_bp.route("/document/<int:doc_id>/summary")
def get_document_summary_route(doc_id: int):
    """
    Returns the saved summary for a document in JSON format.
    Accepts optional ?type=quick|standard|detailed query parameter.
    """
    document = get_document(doc_id)
    if not document:
        return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404

    summary_type = request.args.get("type")
    summary = get_summary(doc_id, summary_type=summary_type)

    if not summary:
        return jsonify({
            "success": False,
            "error": "No summary generated yet for this document.",
            "document_id": doc_id,
        }), 404

    summary_words = len((summary.get("summary_text") or "").split())
    doc_words = document.get("total_words", 0)
    red_pct = round((1.0 - (summary_words / max(doc_words, 1))) * 100, 1) if doc_words > summary_words else 0.0

    return jsonify({
        "success": True,
        "document_id": doc_id,
        "summary": summary,
        "summary_id": summary.get("id"),
        "summary_type": summary.get("summary_type"),
        "summary_text": summary.get("summary_text"),
        "chapter_summaries": summary.get("chapter_summaries") or [],
        "key_concepts": summary.get("key_concepts") or [],
        "keywords": summary.get("keywords") or [],
        "exam_points": summary.get("exam_points") or [],
        "topics": summary.get("topics") or [],
        "model_name": summary.get("model_name"),
        "stats": {
            "original_words": doc_words,
            "summary_words": summary_words,
            "reduction_percentage": max(0.0, red_pct),
        },
    }), 200


@main_bp.route("/document/<int:doc_id>/summary/delete", methods=["POST"])
def delete_summary_route(doc_id: int):
    """
    Deletes the saved summary for a document, allowing regeneration.
    """
    document = get_document(doc_id)
    if not document:
        error_msg = f"Document ID {doc_id} not found."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 404
        flash(error_msg, "error")
        return redirect(url_for("main.documents_list")), 404

    summary_type = request.form.get("summary_type") or request.args.get("summary_type")
    deleted_count = delete_summaries_for_document(doc_id, summary_type=summary_type)

    msg = f"Summary reset successfully ({deleted_count} record(s) removed)."
    if is_json_request():
        return jsonify({"success": True, "message": msg, "deleted_count": deleted_count})

    flash(msg, "success")
    return redirect(url_for("main.document_details", doc_id=doc_id))


@main_bp.route("/document/<int:doc_id>/delete", methods=["POST"])
def delete_document_route(doc_id: int):
    """
    Deletes a document, its database records, stored file from disk,
    associated vector store index, and chat conversation history.
    """
    file_path = delete_document(doc_id)
    if file_path is None:
        error_msg = f"Document with ID {doc_id} does not exist."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 404
        flash(error_msg, "error")
        return redirect(url_for("main.documents_list")), 404

    # Delete physical file from uploads folder securely
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        pass

    # Clean up vector database index files and database chunks
    try:
        vs = get_vector_store()
        vs.delete_document_index(doc_id)
    except Exception as e:
        current_app.logger.warning(f"Error removing vector index for document {doc_id}: {e}")

    try:
        delete_document_chunks(doc_id)
        delete_chat_for_document(doc_id)
        delete_quizzes_for_document(doc_id)
    except Exception as e:
        current_app.logger.warning(f"Error removing chat/chunks/quizzes for document {doc_id}: {e}")

    success_msg = f"Document ID #{doc_id} was successfully deleted."
    if is_json_request():
        return jsonify({"success": True, "message": success_msg, "deleted_id": doc_id})

    flash(success_msg, "success")
    return redirect(url_for("main.documents_list"))


# ------------------------------------------------------------------------------
# Phase 4: AI Chatbot & RAG Routes
# ------------------------------------------------------------------------------

@main_bp.route("/document/<int:doc_id>/chat")
def document_chat(doc_id: int):
    """
    Renders the dedicated AI Study Chatbot interface for a specific document.
    Ensures the document is indexed in the vector store and loads conversation history.
    """
    document = get_document(doc_id)
    if not document:
        if is_json_request():
            return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404
        flash(f"Document with ID {doc_id} not found.", "error")
        return redirect(url_for("main.documents_list")), 404

    # Verify if document has extractable text
    if document.get("processing_status") == "scanned_no_text" or document.get("total_words", 0) == 0:
        error_msg = (
            "This document appears to be a scanned PDF or contains no extractable text. "
            "RAG Study Chatbot requires extractable text to build semantic indices."
        )
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "warning")
        return redirect(url_for("main.document_details", doc_id=doc_id))

    # Ensure document is indexed for RAG
    vs = get_vector_store()
    indexing_status = "already_indexed"
    total_chunks = 0
    try:
        if not vs.is_document_indexed(doc_id):
            idx_res = index_document_for_rag(
                doc_id,
                chunk_size_words=current_app.config.get("RAG_CHUNK_SIZE_WORDS", 120),
                overlap_words=current_app.config.get("RAG_CHUNK_OVERLAP_WORDS", 30),
            )
            indexing_status = idx_res.get("status", "indexed")
            total_chunks = idx_res.get("total_chunks", 0)
        else:
            chunks = get_document_chunks(doc_id)
            total_chunks = len(chunks)
    except Exception as e:
        current_app.logger.error(f"Error ensuring index for document {doc_id}: {e}")

    # Fetch active conversation and message history
    conv = get_or_create_conversation(doc_id)
    messages = get_chat_history(conv["id"])

    # Suggested starter questions
    starter_questions = [
        "What are the main topics and concepts covered in this document?",
        "Can you explain the key principles discussed here?",
        "What are the most important points for exam preparation?",
        "Can you provide a brief overview of this study material?",
    ]

    return render_template(
        "chatbot.html",
        document=document,
        conversation=conv,
        chat_history=messages,
        total_chunks=total_chunks,
        indexing_status=indexing_status,
        starter_questions=starter_questions,
    )


@main_bp.route("/document/<int:doc_id>/chat/ask", methods=["POST"])
def ask_chat_question(doc_id: int):
    """
    RAG API endpoint: Accepts user study questions, performs semantic search in FAISS,
    generates grounded answers with Flan-T5, tracks conversation history, and returns citations.
    """
    # 1. Validate document exists
    document = get_document(doc_id)
    if not document:
        return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404

    if document.get("processing_status") == "scanned_no_text" or document.get("total_words", 0) == 0:
        return jsonify({
            "success": False,
            "error": "This PDF is scanned or contains no extractable text. Chatbot cannot answer questions.",
        }), 400

    # 2. Extract and validate question from request
    question = ""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "")
    elif request.form:
        question = request.form.get("question", "")

    question = (question or "").strip()
    if not question:
        return jsonify({
            "success": False,
            "error": "Please provide a valid question about your study material.",
        }), 400

    # 3. Optional query configuration parameters
    top_k = current_app.config.get("RAG_TOP_K", 4)
    threshold = current_app.config.get("RAG_SIMILARITY_THRESHOLD", 0.28)

    # 4. Execute RAG Question Answering pipeline
    try:
        rag_response = ask_document_question(
            document_id=doc_id,
            question=question,
            top_k=top_k,
            threshold=threshold,
            save_to_history=True,
        )

        # Phase 6: Log student activity
        try:
            q_snippet = question[:60] + "..." if len(question) > 60 else question
            log_activity(
                activity_type="chat",
                title="Asked Study Chatbot",
                description=f"Question on '{rag_response['filename']}': \"{q_snippet}\"",
                document_id=doc_id,
                metadata={"question": q_snippet, "is_grounded": rag_response.get("is_grounded", True)},
            )
        except Exception as log_err:
            current_app.logger.warning(f"Failed to log chat activity: {log_err}")

        return jsonify({
            "success": True,
            "document_id": doc_id,
            "filename": rag_response["filename"],
            "question": rag_response["question"],
            "answer": rag_response["answer"],
            "sources": rag_response.get("sources", []),
            "suggested_questions": rag_response.get("suggested_questions", []),
            "is_grounded": rag_response.get("is_grounded", True),
            "model_name": rag_response.get("model_name"),
            "conversation_id": rag_response.get("conversation_id"),
        }), 200

    except ChatbotError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Unexpected error in ask_chat_question: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to generate answer: {e!s}",
        }), 500


@main_bp.route("/document/<int:doc_id>/chat/history")
def get_chat_history_route(doc_id: int):
    """
    Returns the message history for a document's active conversation in JSON format.
    """
    document = get_document(doc_id)
    if not document:
        return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404

    conv = get_or_create_conversation(doc_id)
    messages = get_chat_history(conv["id"])

    return jsonify({
        "success": True,
        "document_id": doc_id,
        "conversation_id": conv["id"],
        "total_messages": len(messages),
        "messages": messages,
    }), 200


@main_bp.route("/document/<int:doc_id>/chat/clear", methods=["POST"])
def clear_chat_history_route(doc_id: int):
    """
    Clears all conversational messages for the document's active conversation.
    """
    document = get_document(doc_id)
    if not document:
        return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404

    conv = get_or_create_conversation(doc_id)
    deleted_count = clear_chat_history(conv["id"])

    return jsonify({
        "success": True,
        "document_id": doc_id,
        "conversation_id": conv["id"],
        "message": f"Conversation cleared successfully ({deleted_count} messages deleted).",
        "deleted_count": deleted_count,
    }), 200


@main_bp.route("/document/<int:doc_id>/index", methods=["POST"])
def index_document_route(doc_id: int):
    """
    Explicitly creates or forces reindexing of document chunks in the FAISS vector database.
    """
    document = get_document(doc_id)
    if not document:
        return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404

    force = False
    if request.is_json:
        data = request.get_json(silent=True) or {}
        force = bool(data.get("force_reindex", False))

    try:
        result = index_document_for_rag(
            document_id=doc_id,
            chunk_size_words=current_app.config.get("RAG_CHUNK_SIZE_WORDS", 120),
            overlap_words=current_app.config.get("RAG_CHUNK_OVERLAP_WORDS", 30),
            force_reindex=force,
        )
        return jsonify({
            "success": True,
            "status": result.get("status"),
            "total_chunks": result.get("total_chunks", 0),
            "document_id": doc_id,
            "filename": result.get("filename"),
            "message": result.get("message"),
            "result": result,
        }), 200
    except ChatbotError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Indexing failed: {e!s}"}), 500


# ------------------------------------------------------------------------------
# Phase 5: AI Quiz Generation & Assessment Routes
# ------------------------------------------------------------------------------

@main_bp.route("/quizzes")
def quizzes_hub():
    """
    Renders the Quiz Hub / Dashboard displaying available documents,
    previously generated quizzes, and recent quiz attempts.
    """
    documents = get_all_documents()
    quizzes = get_all_quizzes()
    recent_attempts = get_quiz_attempts(limit=10)

    if is_json_request():
        return jsonify({
            "success": True,
            "documents": documents,
            "total_quizzes": len(quizzes),
            "quizzes": quizzes,
            "recent_attempts": recent_attempts,
        })

    return render_template(
        "quiz_config.html",
        documents=documents,
        quizzes=quizzes,
        recent_attempts=recent_attempts,
        selected_doc_id=None,
    )


@main_bp.route("/document/<int:doc_id>/quiz/new")
def new_quiz_config(doc_id: int):
    """
    Renders the Quiz Configuration page pre-selected for a specific document.
    """
    document = get_document(doc_id)
    if not document:
        if is_json_request():
            return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404
        flash(f"Document with ID {doc_id} not found.", "error")
        return redirect(url_for("main.documents_list")), 404

    if document.get("processing_status") == "scanned_no_text" or document.get("total_words", 0) == 0:
        error_msg = "This document appears to be a scanned image or has no extractable text. Cannot generate a quiz."
        if is_json_request():
            return jsonify({"success": False, "error": error_msg}), 400
        flash(error_msg, "warning")
        return redirect(url_for("main.document_details", doc_id=doc_id))

    documents = get_all_documents()
    doc_quizzes = get_all_quizzes(document_id=doc_id)
    doc_attempts = get_quiz_attempts(document_id=doc_id, limit=5)

    if is_json_request():
        return jsonify({
            "success": True,
            "document": document,
            "quizzes": doc_quizzes,
            "attempts": doc_attempts,
        })

    return render_template(
        "quiz_config.html",
        documents=documents,
        document=document,
        selected_doc_id=doc_id,
        quizzes=doc_quizzes,
        recent_attempts=doc_attempts,
    )


@main_bp.route("/document/<int:doc_id>/quiz/generate", methods=["POST"])
def generate_quiz_route(doc_id: int):
    """
    API endpoint: Generates an AI-powered quiz from the specified PDF document
    and saves the quiz and its questions to the database.
    """
    document = get_document(doc_id)
    if not document:
        return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404

    if document.get("processing_status") == "scanned_no_text" or document.get("total_words", 0) == 0:
        return jsonify({
            "success": False,
            "error": "This document contains scanned images or no extractable text. Cannot generate quiz.",
        }), 400

    # Parse request parameters
    data = request.get_json(silent=True) or request.form or {}
    difficulty = str(data.get("difficulty", "medium")).lower().strip()
    num_questions = int(data.get("num_questions", 5))
    custom_title = str(data.get("title", "")).strip()
    time_limit_minutes = int(data.get("time_limit_minutes", 0))

    try:
        quiz_data = generate_quiz_for_document(
            document_id=doc_id,
            num_questions=num_questions,
            difficulty=difficulty,
            custom_title=custom_title or None,
            time_limit_minutes=time_limit_minutes,
        )

        # Save to database
        quiz_id = create_quiz(
            document_id=doc_id,
            title=quiz_data["title"],
            difficulty=quiz_data["difficulty"],
            total_questions=quiz_data["total_questions"],
            time_limit_minutes=quiz_data["time_limit_minutes"],
        )

        add_quiz_questions(quiz_id, quiz_data["questions"])

        # Phase 6: Log student activity
        try:
            log_activity(
                activity_type="quiz_generate",
                title=f"Generated Quiz: {quiz_data['title']}",
                description=f"Created {quiz_data['total_questions']}-question {quiz_data['difficulty'].capitalize()} quiz for '{document['original_filename']}'.",
                document_id=doc_id,
                metadata={"quiz_id": quiz_id, "difficulty": quiz_data["difficulty"], "total_questions": quiz_data["total_questions"]},
            )
        except Exception as log_err:
            current_app.logger.warning(f"Failed to log quiz generation activity: {log_err}")

        success_msg = f"Quiz '{quiz_data['title']}' generated successfully with {quiz_data['total_questions']} questions!"

        if is_json_request():
            return jsonify({
                "success": True,
                "message": success_msg,
                "quiz_id": quiz_id,
                "document_id": doc_id,
                "title": quiz_data["title"],
                "difficulty": quiz_data["difficulty"],
                "total_questions": quiz_data["total_questions"],
                "time_limit_minutes": quiz_data["time_limit_minutes"],
                "redirect_url": url_for("main.take_quiz_view", quiz_id=quiz_id),
            }), 201

        flash(success_msg, "success")
        return redirect(url_for("main.take_quiz_view", quiz_id=quiz_id))

    except QuizGenerationError as e:
        if is_json_request():
            return jsonify({"success": False, "error": str(e)}), 400
        flash(f"Quiz generation error: {e!s}", "error")
        return redirect(url_for("main.new_quiz_config", doc_id=doc_id))
    except Exception as e:
        current_app.logger.error(f"Unexpected error generating quiz: {e}")
        if is_json_request():
            return jsonify({"success": False, "error": f"Failed to generate quiz: {e!s}"}), 500
        flash(f"Failed to generate quiz: {e!s}", "error")
        return redirect(url_for("main.new_quiz_config", doc_id=doc_id))


@main_bp.route("/quiz/<int:quiz_id>")
def take_quiz_view(quiz_id: int):
    """
    Renders the interactive quiz-taking interface.
    Passes questions WITHOUT the correct answer keys for client-side security.
    """
    quiz = get_quiz(quiz_id)
    if not quiz:
        if is_json_request():
            return jsonify({"success": False, "error": f"Quiz ID {quiz_id} not found."}), 404
        flash(f"Quiz ID {quiz_id} not found.", "error")
        return redirect(url_for("main.quizzes_hub")), 404

    # Security: Retrieve questions with answers stripped out
    questions = get_quiz_questions(quiz_id, include_answers=False)
    if not questions:
        flash("This quiz contains no questions.", "warning")
        return redirect(url_for("main.quizzes_hub"))

    return render_template(
        "take_quiz.html",
        quiz=quiz,
        questions=questions,
        total_questions=len(questions),
    )


@main_bp.route("/api/quiz/<int:quiz_id>/questions")
def api_quiz_questions(quiz_id: int):
    """
    JSON API endpoint: Returns quiz questions for active quiz taking.
    CRITICAL SECURITY: include_answers=False so students cannot inspect correct answers.
    """
    quiz = get_quiz(quiz_id)
    if not quiz:
        return jsonify({"success": False, "error": f"Quiz ID {quiz_id} not found."}), 404

    questions = get_quiz_questions(quiz_id, include_answers=False)
    return jsonify({
        "success": True,
        "quiz": {
            "id": quiz["id"],
            "title": quiz["title"],
            "difficulty": quiz["difficulty"],
            "total_questions": len(questions),
            "time_limit_minutes": quiz.get("time_limit_minutes", 0),
            "original_filename": quiz.get("original_filename"),
        },
        "questions": questions,
    }), 200


@main_bp.route("/api/quiz/<int:quiz_id>/submit", methods=["POST"])
def submit_quiz_route(quiz_id: int):
    """
    JSON API endpoint: Receives student's answers, evaluates them on the backend,
    saves the attempt to the database, and returns the result summary.
    """
    quiz = get_quiz(quiz_id)
    if not quiz:
        return jsonify({"success": False, "error": f"Quiz ID {quiz_id} not found."}), 404

    # Retrieve authoritative questions WITH answer keys from backend
    full_questions = get_quiz_questions(quiz_id, include_answers=True)
    if not full_questions:
        return jsonify({"success": False, "error": "This quiz contains no questions to evaluate."}), 400

    data = request.get_json(silent=True) or request.form or {}
    raw_submitted = data.get("answers") or data.get("user_answers") or {}
    if isinstance(raw_submitted, dict):
        submitted_answers: dict[str, Any] = dict(raw_submitted)
    elif isinstance(raw_submitted, str):
        try:
            parsed = json.loads(raw_submitted)
            submitted_answers = dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            submitted_answers = {}
    else:
        submitted_answers = {}
    time_taken_seconds = int(data.get("time_taken_seconds") or data.get("time_spent_seconds") or 0)

    try:
        evaluation = evaluate_quiz_submission(
            quiz_questions=full_questions,
            submitted_answers=submitted_answers,
            time_taken_seconds=time_taken_seconds,
            original_filename=quiz.get("original_filename", ""),
        )

        # Save attempt in database
        attempt_id = save_quiz_attempt(
            quiz_id=quiz_id,
            document_id=quiz["document_id"],
            total_questions=evaluation["total_questions"],
            correct_answers=evaluation["correct_answers"],
            incorrect_answers=evaluation["incorrect_answers"],
            unanswered_count=evaluation["unanswered_count"],
            score=evaluation["score"],
            percentage=evaluation["percentage"],
            passed=evaluation["passed"],
            time_taken_seconds=evaluation["time_taken_seconds"],
            answers=submitted_answers,
        )

        # Phase 6: Log student activity
        try:
            log_activity(
                activity_type="quiz_attempt",
                title=f"Completed Quiz: {quiz['title']}",
                description=f"Scored {evaluation['score']}/{evaluation['total_questions']} ({evaluation['percentage']}%) - {'Passed' if evaluation['passed'] else 'Needs Review'}.",
                document_id=quiz["document_id"],
                metadata={
                    "attempt_id": attempt_id,
                    "quiz_id": quiz_id,
                    "score": evaluation["score"],
                    "total_questions": evaluation["total_questions"],
                    "percentage": evaluation["percentage"],
                    "passed": bool(evaluation["passed"]),
                },
            )
        except Exception as log_err:
            current_app.logger.warning(f"Failed to log quiz attempt activity: {log_err}")

        return jsonify({
            "success": True,
            "attempt_id": attempt_id,
            "quiz_id": quiz_id,
            "score": evaluation["score"],
            "total_questions": evaluation["total_questions"],
            "percentage": evaluation["percentage"],
            "passed": evaluation["passed"],
            "correct_answers": evaluation["correct_answers"],
            "incorrect_answers": evaluation["incorrect_answers"],
            "unanswered_count": evaluation["unanswered_count"],
            "time_taken_formatted": evaluation["time_taken_formatted"],
            "evaluation": evaluation,
            "redirect_url": url_for("main.quiz_results_view", attempt_id=attempt_id),
        }), 200

    except QuizEvaluationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error evaluating quiz submission: {e}")
        return jsonify({"success": False, "error": f"Failed to grade quiz: {e!s}"}), 500


@main_bp.route("/quiz/attempt/<int:attempt_id>")
def quiz_results_view(attempt_id: int):
    """
    Renders the rich results page for a completed quiz attempt,
    including score breakdown, circular chart, and question-by-question review.
    """
    attempt = get_quiz_attempt(attempt_id)
    if not attempt:
        if is_json_request():
            return jsonify({"success": False, "error": f"Attempt ID {attempt_id} not found."}), 404
        flash(f"Attempt ID {attempt_id} not found.", "error")
        return redirect(url_for("main.quizzes_hub")), 404

    # Fetch quiz and full questions for review
    quiz_id = attempt["quiz_id"]
    quiz = get_quiz(quiz_id)
    full_questions = get_quiz_questions(quiz_id, include_answers=True)

    evaluation = evaluate_quiz_submission(
        quiz_questions=full_questions,
        submitted_answers=attempt.get("answers", {}),
        time_taken_seconds=attempt.get("time_taken_seconds", 0),
        original_filename=attempt.get("original_filename", ""),
    )

    if is_json_request():
        return jsonify({
            "success": True,
            "attempt": attempt,
            "evaluation": evaluation,
        }), 200

    return render_template(
        "quiz_results.html",
        attempt=attempt,
        quiz=quiz,
        evaluation=evaluation,
    )


@main_bp.route("/quizzes/history")
def quiz_history_view():
    """
    Renders the global or filtered quiz attempt history page.
    """
    doc_id_arg = request.args.get("document_id")
    document_id = int(doc_id_arg) if doc_id_arg and doc_id_arg.isdigit() else None

    attempts = get_quiz_attempts(document_id=document_id)
    documents = get_all_documents()
    selected_doc = get_document(document_id) if document_id else None

    if is_json_request():
        return jsonify({
            "success": True,
            "total_attempts": len(attempts),
            "attempts": attempts,
        }), 200

    return render_template(
        "quiz_history.html",
        attempts=attempts,
        documents=documents,
        selected_doc=selected_doc,
    )


@main_bp.route("/document/<int:doc_id>/quizzes")
def document_quizzes(doc_id: int):
    """
    Returns or displays quizzes specifically for document <doc_id>.
    """
    document = get_document(doc_id)
    if not document:
        if is_json_request():
            return jsonify({"success": False, "error": f"Document ID {doc_id} not found."}), 404
        flash(f"Document ID {doc_id} not found.", "error")
        return redirect(url_for("main.documents_list")), 404

    quizzes = get_all_quizzes(document_id=doc_id)
    attempts = get_quiz_attempts(document_id=doc_id)

    if is_json_request():
        return jsonify({
            "success": True,
            "document_id": doc_id,
            "quizzes": quizzes,
            "attempts": attempts,
        }), 200

    return render_template(
        "quiz_config.html",
        document=document,
        selected_doc_id=doc_id,
        quizzes=quizzes,
        recent_attempts=attempts,
        documents=get_all_documents(),
    )


@main_bp.route("/quiz/<int:quiz_id>/delete", methods=["POST"])
def delete_quiz_route(quiz_id: int):
    """
    Deletes a quiz and all associated attempts.
    """
    quiz = get_quiz(quiz_id)
    if not quiz:
        if is_json_request():
            return jsonify({"success": False, "error": f"Quiz ID {quiz_id} not found."}), 404
        flash(f"Quiz ID {quiz_id} not found.", "error")
        return redirect(url_for("main.quizzes_hub")), 404

    deleted = delete_quiz(quiz_id)
    msg = f"Quiz '{quiz['title']}' deleted successfully."
    if is_json_request():
        return jsonify({"success": deleted, "message": msg, "quiz_id": quiz_id}), 200

    flash(msg, "success")
    return redirect(url_for("main.quizzes_hub"))


# ------------------------------------------------------------------------------
# Phase 6: Dashboard, Performance Analytics & Study Goals Routes
# ------------------------------------------------------------------------------

@main_bp.route("/api/dashboard/stats")
def api_dashboard_stats():
    """Returns real-time aggregated dashboard statistics in JSON format."""
    stats = get_dashboard_statistics()
    return jsonify({
        "success": True,
        "statistics": stats,
    }), 200


@main_bp.route("/analytics")
@main_bp.route("/progress")
def analytics_view():
    """
    Renders the dedicated Progress & Performance Analytics page.
    Supplies weekly activity, quiz trends, difficulty distribution, and overall progress index.
    """
    analytics = get_performance_analytics()
    documents = get_all_documents()

    if is_json_request():
        return jsonify({
            "success": True,
            "analytics": analytics,
        }), 200

    return render_template(
        "analytics.html",
        analytics=analytics,
        documents=documents,
    )


@main_bp.route("/api/analytics/data")
def api_analytics_data():
    """Returns analytics data (weekly study, quiz trends, difficulty breakdown) in JSON."""
    analytics = get_performance_analytics()
    return jsonify({
        "success": True,
        **analytics,
    }), 200


@main_bp.route("/goals")
def study_goals_view():
    """Renders the Study Goals management interface."""
    in_progress_goals = get_study_goals(status="in_progress")
    completed_goals = get_study_goals(status="completed")
    stats = get_dashboard_statistics()

    if is_json_request():
        return jsonify({
            "success": True,
            "in_progress_goals": in_progress_goals,
            "completed_goals": completed_goals,
            "total_goals": len(in_progress_goals) + len(completed_goals),
        }), 200

    return render_template(
        "goals.html",
        in_progress_goals=in_progress_goals,
        completed_goals=completed_goals,
        stats=stats,
    )


@main_bp.route("/api/goals")
def api_get_goals():
    """JSON API to fetch all study goals with calculated progress."""
    goals = get_study_goals()
    return jsonify({
        "success": True,
        "goals": goals,
        "total_goals": len(goals),
    }), 200


@main_bp.route("/api/goals/create", methods=["POST"])
def api_create_goal():
    """Creates a new study goal."""
    data = request.get_json(silent=True) or request.form or {}
    title = str(data.get("title", "")).strip()
    goal_type = str(data.get("goal_type", "")).strip().lower()
    target_date = data.get("target_date")

    if not title:
        return jsonify({"success": False, "error": "Goal title is required."}), 400
    if not goal_type:
        return jsonify({"success": False, "error": "Goal type is required."}), 400
    raw_target = data.get("target_value")
    if raw_target is None:
        return jsonify({"success": False, "error": "Target value must be a positive number."}), 400
    try:
        parsed_target = float(raw_target)
        if parsed_target <= 0:
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Target value must be a positive number."}), 400

    goal_id = create_study_goal(
        title=title,
        goal_type=goal_type,
        target_value=parsed_target,
        target_date=target_date or None,
    )

    return jsonify({
        "success": True,
        "goal_id": goal_id,
        "message": f"Study goal '{title}' created successfully!",
    }), 201


@main_bp.route("/api/goals/<int:goal_id>/update", methods=["POST"])
def api_update_goal(goal_id: int):
    """Updates status or properties of an existing study goal."""
    goal = get_study_goal(goal_id)
    if not goal:
        return jsonify({"success": False, "error": f"Goal ID {goal_id} not found."}), 404

    data = request.get_json(silent=True) or request.form or {}
    status = str(data["status"]) if "status" in data and data["status"] is not None else None
    title = str(data["title"]).strip() if "title" in data and data["title"] is not None else None
    target_value = data.get("target_value")
    parsed_target: float | None = None
    if target_value is not None:
        try:
            parsed_target = float(target_value)
        except (TypeError, ValueError):
            pass
    target_date = str(data["target_date"]) if "target_date" in data and data["target_date"] is not None else None

    success = update_study_goal(
        goal_id=goal_id,
        title=title,
        target_value=parsed_target,
        target_date=target_date,
        status=status,
    )
    return jsonify({
        "success": success,
        "message": "Goal updated successfully.",
    }), 200


@main_bp.route("/api/goals/<int:goal_id>/delete", methods=["POST"])
def api_delete_goal(goal_id: int):
    """Deletes a study goal by ID."""
    goal = get_study_goal(goal_id)
    if not goal:
        return jsonify({"success": False, "error": f"Goal ID {goal_id} not found."}), 404

    delete_study_goal(goal_id)
    return jsonify({
        "success": True,
        "message": "Study goal deleted successfully.",
    }), 200


@main_bp.route("/activity")
@main_bp.route("/history")
def activity_history_view():
    """
    Renders the unified Activity History page.
    Supports filtering by activity type (upload, summarize, chat, quiz) and date range.
    """
    act_type = request.args.get("activity_type") or request.args.get("type", "all")
    date_from = request.args.get("from")
    date_to = request.args.get("to")
    doc_id = request.args.get("document_id")
    try:
        doc_id = int(doc_id) if doc_id else None
    except ValueError:
        doc_id = None

    activities = get_study_activities(
        limit=100,
        activity_type=act_type,
        date_from=date_from,
        date_to=date_to,
        document_id=doc_id,
    )
    total_count = get_total_activities_count(
        activity_type=act_type,
        date_from=date_from,
        date_to=date_to,
        document_id=doc_id,
    )
    documents = get_all_documents()

    if is_json_request():
        return jsonify({
            "success": True,
            "total_count": total_count,
            "activities": activities,
        }), 200

    return render_template(
        "activity.html",
        activities=activities,
        total_count=total_count,
        documents=documents,
        current_type=act_type,
        date_from=date_from,
        date_to=date_to,
        selected_doc_id=doc_id,
    )


@main_bp.route("/api/activity/history")
def api_activity_history():
    """JSON API returning filtered activity history."""
    act_type = request.args.get("activity_type") or request.args.get("type", "all")
    date_from = request.args.get("from")
    date_to = request.args.get("to")
    activities = get_study_activities(limit=100, activity_type=act_type, date_from=date_from, date_to=date_to)
    total_count = get_total_activities_count(activity_type=act_type, date_from=date_from, date_to=date_to)
    return jsonify({
        "success": True,
        "total_count": total_count,
        "activities": activities,
    }), 200


@main_bp.route("/learning-map")
def learning_map_view():
    """Renders the dedicated interactive Learning Map page."""
    documents = get_all_documents()
    stats = get_dashboard_statistics()
    analytics = get_performance_analytics()
    quizzes = get_all_quizzes()
    return render_template(
        "learning_map.html",
        documents=documents,
        stats=stats,
        analytics=analytics,
        quizzes=quizzes,
    )


@main_bp.route("/settings", methods=["GET", "POST"])
def settings_view():
    """Renders and handles updates for user profile and workspace preferences."""
    if request.method == "POST":
        try:
            if request.is_json:
                data = request.get_json() or {}
            else:
                data = request.form.to_dict()

            updated = update_student_profile(data)
            if is_json_request():
                return jsonify({
                    "success": True,
                    "message": "Student profile and preferences updated successfully!",
                    "profile": updated,
                }), 200

            flash("Student profile and preferences updated successfully!", "success")
            return redirect(url_for("main.settings_view"))
        except ValueError as val_err:
            if is_json_request():
                return jsonify({
                    "success": False,
                    "error": str(val_err),
                }), 400
            flash(str(val_err), "error")
            stats = get_dashboard_statistics()
            profile = get_student_profile()
            return render_template("settings.html", stats=stats, profile=profile), 400
        except Exception as err:
            if is_json_request():
                return jsonify({
                    "success": False,
                    "error": f"Failed to update profile: {err}",
                }), 500
            flash(f"Failed to update profile: {err}", "error")
            stats = get_dashboard_statistics()
            profile = get_student_profile()
            return render_template("settings.html", stats=stats, profile=profile), 500

    stats = get_dashboard_statistics()
    profile = get_student_profile()
    if is_json_request():
        return jsonify({
            "success": True,
            "profile": profile,
            "stats": stats,
        }), 200
    return render_template("settings.html", stats=stats, profile=profile)


@main_bp.route("/api/profile", methods=["GET", "POST"])
def api_student_profile():
    """Dedicated API endpoint for fetching and updating the student profile."""
    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form.to_dict()
        try:
            updated = update_student_profile(data or {})
            return jsonify({
                "success": True,
                "message": "Student profile updated successfully!",
                "profile": updated,
            }), 200
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to update profile: {e}"}), 500

    profile = get_student_profile()
    return jsonify({"success": True, "profile": profile}), 200


@main_bp.route("/chat")
def chat_hub_redirect():
    """Redirects to the active document chat or documents list."""
    docs = get_all_documents(limit=1)
    if docs:
        return redirect(url_for("main.document_chat", doc_id=docs[0]["id"]))
    return redirect(url_for("main.documents_list"))


@main_bp.route("/summarizer", methods=["GET", "POST"], endpoint="summarizer_view")
@main_bp.route("/summarizer", methods=["GET", "POST"], endpoint="summarizer_hub_redirect")
def summarizer_view():
    """
    Dedicated AI Summarizer workspace (Task 1 & Task 2).
    Allows students to access summarization directly from the sidebar without requiring
    a document ID in the URL. Displays available uploaded documents, allows document selection,
    mode selection (Quick, Standard, Detailed), and generates DistilBART summaries.
    """
    if request.method == "POST":
        # Handle POST direct to /summarizer
        doc_id = None
        if request.is_json:
            doc_id = request.json.get("document_id") or request.json.get("doc_id")
        elif request.form:
            doc_id = request.form.get("document_id") or request.form.get("doc_id")

        if not doc_id:
            return jsonify({"success": False, "error": "No document selected for summarization."}), 400

        try:
            doc_id_int = int(doc_id)
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": "Invalid document ID."}), 400

        return summarize_document(doc_id_int)

    # GET request - render dedicated AI Summarizer workspace
    documents = get_all_documents()
    selected_doc_id = request.args.get("doc_id", type=int)

    selected_doc = None
    if selected_doc_id:
        selected_doc = get_document(selected_doc_id)

    # If no doc selected via query param but documents exist, default to the first readable document or first document
    if not selected_doc and documents:
        selected_doc = next((d for d in documents if d.get("total_words", 0) > 0), documents[0])

    current_summary = None
    all_summaries = []
    if selected_doc:
        requested_type = request.args.get("summary_type") or request.args.get("type")
        current_summary = get_summary(selected_doc["id"], summary_type=requested_type)
        all_summaries = get_all_summaries_for_document(selected_doc["id"])

    if is_json_request():
        return jsonify({
            "success": True,
            "documents": documents,
            "selected_document": selected_doc,
            "current_summary": current_summary,
            "all_summaries": all_summaries,
        }), 200

    return render_template(
        "summarizer.html",
        documents=documents,
        selected_doc=selected_doc,
        document=selected_doc,
        current_summary=current_summary,
        all_summaries=all_summaries,
        active_page="summarizer",
    )


# Alias for backward compatibility with templates and tests
summarizer_hub_redirect = summarizer_view


@main_bp.route("/health")
def health_check():
    """
    Health check endpoint to verify backend operational readiness.
    Returns status, timestamp, environment, database ready, summarization readiness,
    Phase 4 RAG vector store and QA model status, Phase 5 AI Quiz Generation,
    and Phase 6 Student Dashboard, Analytics & Goals readiness.
    """
    db_ready = False
    try:
        _ = get_all_documents(limit=1)
        db_ready = True
    except Exception:
        db_ready = False

    summarizer_engine = SummarizerEngine.get_instance()
    qa_engine = get_qa_engine()
    vector_store = get_vector_store()
    vector_stats = vector_store.get_index_stats()

    return jsonify({
        "status": "healthy",
        "service": "AI-Powered Smart Study Assistant",
        "version": "1.0.0",
        "phase": "Phases 1-6 Integrated (Phase 3 Summarization, Phase 4 Chatbot RAG, Phase 5 AI Quizzes, Phase 6 Dashboard & Analytics)",
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": "testing" if current_app.config.get("TESTING") else (
            "development" if current_app.config.get("DEBUG") else "production"
        ),
        "database_ready": db_ready,
        "upload_folder_ready": current_app.config.get("UPLOAD_FOLDER") is not None,
        "summarization_model": current_app.config.get("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6"),
        "model_loaded": summarizer_engine.is_loaded(),
        "summarizer_loaded": summarizer_engine.is_loaded(),
        "rag_embedding_model": current_app.config.get("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "rag_qa_model": current_app.config.get("RAG_QA_MODEL", "google/flan-t5-small"),
        "qa_model_loaded": qa_engine.is_loaded,
        "vector_store": vector_stats,
        "quiz_generator": "ready",
        "quiz_evaluator": "ready",
        "dashboard_analytics": "ready",
        "study_goals": "ready",
        "activity_tracker": "ready",
    }), 200

