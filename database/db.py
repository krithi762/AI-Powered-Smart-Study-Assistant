"""
Database module for AI-Powered Smart Study Assistant.
Provides SQLite connection management, schema initialization, and CRUD operations
for documents and extracted page text with foreign key constraints.
"""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from flask import current_app


def get_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve database path from argument, Flask app config, or default location."""
    if db_path is not None:
        return Path(db_path)
    if current_app:
        configured_path = current_app.config.get("DATABASE_PATH")
        if configured_path:
            return Path(configured_path)
    # Fallback to local database directory
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "database" / "study_assistant.db"


def get_db_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """
    Creates and returns an SQLite database connection.
    Enables foreign key enforcement and sets row_factory to sqlite3.Row.
    """
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # Enforce foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    """
    Initializes the database schema if tables do not exist.
    Creates 'documents', 'document_pages', and 'document_summaries' tables with indexes.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL UNIQUE,
                    file_path TEXT NOT NULL,
                    total_pages INTEGER NOT NULL DEFAULT 0,
                    total_words INTEGER NOT NULL DEFAULT 0,
                    upload_date TEXT NOT NULL,
                    processing_status TEXT NOT NULL DEFAULT 'processed'
                );

                CREATE TABLE IF NOT EXISTS document_pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    extracted_text TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS document_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    summary_type TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    chapter_summaries TEXT,
                    key_concepts TEXT,
                    keywords TEXT,
                    exam_points TEXT,
                    topics TEXT,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    title TEXT NOT NULL DEFAULT 'Study Chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    document_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                -- Phase 5: AI Quiz Generation Tables
                CREATE TABLE IF NOT EXISTS quizzes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    total_questions INTEGER NOT NULL,
                    time_limit_minutes INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS quiz_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quiz_id INTEGER NOT NULL,
                    question_number INTEGER NOT NULL,
                    question_text TEXT NOT NULL,
                    option_a TEXT NOT NULL,
                    option_b TEXT NOT NULL,
                    option_c TEXT NOT NULL,
                    option_d TEXT NOT NULL,
                    correct_option TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    source_page INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS quiz_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quiz_id INTEGER NOT NULL,
                    document_id INTEGER NOT NULL,
                    user_id TEXT DEFAULT 'default_student',
                    total_questions INTEGER NOT NULL,
                    correct_answers INTEGER NOT NULL,
                    incorrect_answers INTEGER NOT NULL,
                    unanswered_count INTEGER NOT NULL,
                    score INTEGER NOT NULL,
                    percentage REAL NOT NULL,
                    passed BOOLEAN NOT NULL,
                    time_taken_seconds INTEGER NOT NULL DEFAULT 0,
                    answers_json TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_pages_doc_id 
                    ON document_pages(document_id);
                CREATE INDEX IF NOT EXISTS idx_pages_doc_page 
                    ON document_pages(document_id, page_number);
                CREATE INDEX IF NOT EXISTS idx_docs_upload_date 
                    ON documents(upload_date DESC);
                CREATE INDEX IF NOT EXISTS idx_summaries_doc_id 
                    ON document_summaries(document_id);
                CREATE INDEX IF NOT EXISTS idx_summaries_doc_type 
                    ON document_summaries(document_id, summary_type);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id 
                    ON document_chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_page 
                    ON document_chunks(document_id, page_number);
                CREATE INDEX IF NOT EXISTS idx_chat_conv_doc 
                    ON chat_conversations(document_id);
                CREATE INDEX IF NOT EXISTS idx_chat_msg_conv 
                    ON chat_messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_chat_msg_doc 
                    ON chat_messages(document_id);

                -- Phase 5 Indexes
                CREATE INDEX IF NOT EXISTS idx_quizzes_doc_id 
                    ON quizzes(document_id);
                CREATE INDEX IF NOT EXISTS idx_quizzes_created 
                    ON quizzes(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_quiz_q_quiz_id 
                    ON quiz_questions(quiz_id);
                CREATE INDEX IF NOT EXISTS idx_quiz_q_number 
                    ON quiz_questions(quiz_id, question_number);
                CREATE INDEX IF NOT EXISTS idx_quiz_att_quiz_id 
                    ON quiz_attempts(quiz_id);
                CREATE INDEX IF NOT EXISTS idx_quiz_att_doc_id 
                    ON quiz_attempts(document_id);
                CREATE INDEX IF NOT EXISTS idx_quiz_att_completed 
                    ON quiz_attempts(completed_at DESC);

                -- Phase 6: Activity History & Study Goals Tables
                CREATE TABLE IF NOT EXISTS study_activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT DEFAULT 'default_student',
                    activity_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    document_id INTEGER,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS study_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT DEFAULT 'default_student',
                    title TEXT NOT NULL,
                    goal_type TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    current_value REAL DEFAULT 0,
                    target_date TEXT,
                    status TEXT DEFAULT 'in_progress',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                -- Phase 6 Indexes
                CREATE INDEX IF NOT EXISTS idx_activity_type 
                    ON study_activities(activity_type);
                CREATE INDEX IF NOT EXISTS idx_activity_created 
                    ON study_activities(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_activity_doc_id 
                    ON study_activities(document_id);
                CREATE INDEX IF NOT EXISTS idx_goals_status 
                    ON study_goals(status);
                CREATE INDEX IF NOT EXISTS idx_goals_created 
                    ON study_goals(created_at DESC);

                -- Student Profile Table
                CREATE TABLE IF NOT EXISTS student_profile (
                    user_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL DEFAULT 'Krishna K',
                    email TEXT NOT NULL DEFAULT 'krishna.k@smartstudy.ai',
                    college_name TEXT NOT NULL DEFAULT 'Institute of Engineering & Technology',
                    department TEXT NOT NULL DEFAULT 'Computer Science & Engineering',
                    specialization TEXT NOT NULL DEFAULT 'Artificial Intelligence & Data Systems',
                    academic_year TEXT NOT NULL DEFAULT 'Year 3 (Junior)',
                    current_semester TEXT NOT NULL DEFAULT 'Semester 6',
                    preferred_summary_mode TEXT NOT NULL DEFAULT 'standard',
                    rag_precision TEXT NOT NULL DEFAULT '3',
                    learning_mode TEXT NOT NULL DEFAULT 'Visual & Concept Mapping',
                    study_target_minutes INTEGER NOT NULL DEFAULT 45,
                    updated_at TEXT NOT NULL
                );
            """)
    finally:
        conn.close()

    # OCR migration: add columns if they don't exist yet (safe on existing DBs)
    _migrate_add_ocr_columns(db_path=db_path)

    # Backfill historical activities if table is currently empty
    backfill_activities_if_empty(db_path=db_path)
    # Seed default student profile if not exists
    seed_default_profile(db_path=db_path)


def _migrate_add_ocr_columns(db_path: str | Path | None = None) -> None:
    """
    Migration helper: adds OCR-related columns to the 'documents' table if
    they don't already exist. Safe to run on both fresh and existing databases.
    """
    ocr_columns: list[tuple[str, str]] = [
        ("ocr_applied", "INTEGER NOT NULL DEFAULT 0"),
        ("ocr_engine", "TEXT NOT NULL DEFAULT ''"),
        ("ocr_confidence", "REAL"),
        ("extraction_method", "TEXT NOT NULL DEFAULT 'pymupdf'"),
    ]

    conn = get_db_connection(db_path)
    try:
        # Fetch existing column names
        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        with conn:
            for col_name, col_def in ocr_columns:
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE documents ADD COLUMN {col_name} {col_def}"
                    )
    finally:
        conn.close()


def seed_default_profile(db_path: str | Path | None = None) -> None:
    """Seeds the default student profile if not already present."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            existing = conn.execute(
                "SELECT user_id FROM student_profile WHERE user_id = 'default_student'"
            ).fetchone()
            if not existing:
                now_str = datetime.now(UTC).isoformat()
                conn.execute(
                    """
                    INSERT INTO student_profile (
                        user_id, full_name, email, college_name, department,
                        specialization, academic_year, current_semester,
                        preferred_summary_mode, rag_precision, learning_mode,
                        study_target_minutes, updated_at
                    ) VALUES (
                        'default_student', 'Krishna K', 'krishna.k@smartstudy.ai',
                        'Institute of Engineering & Technology',
                        'Computer Science & Engineering',
                        'Artificial Intelligence & Data Systems',
                        'Year 3 (Junior)', 'Semester 6',
                        'standard', '3', 'Visual & Concept Mapping', 45, ?
                    )
                    """,
                    (now_str,),
                )
    finally:
        conn.close()


def _format_profile_record(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """Helper to convert a database row/dict to a standardized profile dictionary with computed helpers."""
    data = dict(row)
    full_name = str(data.get("full_name") or "Krishna K").strip()
    name_parts = full_name.split()
    first_name = name_parts[0] if name_parts else "Krishna"

    if len(name_parts) >= 2:
        initials = (name_parts[0][0] + name_parts[-1][0]).upper()
    elif len(name_parts) == 1 and name_parts[0]:
        initials = name_parts[0][0].upper()
    else:
        initials = "K"

    data["full_name"] = full_name
    data["first_name"] = first_name
    data["initials"] = initials
    return data


def get_student_profile(
    user_id: str = "default_student",
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Retrieves the student profile for a given user_id.
    If not found, initializes and returns the default profile.
    """
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM student_profile WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            return _format_profile_record(row)
    finally:
        conn.close()

    # If missing, seed and re-fetch
    seed_default_profile(db_path=db_path)
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM student_profile WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            return _format_profile_record(row)
        return {
            "user_id": user_id,
            "full_name": "Krishna K",
            "first_name": "Krishna",
            "initials": "K",
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
            "updated_at": datetime.now(UTC).isoformat(),
        }
    finally:
        conn.close()


def update_student_profile(
    profile_data: dict[str, Any],
    user_id: str = "default_student",
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Updates the student profile fields for the specified user_id with validation.
    Returns the updated profile dictionary.
    Raises ValueError on validation failure.
    """
    current = get_student_profile(user_id=user_id, db_path=db_path)

    full_name = str(profile_data.get("full_name", current["full_name"])).strip()
    if not full_name or len(full_name) < 2:
        raise ValueError("Full Name must be at least 2 characters.")

    email = str(profile_data.get("email", current["email"])).strip()
    if "@" not in email or "." not in email:
        raise ValueError("A valid email address is required.")

    college_name = str(profile_data.get("college_name", current["college_name"])).strip()
    if not college_name:
        raise ValueError("College Name cannot be empty.")

    department = str(profile_data.get("department", current["department"])).strip()
    if not department:
        raise ValueError("Department cannot be empty.")

    specialization = str(profile_data.get("specialization", current["specialization"])).strip()
    if not specialization:
        raise ValueError("Academic Specialization cannot be empty.")

    academic_year = str(profile_data.get("academic_year", current["academic_year"])).strip()
    current_semester = str(profile_data.get("current_semester", current["current_semester"])).strip()

    preferred_summary_mode = str(profile_data.get("preferred_summary_mode", current["preferred_summary_mode"])).strip().lower()
    if preferred_summary_mode not in ("standard", "quick", "detailed"):
        preferred_summary_mode = "standard"

    rag_precision = str(profile_data.get("rag_precision", current["rag_precision"])).strip()
    if rag_precision not in ("3", "5"):
        rag_precision = "3"

    learning_mode = str(profile_data.get("learning_mode", current["learning_mode"])).strip()
    if not learning_mode:
        learning_mode = "Visual & Concept Mapping"

    try:
        study_target_minutes = int(profile_data.get("study_target_minutes", current.get("study_target_minutes", 45)))
        if study_target_minutes <= 0 or study_target_minutes > 720:
            study_target_minutes = 45
    except (ValueError, TypeError):
        study_target_minutes = 45

    updated_at = datetime.now(UTC).isoformat()

    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO student_profile (
                    user_id, full_name, email, college_name, department,
                    specialization, academic_year, current_semester,
                    preferred_summary_mode, rag_precision, learning_mode,
                    study_target_minutes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    email = excluded.email,
                    college_name = excluded.college_name,
                    department = excluded.department,
                    specialization = excluded.specialization,
                    academic_year = excluded.academic_year,
                    current_semester = excluded.current_semester,
                    preferred_summary_mode = excluded.preferred_summary_mode,
                    rag_precision = excluded.rag_precision,
                    learning_mode = excluded.learning_mode,
                    study_target_minutes = excluded.study_target_minutes,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    full_name,
                    email,
                    college_name,
                    department,
                    specialization,
                    academic_year,
                    current_semester,
                    preferred_summary_mode,
                    rag_precision,
                    learning_mode,
                    study_target_minutes,
                    updated_at,
                ),
            )
    finally:
        conn.close()

    try:
        log_activity(
            activity_type="profile_update",
            title="Updated Student Profile",
            description=f"Updated details for {full_name} ({specialization}).",
            document_id=None,
            metadata={"full_name": full_name, "specialization": specialization},
            user_id=user_id,
            db_path=db_path,
        )
    except Exception:
        pass

    return get_student_profile(user_id=user_id, db_path=db_path)


def insert_document(
    original_filename: str,
    stored_filename: str,
    file_path: str,
    total_pages: int,
    total_words: int,
    processing_status: str = "processed",
    upload_date: str | None = None,
    ocr_applied: bool = False,
    ocr_engine: str = "",
    ocr_confidence: float | None = None,
    extraction_method: str = "pymupdf",
    db_path: str | Path | None = None,
) -> int:
    """
    Inserts a newly uploaded document record into the 'documents' table.
    Returns the newly created document id.

    Extended with OCR metadata fields:
      - ocr_applied: whether OCR was used for any page
      - ocr_engine: name of OCR engine used (e.g. 'tesseract')
      - ocr_confidence: average OCR confidence score (0-100), None if N/A
      - extraction_method: 'pymupdf' | 'ocr' | 'mixed'
    """
    if upload_date is None:
        upload_date = datetime.now(UTC).isoformat()

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (
                    original_filename,
                    stored_filename,
                    file_path,
                    total_pages,
                    total_words,
                    upload_date,
                    processing_status,
                    ocr_applied,
                    ocr_engine,
                    ocr_confidence,
                    extraction_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    original_filename,
                    stored_filename,
                    str(file_path),
                    total_pages,
                    total_words,
                    upload_date,
                    processing_status,
                    1 if ocr_applied else 0,
                    ocr_engine,
                    ocr_confidence,
                    extraction_method,
                ),
            )
            return int(cursor.lastrowid or 0)
    finally:
        conn.close()


def insert_document_pages(
    document_id: int,
    pages_data: list[dict[str, Any]],
    replace_existing: bool = False,
    db_path: str | Path | None = None,
) -> None:
    """
    Inserts page text records for a given document.
    pages_data is expected to be a list of dicts with 'page_number' and 'text'.

    Args:
        document_id:      Document primary key.
        pages_data:       List of {page_number, text} dicts.
        replace_existing: If True, delete existing pages for this document first
                          (used when replacing with OCR text).
        db_path:          Optional explicit database path.
    """
    if not pages_data:
        return

    conn = get_db_connection(db_path)
    try:
        with conn:
            if replace_existing:
                conn.execute(
                    "DELETE FROM document_pages WHERE document_id = ?",
                    (document_id,),
                )
            conn.executemany(
                """
                INSERT INTO document_pages (document_id, page_number, extracted_text)
                VALUES (?, ?, ?)
                """,
                [
                    (document_id, page["page_number"], page["text"])
                    for page in pages_data
                ],
            )
    finally:
        conn.close()


def update_document_ocr_metadata(
    document_id: int,
    total_words: int,
    processing_status: str,
    ocr_applied: bool,
    ocr_engine: str,
    ocr_confidence: float | None,
    extraction_method: str,
    db_path: str | Path | None = None,
) -> None:
    """
    Updates the OCR-related metadata columns for an existing document record.
    Called after OCR is applied post-upload.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE documents
                SET total_words = ?,
                    processing_status = ?,
                    ocr_applied = ?,
                    ocr_engine = ?,
                    ocr_confidence = ?,
                    extraction_method = ?
                WHERE id = ?
                """,
                (
                    total_words,
                    processing_status,
                    1 if ocr_applied else 0,
                    ocr_engine,
                    ocr_confidence,
                    extraction_method,
                    document_id,
                ),
            )
    finally:
        conn.close()


def get_all_documents(
    limit: int | None = None, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Retrieves all documents sorted by upload_date descending."""
    conn = get_db_connection(db_path)
    try:
        query = "SELECT * FROM documents ORDER BY upload_date DESC"
        params: list[Any] = []
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_recent_documents(
    limit: int = 5, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Convenience function to fetch the most recent documents."""
    return get_all_documents(limit=limit, db_path=db_path)


def get_document(
    document_id: int, db_path: str | Path | None = None
) -> dict[str, Any] | None:
    """Retrieves a single document by its primary key ID."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# Alias for backward and cross-module compatibility
get_document_by_id = get_document


def get_document_pages(
    document_id: int, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Retrieves all extracted pages for a document ordered by page_number."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, document_id, page_number, extracted_text
            FROM document_pages
            WHERE document_id = ?
            ORDER BY page_number ASC
            """,
            (document_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_document(
    document_id: int, db_path: str | Path | None = None
) -> str | None:
    """
    Deletes a document from the database and returns the file_path for removal.
    Associated rows in 'document_pages' and 'document_summaries' are automatically
    removed via ON DELETE CASCADE.
    Returns the file_path if found, otherwise None.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "SELECT file_path FROM documents WHERE id = ?", (document_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            file_path = row["file_path"]
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return file_path
    finally:
        conn.close()


def save_summary(
    document_id: int,
    summary_type: str,
    summary_text: str,
    chapter_summaries: Any | None = None,
    key_concepts: Any | None = None,
    keywords: Any | None = None,
    exam_points: Any | None = None,
    topics: Any | None = None,
    model_name: str = "sshleifer/distilbart-cnn-12-6",
    created_at: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """
    Inserts or updates a summary for a given document and summary_type.
    Serializes list/dict fields to JSON strings.
    Returns the summary record ID.
    """
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()

    def _to_json(val: Any) -> str | None:
        if val is None:
            return None
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return str(val)

    chapter_summaries_json = _to_json(chapter_summaries)
    key_concepts_json = _to_json(key_concepts)
    keywords_json = _to_json(keywords)
    exam_points_json = _to_json(exam_points)
    topics_json = _to_json(topics)

    conn = get_db_connection(db_path)
    try:
        with conn:
            # Check if summary already exists for this document and type
            cursor = conn.execute(
                "SELECT id FROM document_summaries WHERE document_id = ? AND summary_type = ?",
                (document_id, summary_type),
            )
            existing = cursor.fetchone()

            if existing:
                summary_id = existing["id"]
                conn.execute(
                    """
                    UPDATE document_summaries
                    SET summary_text = ?,
                        chapter_summaries = ?,
                        key_concepts = ?,
                        keywords = ?,
                        exam_points = ?,
                        topics = ?,
                        model_name = ?,
                        created_at = ?
                    WHERE id = ?
                    """,
                    (
                        summary_text,
                        chapter_summaries_json,
                        key_concepts_json,
                        keywords_json,
                        exam_points_json,
                        topics_json,
                        model_name,
                        created_at,
                        summary_id,
                    ),
                )
                return summary_id
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO document_summaries (
                        document_id,
                        summary_type,
                        summary_text,
                        chapter_summaries,
                        key_concepts,
                        keywords,
                        exam_points,
                        topics,
                        model_name,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        summary_type,
                        summary_text,
                        chapter_summaries_json,
                        key_concepts_json,
                        keywords_json,
                        exam_points_json,
                        topics_json,
                        model_name,
                        created_at,
                    ),
                )
                return int(cursor.lastrowid or 0)
    finally:
        conn.close()


def _format_summary_record(row: sqlite3.Row) -> dict[str, Any]:
    """Helper to convert a summary row to a dict with deserialized JSON fields."""
    d = dict(row)
    for field in ("chapter_summaries", "key_concepts", "keywords", "exam_points", "topics"):
        val = d.get(field)
        if val and isinstance(val, str):
            try:
                d[field] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
    return d


def get_summary(
    document_id: int,
    summary_type: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """
    Retrieves a summary for a document.
    If summary_type is specified, retrieves that specific type ('quick', 'standard', 'detailed').
    Otherwise retrieves the most recently created summary for the document.
    """
    conn = get_db_connection(db_path)
    try:
        if summary_type:
            cursor = conn.execute(
                """
                SELECT * FROM document_summaries 
                WHERE document_id = ? AND summary_type = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (document_id, summary_type),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM document_summaries 
                WHERE document_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (document_id,),
            )
        row = cursor.fetchone()
        return _format_summary_record(row) if row else None
    finally:
        conn.close()


def get_all_summaries_for_document(
    document_id: int, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Retrieves all generated summaries for a document, ordered by created_at DESC."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT * FROM document_summaries 
            WHERE document_id = ?
            ORDER BY created_at DESC
            """,
            (document_id,),
        )
        rows = cursor.fetchall()
        return [_format_summary_record(row) for row in rows]
    finally:
        conn.close()


def delete_summaries_for_document(
    document_id: int,
    summary_type: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """
    Deletes summaries for a document. If summary_type is given, deletes only that type.
    Returns the number of deleted records.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            if summary_type:
                cursor = conn.execute(
                    "DELETE FROM document_summaries WHERE document_id = ? AND summary_type = ?",
                    (document_id, summary_type),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM document_summaries WHERE document_id = ?",
                    (document_id,),
                )
            return cursor.rowcount
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# Phase 4 - RAG Document Chunks & Chatbot Persistence
# ------------------------------------------------------------------------------

def save_document_chunks(
    chunks: list[dict[str, Any]], db_path: str | Path | None = None
) -> int:
    """
    Saves a batch of document chunks to the database.
    Each chunk dict must have: document_id, page_number, chunk_index, chunk_id, chunk_text, word_count.
    Returns the number of chunks saved.
    """
    if not chunks:
        return 0

    now_iso = datetime.now(UTC).isoformat()
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.executemany(
                """
                INSERT INTO document_chunks (
                    document_id, page_number, chunk_index, chunk_id, chunk_text, word_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c["document_id"],
                        c["page_number"],
                        c["chunk_index"],
                        c["chunk_id"],
                        c["chunk_text"],
                        c.get("word_count", len(c["chunk_text"].split())),
                        now_iso,
                    )
                    for c in chunks
                ],
            )
            return cursor.rowcount
    finally:
        conn.close()


def get_document_chunks(
    document_id: int, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Retrieves all chunks for a document ordered by page_number and chunk_index."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT * FROM document_chunks
            WHERE document_id = ?
            ORDER BY page_number ASC, chunk_index ASC
            """,
            (document_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_document_chunks(
    document_id: int, db_path: str | Path | None = None
) -> int:
    """Deletes all chunks for a document. Returns number of deleted rows."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "DELETE FROM document_chunks WHERE document_id = ?",
                (document_id,),
            )
            return cursor.rowcount
    finally:
        conn.close()


def get_or_create_conversation(
    document_id: int,
    title: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Retrieves the active conversation for a document or creates a new one.
    Returns the conversation record dict.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT * FROM chat_conversations
            WHERE document_id = ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (document_id,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)

        # Create new conversation
        now_iso = datetime.now(UTC).isoformat()
        conv_title = title or "Study Chat"
        with conn:
            insert_cur = conn.execute(
                """
                INSERT INTO chat_conversations (document_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (document_id, conv_title, now_iso, now_iso),
            )
            conv_id = insert_cur.lastrowid
            return {
                "id": conv_id,
                "document_id": document_id,
                "title": conv_title,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
    finally:
        conn.close()


def save_chat_message(
    conversation_id: int,
    document_id: int,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
    db_path: str | Path | None = None,
) -> int:
    """
    Appends a user or assistant message to a conversation.
    Updates the conversation's updated_at timestamp.
    Returns the new message ID.
    """
    now_iso = datetime.now(UTC).isoformat()
    sources_str = json.dumps(sources) if sources else None
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_messages (
                    conversation_id, document_id, role, content, sources_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, document_id, role, content, sources_str, now_iso),
            )
            msg_id = int(cursor.lastrowid or 0)

            conn.execute(
                "UPDATE chat_conversations SET updated_at = ? WHERE id = ?",
                (now_iso, conversation_id),
            )
            return msg_id
    finally:
        conn.close()


def get_chat_history(
    conversation_id: int, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Retrieves all messages for a conversation ordered chronologically."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT * FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (conversation_id,),
        )
        rows = cursor.fetchall()
        result = []
        for r in rows:
            item = dict(r)
            if item.get("sources_json"):
                try:
                    item["sources"] = json.loads(item["sources_json"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    item["sources"] = []
            else:
                item["sources"] = []
            result.append(item)
        return result
    finally:
        conn.close()


def clear_chat_history(
    conversation_id: int, db_path: str | Path | None = None
) -> int:
    """Deletes all messages from a conversation. Returns the number of deleted messages."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "DELETE FROM chat_messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            now_iso = datetime.now(UTC).isoformat()
            conn.execute(
                "UPDATE chat_conversations SET updated_at = ? WHERE id = ?",
                (now_iso, conversation_id),
            )
            return cursor.rowcount
    finally:
        conn.close()


def delete_chat_for_document(
    document_id: int, db_path: str | Path | None = None
) -> int:
    """Deletes all conversations and messages for a document."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "DELETE FROM chat_conversations WHERE document_id = ?",
                (document_id,),
            )
            return cursor.rowcount
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# Phase 5 - AI Quiz Generation & Assessment Persistence
# ------------------------------------------------------------------------------

def create_quiz(
    document_id: int,
    title: str,
    difficulty: str,
    total_questions: int,
    time_limit_minutes: int = 0,
    created_at: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """
    Creates a new quiz entry associated with a document.
    Returns the newly created quiz ID.
    """
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO quizzes (
                    document_id, title, difficulty, total_questions, time_limit_minutes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    title.strip(),
                    difficulty.lower().strip(),
                    total_questions,
                    int(time_limit_minutes or 0),
                    created_at,
                ),
            )
            return int(cursor.lastrowid or 0)
    finally:
        conn.close()


def add_quiz_questions(
    quiz_id: int,
    questions_data: list[dict[str, Any]],
    db_path: str | Path | None = None,
) -> int:
    """
    Inserts a list of questions for a specific quiz.
    Each item must contain:
      - question_text: str
      - option_a: str
      - option_b: str
      - option_c: str
      - option_d: str
      - correct_option: str ('A', 'B', 'C', or 'D')
      - explanation: str
      - difficulty: str ('easy', 'medium', 'hard')
      - source_page: Optional[int]
    Returns the number of questions added.
    """
    if not questions_data:
        return 0

    now_iso = datetime.now(UTC).isoformat()
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.executemany(
                """
                INSERT INTO quiz_questions (
                    quiz_id, question_number, question_text,
                    option_a, option_b, option_c, option_d,
                    correct_option, explanation, difficulty,
                    source_page, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        quiz_id,
                        q.get("question_number", idx + 1),
                        q["question_text"].strip(),
                        q["option_a"].strip(),
                        q["option_b"].strip(),
                        q["option_c"].strip(),
                        q["option_d"].strip(),
                        q["correct_option"].upper().strip(),
                        q.get("explanation", "").strip(),
                        q.get("difficulty", "medium").lower().strip(),
                        q.get("source_page"),
                        now_iso,
                    )
                    for idx, q in enumerate(questions_data)
                ],
            )
            return cursor.rowcount
    finally:
        conn.close()


def get_quiz(
    quiz_id: int, db_path: str | Path | None = None
) -> dict[str, Any] | None:
    """
    Retrieves a quiz record joined with its document's original_filename.
    Returns None if the quiz does not exist.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT q.*, d.original_filename, d.total_pages
            FROM quizzes q
            JOIN documents d ON q.document_id = d.id
            WHERE q.id = ?
            """,
            (quiz_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_quiz_questions(
    quiz_id: int,
    include_answers: bool = False,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieves all questions for a given quiz ordered by question_number.
    
    SECURITY:
    When include_answers=False (during active quiz taking), 'correct_option' and
    'explanation' are stripped from the returned dictionary so correct answers
    can never be inspected by the client before submission.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT * FROM quiz_questions
            WHERE quiz_id = ?
            ORDER BY question_number ASC, id ASC
            """,
            (quiz_id,),
        )
        rows = cursor.fetchall()
        result = []
        for r in rows:
            item = dict(r)
            if not include_answers:
                # Strip answer key from client response during quiz attempt
                item.pop("correct_option", None)
                item.pop("explanation", None)
            result.append(item)
        return result
    finally:
        conn.close()


def get_all_quizzes(
    document_id: int | None = None,
    limit: int | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieves quizzes ordered by created_at DESC.
    Optionally filters by document_id and limits result count.
    Also retrieves the number of completed attempts for each quiz.
    """
    conn = get_db_connection(db_path)
    try:
        query = """
            SELECT q.*, d.original_filename,
                   (SELECT COUNT(*) FROM quiz_attempts qa WHERE qa.quiz_id = q.id) AS attempt_count,
                   (SELECT MAX(qa.percentage) FROM quiz_attempts qa WHERE qa.quiz_id = q.id) AS best_percentage
            FROM quizzes q
            JOIN documents d ON q.document_id = d.id
        """
        params: list[Any] = []
        if document_id is not None:
            query += " WHERE q.document_id = ?"
            params.append(document_id)

        query += " ORDER BY q.created_at DESC"
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_quiz_attempt(
    quiz_id: int,
    document_id: int,
    total_questions: int,
    correct_answers: int,
    incorrect_answers: int,
    unanswered_count: int,
    score: int,
    percentage: float,
    passed: bool,
    time_taken_seconds: int,
    answers: dict[str, Any],
    user_id: str = "default_student",
    completed_at: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """
    Saves an evaluated quiz attempt into 'quiz_attempts'.
    Returns the newly created attempt ID.
    """
    if completed_at is None:
        completed_at = datetime.now(UTC).isoformat()

    answers_json = json.dumps(answers, ensure_ascii=False)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO quiz_attempts (
                    quiz_id, document_id, user_id,
                    total_questions, correct_answers, incorrect_answers, unanswered_count,
                    score, percentage, passed, time_taken_seconds,
                    answers_json, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quiz_id,
                    document_id,
                    user_id,
                    total_questions,
                    correct_answers,
                    incorrect_answers,
                    unanswered_count,
                    score,
                    round(float(percentage), 2),
                    1 if passed else 0,
                    int(time_taken_seconds or 0),
                    answers_json,
                    completed_at,
                ),
            )
            return int(cursor.lastrowid or 0)
    finally:
        conn.close()


def get_quiz_attempt(
    attempt_id: int, db_path: str | Path | None = None
) -> dict[str, Any] | None:
    """
    Retrieves a single quiz attempt by ID, joined with quiz title and document filename.
    Deserializes answers_json into a dictionary.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT qa.*, q.title AS quiz_title, q.difficulty, q.time_limit_minutes,
                   d.original_filename
            FROM quiz_attempts qa
            JOIN quizzes q ON qa.quiz_id = q.id
            JOIN documents d ON qa.document_id = d.id
            WHERE qa.id = ?
            """,
            (attempt_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        attempt = dict(row)
        if attempt.get("answers_json"):
            try:
                attempt["answers"] = json.loads(attempt["answers_json"])
            except (json.JSONDecodeError, TypeError, ValueError):
                attempt["answers"] = {}
        else:
            attempt["answers"] = {}
        return attempt
    finally:
        conn.close()


def get_quiz_attempts(
    quiz_id: int | None = None,
    document_id: int | None = None,
    limit: int | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieves past quiz attempts ordered by completed_at DESC.
    Supports filtering by quiz_id or document_id.
    """
    conn = get_db_connection(db_path)
    try:
        query = """
            SELECT qa.*, q.title AS quiz_title, q.difficulty,
                   d.original_filename
            FROM quiz_attempts qa
            JOIN quizzes q ON qa.quiz_id = q.id
            JOIN documents d ON qa.document_id = d.id
        """
        conditions = []
        params: list[Any] = []
        if quiz_id is not None:
            conditions.append("qa.quiz_id = ?")
            params.append(quiz_id)
        if document_id is not None:
            conditions.append("qa.document_id = ?")
            params.append(document_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY qa.completed_at DESC"
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        result = []
        for r in rows:
            item = dict(r)
            if item.get("answers_json"):
                try:
                    item["answers"] = json.loads(item["answers_json"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    item["answers"] = {}
            else:
                item["answers"] = {}
            result.append(item)
        return result
    finally:
        conn.close()


def delete_quiz(
    quiz_id: int, db_path: str | Path | None = None
) -> bool:
    """
    Deletes a quiz. Questions and attempts are cascaded.
    Returns True if a record was deleted, False otherwise.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def delete_quizzes_for_document(
    document_id: int, db_path: str | Path | None = None
) -> int:
    """Deletes all quizzes for a document."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM quizzes WHERE document_id = ?", (document_id,))
            return cursor.rowcount
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# Phase 6: Activity History, Study Goals, Dashboard & Analytics Operations
# ------------------------------------------------------------------------------

def log_activity(
    activity_type: str,
    title: str,
    description: str | None = None,
    document_id: int | None = None,
    metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
    user_id: str = "default_student",
    db_path: str | Path | None = None,
) -> int:
    """
    Logs an authoritative student learning activity record to 'study_activities'.
    Automatically triggers active goal progress evaluation.
    Returns the newly created activity primary key ID.
    """
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    meta_json = json.dumps(metadata) if metadata else "{}"

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO study_activities (
                    user_id, activity_type, title, description,
                    document_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    activity_type,
                    title,
                    description or "",
                    document_id,
                    meta_json,
                    created_at,
                ),
            )
            activity_id = int(cursor.lastrowid or 0)
        # Synchronize any goals impacted by this activity
        sync_study_goals_progress(db_path=db_path)
        return activity_id
    finally:
        conn.close()


def get_study_activities(
    limit: int | None = 50,
    offset: int = 0,
    activity_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    document_id: int | None = None,
    user_id: str = "default_student",
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieves filtered learning activities sorted descending by created_at.
    Joins documents table to resolve source PDF filenames.
    """
    conn = get_db_connection(db_path)
    try:
        query = """
            SELECT 
                sa.id, sa.user_id, sa.activity_type, sa.title, sa.description,
                sa.document_id, sa.metadata_json, sa.created_at,
                d.original_filename
            FROM study_activities sa
            LEFT JOIN documents d ON sa.document_id = d.id
        """
        conditions: list[str] = ["sa.user_id = ?"]
        params: list[Any] = [user_id]

        if activity_type and activity_type.lower() != "all":
            conditions.append("sa.activity_type = ?")
            params.append(activity_type.lower())
        if date_from:
            conditions.append("sa.created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("sa.created_at <= ?")
            params.append(date_to + "T23:59:59")
        if document_id:
            conditions.append("sa.document_id = ?")
            params.append(document_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY sa.created_at DESC"
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)
            if offset > 0:
                query += " OFFSET ?"
                params.append(offset)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        result = []
        for r in rows:
            item = dict(r)
            if item.get("metadata_json"):
                try:
                    item["metadata"] = json.loads(item["metadata_json"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    item["metadata"] = {}
            else:
                item["metadata"] = {}
            result.append(item)
        return result
    finally:
        conn.close()


def get_total_activities_count(
    activity_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    document_id: int | None = None,
    user_id: str = "default_student",
    db_path: str | Path | None = None,
) -> int:
    """Returns total count of activities matching filter conditions."""
    conn = get_db_connection(db_path)
    try:
        query = "SELECT COUNT(*) FROM study_activities WHERE user_id = ?"
        params: list[Any] = [user_id]
        if activity_type and activity_type.lower() != "all":
            query += " AND activity_type = ?"
            params.append(activity_type.lower())
        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND created_at <= ?"
            params.append(date_to + "T23:59:59")
        if document_id:
            query += " AND document_id = ?"
            params.append(document_id)

        cursor = conn.execute(query, params)
        return cursor.fetchone()[0]
    finally:
        conn.close()


def get_dashboard_statistics(db_path: str | Path | None = None) -> dict[str, Any]:
    """
    Computes real, authoritative statistics for the main Student Dashboard directly from SQLite.
    Includes overall learning progress calculation and recent item feeds.
    """
    conn = get_db_connection(db_path)
    try:
        # 1. Total uploaded documents
        c_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        # 2. Total summaries generated
        c_sums = conn.execute("SELECT COUNT(*) FROM document_summaries").fetchone()[0]
        # 3. Total chatbot questions asked
        c_chats = conn.execute("SELECT COUNT(*) FROM chat_messages WHERE role = 'user'").fetchone()[0]
        # 4. Total quizzes generated
        c_quizzes = conn.execute("SELECT COUNT(*) FROM quizzes").fetchone()[0]
        # 5. Total quizzes completed (attempts)
        c_attempts = conn.execute("SELECT COUNT(*) FROM quiz_attempts").fetchone()[0]
        # 6. Average quiz score (percentage)
        avg_row = conn.execute("SELECT AVG(percentage) FROM quiz_attempts").fetchone()[0]
        avg_quiz_score = round(float(avg_row), 1) if avg_row is not None else 0.0

        # 7. Total study sessions (distinct dates of learning activity across all pillars)
        session_row = conn.execute("""
            SELECT COUNT(DISTINCT activity_date) FROM (
                SELECT substr(created_at, 1, 10) AS activity_date FROM study_activities
                UNION
                SELECT substr(upload_date, 1, 10) AS activity_date FROM documents
                UNION
                SELECT substr(created_at, 1, 10) AS activity_date FROM chat_messages
                UNION
                SELECT substr(completed_at, 1, 10) AS activity_date FROM quiz_attempts
            ) WHERE activity_date IS NOT NULL
        """).fetchone()[0]
        total_sessions = int(session_row or 0)

        # 8. Overall Learning Progress Index (0.0 to 100.0%)
        # Explicit, transparent mathematical weighting across 5 fundamental learning activities:
        # - Documents Processed (Target: 5 docs)  -> Max 20%
        # - Summaries Created (Target: 5 sums)    -> Max 20%
        # - Chat Interactions (Target: 10 Qs)     -> Max 20%
        # - Quizzes Attempted (Target: 5 quizzes) -> Max 20%
        # - Quiz Performance (Avg percentage)     -> Max 20%
        p_docs = min(20.0, (c_docs / 5.0) * 20.0) if c_docs else 0.0
        p_sums = min(20.0, (c_sums / 5.0) * 20.0) if c_sums else 0.0
        p_chats = min(20.0, (c_chats / 10.0) * 20.0) if c_chats else 0.0
        p_quizzes = min(20.0, (c_attempts / 5.0) * 20.0) if c_attempts else 0.0
        p_mastery = min(20.0, (avg_quiz_score / 100.0) * 20.0) if c_attempts > 0 else 0.0

        overall_progress = round(p_docs + p_sums + p_chats + p_quizzes + p_mastery, 1)

        # 9. Recent entities for dashboard display
        recent_docs = [dict(r) for r in conn.execute(
            "SELECT id, original_filename, total_pages, total_words, upload_date FROM documents ORDER BY upload_date DESC LIMIT 5"
        ).fetchall()]

        recent_summaries_rows = conn.execute("""
            SELECT s.id, s.document_id, s.summary_type, s.summary_text, s.created_at, d.original_filename
            FROM document_summaries s
            JOIN documents d ON s.document_id = d.id
            ORDER BY s.created_at DESC LIMIT 5
        """).fetchall()
        recent_summaries = []
        for r in recent_summaries_rows:
            item = dict(r)
            item["word_count"] = len((item.get("summary_text") or "").split())
            recent_summaries.append(item)

        recent_chats = [dict(r) for r in conn.execute("""
            SELECT m.id, m.document_id, m.content, m.created_at, d.original_filename
            FROM chat_messages m
            JOIN documents d ON m.document_id = d.id
            WHERE m.role = 'user'
            ORDER BY m.created_at DESC LIMIT 5
        """).fetchall()]

        recent_attempts = [dict(r) for r in conn.execute("""
            SELECT qa.id, qa.quiz_id, qa.document_id, qa.score, qa.total_questions,
                   qa.percentage, qa.passed, qa.completed_at, q.title AS quiz_title,
                   q.difficulty, d.original_filename
            FROM quiz_attempts qa
            JOIN quizzes q ON qa.quiz_id = q.id
            JOIN documents d ON qa.document_id = d.id
            ORDER BY qa.completed_at DESC LIMIT 5
        """).fetchall()]

        return {
            "total_documents": c_docs,
            "total_summaries": c_sums,
            "total_chat_questions": c_chats,
            "total_quizzes_generated": c_quizzes,
            "total_quizzes_completed": c_attempts,
            "average_quiz_score": avg_quiz_score,
            "total_study_sessions": total_sessions,
            "overall_progress": overall_progress,
            "progress_breakdown": {
                "documents_score": round(p_docs, 1),
                "summaries_score": round(p_sums, 1),
                "chat_score": round(p_chats, 1),
                "quizzes_score": round(p_quizzes, 1),
                "mastery_score": round(p_mastery, 1),
            },
            "recent_documents": recent_docs,
            "recent_summaries": recent_summaries,
            "recent_chats": recent_chats,
            "recent_attempts": recent_attempts,
        }
    finally:
        conn.close()


def get_performance_analytics(db_path: str | Path | None = None) -> dict[str, Any]:
    """
    Computes structured analytics datasets for Progress & Analytics:
    - Weekly study activity (Monday through Sunday for the current ISO week)
    - Quiz performance timeline (chronological scores and percentages)
    - Quiz difficulty analysis (Easy, Medium, Hard breakdown)
    - Learning statistics (highest, lowest, totals, correct/incorrect)
    - Transparent progress explanation
    """
    conn = get_db_connection(db_path)
    try:
        # A. Weekly Study Activity (Mon to Sun for current week)
        today = datetime.now(UTC).date()
        monday = today - timedelta(days=today.weekday())
        week_days = [monday + timedelta(days=i) for i in range(7)]
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_counts = []

        for d in week_days:
            d_str = d.isoformat()
            c = conn.execute("""
                SELECT COUNT(*) FROM (
                    SELECT id FROM study_activities WHERE substr(created_at, 1, 10) = ?
                    UNION ALL
                    SELECT id FROM documents WHERE substr(upload_date, 1, 10) = ?
                    UNION ALL
                    SELECT id FROM chat_messages WHERE substr(created_at, 1, 10) = ? AND role = 'user'
                    UNION ALL
                    SELECT id FROM quiz_attempts WHERE substr(completed_at, 1, 10) = ?
                )
            """, (d_str, d_str, d_str, d_str)).fetchone()[0]
            day_counts.append(int(c))

        # B. Quiz Performance History (Chronological)
        quiz_history_rows = conn.execute("""
            SELECT qa.id, qa.score, qa.total_questions, qa.percentage, qa.passed,
                   qa.completed_at, q.title AS quiz_title, q.difficulty,
                   d.original_filename
            FROM quiz_attempts qa
            JOIN quizzes q ON qa.quiz_id = q.id
            JOIN documents d ON qa.document_id = d.id
            ORDER BY qa.completed_at ASC
        """).fetchall()

        quiz_timeline = []
        for r in quiz_history_rows:
            quiz_timeline.append({
                "attempt_id": r["id"],
                "quiz_title": r["quiz_title"],
                "filename": r["original_filename"],
                "difficulty": r["difficulty"],
                "score": r["score"],
                "total_questions": r["total_questions"],
                "percentage": r["percentage"],
                "passed": bool(r["passed"]),
                "date": r["completed_at"][:10],
                "time": r["completed_at"][11:16] if len(r["completed_at"]) > 16 else "",
            })

        # C. Quiz Difficulty Breakdown (Easy, Medium, Hard)
        difficulty_breakdown = {}
        for diff in ["easy", "medium", "hard"]:
            diff_rows = conn.execute("""
                SELECT qa.percentage, qa.passed
                FROM quiz_attempts qa
                JOIN quizzes q ON qa.quiz_id = q.id
                WHERE LOWER(q.difficulty) = ?
            """, (diff,)).fetchall()
            
            d_count = len(diff_rows)
            d_passed = sum(1 for row in diff_rows if row["passed"])
            d_failed = d_count - d_passed
            d_avg = round(sum(row["percentage"] for row in diff_rows) / d_count, 1) if d_count > 0 else 0.0
            d_pass_rate = round((d_passed / d_count) * 100, 1) if d_count > 0 else 0.0

            difficulty_breakdown[diff] = {
                "attempts": d_count,
                "passed": d_passed,
                "failed": d_failed,
                "average_percentage": d_avg,
                "pass_rate": d_pass_rate,
            }

        # D. Aggregate Learning Statistics
        total_acts = conn.execute("SELECT COUNT(*) FROM study_activities").fetchone()[0]
        total_quizzes = len(quiz_history_rows)
        avg_score_row = conn.execute("SELECT AVG(percentage) FROM quiz_attempts").fetchone()[0]
        max_score_row = conn.execute("SELECT MAX(percentage) FROM quiz_attempts").fetchone()[0]
        min_score_row = conn.execute("SELECT MIN(percentage) FROM quiz_attempts").fetchone()[0]
        
        sum_correct = conn.execute("SELECT SUM(correct_answers) FROM quiz_attempts").fetchone()[0] or 0
        sum_incorrect = conn.execute("SELECT SUM(incorrect_answers) FROM quiz_attempts").fetchone()[0] or 0
        sum_unanswered = conn.execute("SELECT SUM(unanswered_count) FROM quiz_attempts").fetchone()[0] or 0

        # Overall progress
        stats = get_dashboard_statistics(db_path=db_path)

        return {
            "weekly_activity": {
                "labels": day_labels,
                "dates": [d.isoformat() for d in week_days],
                "counts": day_counts,
                "total_this_week": sum(day_counts),
            },
            "quiz_performance_history": quiz_timeline,
            "difficulty_breakdown": difficulty_breakdown,
            "learning_stats": {
                "total_activities": total_acts,
                "total_quizzes_completed": total_quizzes,
                "average_score": round(float(avg_score_row), 1) if avg_score_row is not None else 0.0,
                "highest_score": round(float(max_score_row), 1) if max_score_row is not None else 0.0,
                "lowest_score": round(float(min_score_row), 1) if min_score_row is not None else 0.0,
                "total_correct_answers": int(sum_correct),
                "total_incorrect_answers": int(sum_incorrect),
                "total_unanswered": int(sum_unanswered),
            },
            "overall_progress": stats["overall_progress"],
            "progress_breakdown": stats["progress_breakdown"],
            "progress_explanation": (
                "The Overall Learning Progress Index (0–100%) is calculated across five core academic activities: "
                "1) Document Uploads (20% for 5 PDFs), "
                "2) AI Summaries (20% for 5 summaries), "
                "3) Chatbot Inquiries (20% for 10 questions), "
                "4) Quizzes Attempted (20% for 5 quizzes), and "
                "5) Quiz Mastery (20% scaled directly by your average quiz score percentage)."
            ),
        }
    finally:
        conn.close()


def create_study_goal(
    title: str,
    goal_type: str,
    target_value: float,
    target_date: str | None = None,
    user_id: str = "default_student",
    db_path: str | Path | None = None,
) -> int:
    """
    Creates a new study goal and immediately evaluates initial progress.
    Returns the newly created goal ID.
    """
    now_iso = datetime.now(UTC).isoformat()
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO study_goals (
                    user_id, title, goal_type, target_value, current_value,
                    target_date, status, created_at
                ) VALUES (?, ?, ?, ?, 0, ?, 'in_progress', ?)
                """,
                (user_id, title.strip(), goal_type.strip(), float(target_value), target_date, now_iso),
            )
            goal_id = int(cursor.lastrowid or 0)
        # Synchronize progress immediately
        sync_study_goals_progress(db_path=db_path)
        return goal_id
    finally:
        conn.close()


def get_study_goals(
    status: str | None = None,
    user_id: str = "default_student",
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieves all study goals for the user with freshly computed progress percentages.
    """
    # First sync goal progress against live database counts
    sync_study_goals_progress(user_id=user_id, db_path=db_path)

    conn = get_db_connection(db_path)
    try:
        query = "SELECT * FROM study_goals WHERE user_id = ?"
        params: list[Any] = [user_id]
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY CASE WHEN status = 'in_progress' THEN 0 ELSE 1 END, created_at DESC"
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        result = []
        today = datetime.now(UTC).date()

        for r in rows:
            item = dict(r)
            target_val = float(item["target_value"]) if item["target_value"] else 1.0
            current_val = float(item["current_value"]) if item["current_value"] else 0.0
            progress_pct = min(100.0, round((current_val / target_val) * 100.0, 1))
            item["progress_percentage"] = progress_pct

            days_remaining = None
            if item.get("target_date"):
                try:
                    t_date = datetime.strptime(item["target_date"][:10], "%Y-%m-%d").date()
                    days_remaining = (t_date - today).days
                except (ValueError, TypeError):
                    days_remaining = None
            item["days_remaining"] = days_remaining
            result.append(item)
        return result
    finally:
        conn.close()


def get_study_goal(
    goal_id: int,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Retrieves a single study goal by its primary key ID."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute("SELECT * FROM study_goals WHERE id = ?", (goal_id,))
        row = cursor.fetchone()
        if not row:
            return None
        item = dict(row)
        target_val = float(item["target_value"]) if item["target_value"] else 1.0
        current_val = float(item["current_value"]) if item["current_value"] else 0.0
        item["progress_percentage"] = min(100.0, round((current_val / target_val) * 100.0, 1))
        return item
    finally:
        conn.close()


def sync_study_goals_progress(
    user_id: str = "default_student",
    db_path: str | Path | None = None,
) -> None:
    """
    Evaluates current real database metrics for all in-progress study goals.
    Automatically marks goals as completed if target_value is reached.
    """
    conn = get_db_connection(db_path)
    try:
        # Fetch live metric counters
        c_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        c_sums = conn.execute("SELECT COUNT(*) FROM document_summaries").fetchone()[0]
        c_chats = conn.execute("SELECT COUNT(*) FROM chat_messages WHERE role = 'user'").fetchone()[0]
        c_attempts = conn.execute("SELECT COUNT(*) FROM quiz_attempts").fetchone()[0]
        avg_row = conn.execute("SELECT AVG(percentage) FROM quiz_attempts").fetchone()[0]
        avg_score = round(float(avg_row), 1) if avg_row is not None else 0.0

        session_row = conn.execute("""
            SELECT COUNT(DISTINCT activity_date) FROM (
                SELECT substr(created_at, 1, 10) AS activity_date FROM study_activities
                UNION
                SELECT substr(upload_date, 1, 10) AS activity_date FROM documents
                UNION
                SELECT substr(created_at, 1, 10) AS activity_date FROM chat_messages
                UNION
                SELECT substr(completed_at, 1, 10) AS activity_date FROM quiz_attempts
            ) WHERE activity_date IS NOT NULL
        """).fetchone()[0]
        c_sessions = int(session_row or 0)

        goals = conn.execute("SELECT id, goal_type, target_value, status FROM study_goals WHERE user_id = ?", (user_id,)).fetchall()
        now_iso = datetime.now(UTC).isoformat()

        with conn:
            for g in goals:
                g_id = g["id"]
                g_type = g["goal_type"]
                target = float(g["target_value"])
                current = 0.0

                if g_type == "upload_documents":
                    current = float(c_docs)
                elif g_type == "generate_summaries":
                    current = float(c_sums)
                elif g_type == "complete_quizzes":
                    current = float(c_attempts)
                elif g_type == "quiz_score":
                    current = float(avg_score)
                elif g_type == "study_days":
                    current = float(c_sessions)
                elif g_type == "chat_questions":
                    current = float(c_chats)
                else:
                    # Custom goal, leave current_value as is
                    continue

                is_completed = current >= target
                new_status = "completed" if is_completed else "in_progress"
                completed_at_val = now_iso if (is_completed and g["status"] != "completed") else None

                if completed_at_val:
                    conn.execute("""
                        UPDATE study_goals
                        SET current_value = ?, status = ?, completed_at = ?
                        WHERE id = ?
                    """, (current, new_status, completed_at_val, g_id))
                else:
                    conn.execute("""
                        UPDATE study_goals
                        SET current_value = ?, status = ?
                        WHERE id = ?
                    """, (current, new_status, g_id))
    finally:
        conn.close()


def update_study_goal(
    goal_id: int,
    title: str | None = None,
    target_value: float | None = None,
    target_date: str | None = None,
    status: str | None = None,
    current_value: float | None = None,
    db_path: str | Path | None = None,
) -> bool:
    """Updates fields of an existing study goal."""
    conn = get_db_connection(db_path)
    try:
        fields = []
        params = []
        if title is not None:
            fields.append("title = ?")
            params.append(title.strip())
        if target_value is not None:
            fields.append("target_value = ?")
            params.append(float(target_value))
        if target_date is not None:
            fields.append("target_date = ?")
            params.append(target_date)
        if status is not None:
            fields.append("status = ?")
            params.append(status)
            if status == "completed":
                fields.append("completed_at = ?")
                params.append(datetime.now(UTC).isoformat())
        if current_value is not None:
            fields.append("current_value = ?")
            params.append(float(current_value))

        if not fields:
            return False

        params.append(goal_id)
        with conn:
            cursor = conn.execute(f"UPDATE study_goals SET {', '.join(fields)} WHERE id = ?", params)
            return cursor.rowcount > 0
    finally:
        conn.close()


def delete_study_goal(
    goal_id: int,
    db_path: str | Path | None = None,
) -> bool:
    """Deletes a study goal by primary key ID."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM study_goals WHERE id = ?", (goal_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def backfill_activities_if_empty(db_path: str | Path | None = None) -> int:
    """
    Backfills study_activities table from existing database records if currently empty.
    Ensures existing documents, summaries, chats, and quiz attempts appear immediately
    in the dashboard and activity history without requiring re-execution.
    """
    conn = get_db_connection(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM study_activities").fetchone()[0]
        if count > 0:
            return 0  # Already populated

        backfilled = 0
        with conn:
            # 1. Documents
            docs = conn.execute("SELECT id, original_filename, total_pages, total_words, upload_date FROM documents ORDER BY upload_date ASC").fetchall()
            for d in docs:
                conn.execute("""
                    INSERT INTO study_activities (user_id, activity_type, title, description, document_id, metadata_json, created_at)
                    VALUES ('default_student', 'upload', ?, ?, ?, ?, ?)
                """, (
                    f"Uploaded '{d['original_filename']}'",
                    f"Processed {d['total_pages']} pages, {d['total_words']} words.",
                    d["id"],
                    json.dumps({"total_pages": d["total_pages"], "total_words": d["total_words"]}),
                    d["upload_date"],
                ))
                backfilled += 1

            # 2. Summaries
            sums = conn.execute("""
                SELECT s.id, s.document_id, s.summary_type, s.summary_text, s.created_at, d.original_filename
                FROM document_summaries s
                JOIN documents d ON s.document_id = d.id
                ORDER BY s.created_at ASC
            """).fetchall()
            for s in sums:
                w_count = len((s["summary_text"] or "").split())
                conn.execute("""
                    INSERT INTO study_activities (user_id, activity_type, title, description, document_id, metadata_json, created_at)
                    VALUES ('default_student', 'summarize', ?, ?, ?, ?, ?)
                """, (
                    f"Generated {s['summary_type'].capitalize()} Summary",
                    f"Created {s['summary_type']} study summary for '{s['original_filename']}' ({w_count} words).",
                    s["document_id"],
                    json.dumps({"summary_type": s["summary_type"], "word_count": w_count}),
                    s["created_at"],
                ))
                backfilled += 1

            # 3. Chat Messages (user questions)
            chats = conn.execute("""
                SELECT m.id, m.document_id, m.content, m.created_at, d.original_filename
                FROM chat_messages m
                JOIN documents d ON m.document_id = d.id
                WHERE m.role = 'user'
                ORDER BY m.created_at ASC
            """).fetchall()
            for c in chats:
                snippet = c["content"][:60] + "..." if len(c["content"]) > 60 else c["content"]
                conn.execute("""
                    INSERT INTO study_activities (user_id, activity_type, title, description, document_id, metadata_json, created_at)
                    VALUES ('default_student', 'chat', ?, ?, ?, ?, ?)
                """, (
                    "Asked Study Chatbot",
                    f"Question on '{c['original_filename']}': \"{snippet}\"",
                    c["document_id"],
                    json.dumps({"content_snippet": snippet}),
                    c["created_at"],
                ))
                backfilled += 1

            # 4. Quizzes
            quizzes = conn.execute("""
                SELECT q.id, q.document_id, q.title, q.difficulty, q.total_questions, q.created_at, d.original_filename
                FROM quizzes q
                JOIN documents d ON q.document_id = d.id
                ORDER BY q.created_at ASC
            """).fetchall()
            for q in quizzes:
                conn.execute("""
                    INSERT INTO study_activities (user_id, activity_type, title, description, document_id, metadata_json, created_at)
                    VALUES ('default_student', 'quiz_generate', ?, ?, ?, ?, ?)
                """, (
                    f"Generated Quiz: {q['title']}",
                    f"Created {q['total_questions']}-question {q['difficulty'].capitalize()} quiz for '{q['original_filename']}'.",
                    q["document_id"],
                    json.dumps({"difficulty": q["difficulty"], "total_questions": q["total_questions"]}),
                    q["created_at"],
                ))
                backfilled += 1

            # 5. Quiz Attempts
            attempts = conn.execute("""
                SELECT qa.id, qa.quiz_id, qa.document_id, qa.score, qa.total_questions,
                       qa.percentage, qa.passed, qa.completed_at, q.title AS quiz_title,
                       d.original_filename
                FROM quiz_attempts qa
                JOIN quizzes q ON qa.quiz_id = q.id
                JOIN documents d ON qa.document_id = d.id
                ORDER BY qa.completed_at ASC
            """).fetchall()
            for a in attempts:
                conn.execute("""
                    INSERT INTO study_activities (user_id, activity_type, title, description, document_id, metadata_json, created_at)
                    VALUES ('default_student', 'quiz_attempt', ?, ?, ?, ?, ?)
                """, (
                    f"Completed Quiz: {a['quiz_title']}",
                    f"Scored {a['score']}/{a['total_questions']} ({a['percentage']}%) - {'Passed' if a['passed'] else 'Needs Review'}.",
                    a["document_id"],
                    json.dumps({"attempt_id": a["id"], "score": a["score"], "percentage": a["percentage"], "passed": bool(a["passed"])}),
                    a["completed_at"],
                ))
                backfilled += 1

        return backfilled
    finally:
        conn.close()




